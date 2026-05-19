"""Phase 0: prove the loader correctly parses config.json and a safetensors file.

Uses a hand-crafted synthetic checkpoint so this test does not depend on the
gated meta-llama HF download. Real-1B integration lives in test_real_smoke.py.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import safetensors.numpy
import safetensors.torch
import torch

import llmengine


LLAMA_3_2_1B_LIKE_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "tie_word_embeddings": True,
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },
    "eos_token_id": [128001, 128008, 128009],
}


def _write_minimal_checkpoint(out_dir: Path, *, tied: bool) -> None:
    """Write config.json + a tiny model.safetensors mimicking llama field names."""
    cfg = dict(LLAMA_3_2_1B_LIKE_CONFIG)
    cfg["tie_word_embeddings"] = tied
    (out_dir / "config.json").write_text(json.dumps(cfg))

    # Tiny tensor shapes for the loader test — we only check parsing/aliasing,
    # not values. Two layers' worth of names.
    tensors = {
        "model.embed_tokens.weight":          np.zeros((8, 4),  dtype=np.float16),
        "model.layers.0.input_layernorm.weight": np.ones((4,),  dtype=np.float16),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((4, 4), dtype=np.float16),
        "model.norm.weight":                  np.ones((4,),  dtype=np.float16),
    }
    if not tied:
        tensors["lm_head.weight"] = np.zeros((8, 4), dtype=np.float16)

    safetensors.numpy.save_file(tensors, str(out_dir / "model.safetensors"))


def test_config_fields_parse(tmp_path: Path) -> None:
    _write_minimal_checkpoint(tmp_path, tied=True)
    e = llmengine.Engine(str(tmp_path))

    assert e.cfg.hidden_size == 2048
    assert e.cfg.num_hidden_layers == 16
    assert e.cfg.num_attention_heads == 32
    assert e.cfg.num_key_value_heads == 8
    assert e.cfg.head_dim == 64
    assert e.cfg.vocab_size == 128256
    assert e.cfg.tie_word_embeddings is True

    # Multi-EOS — Phase 0 deliverable: 3-element set.
    assert e.cfg.eos_token_ids == {128001, 128008, 128009}

    # RoPE scaling parsed.
    assert e.cfg.rope_has_scaling
    assert e.cfg.rope_factor == pytest.approx(32.0)
    assert e.cfg.rope_low_freq_factor == pytest.approx(1.0)
    assert e.cfg.rope_high_freq_factor == pytest.approx(4.0)
    assert e.cfg.rope_original_max_pos == 8192


def test_weight_names_load(tmp_path: Path) -> None:
    _write_minimal_checkpoint(tmp_path, tied=True)
    e = llmengine.Engine(str(tmp_path))
    names = e.weight_names()
    assert "model.embed_tokens.weight" in names
    assert "model.norm.weight" in names
    assert e.has_weight("model.layers.0.self_attn.q_proj.weight")


_HAS_DEBUG_PTR = hasattr(llmengine.Engine, "_debug_weight_ptr")
_DEBUG_PTR_SKIP = pytest.mark.skipif(
    not _HAS_DEBUG_PTR,
    reason="_debug_weight_ptr binding disabled (LLMENGINE_DEBUG_BINDINGS=OFF)",
)


@_DEBUG_PTR_SKIP
def test_tied_lm_head_alias_when_omitted(tmp_path: Path) -> None:
    """tie_word_embeddings=True with no lm_head.weight in the checkpoint:
    loader must alias lm_head.weight to model.embed_tokens.weight."""
    _write_minimal_checkpoint(tmp_path, tied=True)
    e = llmengine.Engine(str(tmp_path))

    embed_ptr = e._debug_weight_ptr("model.embed_tokens.weight")
    lm_ptr    = e._debug_weight_ptr("lm_head.weight")
    assert embed_ptr == lm_ptr, "tied alias not built when lm_head.weight is absent"


@_DEBUG_PTR_SKIP
def test_tied_lm_head_alias_when_present(tmp_path: Path) -> None:
    """tie_word_embeddings=True with lm_head.weight present: loader should
    alias to embed (rather than keep an independent copy) for memory savings."""
    _write_minimal_checkpoint(tmp_path, tied=True)
    # Write an lm_head.weight too, with the same shape/dtype.
    import safetensors.torch as st
    payload = st.load_file(str(tmp_path / "model.safetensors"))
    payload["lm_head.weight"] = torch.zeros((8, 4), dtype=torch.float16)
    st.save_file(payload, str(tmp_path / "model.safetensors"))

    e = llmengine.Engine(str(tmp_path))
    embed_ptr = e._debug_weight_ptr("model.embed_tokens.weight")
    lm_ptr    = e._debug_weight_ptr("lm_head.weight")
    assert embed_ptr == lm_ptr, "loader did not alias lm_head to embed"


def test_safetensors_byte_count_mismatch_rejected(tmp_path: Path) -> None:
    """A tensor whose data_offsets cover fewer bytes than shape*dtype must be
    rejected — before the fix the loader accepted shape=[8,4] dtype=F16
    (64 bytes expected) with only 2 bytes of data."""
    import json
    import struct

    out = tmp_path / "evil_size"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(LLAMA_3_2_1B_LIKE_CONFIG))

    # data section is 2 bytes total; header declares an 8x4 F16 tensor
    # (= 64 bytes). data_offsets [0, 2] fits in the data section but
    # disagrees with shape*dtype.
    tail = b"\x00\x00"
    header = {
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [8, 4],
            "data_offsets": [0, 2],
        }
    }
    header_bytes = json.dumps(header).encode()
    body = struct.pack("<Q", len(header_bytes)) + header_bytes + tail
    (out / "model.safetensors").write_bytes(body)

    with pytest.raises(RuntimeError, match="byte-count mismatch"):
        llmengine.Engine(str(out))


def test_safetensors_negative_shape_dim_rejected(tmp_path: Path) -> None:
    """A negative shape dim must be rejected. Numel would wrap when treated
    as uint64 and could pass the byte-count check."""
    import json
    import struct

    out = tmp_path / "evil_dim"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(LLAMA_3_2_1B_LIKE_CONFIG))

    header = {
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [-1, 4],
            "data_offsets": [0, 0],
        }
    }
    header_bytes = json.dumps(header).encode()
    body = struct.pack("<Q", len(header_bytes)) + header_bytes
    (out / "model.safetensors").write_bytes(body)

    with pytest.raises(RuntimeError, match="negative shape"):
        llmengine.Engine(str(out))


def test_safetensors_offset_past_data_section_rejected(tmp_path: Path) -> None:
    """data_offsets are relative to the data section (file size minus 8
    minus header_len), not the whole file. A malformed file pointing past
    the data section but still inside the file must be rejected — the
    previous bound was too permissive."""
    import json
    import struct

    # Build a minimal safetensors file with one tensor whose data_offsets
    # exceed the data section but fall inside the overall file size.
    out = tmp_path / "evil"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(LLAMA_3_2_1B_LIKE_CONFIG))

    # 8 bytes of data section ("tail" padding), header references offsets
    # 0..16 which is past data_section_bytes=8 but inside the file size.
    tail = b"\x00" * 8
    header = {
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [8, 4],
            "data_offsets": [0, 16],   # past the 8-byte data section
        }
    }
    header_bytes = json.dumps(header).encode()
    body = struct.pack("<Q", len(header_bytes)) + header_bytes + tail
    (out / "model.safetensors").write_bytes(body)

    with pytest.raises(RuntimeError, match="offsets out of range"):
        llmengine.Engine(str(out))


@_DEBUG_PTR_SKIP
def test_untied_lm_head_keeps_independent_buffer(tmp_path: Path) -> None:
    _write_minimal_checkpoint(tmp_path, tied=False)
    e = llmengine.Engine(str(tmp_path))
    embed_ptr = e._debug_weight_ptr("model.embed_tokens.weight")
    lm_ptr    = e._debug_weight_ptr("lm_head.weight")
    assert embed_ptr != lm_ptr, "untied checkpoint should NOT alias"
