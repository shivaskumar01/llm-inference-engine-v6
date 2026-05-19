"""Phase 2: smoke test on the real Llama 3.2 1B-Instruct checkpoint.

This is the only place we touch real 1B in correctness mode — scalar FP32
forward of a real 1B model takes a few seconds per call. The full Phase 1
test suite stays on the tiny fixture for fast iteration.

Skipped if the model is not downloaded. Run
    python tools/download_weights.py --mirror unsloth
to pull the ungated mirror.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llmengine
from llmengine._llmengine import kernels as K


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "data" / "llama-3.2-1b-instruct"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists(),
    reason=f"Real Llama 3.2 1B-Instruct not downloaded at {MODEL_DIR}",
)


@pytest.fixture(scope="module")
def real_engine() -> llmengine.Engine:
    return llmengine.Engine(str(MODEL_DIR))


@pytest.fixture(scope="module")
def hf_model():
    return AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    ).eval()


@pytest.fixture(scope="module")
def hf_tokenizer():
    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


# ----- Loader: real-1B config + tied LM head alias --------------------------

def test_config_loads(real_engine: llmengine.Engine) -> None:
    cfg = real_engine.cfg
    assert cfg.hidden_size == 2048
    assert cfg.intermediate_size == 8192
    assert cfg.num_hidden_layers == 16
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 64
    assert cfg.vocab_size == 128256
    assert cfg.tie_word_embeddings is True
    assert cfg.rope_theta == pytest.approx(500000.0)
    assert cfg.rope_has_scaling is True
    assert cfg.rope_factor == pytest.approx(32.0)
    assert cfg.rope_low_freq_factor == pytest.approx(1.0)
    assert cfg.rope_high_freq_factor == pytest.approx(4.0)
    assert cfg.rope_original_max_pos == 8192


def test_eos_set_merged_from_generation_config(real_engine: llmengine.Engine) -> None:
    # Llama 3.2 1B-Instruct EOS set: <|end_of_text|>, <|eom_id|>, <|eot_id|>.
    assert real_engine.cfg.eos_token_ids == {128001, 128008, 128009}


@pytest.mark.skipif(
    not hasattr(llmengine.Engine, "_debug_weight_ptr"),
    reason="_debug_weight_ptr disabled in perf build (LLMENGINE_DEBUG_BINDINGS=OFF)",
)
def test_tied_lm_head_alias(real_engine: llmengine.Engine) -> None:
    """Real-1B saves only model.embed_tokens.weight (tied); the loader must
    alias lm_head.weight to the same buffer."""
    embed_ptr = real_engine._debug_weight_ptr("model.embed_tokens.weight")
    lm_ptr    = real_engine._debug_weight_ptr("lm_head.weight")
    assert embed_ptr == lm_ptr


@pytest.mark.skipif(
    not hasattr(llmengine.Engine, "_debug_lm_head_f32_ptr"),
    reason="_debug_lm_head_f32_ptr disabled in perf build",
)
def test_tied_lm_head_shares_buffer_at_runtime(real_engine: llmengine.Engine) -> None:
    """Pre-fix, ModelWeightsRef::load copied embed_tokens into a separate
    lm_head_f32_ buffer — ~1 GB redundant on real 1B. After the fix, the
    F32 lm_head pointer aliases embed_tokens. Without this test the
    raw-checkpoint alias passes (because of the WeightMap fixup) while the
    runtime materialization silently duplicates the buffer."""
    embed_runtime = real_engine._debug_embed_tokens_ptr()
    lm_runtime    = real_engine._debug_lm_head_f32_ptr()
    assert embed_runtime == lm_runtime, (
        "tied F32 lm_head is materialized as a separate buffer instead of "
        "aliasing embed_tokens — this is the ~1 GB memory regression"
    )


# ----- RoPE sin/cos validated against HF at multiple positions --------------

def test_rope_tables_match_hf_at_positions() -> None:
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    cfg = LlamaConfig.from_pretrained(str(MODEL_DIR))
    rope = LlamaRotaryEmbedding(config=cfg)

    inv_freq = K.llama3_inv_freq(
        cfg.head_dim, 500000.0, True, 32.0, 1.0, 4.0, 8192,
    )
    # Build cos/sin for a wide-enough range to cover the asserts below.
    max_pos = 8200
    cos_t, sin_t = K.build_rope_tables(inv_freq, max_pos, cfg.head_dim)

    positions = torch.tensor([0, 100, 1000, 8000], dtype=torch.long)
    x = torch.zeros(1, 1, cfg.head_dim)
    hf_cos, hf_sin = rope(x, positions[None, :])
    hf_cos = hf_cos[0].numpy()
    hf_sin = hf_sin[0].numpy()

    # Tolerance widens with position. Our inv_freq is computed in FP64 then
    # cast to FP32, while HF stays FP32 throughout (including `torch.pow`,
    # whose libm path differs from std::pow). The HF FP32 angle for large p
    # carries ~p × FP32-epsilon ≈ 1e-7·p ULPs of accumulated error; our
    # values are more precise, this test just verifies structural agreement.
    for i, p in enumerate(positions.tolist()):
        atol = max(1e-6, 1e-7 * (p if p > 0 else 1))
        np.testing.assert_allclose(cos_t[p], hf_cos[i], atol=atol, rtol=atol,
            err_msg=f"cos mismatch at position {p}")
        np.testing.assert_allclose(sin_t[p], hf_sin[i], atol=atol, rtol=atol,
            err_msg=f"sin mismatch at position {p}")


# ----- Single-prompt forward smoke vs HF FP32 -------------------------------

def test_short_prompt_logits(real_engine: llmengine.Engine,
                             hf_model,
                             hf_tokenizer) -> None:
    prompt = "The capital of France is"
    ids = hf_tokenizer.encode(prompt, add_special_tokens=True)
    assert len(ids) >= 4

    ours = real_engine.forward_logits(ids)

    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long)
        theirs = hf_model(x).logits[0].float().numpy()

    assert ours.shape == theirs.shape

    # Per-position checks. Tight on top-1, set-equality on top-5, max-abs
    # reported as a debug stat.
    max_abs_seen = 0.0
    for t in range(len(ids)):
        ours_top5   = set(np.argsort(ours[t])[-5:].tolist())
        theirs_top5 = set(np.argsort(theirs[t])[-5:].tolist())
        assert ours_top5 == theirs_top5, (
            f"top-5 set differs at position {t}: ours={sorted(ours_top5)}, "
            f"hf={sorted(theirs_top5)}"
        )
        assert int(np.argmax(ours[t])) == int(np.argmax(theirs[t])), (
            f"top-1 differs at position {t}"
        )
        max_abs_seen = max(max_abs_seen, float(np.max(np.abs(ours[t] - theirs[t]))))

    print(f"\n[real_smoke] T={len(ids)}, max_abs_logit_diff={max_abs_seen:.3e}")
