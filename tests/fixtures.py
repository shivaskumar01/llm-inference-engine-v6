"""Test fixtures: synthetic Llama checkpoints for kernel + forward equality.

The tiny Llama is randomly initialized (seeded) and used as the daily
correctness gate. It runs in milliseconds end-to-end, so we can iterate on
kernels and the forward pass without touching the real 1B model.
"""
from __future__ import annotations

from pathlib import Path

import safetensors.torch
import torch
from transformers import LlamaConfig, LlamaForCausalLM


TINY_CONFIG_KWARGS = dict(
    hidden_size=128,
    intermediate_size=384,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    vocab_size=512,
    max_position_embeddings=256,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    # original_max_position_embeddings must be < max_position_embeddings
    # in HF's validator; at this scale the scaling math is exercised the
    # same way regardless of the absolute number.
    rope_scaling={
        "rope_type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 128,
    },
    tie_word_embeddings=False,    # avoid safetensors shared-storage friction
    eos_token_id=[1, 2, 3],
    torch_dtype=torch.float32,
)


def make_tiny_llama_weights(out_dir: Path | str, seed: int = 42
                            ) -> tuple[LlamaConfig, LlamaForCausalLM]:
    """Build + save a tiny Llama and return (config, hf_model_in_eval_mode)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    cfg = LlamaConfig(**TINY_CONFIG_KWARGS)
    model = LlamaForCausalLM(cfg).eval()

    safetensors.torch.save_file(model.state_dict(), str(out_dir / "model.safetensors"))
    cfg.save_pretrained(str(out_dir))
    return cfg, model
