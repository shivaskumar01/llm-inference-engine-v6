"""Phase 1: per-kernel equality vs torch FP32 reference (atol=1e-5).

Each kernel is exercised through the pybind11 ``llmengine._llmengine.kernels``
submodule, exposed for testing only. The torch reference is FP32 throughout
so we don't need to make any precision allowance for dtype mismatch.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from llmengine._llmengine import kernels as K


# ----- RMSNorm ---------------------------------------------------------------

def torch_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * w


@pytest.mark.parametrize("dim", [16, 64, 128, 257])
def test_rmsnorm(dim: int) -> None:
    rng = np.random.default_rng(0xA5A5 ^ dim)
    x = rng.standard_normal(dim, dtype=np.float32)
    w = rng.standard_normal(dim, dtype=np.float32)
    eps = 1e-5

    got = K.rmsnorm_f32(x, w, eps)
    want = torch_rmsnorm(torch.from_numpy(x), torch.from_numpy(w), eps).numpy()
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


# ----- SiLU ------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 16, 384])
def test_silu(n: int) -> None:
    rng = np.random.default_rng(0x5151 ^ n)
    x = rng.standard_normal(n, dtype=np.float32) * 3.0   # cover broader range

    got = K.silu_f32(x)
    want = F.silu(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


# ----- Matmul ---------------------------------------------------------------

@pytest.mark.parametrize("M,N,Kdim", [(1, 8, 16), (1, 128, 128), (4, 64, 32)])
def test_matmul(M: int, N: int, Kdim: int) -> None:
    rng = np.random.default_rng(M * 1000 + N * 10 + Kdim)
    a = rng.standard_normal((M, Kdim), dtype=np.float32)
    w = rng.standard_normal((N, Kdim), dtype=np.float32)

    got = K.matmul_f32(a, w)
    # torch convention: F.linear(x, W) computes x @ W.T  with W: [out, in]
    want = F.linear(torch.from_numpy(a), torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-5)


# ----- Embedding lookup -----------------------------------------------------

def test_embed_lookup() -> None:
    rng = np.random.default_rng(7)
    embedding = rng.standard_normal((128, 32), dtype=np.float32)
    for token in [0, 1, 7, 64, 127]:
        got = K.embed_lookup_f32(embedding, token)
        np.testing.assert_array_equal(got, embedding[token])


# ----- RoPE: inv_freq + sin/cos tables vs HF --------------------------------

def _hf_rope_cos_sin(cfg, position_ids):
    """Mirror transformers' LlamaRotaryEmbedding output for a position sequence."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    rope = LlamaRotaryEmbedding(config=cfg)
    # The module needs an `x` for dtype/device. Use a dummy.
    x = torch.zeros(1, 1, cfg.head_dim)
    cos, sin = rope(x, position_ids[None, :])
    return cos[0].numpy(), sin[0].numpy()


@pytest.mark.parametrize("head_dim", [32, 64])
def test_llama3_rope_tables_match_hf(head_dim: int) -> None:
    from transformers import LlamaConfig

    cfg = LlamaConfig(
        hidden_size=head_dim * 4, head_dim=head_dim,
        num_attention_heads=4, num_key_value_heads=2,
        num_hidden_layers=1, intermediate_size=32,
        vocab_size=64, max_position_embeddings=256,
        rope_theta=500000.0,
        rope_scaling={"rope_type": "llama3", "factor": 32.0,
                      "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                      "original_max_position_embeddings": 8192},
    )

    inv_freq = K.llama3_inv_freq(head_dim, 500000.0, True, 32.0, 1.0, 4.0, 8192)
    cos_t, sin_t = K.build_rope_tables(inv_freq, 64, head_dim)

    positions = torch.arange(64, dtype=torch.long)
    hf_cos, hf_sin = _hf_rope_cos_sin(cfg, positions)

    np.testing.assert_allclose(cos_t, hf_cos, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(sin_t, hf_sin, atol=1e-5, rtol=1e-5)


# ----- RoPE apply: rotate_half formulation matches HF ------------------------

def _hf_rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _hf_apply_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return q * cos + _hf_rotate_half(q) * sin


def test_apply_rope_matches_hf() -> None:
    head_dim = 32
    rng = np.random.default_rng(99)

    inv_freq = K.llama3_inv_freq(head_dim, 500000.0, True, 32.0, 1.0, 4.0, 8192)
    cos_t, sin_t = K.build_rope_tables(inv_freq, 16, head_dim)

    for pos in [0, 1, 5, 15]:
        q = rng.standard_normal(head_dim, dtype=np.float32)
        got = K.apply_rope_f32(q, cos_t[pos], sin_t[pos])
        want = _hf_apply_rope(torch.from_numpy(q),
                              torch.from_numpy(cos_t[pos]),
                              torch.from_numpy(sin_t[pos])).numpy()
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


# ----- Attention (GQA) ------------------------------------------------------

def torch_gqa_attention(q, K_hist, V_hist):
    """Reference scalar GQA attention matching our kernel's contract.

    q:      [num_q, hd]
    K/V:    [seq_len, num_kv, hd]
    Returns [num_q, hd].
    """
    num_q, hd = q.shape
    seq_len, num_kv, _ = K_hist.shape
    group_size = num_q // num_kv
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(hd)
    for h in range(num_q):
        kv_h = h // group_size
        scores = (K_hist[:, kv_h] @ q[h]) * scale         # [seq_len]
        attn = F.softmax(scores, dim=0)                   # [seq_len]
        out[h] = (attn[:, None] * V_hist[:, kv_h]).sum(dim=0)
    return out


# ----- INT8 W8A32 matmul vs torch FP32 dequant reference --------------------

def _quantize_per_row(w_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric INT8 quantization matching model.cpp."""
    max_abs = np.abs(w_fp32).max(axis=1)
    scales = np.where(max_abs > 1e-12, max_abs / 127.0, 1.0).astype(np.float32)
    w_q = np.rint(w_fp32 / scales[:, None]).clip(-128, 127).astype(np.int8)
    return w_q, scales


@pytest.mark.parametrize("M,N,Kdim", [(1, 8, 16), (1, 128, 128), (4, 64, 32)])
def test_matmul_int8w_f32a(M: int, N: int, Kdim: int) -> None:
    rng = np.random.default_rng(M * 1000 + N * 10 + Kdim)
    a = rng.standard_normal((M, Kdim), dtype=np.float32)
    w_fp32 = rng.standard_normal((N, Kdim), dtype=np.float32)

    w_q, scales_f32 = _quantize_per_row(w_fp32)
    scales_f16_bits = np.frombuffer(
        scales_f32.astype(np.float16).tobytes(), dtype=np.uint16)

    # Reference: torch FP32 matmul on the dequantized weights.
    w_dq = w_q.astype(np.float32) * scales_f32[:, None]
    want = F.linear(torch.from_numpy(a), torch.from_numpy(w_dq)).numpy()

    got = K.matmul_int8w_f32a(a, w_q, scales_f16_bits)
    np.testing.assert_allclose(got, want, atol=1e-3, rtol=1e-3)

    if hasattr(K, "matmul_int8w_f32a_neon"):
        got_neon = K.matmul_int8w_f32a_neon(a, w_q, scales_f16_bits)
        np.testing.assert_allclose(got_neon, want, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("num_q,num_kv,hd,seq_len",
                         [(4, 2, 32, 1), (4, 2, 32, 7), (8, 2, 16, 13)])
def test_attention_gqa(num_q: int, num_kv: int, hd: int, seq_len: int) -> None:
    rng = np.random.default_rng((num_q << 16) | (num_kv << 8) | seq_len)
    q       = rng.standard_normal((num_q, hd),               dtype=np.float32)
    K_hist  = rng.standard_normal((seq_len, num_kv, hd),     dtype=np.float32)
    V_hist  = rng.standard_normal((seq_len, num_kv, hd),     dtype=np.float32)

    got = K.attention_f32(q, K_hist, V_hist, num_q, num_kv, hd)
    want = torch_gqa_attention(torch.from_numpy(q),
                               torch.from_numpy(K_hist),
                               torch.from_numpy(V_hist)).numpy()
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


def test_attention_gqa_long_context_threaded() -> None:
    """Long context (seq_len=512, 32 q-heads, real Llama-1B attention shape)
    pushes attention_f32 past the parallel-work threshold, so the head loop
    runs across the thread pool. Heads write disjoint out[h] slices, so the
    threaded result must still match the torch GQA reference. The looser atol
    vs the short cases is only because a 512-term softmax + weighted sum
    accumulates more FP error, the head split itself is bit-identical to
    serial."""
    num_q, num_kv, hd, seq_len = 32, 8, 64, 512
    rng = np.random.default_rng(20250530)
    q      = rng.standard_normal((num_q, hd),           dtype=np.float32)
    K_hist = rng.standard_normal((seq_len, num_kv, hd), dtype=np.float32)
    V_hist = rng.standard_normal((seq_len, num_kv, hd), dtype=np.float32)

    got  = K.attention_f32(q, K_hist, V_hist, num_q, num_kv, hd)
    want = torch_gqa_attention(torch.from_numpy(q),
                               torch.from_numpy(K_hist),
                               torch.from_numpy(V_hist)).numpy()
    np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-4)
