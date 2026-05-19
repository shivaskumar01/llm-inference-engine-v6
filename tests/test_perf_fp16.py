"""Phase 4.2: FP16 weight storage drift gates.

The engine's FP32 path is the reference; the FP16 path stores linear weights
as __fp16 and consumes them via matmul_f16w_f32a_neon. Weights are quantized
to FP16 at load time, so we expect ~FP16-ULP perturbation in logits — small
but measurable. Tests assert top-1 stability and a bounded max-abs drift,
matching the v6 plan's "soft gates" framing for Phase 4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import llmengine
from fixtures import make_tiny_llama_weights


@pytest.fixture(scope="module")
def tiny_paths(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny_fp16")
    make_tiny_llama_weights(out)
    return str(out)


def test_tiny_fp16_top1_match_fp32(tiny_paths) -> None:
    """FP16 storage should reproduce FP32's argmax on every position of a
    short random sequence (tiny weights are small enough that FP16 rounding
    rarely flips the top-1)."""
    eng_f32 = llmengine.Engine(tiny_paths, dtype="fp32")
    eng_f16 = llmengine.Engine(tiny_paths, dtype="fp16")

    rng = np.random.default_rng(0)
    for s in range(20):
        T = int(rng.integers(2, 16))
        ids = rng.integers(0, eng_f32.cfg.vocab_size, size=T).tolist()
        l32 = eng_f32.forward_logits(ids)
        l16 = eng_f16.forward_logits(ids)
        for t in range(T):
            assert int(np.argmax(l32[t])) == int(np.argmax(l16[t])), (
                f"seq {s} pos {t}: top-1 diverged"
            )


def test_tiny_fp16_drift_bounded(tiny_paths) -> None:
    """Per-position max-abs logit drift should be well-bounded for tiny
    weights. Reported as a debug stat so we can see if it ever blows up."""
    eng_f32 = llmengine.Engine(tiny_paths, dtype="fp32")
    eng_f16 = llmengine.Engine(tiny_paths, dtype="fp16")

    rng = np.random.default_rng(1)
    T = 8
    ids = rng.integers(0, eng_f32.cfg.vocab_size, size=T).tolist()

    l32 = eng_f32.forward_logits(ids)
    l16 = eng_f16.forward_logits(ids)

    max_abs = float(np.max(np.abs(l32 - l16)))
    print(f"\n[tiny_fp16_drift] T={T}, max_abs={max_abs:.3e}")
    # FP16 mantissa is 10 bits; weights round to ~5e-4 relative. Logits are
    # O(1) magnitudes, so drift up to ~1e-2 is plausible.
    assert max_abs < 5e-2


# ----- Real-1B drift: top-5 set equality is the v6 Phase 4 soft gate ---------

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL_DIR = REPO_ROOT / "data" / "llama-3.2-1b-instruct"


@pytest.mark.skipif(
    not (REAL_MODEL_DIR / "model.safetensors").exists(),
    reason="Real Llama 3.2 1B-Instruct not downloaded",
)
def test_real_1b_fp16_short_prompt_top5() -> None:
    """FP16 storage of Llama 3.2 1B-Instruct: top-5 set equality on the
    capital-of-France prompt. Top-1 should match exactly for a short prompt
    where the model is confident."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REAL_MODEL_DIR))
    ids = tok.encode("The capital of France is", add_special_tokens=True)

    eng_f32 = llmengine.Engine(str(REAL_MODEL_DIR), dtype="fp32")
    l32 = eng_f32.forward_logits(ids)

    eng_f16 = llmengine.Engine(str(REAL_MODEL_DIR), dtype="fp16")
    l16 = eng_f16.forward_logits(ids)

    max_abs = 0.0
    for t in range(len(ids)):
        top5_32 = set(np.argsort(l32[t])[-5:].tolist())
        top5_16 = set(np.argsort(l16[t])[-5:].tolist())
        assert top5_32 == top5_16, (
            f"pos {t}: top-5 differs.\n  f32: {sorted(top5_32)}\n"
            f"  f16: {sorted(top5_16)}"
        )
        assert int(np.argmax(l32[t])) == int(np.argmax(l16[t])), (
            f"pos {t}: top-1 differs"
        )
        max_abs = max(max_abs, float(np.max(np.abs(l32[t] - l16[t]))))
    print(f"\n[real_1b_fp16] T={len(ids)} max_abs={max_abs:.3e}")


@pytest.mark.skipif(
    not (REAL_MODEL_DIR / "model.safetensors").exists(),
    reason="Real Llama 3.2 1B-Instruct not downloaded",
)
def test_real_1b_fp16_greedy_decode_matches_fp32() -> None:
    """End-to-end check: greedy decode of 3 tokens should produce the same
    text in FP16 and FP32 modes for a confident prompt."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REAL_MODEL_DIR))
    prompt = tok.encode("The capital of France is", add_special_tokens=True)

    eng32 = llmengine.Engine(str(REAL_MODEL_DIR), dtype="fp32")
    out32, _ = eng32.generate(prompt, max_new_tokens=3)

    eng16 = llmengine.Engine(str(REAL_MODEL_DIR), dtype="fp16")
    out16, _ = eng16.generate(prompt, max_new_tokens=3)

    text32 = tok.decode(out32[len(prompt):])
    text16 = tok.decode(out16[len(prompt):])
    print(f"\n[real_1b_fp16_decode] f32={text32!r} f16={text16!r}")
    assert out32 == out16, f"decode diverged: f32={text32!r} f16={text16!r}"
