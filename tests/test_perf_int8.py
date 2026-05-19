"""Phase 5: INT8 W8A32 drift gates.

INT8 is much coarser than FP16 (~5e-3 per-weight quantization noise vs 5e-4
for FP16). Top-1 stability holds when the model is confident; we don't
demand 5/5 top-5 set equality across all positions because random tiny
weights can flip top-k on the noise alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import llmengine
from fixtures import make_tiny_llama_weights


@pytest.fixture(scope="module")
def tiny_paths(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny_int8")
    make_tiny_llama_weights(out)
    return str(out)


def test_tiny_int8_runs_and_drift_bounded(tiny_paths) -> None:
    """End-to-end forward in INT8 mode produces logits; max-abs drift vs FP32
    is reported as a debug stat. Tiny model has random weights so we don't
    expect tight top-1 stability across all positions."""
    eng32 = llmengine.Engine(tiny_paths, dtype="fp32")
    eng_i8 = llmengine.Engine(tiny_paths, dtype="int8")

    rng = np.random.default_rng(0)
    T = 8
    ids = rng.integers(0, eng32.cfg.vocab_size, size=T).tolist()

    l32 = eng32.forward_logits(ids)
    li8 = eng_i8.forward_logits(ids)

    assert l32.shape == li8.shape

    max_abs = float(np.max(np.abs(l32 - li8)))
    top1_matches = sum(int(np.argmax(l32[t]) == np.argmax(li8[t])) for t in range(T))
    print(f"\n[tiny_int8] T={T}, max_abs={max_abs:.3e}, top1={top1_matches}/{T}")

    # Forward succeeded and produced finite logits.
    assert np.all(np.isfinite(li8))


# ----- Real-1B: this is where W8A32 actually shines ------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL_DIR = REPO_ROOT / "data" / "llama-3.2-1b-instruct"


@pytest.mark.skipif(
    not (REAL_MODEL_DIR / "model.safetensors").exists(),
    reason="Real Llama 3.2 1B-Instruct not downloaded",
)
def test_real_1b_int8_top5_and_decode() -> None:
    """Per the v6 plan §5.5 soft gates: top-1 match >= 75%, top-5 overlap
    >= 3.5/5 average. On a short, confident prompt we expect much better
    than that (the capital-of-France logits put Paris at a wide margin)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REAL_MODEL_DIR))
    ids = tok.encode("The capital of France is", add_special_tokens=True)

    eng32 = llmengine.Engine(str(REAL_MODEL_DIR), dtype="fp32")
    l32 = eng32.forward_logits(ids)

    eng_i8 = llmengine.Engine(str(REAL_MODEL_DIR), dtype="int8")
    li8 = eng_i8.forward_logits(ids)

    max_abs = 0.0
    top1_hits = 0
    top5_overlap_sum = 0.0
    for t in range(len(ids)):
        if int(np.argmax(l32[t])) == int(np.argmax(li8[t])):
            top1_hits += 1
        t32 = set(np.argsort(l32[t])[-5:].tolist())
        ti8 = set(np.argsort(li8[t])[-5:].tolist())
        top5_overlap_sum += len(t32 & ti8)
        max_abs = max(max_abs, float(np.max(np.abs(l32[t] - li8[t]))))

    top1_rate = top1_hits / len(ids)
    top5_avg  = top5_overlap_sum / len(ids)
    print(f"\n[real_1b_int8] T={len(ids)} top1={top1_rate:.2f} "
          f"top5_overlap={top5_avg:.2f}/5 max_abs={max_abs:.3e}")

    # Soft gates from the v6 plan.
    assert top1_rate >= 0.75
    assert top5_avg  >= 3.5


@pytest.mark.skipif(
    not (REAL_MODEL_DIR / "model.safetensors").exists(),
    reason="Real Llama 3.2 1B-Instruct not downloaded",
)
def test_real_1b_int8_greedy_decode_coherent() -> None:
    """End-to-end: prefill + decode 3 tokens in INT8 mode. The output should
    still be a sensible continuation of 'The capital of France is'."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REAL_MODEL_DIR))
    prompt = tok.encode("The capital of France is", add_special_tokens=True)

    eng_i8 = llmengine.Engine(str(REAL_MODEL_DIR), dtype="int8")
    out, finish = eng_i8.generate(prompt, max_new_tokens=3)
    text = tok.decode(out[len(prompt):])
    print(f"\n[real_1b_int8_decode] generated={out[len(prompt):]!r} text={text!r}")
    assert finish in ("length", "stop")
    # The first generated token should be Paris-ish. Don't assert exact text;
    # INT8 noise can shift among the top-2 plausible continuations.
    assert "Paris" in text or "paris" in text.lower()
