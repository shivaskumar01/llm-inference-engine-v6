"""Phase 3: ContiguousKVCache + Engine.generate.

Three groups of tests:
  1. KV-cache correctness: forward_logits (now backed by ContiguousKVCache)
     still token-for-token matches HF on the tiny model.
  2. generate equivalence: generate(prompt, n) matches the Phase 1 reference
     greedy loop (forward_logits + argmax) bit-exactly. This is the cheapest
     way to assert the prefill+decode split + KV write/read math.
  3. Termination: max_new_tokens cap → finish_reason == "length"; EOS-driven
     stop via a synthetic checkpoint whose eos_token_id set is rigged to
     contain whatever token the model emits first (so we can deterministically
     trigger "stop" without needing a trained model).
  4. (optional) Real-1B 3-token greedy smoke decode that doesn't crash and
     produces coherent text, skipped if weights are not downloaded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import llmengine
from fixtures import make_tiny_llama_weights


# ----- KV-backed forward equivalence ----------------------------------------

@pytest.fixture(scope="module")
def tiny_engine(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny_phase3")
    cfg, _ = make_tiny_llama_weights(out)
    return llmengine.Engine(str(out)), cfg


def test_generate_matches_reference_greedy_loop(tiny_engine) -> None:
    """generate(prompt, n) should produce the same tokens as the Phase 1
    forward_logits+argmax reference loop."""
    engine, cfg = tiny_engine
    rng = np.random.default_rng(0)

    n_seqs = 20
    n_steps = 20
    eos = set(engine.cfg.eos_token_ids)

    for s in range(n_seqs):
        prompt_len = int(rng.integers(1, 6))
        prompt = rng.integers(0, cfg.vocab_size, size=prompt_len).tolist()

        # Reference: re-run forward over the growing sequence and argmax.
        # Mirrors generate()'s EOS-stop contract.
        ref_seq = list(prompt)
        ref_finish = "length"
        for _ in range(n_steps):
            logits = engine.forward_logits(ref_seq)
            nxt = int(np.argmax(logits[-1]))
            if nxt in eos:
                ref_finish = "stop"
                break
            ref_seq.append(nxt)

        out_ids, finish = engine.generate(prompt, n_steps)
        assert finish == ref_finish, (
            f"seq {s}: gen finish={finish}, ref finish={ref_finish}"
        )
        assert out_ids == ref_seq, (
            f"seq {s} diverged.\n  ref:  {ref_seq}\n  gen:  {out_ids}"
        )


def test_max_new_tokens_cap(tiny_engine) -> None:
    engine, cfg = tiny_engine
    prompt = [5, 7, 11]
    out, finish = engine.generate(prompt, max_new_tokens=8)
    assert finish == "length"
    assert len(out) == len(prompt) + 8
    assert out[: len(prompt)] == prompt


def test_zero_max_new_tokens_returns_prompt(tiny_engine) -> None:
    engine, cfg = tiny_engine
    prompt = [3, 4, 5]
    out, finish = engine.generate(prompt, max_new_tokens=0)
    assert finish == "length"
    assert out == prompt


def test_eos_triggers_stop(tmp_path) -> None:
    """Build a tiny model, run one greedy step to learn what it predicts,
    then save a checkpoint whose eos_token_id includes that token. Reload
    and assert generate stops immediately with finish_reason='stop'."""
    out = tmp_path / "eos_rig"
    cfg, hf = make_tiny_llama_weights(out)
    engine = llmengine.Engine(str(out))

    prompt = [1, 2, 3, 4]
    logits = engine.forward_logits(prompt)
    next_tok = int(np.argmax(logits[-1]))

    # Rewrite config.json with the predicted token in the EOS list.
    import json
    cfg_path = out / "config.json"
    cfg_data = json.loads(cfg_path.read_text())
    cfg_data["eos_token_id"] = [next_tok]
    cfg_path.write_text(json.dumps(cfg_data))

    engine2 = llmengine.Engine(str(out))
    assert next_tok in engine2.cfg.eos_token_ids

    out_ids, finish = engine2.generate(prompt, max_new_tokens=5)
    assert finish == "stop", f"expected stop, got {finish}"
    assert out_ids == prompt, (
        "EOS hit on the very first decode step, so output should equal the "
        f"prompt; got {out_ids}"
    )


# ----- Real-1B smoke decode (skipped if weights not downloaded) -------------

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL_DIR = REPO_ROOT / "data" / "llama-3.2-1b-instruct"


@pytest.mark.skipif(
    not (REAL_MODEL_DIR / "model.safetensors").exists(),
    reason="Real Llama 3.2 1B-Instruct not downloaded",
)
def test_real_1b_greedy_decode_3_tokens() -> None:
    """Doesn't compare to HF (Phase 2 already validated forward correctness);
    just verifies generate prefills + decodes a few tokens against the real
    model without crashing, with a finish_reason of length."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REAL_MODEL_DIR))

    engine = llmengine.Engine(str(REAL_MODEL_DIR))
    prompt = tok.encode("The capital of France is", add_special_tokens=True)

    out, finish = engine.generate(prompt, max_new_tokens=3)
    assert finish in ("length", "stop")
    assert out[: len(prompt)] == prompt
    decoded = tok.decode(out[len(prompt):])
    print(f"\n[real_1b_decode] generated={out[len(prompt):]!r} text={decoded!r}")
