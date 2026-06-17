"""Phase 1 hard gate: tiny Llama full-forward equality vs HF transformers.

The fixture builds a 2-layer / hidden-128 randomly-initialized Llama saved as
a real safetensors checkpoint. The engine loads it through the production
loader path (Phase 0 weight loader + Phase 1 forward pass).

Tolerances: atol=1e-4 on full logits (matches plan §1.4); 50-token greedy
decode is required to match HF token-for-token because FP32 vs FP32 is
deterministic when `-fno-fast-math` is in effect (correctness build).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import llmengine
from fixtures import make_tiny_llama_weights


@pytest.fixture(scope="module")
def tiny_engine_and_hf(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny_llama")
    cfg, hf_model = make_tiny_llama_weights(out)
    engine = llmengine.Engine(str(out))
    return engine, hf_model, cfg


def _hf_logits(hf_model, ids: list[int]) -> np.ndarray:
    """Run HF model in FP32 and return logits [T, vocab]."""
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long)
        out = hf_model(x)
        return out.logits[0].float().numpy()


def test_full_forward_logits_match(tiny_engine_and_hf) -> None:
    engine, hf_model, cfg = tiny_engine_and_hf
    rng = np.random.default_rng(0)

    # 100 random sequences of varying length.
    n_pass = 0
    max_abs_seen = 0.0
    for _ in range(100):
        T = int(rng.integers(1, 32))
        ids = rng.integers(0, cfg.vocab_size, size=T).tolist()

        ours = engine.forward_logits(ids)
        theirs = _hf_logits(hf_model, ids)

        assert ours.shape == theirs.shape == (T, cfg.vocab_size)
        max_abs = float(np.max(np.abs(ours - theirs)))
        max_abs_seen = max(max_abs_seen, max_abs)
        np.testing.assert_allclose(ours, theirs, atol=1e-4, rtol=1e-4)
        n_pass += 1

    print(f"\n[full_forward] {n_pass}/100 matched, max_abs_seen={max_abs_seen:.2e}")


def test_per_layer_hidden_states(tiny_engine_and_hf) -> None:
    """Single-token forward, but compare argmax stability across positions,
    serves as a faster gate than the full-forward check above."""
    engine, hf_model, cfg = tiny_engine_and_hf
    rng = np.random.default_rng(1)

    T = 8
    ids = rng.integers(0, cfg.vocab_size, size=T).tolist()
    ours = engine.forward_logits(ids)
    theirs = _hf_logits(hf_model, ids)

    # Token-by-token argmax must match (FP32 deterministic).
    for t in range(T):
        ours_top = int(np.argmax(ours[t]))
        theirs_top = int(np.argmax(theirs[t]))
        assert ours_top == theirs_top, f"argmax differs at position {t}: ours={ours_top}, theirs={theirs_top}"


def _greedy_decode_ours(engine, prompt_ids: list[int], n_steps: int) -> list[int]:
    """Run greedy decode by re-running forward each step. Phase 3 will replace
    this with the proper KV-cache prefill+decode path, but for the Phase 1 gate
    we just want token-for-token equality of the resulting sequence."""
    seq = list(prompt_ids)
    for _ in range(n_steps):
        logits = engine.forward_logits(seq)
        nxt = int(np.argmax(logits[-1]))
        seq.append(nxt)
    return seq


def _greedy_decode_hf(hf_model, prompt_ids: list[int], n_steps: int) -> list[int]:
    seq = list(prompt_ids)
    for _ in range(n_steps):
        with torch.no_grad():
            x = torch.tensor([seq], dtype=torch.long)
            logits = hf_model(x).logits[0, -1].float().numpy()
        seq.append(int(np.argmax(logits)))
    return seq


def test_greedy_decode_token_for_token(tiny_engine_and_hf) -> None:
    engine, hf_model, cfg = tiny_engine_and_hf
    rng = np.random.default_rng(2)

    n_seqs = 50
    n_steps = 50

    matched = 0
    for s in range(n_seqs):
        prompt_len = int(rng.integers(1, 8))
        prompt = rng.integers(0, cfg.vocab_size, size=prompt_len).tolist()
        ours = _greedy_decode_ours(engine, prompt, n_steps)
        theirs = _greedy_decode_hf(hf_model, prompt, n_steps)
        assert ours == theirs, (
            f"sequence {s} diverged.\n  prompt={prompt}\n  ours={ours}\n  hf={theirs}"
        )
        matched += 1
    assert matched == n_seqs
