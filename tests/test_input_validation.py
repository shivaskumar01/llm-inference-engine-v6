"""Boundary tests for the public Python surface.

Covers four review findings:
  - token IDs outside [0, vocab_size) at every public entry point
  - prompt_len + max_new_tokens overflow at the int boundary
  - streaming worker exception path produces a clean SSE close
"""
from __future__ import annotations

from pathlib import Path

import pytest

import llmengine
from fixtures import make_tiny_llama_weights


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    out = tmp_path_factory.mktemp("input_validation_tiny")
    make_tiny_llama_weights(out)
    return llmengine.Engine(str(out))


def _make_mgr(engine):
    cfg = engine.cfg
    return llmengine.BlockManager(
        num_blocks=8, block_size=4,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )


# ----- token id bounds ----------------------------------------------------

@pytest.mark.parametrize("bad", [-1, -100])
def test_engine_forward_logits_rejects_negative_token(engine, bad) -> None:
    with pytest.raises(ValueError):
        engine.forward_logits([bad])


def test_engine_forward_logits_rejects_oor_token(engine) -> None:
    V = engine.cfg.vocab_size
    with pytest.raises(ValueError):
        engine.forward_logits([V])
    with pytest.raises(ValueError):
        engine.forward_logits([V + 100])


@pytest.mark.parametrize("bad", [-1, 9999999])
def test_engine_generate_rejects_invalid_token(engine, bad) -> None:
    with pytest.raises(ValueError):
        engine.generate([bad], 0)


def test_engine_forward_logits_paged_rejects_invalid_token(engine) -> None:
    mgr = _make_mgr(engine)
    V = engine.cfg.vocab_size
    with pytest.raises(ValueError):
        engine.forward_logits_paged([V], mgr)
    with pytest.raises(ValueError):
        engine.forward_logits_paged([-1], mgr)


@pytest.mark.parametrize("which", ["static", "continuous"])
def test_scheduler_enqueue_rejects_invalid_token(engine, which) -> None:
    mgr = _make_mgr(engine)
    sched = (llmengine.StaticBatchScheduler(engine, mgr)
             if which == "static"
             else llmengine.ContinuousBatchScheduler(engine, mgr))
    with pytest.raises(ValueError):
        sched.enqueue([engine.cfg.vocab_size], 1)
    with pytest.raises(ValueError):
        sched.enqueue([-1, 0], 1)


# ----- prompt + max_new overflow -----------------------------------------

INT_MAX = 2**31 - 1


def test_engine_generate_overflow_caught(engine) -> None:
    # Before the int64 widening, max_new = INT_MAX caused prompt_len +
    # max_new to wrap negative, bypass the RoPE guard, and trip
    # ContiguousKVCache: non-positive dim. After the fix it must throw a
    # plain RoPE-exceeded RuntimeError.
    with pytest.raises(RuntimeError, match="RoPE max_pos"):
        engine.generate([1], INT_MAX)


@pytest.mark.parametrize("which", ["static", "continuous"])
def test_scheduler_enqueue_overflow_caught(engine, which) -> None:
    mgr = _make_mgr(engine)
    sched = (llmengine.StaticBatchScheduler(engine, mgr)
             if which == "static"
             else llmengine.ContinuousBatchScheduler(engine, mgr))
    with pytest.raises(ValueError, match="RoPE max_pos"):
        sched.enqueue([1], INT_MAX)


# ----- engine.generate validates inputs BEFORE model materialization ------

def test_engine_generate_input_errors_before_model_load(tmp_path) -> None:
    """A partially-built checkpoint (config + no tensors) used to fail with
    a weight-loader 'missing weight' message even when the input was
    invalid. After the fix, input validation runs before ensure_model_loaded,
    so the user sees the actual input error."""
    import json
    cfg = {
        "hidden_size": 4, "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
        "vocab_size": 8, "max_position_embeddings": 16,
        "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
        "tie_word_embeddings": False, "eos_token_id": [1, 2, 3],
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    # An empty .safetensors with just the 8-byte header-length prefix +
    # empty {} header is the minimal "loads but has no tensors" payload.
    import struct
    empty_header = b"{}"
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(empty_header)) + empty_header)

    e = llmengine.Engine(str(tmp_path))

    # Empty prompt: must surface as a clean ValueError, not 'missing weight'.
    with pytest.raises(ValueError, match="prompt"):
        e.generate([], 1)
    # Negative max_new_tokens: same.
    with pytest.raises(ValueError, match="max_new_tokens"):
        e.generate([1], -1)
    # Out-of-range token id: same — token validation also happens upfront.
    with pytest.raises(ValueError, match="out of range"):
        e.generate([999], 1)


# ----- streaming worker error ----------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL_DIR = REPO_ROOT / "data" / "llama-3.2-1b-instruct"


@pytest.mark.skipif(
    not (REAL_MODEL_DIR / "model.safetensors").exists(),
    reason="Real Llama 3.2 1B-Instruct not downloaded",
)
def test_http_streaming_overflow_does_not_hang() -> None:
    """An overflow-large max_tokens on a streaming request used to slip
    through to the worker, which raised and left the SSE bridge waiting
    forever for a DONE that never came. After the v6 pre-stream validation
    landed, the same request is caught synchronously and the response is
    a structured 422 — the test still asserts 'doesn't hang' (the regression
    we care about), plus the better-shaped 422 status it now returns."""
    import time
    from fastapi.testclient import TestClient
    from llmengine.server import build_app

    app = build_app(str(REAL_MODEL_DIR), dtype="int8",
                    model_name="llama-3.2-1b-instruct")
    with TestClient(app) as client:
        t0 = time.perf_counter()
        r = client.post("/v1/chat/completions", json={
            "model": "llama-3.2-1b-instruct",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": INT_MAX,
            "stream": True,
        })
        elapsed = time.perf_counter() - t0

    assert elapsed < 30, f"streaming hung for {elapsed:.1f}s"
    assert r.status_code == 422, r.text
    body = r.json()
    assert "RoPE max_pos" in body["error"]["message"]
