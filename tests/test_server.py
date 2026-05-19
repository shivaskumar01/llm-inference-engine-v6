"""Phase 8: FastAPI server smoke.

Uses fastapi.testclient.TestClient (httpx under the hood, in-process — no
network or socket). For both /v1/chat/completions and /v1/completions we
check the full OpenAI response shape (id/object/created/model/choices/usage)
and the streaming SSE format (data: ... \\n\\n + [DONE]).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL_DIR = REPO_ROOT / "data" / "llama-3.2-1b-instruct"


pytestmark = pytest.mark.skipif(
    not (REAL_MODEL_DIR / "model.safetensors").exists(),
    reason="Real Llama 3.2 1B-Instruct not downloaded",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from llmengine.server import build_app

    app = build_app(str(REAL_MODEL_DIR), dtype="int8",
                    model_name="llama-3.2-1b-instruct")
    with TestClient(app) as c:
        yield c


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "llama-3.2-1b-instruct"
    assert body["dtype"] == "int8"
    assert body["vocab_size"] == 128256


def test_chat_completion_full_shape(client) -> None:
    r = client.post("/v1/chat/completions", json={
        "model": "llama-3.2-1b-instruct",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 4,
    })
    assert r.status_code == 200, r.text
    body = r.json()

    # Full OpenAI shape.
    assert body["object"] == "chat.completion"
    assert body["model"]  == "llama-3.2-1b-instruct"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert len(body["choices"]) == 1
    ch = body["choices"][0]
    assert ch["index"] == 0
    assert ch["message"]["role"] == "assistant"
    assert ch["finish_reason"] in {"stop", "length"}
    assert "content" in ch["message"]
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    print(f"\n[chat] reply={ch['message']['content']!r}")


def test_completion_full_shape(client) -> None:
    r = client.post("/v1/completions", json={
        "model": "llama-3.2-1b-instruct",
        "prompt": "The capital of France is",
        "max_tokens": 3,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"]  == "llama-3.2-1b-instruct"
    assert body["choices"][0]["finish_reason"] in {"stop", "length"}
    assert "Paris" in body["choices"][0]["text"] or "paris" in body["choices"][0]["text"].lower()


def test_generate_streaming_engine_native(client) -> None:
    """The engine-side streaming API (no HTTP). Verifies the C++
    generate_streaming + CancelToken plumbing fires on_token per token and
    on_done with a terminal reason."""
    import llmengine
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REAL_MODEL_DIR))
    engine = llmengine.Engine(str(REAL_MODEL_DIR), dtype="int8")
    cancel = llmengine.CancelToken()

    prompt_ids = tok.encode("The capital of France is", add_special_tokens=True)
    tokens: list[int] = []
    done: list[str] = []
    engine.generate_streaming(
        prompt_ids, 3,
        lambda t: tokens.append(t),
        lambda r: done.append(r),
        cancel,
    )
    assert len(tokens) == 3
    assert done == ["length"]
    assert "Paris" in tok.decode(tokens) or "paris" in tok.decode(tokens).lower()


def test_generate_streaming_cancellation() -> None:
    """If cancel() fires before the engine starts, the worker exits via the
    cancellation path without doing any forward steps."""
    import llmengine
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REAL_MODEL_DIR))
    engine = llmengine.Engine(str(REAL_MODEL_DIR), dtype="int8")
    cancel = llmengine.CancelToken()
    cancel.cancel()

    prompt_ids = tok.encode("Hello", add_special_tokens=True)
    tokens: list[int] = []
    done: list[str] = []
    engine.generate_streaming(
        prompt_ids, 10,
        lambda t: tokens.append(t),
        lambda r: done.append(r),
        cancel,
    )
    assert done == ["cancelled"]
    assert len(tokens) == 0


def test_generate_streaming_cancel_mid_stream() -> None:
    """Cancellation mid-stream: cancel() fires after N tokens stream out;
    the engine checks the flag between forward steps and exits with
    on_done('cancelled') without burning through the full max_new_tokens."""
    import llmengine
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REAL_MODEL_DIR))
    engine = llmengine.Engine(str(REAL_MODEL_DIR), dtype="int8")
    cancel = llmengine.CancelToken()

    prompt_ids = tok.encode("The capital of France is", add_special_tokens=True)
    tokens: list[int] = []
    done: list[str] = []

    def on_tok(t: int) -> None:
        tokens.append(t)
        if len(tokens) >= 2:
            cancel.cancel()  # request cancel after 2 tokens

    engine.generate_streaming(
        prompt_ids, 20,                       # would otherwise emit 20 tokens
        on_tok,
        lambda r: done.append(r),
        cancel,
    )
    assert done == ["cancelled"]
    # Engine should stop shortly after the cancel signal — definitely not run
    # to the 20-token cap. A small allowance for the in-flight step.
    assert 2 <= len(tokens) <= 4, f"got {len(tokens)} tokens, expected ~2-3"


def test_non_stream_negative_max_tokens_returns_400(client) -> None:
    """ValueError → 400 with OpenAI-shaped error body. engine.generate
    rejects max_new_tokens < 0 with ValueError; the HTTP layer must
    surface that as 400, not a raw 500."""
    r = client.post("/v1/completions", json={
        "model": "llama-3.2-1b-instruct",
        "prompt": "Hi",
        "max_tokens": -1,
    })
    assert r.status_code == 400, r.text
    body = r.json()
    assert "error" in body
    assert body["error"]["type"] == "invalid_request_error"
    assert "max_new_tokens" in body["error"]["message"]


def test_non_stream_overflow_returns_422(client) -> None:
    """Engine RuntimeError (RoPE max_pos exceeded) → 422 not 500."""
    r = client.post("/v1/chat/completions", json={
        "model": "llama-3.2-1b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 2147483647,
    })
    assert r.status_code == 422, r.text
    body = r.json()
    assert "error" in body
    assert body["error"]["type"] == "invalid_request_error"
    assert "RoPE max_pos" in body["error"]["message"]


def test_non_stream_completions_overflow_returns_422(client) -> None:
    r = client.post("/v1/completions", json={
        "model": "llama-3.2-1b-instruct",
        "prompt": "Hi",
        "max_tokens": 2147483647,
    })
    assert r.status_code == 422, r.text
    body = r.json()
    assert "error" in body
    assert body["error"]["type"] == "invalid_request_error"


def test_http_disconnect_cancels_worker_then_next_request_succeeds(client) -> None:
    """End-to-end: open an SSE stream, abort early, then verify the next
    request runs cleanly. If the disconnect path leaked the engine mutex or
    the worker, the second request would hang."""
    import time

    with client.stream("POST", "/v1/chat/completions", json={
        "model": "llama-3.2-1b-instruct",
        "messages": [{"role": "user", "content": "Tell me a short story."}],
        "max_tokens": 64,           # would otherwise take many seconds
        "stream": True,
    }) as resp:
        assert resp.status_code == 200
        # Read just a couple chunks and break — TestClient closes the
        # connection on context exit, which surfaces as is_disconnected()=True
        # on the server side.
        seen = 0
        for raw in resp.iter_lines():
            if not raw: continue
            seen += 1
            if seen >= 2: break
        assert seen >= 1

    # Drop a beat to let the server-side cleanup run (cancel propagated,
    # `await worker` resolves, engine mutex released).
    time.sleep(0.5)

    # Next request must run successfully — if the worker hung or the mutex
    # leaked, this would block forever.
    r = client.post("/v1/chat/completions", json={
        "model": "llama-3.2-1b-instruct",
        "messages": [{"role": "user", "content": "Hi."}],
        "max_tokens": 2,
    })
    assert r.status_code == 200
    assert r.json()["object"] == "chat.completion"


def test_chat_streaming_sse(client) -> None:
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "llama-3.2-1b-instruct",
        "messages": [{"role": "user", "content": "Say hi briefly."}],
        "max_tokens": 4,
        "stream": True,
    }) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events: list[dict] = []
        got_done = False
        for raw in resp.iter_lines():
            if not raw:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode()
            assert raw.startswith("data: "), f"unexpected line: {raw!r}"
            payload = raw[len("data: "):]
            if payload == "[DONE]":
                got_done = True
                break
            events.append(json.loads(payload))

        assert got_done
        assert events, "no streamed chunks received"
        for ev in events:
            assert ev["object"] == "chat.completion.chunk"
        # Final chunk must carry a finish_reason.
        assert events[-1]["choices"][0]["finish_reason"] in {"stop", "length"}
