"""OpenAI-compatible HTTP server (Phase 8 + v6 §8.3 streaming).

POST /v1/completions and /v1/chat/completions, with streaming via SSE. The
engine itself only ever sees token IDs; tokenization lives in this Python
layer using HuggingFace AutoTokenizer.

The streaming path uses the v6 design:
  - engine.generate_streaming runs on a background thread, fires on_token
    per sampled token and on_done with the terminal reason.
  - A janus.Queue bridges the engine thread → asyncio loop.
  - The SSE loop polls request.is_disconnected each iteration and waits for
    the next token with a 100 ms timeout (try/except inside the loop so a
    quiet 100 ms doesn't terminate the stream, the v6 streaming-bug fix).
  - On disconnect, cancel_token.cancel() flips a flag the engine checks
    between forward steps, draining the worker without waste.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import janus
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

import llmengine


# Internal finish_reason → OpenAI's enum {stop, length, content_filter,
# tool_calls, function_call}. Internal "capacity" maps to "length" (closest
# semantic, we ran out of context room); "cancelled" maps to "stop" so
# disconnecting clients don't see an unknown enum value.
class ChatMessage(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    """OpenAI /v1/completions request body (subset). Pydantic enforces the
    schema so bad types (prompt=list, max_tokens=str, …) become 422s
    instead of bubbling up as 500s from the tokenizer or int()."""
    model: Optional[str] = None
    prompt: str
    max_tokens: int = Field(default=16)
    stream: bool = False


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64)
    stream: bool = False


def _to_openai_finish(internal: str) -> str:
    # The OpenAI enum is {stop, length, content_filter, tool_calls,
    # function_call}. We map our internal terminal reasons onto it:
    #   capacity  → length  (ran out of KV / context room)
    #   cancelled → stop    (client disconnected, no error visible to user)
    #   error     → stop    (worker raised; we already closed the stream)
    return {"capacity": "length",
            "cancelled": "stop",
            "error": "stop"}.get(internal, internal)


def build_app(model_dir: Optional[str] = None,
              dtype: str = "fp32",
              model_name: Optional[str] = None) -> FastAPI:
    from transformers import AutoTokenizer

    model_dir  = model_dir  or os.environ.get("LLMENGINE_MODEL", "")
    dtype      = os.environ.get("LLMENGINE_DTYPE", dtype)
    model_name = model_name or os.path.basename(os.path.normpath(model_dir or "model"))

    if not model_dir:
        raise RuntimeError(
            "LLMENGINE_MODEL env var (or model_dir kwarg) must be set "
            "to the directory containing config.json + model.safetensors")

    tok    = AutoTokenizer.from_pretrained(model_dir)
    engine = llmengine.Engine(model_dir, dtype=dtype)

    # The Engine serializes forward_logits/generate/generate_streaming via
    # an internal mutex, so concurrent HTTP requests are correctness-safe.
    # We additionally pin engine work to a single-thread executor so we
    # don't pile up worker threads queued on the same mutex, FastAPI's
    # asyncio.to_thread default would use the shared executor (40+ threads
    # in 3.12) and stack many concurrent calls in front of a single lock.
    engine_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="llmengine-engine")

    async def _engine_call(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(
            engine_pool, fn, *args)

    def _error_response(status: int, exc: BaseException,
                         err_type: str = "invalid_request_error") -> JSONResponse:
        """OpenAI-shaped error body. Matches python-openai's expectations
        well enough that client.with_raw_response.parse_error() lights up."""
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "message": str(exc),
                    "type":    err_type,
                    "param":   None,
                    "code":    None,
                },
            },
        )

    # Lifespan replaces the deprecated @app.on_event("shutdown") API.
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            engine_pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="llm-engine OpenAI-compatible server",
                  lifespan=lifespan)

    # ----- non-streaming response shapes -----

    def _chat_completion_response(prompt_ids, out_ids, finish, model_used):
        generated = out_ids[len(prompt_ids):]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_used,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": tok.decode(generated)},
                "finish_reason": _to_openai_finish(finish),
            }],
            "usage": {
                "prompt_tokens":     len(prompt_ids),
                "completion_tokens": len(generated),
                "total_tokens":      len(out_ids),
            },
        }

    def _completion_response(prompt_ids, out_ids, finish, model_used):
        generated = out_ids[len(prompt_ids):]
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_used,
            "choices": [{
                "index": 0,
                "text": tok.decode(generated),
                "logprobs": None,
                "finish_reason": _to_openai_finish(finish),
            }],
            "usage": {
                "prompt_tokens":     len(prompt_ids),
                "completion_tokens": len(generated),
                "total_tokens":      len(out_ids),
            },
        }

    # ----- streaming helper (v6 §8.3) -----

    async def _stream_sse(prompt_ids, max_tokens, *, chat, model_used,
                          request: Request):
        q = janus.Queue()
        cancel_token = llmengine.CancelToken()

        # Engine-thread callbacks. Both must use the thread-safe sync end of
        # the janus queue; the asyncio task awaits on the async end.
        DONE = object()
        def on_token(tok_id: int):
            q.sync_q.put(tok_id)
        def on_done(reason: str):
            q.sync_q.put((DONE, reason))

        loop = asyncio.get_running_loop()
        worker = loop.run_in_executor(
            engine_pool,
            engine.generate_streaming,
            prompt_ids, max_tokens, on_token, on_done, cancel_token)

        chunk_id = (("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24])
        created = int(time.time())

        def chunk(delta, finish):
            if chat:
                body = {
                    "id": chunk_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_used,
                    "choices": [{
                        "index": 0,
                        "delta": ({"content": delta} if delta is not None else {}),
                        "finish_reason": finish,
                    }],
                }
            else:
                body = {
                    "id": chunk_id, "object": "text_completion",
                    "created": created, "model": model_used,
                    "choices": [{
                        "index": 0,
                        "text": delta if delta is not None else "",
                        "finish_reason": finish,
                    }],
                }
            return f"data: {json.dumps(body)}\n\n"

        buffer: list[int] = []
        prev_text = ""

        try:
            while True:
                # Cooperative cancel: HTTP disconnect → flip the engine's
                # CancelToken. The engine drains and fires on_done("cancelled").
                if await request.is_disconnected():
                    cancel_token.cancel()

                # Poll the engine thread. A 100 ms quiet window is normal on
                # CPU inference, DO NOT exit the stream; the try/except sits
                # INSIDE the loop and continues (v6 streaming-bug fix).
                try:
                    item = await asyncio.wait_for(q.async_q.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # Safety net: if the worker raised before reaching
                    # on_done (e.g. a bad input that throws inside the
                    # engine), no DONE will ever land in the queue. Engine
                    # code wraps generate_streaming in try/catch so this
                    # path is rare, but a defensive check here keeps the
                    # SSE response from hanging forever on any escape.
                    if worker.done():
                        exc = worker.exception()
                        if exc is not None:
                            yield chunk(None, _to_openai_finish("error"))
                            break
                        # Worker finished cleanly without enqueuing DONE,
                        # shouldn't happen, but treat as a normal close.
                        yield chunk(None, "stop")
                        break
                    continue

                if isinstance(item, tuple) and item[0] is DONE:
                    yield chunk(None, _to_openai_finish(item[1]))
                    break

                # Token arrived. Buffer-and-diff decode so multi-byte UTF-8
                # boundaries don't emit `�`.
                buffer.append(item)
                text = tok.decode(buffer)
                if text.endswith("�"):  # partial UTF-8, hold
                    continue
                delta = text[len(prev_text):]
                prev_text = text
                if delta:
                    yield chunk(delta, None)
        finally:
            # If the client bailed before on_done arrived, make sure the
            # engine wraps up and the worker future completes.
            cancel_token.cancel()
            try:
                await worker
            except Exception:
                pass

        yield "data: [DONE]\n\n"

    # ----- endpoints -----

    def _validate_generate_inputs(prompt_ids, max_tokens):
        """Sync pre-flight matching engine.generate's contract. Run BEFORE
        returning StreamingResponse so a bad streaming request surfaces as
        400/422, not as a 200 stream that closes with finish_reason=stop
        from the worker."""
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if not prompt_ids:
            raise ValueError("prompt is empty")
        if len(prompt_ids) + max_tokens > engine.max_pos:
            raise RuntimeError(
                f"prompt + max_tokens exceeds RoPE max_pos ({engine.max_pos})")

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        try:
            used_name = body.model or model_name
            prompt_ids = tok.encode(body.prompt, add_special_tokens=True)
            _validate_generate_inputs(prompt_ids, body.max_tokens)
            if body.stream:
                return StreamingResponse(
                    _stream_sse(prompt_ids, body.max_tokens, chat=False,
                                model_used=used_name, request=request),
                    media_type="text/event-stream")
            out, finish = await _engine_call(engine.generate, prompt_ids, body.max_tokens)
            return _completion_response(prompt_ids, out, finish, used_name)
        except ValueError as e:
            # Bad inputs (empty prompt, negative max_tokens, out-of-range
            # token ids if any slip through the tokenizer).
            return _error_response(400, e)
        except RuntimeError as e:
            # Engine runtime constraint violations (RoPE max_pos, etc.).
            return _error_response(422, e)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        try:
            used_name = body.model or model_name
            messages = [m.model_dump() for m in body.messages]
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            _validate_generate_inputs(prompt_ids, body.max_tokens)
            if body.stream:
                return StreamingResponse(
                    _stream_sse(prompt_ids, body.max_tokens, chat=True,
                                model_used=used_name, request=request),
                    media_type="text/event-stream")
            out, finish = await _engine_call(engine.generate, prompt_ids, body.max_tokens)
            return _chat_completion_response(prompt_ids, out, finish, used_name)
        except ValueError as e:
            return _error_response(400, e)
        except RuntimeError as e:
            return _error_response(422, e)

    @app.get("/health")
    async def health():
        return {"status": "ok",
                "model": model_name,
                "dtype": engine.dtype,
                "vocab_size": engine.cfg.vocab_size}

    return app


# Import-time app construction when run via uvicorn.
if os.environ.get("LLMENGINE_MODEL"):
    app = build_app()
