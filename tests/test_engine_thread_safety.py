"""Concurrent access to a single Engine must produce deterministic output.

Before the fix, Engine's mutable scratch buffers (h_, h_norm_, q_buf_,
k_buf_, v_buf_, attn_out_, o_out_, gate_, up_, ffn_out_, k_gather_,
v_gather_) were racy whenever pybind released the GIL around forward_step
and FastAPI's asyncio.to_thread dispatched multiple requests onto its
shared executor pool. Two parallel engine.generate() calls on the same
prompt would diverge from the serial reference.

This test pins that invariant: from N worker threads, each calling
engine.generate(prompt, k) on the same engine, every output must equal
the single-threaded reference.
"""
from __future__ import annotations

import threading

import pytest

import llmengine
from fixtures import make_tiny_llama_weights


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    out = tmp_path_factory.mktemp("thread_safety_tiny")
    make_tiny_llama_weights(out)
    return llmengine.Engine(str(out))


@pytest.mark.parametrize("n_workers", [2, 4, 8])
def test_concurrent_generate_matches_serial(engine, n_workers) -> None:
    prompt = [1, 5, 7, 11, 13, 17, 19, 23]
    max_new = 8

    ref_ids, ref_finish = engine.generate(prompt, max_new)

    outputs: list[tuple[list[int], str]] = []
    outputs_lock = threading.Lock()
    errors: list[Exception] = []

    def worker() -> None:
        try:
            o, f = engine.generate(prompt, max_new)
            with outputs_lock:
                outputs.append((o, f))
        except Exception as e:           # noqa: BLE001
            with outputs_lock:
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n_workers)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert not errors, f"worker exceptions: {errors}"
    assert len(outputs) == n_workers
    for i, (o, f) in enumerate(outputs):
        assert f == ref_finish, f"worker {i} finish={f}, ref={ref_finish}"
        assert o == ref_ids, (
            f"worker {i} diverged from serial reference.\n"
            f"  ref:  {ref_ids}\n  got:  {o}"
        )


def test_concurrent_scheduler_and_generate_dont_corrupt(engine) -> None:
    """The scheduler calls Engine::forward_step_paged through pybind with the
    GIL released. forward_step_paged must lock forward_mu_ so a parallel
    engine.generate() (which DOES lock) doesn't trample the shared scratch
    buffers. Without the fix, the scheduler's per-token outputs diverge from
    its serial reference."""
    cfg = engine.cfg
    prompt = [3, 5, 7, 11, 13, 17]
    max_new = 8

    mgr = llmengine.BlockManager(
        num_blocks=8, block_size=4,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )
    sched = llmengine.StaticBatchScheduler(engine, mgr)
    sched.enqueue(prompt, max_new)
    sched.run_until_done()
    ref_seq = sched.results[0].token_ids
    sched.clear_results()

    # Now run the scheduler concurrently with direct engine.generate calls.
    # The scheduler's seq output must still match the serial reference.
    errors: list[Exception] = []
    out_sched: list[list[int]] = []

    def run_sched() -> None:
        try:
            mgr2 = llmengine.BlockManager(
                num_blocks=8, block_size=4,
                num_layers=cfg.num_hidden_layers,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
            )
            s = llmengine.StaticBatchScheduler(engine, mgr2)
            s.enqueue(prompt, max_new)
            s.run_until_done()
            out_sched.append(s.results[0].token_ids)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def run_gen() -> None:
        try:
            for _ in range(3):
                engine.generate(prompt, max_new)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run_sched) for _ in range(2)] + \
              [threading.Thread(target=run_gen)   for _ in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert not errors, f"worker exceptions: {errors}"
    assert len(out_sched) == 2
    for i, o in enumerate(out_sched):
        assert o == ref_seq, (
            f"scheduler run {i} diverged under concurrent generate().\n"
            f"  ref: {ref_seq}\n  got: {o}"
        )


def test_streaming_callback_can_reenter_engine(engine) -> None:
    """on_token must be invoked without forward_mu_ held: a callback that
    calls back into engine.forward_logits / engine.generate would otherwise
    deadlock on std::mutex. Re-enter from inside the callback and verify
    the streaming run completes."""
    prompt = [1, 2, 3, 4]
    nested_logits: list[int] = []

    def on_tok(t: int) -> None:
        # Re-enter the engine from inside the streaming callback. If
        # forward_mu_ was still held this would deadlock.
        lg = engine.forward_logits([t])
        # `lg` is shape [1, vocab]; record its argmax just to use the result.
        import numpy as np
        nested_logits.append(int(np.argmax(lg[0])))

    done: list[str] = []
    cancel = llmengine.CancelToken()
    engine.generate_streaming(
        prompt, 3, on_tok, lambda r: done.append(r), cancel)

    assert done == ["length"]
    assert len(nested_logits) == 3  # callback ran on each of the 3 tokens


def test_concurrent_streaming_each_stream_matches_serial(engine) -> None:
    """The streaming API serializes through the same mutex; each stream's
    full token list must equal the single-threaded generate output."""
    prompt = [2, 3, 5, 7, 11, 13]
    max_new = 6
    ref_ids, _ = engine.generate(prompt, max_new)
    ref_generated = ref_ids[len(prompt):]

    streams: list[list[int]] = []
    streams_lock = threading.Lock()

    def worker(idx: int) -> None:
        tokens: list[int] = []
        cancel = llmengine.CancelToken()
        engine.generate_streaming(
            prompt, max_new,
            lambda t: tokens.append(t),
            lambda _r: None,
            cancel,
        )
        with streams_lock:
            streams.append(tokens)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    for i, s in enumerate(streams):
        assert s == ref_generated, (
            f"stream {i} diverged from serial reference.\n"
            f"  ref:  {ref_generated}\n  got:  {s}"
        )
