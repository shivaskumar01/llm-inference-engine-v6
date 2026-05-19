"""Phase 7: scheduler correctness.

The scheduler equivalence gate is per-sequence determinism: running a set of
prompts through ContinuousBatchScheduler must produce the same token streams
that the single-sequence engine.generate() path would, because the math is
identical (paged-KV equivalence was established in Phase 6).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import llmengine
from fixtures import make_tiny_llama_weights


@pytest.fixture(scope="module")
def tiny_setup(tmp_path_factory):
    out = tmp_path_factory.mktemp("scheduler_tiny")
    cfg, _ = make_tiny_llama_weights(out)
    engine = llmengine.Engine(str(out))
    return engine, cfg


def _make_mgr(cfg, num_blocks=64, block_size=4):
    return llmengine.BlockManager(
        num_blocks=num_blocks, block_size=block_size,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )


def _engine_generate(engine, prompts, max_new):
    """Reference path: one call to engine.generate per prompt."""
    out = []
    for p in prompts:
        ids, finish = engine.generate(p, max_new)
        out.append((ids, finish))
    return out


# ----- Static batching matches single-sequence generate --------------------

def test_static_batch_matches_single_seq(tiny_setup) -> None:
    engine, cfg = tiny_setup
    rng = np.random.default_rng(0)
    prompts = [rng.integers(0, cfg.vocab_size, size=int(rng.integers(2, 6))).tolist()
               for _ in range(5)]
    max_new = 10

    ref = _engine_generate(engine, prompts, max_new)

    mgr = _make_mgr(cfg)
    sched = llmengine.StaticBatchScheduler(engine, mgr)
    for p in prompts:
        sched.enqueue(p, max_new)
    sched.run_until_done()
    results = sched.results

    assert len(results) == len(prompts)
    for i, (r, (ref_ids, ref_finish)) in enumerate(zip(results, ref)):
        assert r.token_ids == ref_ids, f"seq {i} divergent token stream"
        assert r.finish_reason == ref_finish, f"seq {i} finish reason"
    # KV pool must drain to full capacity.
    assert mgr.free_blocks == mgr.num_blocks


# ----- Continuous batching matches single-sequence generate ----------------

def test_continuous_batch_matches_single_seq(tiny_setup) -> None:
    engine, cfg = tiny_setup
    rng = np.random.default_rng(1)
    prompts = [rng.integers(0, cfg.vocab_size, size=int(rng.integers(2, 6))).tolist()
               for _ in range(6)]
    max_new = 10

    ref = _engine_generate(engine, prompts, max_new)

    mgr = _make_mgr(cfg)
    sched = llmengine.ContinuousBatchScheduler(
        engine, mgr, max_concurrent=8, max_prefill_tokens_per_step=4)
    for p in prompts:
        sched.enqueue(p, max_new)
    sched.run_until_idle()
    results = sorted(sched.results, key=lambda r: r.seq_id)

    assert len(results) == len(prompts)
    for i, (r, (ref_ids, ref_finish)) in enumerate(zip(results, ref)):
        assert r.token_ids == ref_ids, f"seq {i} divergent token stream"
        assert r.finish_reason == ref_finish, f"seq {i} finish reason"
    assert mgr.free_blocks == mgr.num_blocks


# ----- max_new_tokens=1 must produce exactly one generated token ----------

@pytest.mark.parametrize("max_concurrent,budget", [
    (0, 16), (-1, 16), (4, 0), (4, -2),
])
def test_continuous_scheduler_rejects_invalid_knobs(tiny_setup,
                                                     max_concurrent, budget) -> None:
    """Without validation, run_until_idle() would spin forever:
    max_concurrent<=0 blocks every admission; budget<=0 blocks every
    prefill advance — step() keeps returning 'not idle' with no progress."""
    engine, cfg = tiny_setup
    mgr = _make_mgr(cfg)
    with pytest.raises(ValueError):
        llmengine.ContinuousBatchScheduler(
            engine, mgr,
            max_concurrent=max_concurrent,
            max_prefill_tokens_per_step=budget)


def test_continuous_scheduler_budget_setter_validates(tiny_setup) -> None:
    engine, cfg = tiny_setup
    mgr = _make_mgr(cfg)
    sched = llmengine.ContinuousBatchScheduler(
        engine, mgr, max_concurrent=4, max_prefill_tokens_per_step=16)
    sched.budget = 1               # legal
    with pytest.raises(ValueError):
        sched.budget = 0
    with pytest.raises(ValueError):
        sched.budget = -7


@pytest.mark.parametrize("which", ["static", "continuous"])
@pytest.mark.parametrize("prompt,max_new", [
    ([], 1),
    ([1, 2], -1),
])
def test_scheduler_enqueue_input_validation(tiny_setup, which, prompt, max_new) -> None:
    """Schedulers should reject the same inputs engine.generate does:
    empty prompt and negative max_new_tokens. Otherwise enqueue silently
    creates a sequence that produces nonsense output downstream."""
    engine, cfg = tiny_setup
    mgr = _make_mgr(cfg)
    sched = (llmengine.StaticBatchScheduler(engine, mgr)
             if which == "static"
             else llmengine.ContinuousBatchScheduler(engine, mgr))
    with pytest.raises(ValueError):
        sched.enqueue(prompt, max_new)


@pytest.mark.parametrize("which", ["static", "continuous"])
def test_scheduler_enqueue_rejects_over_rope_request(tiny_setup, which) -> None:
    """Tiny fixture caps RoPE at 128 (rope_scaling.original_max=128 wins over
    max_position_embeddings=256). engine.generate([1], 256) already throws;
    scheduler.enqueue must throw too or the scheduler will eventually drive
    pos past the RoPE table and hit an unchecked cos_row(pos) access."""
    engine, cfg = tiny_setup

    # Confirm the engine-level contract first so the test isn't silently
    # asserting against a moved goalpost.
    with pytest.raises(RuntimeError):
        engine.generate([1], 256)

    mgr = _make_mgr(cfg)
    sched = (llmengine.StaticBatchScheduler(engine, mgr)
             if which == "static"
             else llmengine.ContinuousBatchScheduler(engine, mgr))
    with pytest.raises(ValueError):
        sched.enqueue([1], 256)


def test_continuous_scheduler_results_in_enqueue_order(tiny_setup) -> None:
    """scheduler.hpp documents results in enqueue order. The internal
    storage is completion order — a long seq enqueued before a short one
    would otherwise land in results[1]. The accessor sorts by seq_id."""
    engine, cfg = tiny_setup
    mgr = _make_mgr(cfg, num_blocks=32, block_size=4)
    sched = llmengine.ContinuousBatchScheduler(
        engine, mgr, max_concurrent=4, max_prefill_tokens_per_step=16)

    # Long seq first (more tokens to generate → finishes last); short
    # second (finishes first). Without the sort, results would be
    # [short, long] = [seq_id=1, seq_id=0].
    sched.enqueue([1, 2, 3, 4], 12)
    sched.enqueue([5, 6, 7, 8], 1)
    sched.run_until_idle()

    results = sched.results
    assert len(results) == 2
    assert [r.seq_id for r in results] == [0, 1], (
        f"results not in enqueue order: {[r.seq_id for r in results]}"
    )


def test_max_new_zero_keeps_length_even_if_first_sample_would_be_eos(
        tiny_setup) -> None:
    """The exact repro from the code review: engine.generate([88], 0) returns
    ([88], "length") because the decode loop never samples. Schedulers used
    to sample first and could report "stop" when the would-be-first token
    landed in the EOS set."""
    engine, cfg = tiny_setup
    prompt = [88]

    ref_ids, ref_finish = engine.generate(prompt, 0)
    assert ref_finish == "length"
    assert ref_ids == prompt

    mgr = _make_mgr(cfg)
    sched_s = llmengine.StaticBatchScheduler(engine, mgr)
    sched_s.enqueue(prompt, 0)
    sched_s.run_until_done()
    r = sched_s.results[0]
    assert r.token_ids == prompt
    assert r.finish_reason == "length"

    mgr_c = _make_mgr(cfg)
    sched_c = llmengine.ContinuousBatchScheduler(engine, mgr_c)
    sched_c.enqueue(prompt, 0)
    sched_c.run_until_idle()
    r = sched_c.results[0]
    assert r.token_ids == prompt
    assert r.finish_reason == "length"


@pytest.mark.parametrize("max_new", [0, 1, 2, 5])
def test_schedulers_match_engine_generate_token_count(tiny_setup, max_new) -> None:
    """Regression for two prefill-end off-by-ones:
      - max_new_tokens=1: schedulers used to emit two tokens because the
        prefill-end sample didn't check the cap before the decode loop.
      - max_new_tokens=0: engine.generate skips sampling entirely and
        returns "length"; schedulers used to sample-then-cap-check and
        could return "stop" if the first sample happened to be an EOS id.
    Both must match engine.generate exactly."""
    engine, cfg = tiny_setup
    rng = np.random.default_rng(7)
    prompt = rng.integers(0, cfg.vocab_size, size=4).tolist()

    ref_ids, ref_finish = engine.generate(prompt, max_new)

    mgr_s = _make_mgr(cfg)
    sched_s = llmengine.StaticBatchScheduler(engine, mgr_s)
    sched_s.enqueue(prompt, max_new)
    sched_s.run_until_done()
    r_static = sched_s.results[0]
    assert r_static.token_ids == ref_ids
    assert r_static.finish_reason == ref_finish

    mgr_c = _make_mgr(cfg)
    sched_c = llmengine.ContinuousBatchScheduler(
        engine, mgr_c, max_concurrent=1, max_prefill_tokens_per_step=4)
    sched_c.enqueue(prompt, max_new)
    sched_c.run_until_idle()
    r_cont = sched_c.results[0]
    assert r_cont.token_ids == ref_ids
    assert r_cont.finish_reason == ref_finish


# ----- Capacity exhaustion is non-fatal — seq terminates with reason -------

def test_continuous_capacity_termination(tiny_setup) -> None:
    """Tiny pool that can hold one short seq but not two. The second seq
    should terminate with finish_reason='capacity'; the first should complete
    normally."""
    engine, cfg = tiny_setup

    # Block size 4, only 4 blocks => 16 token-positions of total capacity.
    # Each seq with prompt_len=4 + max_new=12 = 16 tokens fits exactly in 4
    # blocks. Two simultaneous seqs would need 8 blocks → second OOMs.
    mgr = _make_mgr(cfg, num_blocks=4, block_size=4)
    sched = llmengine.ContinuousBatchScheduler(
        engine, mgr, max_concurrent=2, max_prefill_tokens_per_step=4)

    sched.enqueue([1, 2, 3, 4], 12)
    sched.enqueue([5, 6, 7, 8], 12)
    sched.run_until_idle()
    results = {r.seq_id: r for r in sched.results}

    finishes = {r.finish_reason for r in results.values()}
    # At least one seq should hit capacity (the second), the other should
    # complete by length or stop.
    assert "capacity" in finishes
    assert mgr.free_blocks == mgr.num_blocks


# ----- Prefill budget actually chunks long prompts ------------------------

def test_continuous_budget_chunks_prefill(tiny_setup) -> None:
    """A prompt longer than the per-step budget should still complete; the
    scheduler advances the prefill cursor by `budget` tokens per step."""
    engine, cfg = tiny_setup

    mgr = _make_mgr(cfg, num_blocks=32, block_size=4)
    sched = llmengine.ContinuousBatchScheduler(
        engine, mgr, max_concurrent=1, max_prefill_tokens_per_step=2)

    prompt = list(range(1, 21))  # 20 tokens; budget=2 → ~10 prefill steps
    sched.enqueue(prompt, 4)
    sched.run_until_idle()

    results = sched.results
    assert len(results) == 1
    r = results[0]
    assert r.finish_reason == "length"
    assert r.prompt_len == 20
    assert len(r.token_ids) == 24
    assert mgr.free_blocks == mgr.num_blocks


# ----- Real-1B smoke: continuous batching produces sensible output --------

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL_DIR = REPO_ROOT / "data" / "llama-3.2-1b-instruct"


@pytest.mark.skipif(
    not (REAL_MODEL_DIR / "model.safetensors").exists(),
    reason="Real Llama 3.2 1B-Instruct not downloaded",
)
def test_real_1b_continuous_batch_smoke() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REAL_MODEL_DIR))
    engine = llmengine.Engine(str(REAL_MODEL_DIR), dtype="int8")

    cfg = engine.cfg
    mgr = llmengine.BlockManager(
        num_blocks=32, block_size=16,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )
    sched = llmengine.ContinuousBatchScheduler(
        engine, mgr, max_concurrent=2, max_prefill_tokens_per_step=8)

    p1 = tok.encode("The capital of France is", add_special_tokens=True)
    p2 = tok.encode("Two plus two equals", add_special_tokens=True)
    sched.enqueue(p1, 3)
    sched.enqueue(p2, 3)
    sched.run_until_idle()

    results = sorted(sched.results, key=lambda r: r.seq_id)
    assert len(results) == 2
    for r in results:
        gen = r.token_ids[r.prompt_len:]
        text = tok.decode(gen)
        print(f"\n[real_1b_cb] seq {r.seq_id} finish={r.finish_reason} text={text!r}")
        assert r.finish_reason in ("length", "stop", "capacity")
    assert mgr.free_blocks == mgr.num_blocks
