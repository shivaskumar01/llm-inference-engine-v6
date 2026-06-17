"""Multi-sequence throughput benchmark (Phase 7).

Compares static-batch vs continuous-batch aggregate decode tok/s on a
synthetic workload of N requests with output-length variance. Also sweeps
the continuous batcher's max_prefill_tokens_per_step ("budget") to show
the latency/throughput trade-off the v6 plan promised.

Defaults are sized for the tiny fixture so this runs in seconds. Pass
--real to switch to Llama 3.2 1B-Instruct (much slower).

Usage:
    python benchmarks/bench_throughput.py
    python benchmarks/bench_throughput.py --real --requests 8
    python benchmarks/bench_throughput.py --real --budget-sweep
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import numpy as np

import llmengine


def _make_mgr(cfg, num_blocks: int, block_size: int):
    return llmengine.BlockManager(
        num_blocks=num_blocks, block_size=block_size,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )


def _aggregate_tok_s(results, elapsed_s: float) -> tuple[float, int]:
    total_gen = sum(len(r.token_ids) - r.prompt_len for r in results)
    return (total_gen / max(elapsed_s, 1e-9), total_gen)


def _per_seq_latency_ms(results) -> dict:
    """Returns p50/p99 inter-token latency over the workload's generated tokens.
    Per-token timing isn't exposed by the scheduler, so we approximate as
    elapsed_s / total_generated. Real p99 wiring is a follow-up."""
    return {}  # placeholder, see results.md note about scheduler-side timing


def bench_workload(engine, cfg, prompts: list[list[int]], max_new: int,
                   *, budget: int):
    """Run the workload through both static and continuous scheduling."""
    rows = []

    mgr_s = _make_mgr(cfg, num_blocks=max(64, len(prompts) * 8), block_size=16)
    sched_s = llmengine.StaticBatchScheduler(engine, mgr_s)
    for p in prompts:
        sched_s.enqueue(p, max_new)
    t0 = time.perf_counter()
    sched_s.run_until_done()
    static_s = time.perf_counter() - t0
    static_tok_s, static_gen = _aggregate_tok_s(sched_s.results, static_s)
    rows.append({"mode": "static",
                 "elapsed_s": static_s,
                 "total_generated": static_gen,
                 "aggregate_tok_s": static_tok_s,
                 "budget": None})

    mgr_c = _make_mgr(cfg, num_blocks=max(64, len(prompts) * 8), block_size=16)
    sched_c = llmengine.ContinuousBatchScheduler(
        engine, mgr_c,
        max_concurrent=len(prompts),
        max_prefill_tokens_per_step=budget)
    for p in prompts:
        sched_c.enqueue(p, max_new)
    t0 = time.perf_counter()
    sched_c.run_until_idle()
    cont_s = time.perf_counter() - t0
    cont_tok_s, cont_gen = _aggregate_tok_s(sched_c.results, cont_s)
    rows.append({"mode": f"continuous@budget={budget}",
                 "elapsed_s": cont_s,
                 "total_generated": cont_gen,
                 "aggregate_tok_s": cont_tok_s,
                 "budget": budget})

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="Use Llama 3.2 1B-Instruct instead of the tiny fixture.")
    ap.add_argument("--dtype", default="int8")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--budget", type=int, default=16,
                    help="max_prefill_tokens_per_step for continuous batching.")
    ap.add_argument("--budget-sweep", action="store_true",
                    help="Sweep budgets [4, 16, 64, 256] for continuous mode.")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    if args.real:
        real_dir = REPO_ROOT / "data" / "llama-3.2-1b-instruct"
        if not (real_dir / "model.safetensors").exists():
            print("(real model not downloaded; abort)", file=sys.stderr)
            return 1
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(real_dir))
        engine = llmengine.Engine(str(real_dir), dtype=args.dtype)

        # Mixed prompt lengths to exercise variance and the prefill budget.
        seed_prompts = [
            "The capital of France is",
            "Two plus two equals",
            "Once upon a time, in a land far away,",
            "Roses are red, violets are",
            "Python is a programming language that",
            "The mitochondria is the",
            "Hello! How can I help you today?",
            "Write a haiku about autumn:",
        ]
        prompts = [tok.encode(p, add_special_tokens=True)
                   for p in seed_prompts[:args.requests]]
        max_new = args.max_new
        model_name = "llama-3.2-1b-instruct[" + args.dtype + "]"
    else:
        from fixtures import make_tiny_llama_weights
        out = REPO_ROOT / "data" / "_bench_throughput_tiny"
        out.mkdir(parents=True, exist_ok=True)
        cfg, _ = make_tiny_llama_weights(out)
        engine = llmengine.Engine(str(out), dtype="fp32")
        prompts = [rng.integers(0, cfg.vocab_size,
                                size=int(rng.integers(4, 12))).tolist()
                   for _ in range(args.requests)]
        max_new = args.max_new
        model_name = "tiny-llama-2L-128h"

    cfg = engine.cfg

    print(f"\nworkload: {model_name}   N={len(prompts)}   "
          f"max_new={max_new}   "
          f"prompt_lens={[len(p) for p in prompts]}\n")

    if args.budget_sweep:
        for b in [4, 16, 64, 256]:
            rows = bench_workload(engine, cfg, prompts, max_new, budget=b)
            for r in rows:
                print(f"  {r['mode']:30s} elapsed={r['elapsed_s']:7.3f}s  "
                      f"gen={r['total_generated']:>4d}  "
                      f"agg_tok/s={r['aggregate_tok_s']:.2f}")
            print()
    else:
        rows = bench_workload(engine, cfg, prompts, max_new, budget=args.budget)
        for r in rows:
            print(f"  {r['mode']:30s} elapsed={r['elapsed_s']:7.3f}s  "
                  f"gen={r['total_generated']:>4d}  "
                  f"agg_tok/s={r['aggregate_tok_s']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
