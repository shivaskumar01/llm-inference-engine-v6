"""End-to-end decode throughput benchmark.

Runs Engine.generate(prefill, max_new_tokens) on the tiny fixture and, if
available, on real Llama 3.2 1B-Instruct. Reports decode tok/s and prefill
TTFT. Appends one row per run to benchmarks/results.md so the perf
progression across phases (scalar → NEON → FP16 → INT8 → …) is auditable.

Build mode is detected from a compile-time constant exposed via the engine.
Usage:
    python benchmarks/bench_e2e.py             # tiny only
    python benchmarks/bench_e2e.py --real      # also runs real 1B
    python benchmarks/bench_e2e.py --record    # appends to results.md
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "tests"))


def _build_mode() -> str:
    return os.environ.get("LLMENGINE_BUILD") or "perf"


def _hw_label() -> str:
    return f"{platform.machine()} / {platform.system()}"


def _bench(engine, prompt_ids: list[int], new_tokens: int) -> tuple[float, float]:
    """Returns (prefill_seconds, decode_tok_per_sec).

    The first generate() call triggers a one-time ModelWeightsRef
    materialization (~seconds for 1B). We warm up with a discarded call,
    then time prefill (max_new_tokens=0) and a full generate separately so
    decode time can be isolated.
    """
    # Warmup: ensure model + scratch buffers are built.
    engine.generate(prompt_ids, 0)

    t0 = time.perf_counter()
    engine.generate(prompt_ids, 0)
    t1 = time.perf_counter()
    prefill_s = t1 - t0

    t0 = time.perf_counter()
    out, _ = engine.generate(prompt_ids, new_tokens)
    t1 = time.perf_counter()
    total_s = t1 - t0

    decoded = len(out) - len(prompt_ids)
    decode_s = max(total_s - prefill_s, 1e-9)
    decode_tok_s = decoded / decode_s if decoded > 0 else 0.0
    return prefill_s, decode_tok_s


def bench_tiny(dtype: str = "fp32") -> dict:
    import llmengine
    from fixtures import make_tiny_llama_weights

    out = REPO_ROOT / "data" / "_bench_tiny"
    out.mkdir(parents=True, exist_ok=True)
    make_tiny_llama_weights(out)
    engine = llmengine.Engine(str(out), dtype=dtype)

    prompt_ids = [1, 5, 7, 11, 13, 17, 19, 23]
    return {"model": f"tiny-llama-2L-128h[{dtype}]",
            **_bench_dict(_bench(engine, prompt_ids, 16), len(prompt_ids), 16)}


def bench_real_1b(dtype: str = "fp32") -> dict | None:
    real_dir = REPO_ROOT / "data" / "llama-3.2-1b-instruct"
    if not (real_dir / "model.safetensors").exists():
        return None
    import llmengine
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(real_dir))
    engine = llmengine.Engine(str(real_dir), dtype=dtype)
    prompt_ids = tok.encode("The capital of France is", add_special_tokens=True)
    return {"model": f"llama-3.2-1b-instruct[{dtype}]",
            **_bench_dict(_bench(engine, prompt_ids, 5), len(prompt_ids), 5)}


def _bench_dict(timing: tuple[float, float], prompt_len: int, gen: int) -> dict:
    prefill_s, decode_tok_s = timing
    prefill_tok_s = prompt_len / prefill_s if prefill_s > 0 else 0.0
    return {
        "prompt_len": prompt_len,
        "gen_tokens": gen,
        "prefill_s": prefill_s,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "int8", "all"],
                    default="all")
    args = ap.parse_args()

    dtypes = (["fp32", "fp16", "int8"] if args.dtype == "all" else [args.dtype])
    rows = []
    for dt in dtypes:
        rows.append(bench_tiny(dt))
    if args.real:
        for dt in dtypes:
            r = bench_real_1b(dt)
            if r is None:
                print("(real model not downloaded; skipping)")
                break
            rows.append(r)

    print(f"\nbuild={_build_mode()}  hw={_hw_label()}\n")
    print(f"{'model':30s}  {'prompt':>6s}  {'gen':>4s}  "
          f"{'prefill_s':>9s}  {'prefill_tok/s':>13s}  {'decode_tok/s':>12s}")
    for r in rows:
        print(f"{r['model']:30s}  {r['prompt_len']:>6d}  {r['gen_tokens']:>4d}  "
              f"{r['prefill_s']:>9.3f}  {r['prefill_tok_s']:>13.2f}  "
              f"{r['decode_tok_s']:>12.2f}")

    if args.record:
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        rfile = REPO_ROOT / "benchmarks" / "results.md"
        new = not rfile.exists()
        with rfile.open("a") as f:
            if new:
                f.write("| date | hw | build | model | prompt_len | gen | "
                        "prefill_s | prefill_tok/s | decode_tok/s |\n")
                f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in rows:
                f.write(f"| {ts} | {_hw_label()} | {_build_mode()} | "
                        f"{r['model']} | {r['prompt_len']} | {r['gen_tokens']} | "
                        f"{r['prefill_s']:.3f} | {r['prefill_tok_s']:.2f} | "
                        f"{r['decode_tok_s']:.2f} |\n")
        print(f"\nappended to {rfile}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
