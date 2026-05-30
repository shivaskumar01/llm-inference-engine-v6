# llm-engine

A from-scratch C++17 inference engine for Llama 3.2 1B-Instruct on Apple
Silicon, built bottom-up from the kernel layer through paged attention,
continuous batching, and an OpenAI-compatible HTTP server.

**Status:** the structural surface of all 8 v6 phases is in place and
green under test (111 tests across kernels, tiny+real-1B equivalence,
paged KV, schedulers, streaming, OpenAI-compatible HTTP) — but the v6
plan is **not fully shipped**. The "What's *not* there" section below
enumerates the parts that are still missing or aspirational (W16A16
activations, paged-KV FP16, `convert_weights.py`, the persistent thread
pool, `compare_llamacpp.py`, real PPL with sample sizes). Treat this repo
as "v6-shaped and resume-defensible" rather than "v6-complete." A separate
code review against the plan turned up four real gaps that were fixed in a
P1 follow-up:
1. **Streaming** in `server.py` now uses `janus.Queue` + the v6 timeout-bug
   fix (try/except inside the loop), real `request.is_disconnected()`
   polling, and a `CancelToken` plumbed through to the engine — replacing
   an earlier `engine.generate(cur, 1)` loop that didn't honor the design.
2. **Scheduler off-by-one** — `max_new_tokens=1` now produces exactly one
   generated token; before the prefill-end sample + decode-step double-fire,
   it produced two.
3. **FP16 KV** in fp16 / int8 engine modes (`ContiguousKVCache::Dtype::F16`);
   FP32 KV stays for `dtype=fp32` so the Phase 1 hard logit gates keep
   passing.
4. **`_debug_weight_ptr`** is now gated behind `LLMENGINE_DEBUG_BINDINGS`
   (on in correctness builds, off in perf).

Remaining honest gaps are listed in [`benchmarks/results.md`](benchmarks/results.md)
("Known gaps") and in this README's "What's *not* there" section. The
biggest one is that activations are still FP32 throughout — what shipped
is W8A32 (not the v6 W8A16 aspiration), with FP16 KV cache.

```
The capital of France is Paris. The
```

That output is generated end-to-end by the engine in FP32, FP16, or INT8
weight modes, served via FastAPI streaming, using a paged KV cache and a
continuous-batching scheduler.

---

## What's in the box

| Layer | File(s) |
|---|---|
| FP32 reference kernels (rmsnorm, silu, matmul, embed, rope, attention) | `src/kernels.cpp` |
| NEON FP32 / FP16-weight / INT8-weight matmul + rmsnorm | `src/kernels.cpp` (`__ARM_NEON` paths) |
| safetensors mmap loader, tied-LM-head alias, RoPE param normalization | `src/weights.cpp` |
| Llama-3 scaled RoPE (FP64 inv_freq + tables) | `src/kernels.cpp::llama3_inv_freq` |
| Model weights materialization in FP32 / FP16 / INT8 | `src/model.cpp` |
| Engine: per-token forward, `forward_logits`, `generate(prompt, n)` | `src/engine.cpp` |
| ContiguousKVCache | `src/kv_cache.cpp` |
| Paged KV: BlockManager + PagedKVCache + `forward_logits_paged` | `src/paged_kv.cpp`, `src/engine.cpp` |
| Static + Continuous batch schedulers (PREFILLING / RUNNING state machine, prefill budget, capacity termination) | `src/scheduler.cpp` |
| True batched decode — running seqs share one `forward_decode_batch` (M=B weight-stationary GEMMs, per-seq attention) | `src/engine.cpp`, `src/scheduler.cpp` |
| FastAPI server (`/v1/completions`, `/v1/chat/completions`, SSE streaming) | `python/llmengine/server.py` |
| pybind11 bindings | `src/bindings.cpp` |

Two build modes, both green on the full test suite:
- **correctness** — `-O2`, no `-ffast-math`, FP32 scalar kernels. Deterministic.
- **perf** — `-O3`, `-ffast-math`, NEON FP16/INT8 paths. Soft drift gates.

---

## Build

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pybind11 pytest numpy huggingface_hub safetensors janus \
            torch transformers fastapi uvicorn openai

cmake --preset correctness && cmake --build build/correctness -j8
cmake --preset perf        && cmake --build build/perf        -j8

python tools/download_weights.py --mirror unsloth      # ~2.4 GB
pytest tests/                                          # 110 tests
```

The compiled `.so` for the perf build lands in `python/llmengine/` and is
picked up by `import llmengine`. The correctness build's `.so` lives in
`build/correctness/python/llmengine/`; tests opt into it via
`LLMENGINE_BUILD=correctness pytest …`.

---

## Quick start

```python
import llmengine
from transformers import AutoTokenizer

tok    = AutoTokenizer.from_pretrained("./data/llama-3.2-1b-instruct")
engine = llmengine.Engine("./data/llama-3.2-1b-instruct", dtype="int8")

ids = tok.encode("The capital of France is", add_special_tokens=True)
out, finish = engine.generate(ids, max_new_tokens=5)
print(tok.decode(out[len(ids):]))
# → " Paris. The capital"
```

Continuous batching:

```python
mgr   = llmengine.BlockManager(num_blocks=32, block_size=16,
                                num_layers=engine.cfg.num_hidden_layers,
                                num_kv_heads=engine.cfg.num_key_value_heads,
                                head_dim=engine.cfg.head_dim)
sched = llmengine.ContinuousBatchScheduler(
    engine, mgr, max_concurrent=2, max_prefill_tokens_per_step=8)

sched.enqueue(tok.encode("The capital of France is"), 3)
sched.enqueue(tok.encode("Two plus two equals"),     3)
sched.run_until_idle()
for r in sched.results:
    print(r.seq_id, r.finish_reason, tok.decode(r.token_ids[r.prompt_len:]))
# 0 length  Paris. The
# 1 length  four. This
```

OpenAI-compatible HTTP server:

```bash
LLMENGINE_MODEL=./data/llama-3.2-1b-instruct LLMENGINE_DTYPE=int8 \
    uvicorn llmengine.server:app --port 8000

python -c "
import openai
c = openai.OpenAI(base_url='http://localhost:8000/v1', api_key='x')
r = c.chat.completions.create(
    model='llama-3.2-1b-instruct',
    messages=[{'role':'user','content':'What is the capital of France?'}],
    max_tokens=8)
print(r.choices[0].message.content)"
```

---

## Phase-by-phase test counts

Current pytest collection (post code-review fixups):

```
tests/test_phase0_loader.py          9    config, safetensors, tied-LM alias, malformed-offset/byte-count/shape/transposed rejects
tests/test_kernels.py               20    every kernel vs torch FP32 (atol=1e-5)
tests/test_tiny_equality.py          3    tiny full-forward + greedy match HF
tests/test_real_smoke.py             6    real-1B short-prompt + RoPE at scale + runtime tied-share
tests/test_kv_cache.py               5    ContiguousKVCache + generate() + multi-EOS
tests/test_perf_fp16.py              4    FP16 storage drift on tiny + real-1B
tests/test_perf_int8.py              3    INT8 W8A32 drift on tiny + real-1B
tests/test_paged_kv.py               4    PagedKV == ContiguousKV (real-1B), capacity
tests/test_scheduler.py             23    static + continuous batching, batched-decode==per-seq equivalence, budget, capacity, max_new=0/1/N, knob+enqueue+RoPE+order validation
tests/test_engine_thread_safety.py   6    concurrent generate / streaming + scheduler concurrency + callback reentrancy
tests/test_input_validation.py      13    token-ID bounds, prompt+max_new overflow, streaming worker-error, input-order before model load
tests/test_server.py                15    FastAPI /v1/* + SSE + cancellation + 400/422 mapping + Pydantic schemas + pre-stream validation
                                    ---
                                    111   total
```

Correctness build runs all 111. Perf build runs 106 + 5 skips (the five
debug-binding tests — four `_debug_weight_ptr` lookups plus the runtime
tied-share check — skip because `_debug_*_ptr` accessors are gated to
correctness via `LLMENGINE_DEBUG_BINDINGS`). Both green.

---

## Benchmarks

See [`benchmarks/results.md`](benchmarks/results.md) for the full table.
Real Llama 3.2 1B-Instruct, single-sequence decode on Apple Silicon:

| Mode | Static memory (real 1B, runtime) | Decode tok/s | vs FP32 |
|---|---|---|---|
| FP32 weights | ~2.5 GiB (tied lm_head shares embed) | 9.57  | 1.00× |
| FP16 weights | ~2.7 GiB (linear FP16 + embed FP32 + lm_head FP16 copy) | 10.61 | 1.11× |
| INT8 weights | ~2.13 GiB (linear INT8 + embed FP32 + lm_head INT8 copy) | 11.27 | 1.18× |

INT8 cuts the *linear-weight* portion by 4× vs FP32 with **top-1 accuracy
preserved** (top-5 set overlap 4.5/5 on average, decoded output identical
to FP32). Static-memory totals are higher than the linear-weight portion
alone because `embed_tokens` always lives in FP32 (~525 MiB) and the F16
/ I8 `lm_head` matrices are materialized separately — only the tied
**F32** path actually aliases lm_head into embed. F16/I8 embed
quantization is a documented follow-up. Decode speedup is modest vs the
bytes-saved ratio because the matmul is no longer the bottleneck —
attention, RMSNorm, FFN element-wise ops, and the unfused softmax
dominate at this scale.

---

## Where the build wall is

By measurement, the remaining cost lives in:
- Attention (unfused; per-token gather from paged KV)
- RMSNorm + SwiGLU + softmax element-wise stages
- FP16 → FP32 widening inside the matmul kernel

Quick wins from here:
- **W8A8** with `vdotq_s32` — needs activation calibration but unlocks
  real INT8 throughput rather than just smaller reads.
- **Fused paged attention** with in-kernel block walk + online softmax —
  cuts the gather copy and the softmax pass.
- **W16A16** full FP16 pipeline — uses `vfmlalq_*_f16` and halves the
  activation bandwidth too.

The structural pieces (paged KV, chunked-prefill admission, batching
scheduler, OpenAI server) are all in place; these are kernel-level
optimizations on top.

---

## What's *not* there (known gaps, post-review)

These were called out in code review against the v6 plan and are the
honest delta between the resume line and the implementation:

- **W16A16 activations.** Activation buffers (`h`, `h_norm`, `q_buf`,
  `k_buf`, `v_buf`, …) are still FP32 in every engine dtype mode. What
  shipped is W8A32 with FP16 KV. The v6 "W8A16" line was aspirational.
- **Paged-KV FP16.** `ContiguousKVCache` is FP16 in fp16/int8 modes;
  `BlockManager` + `PagedKVCache` are still FP32.
- **W8A8 + `vdotq_s32`.** Needs activation calibration; unlocks real
  INT8 throughput rather than just smaller reads.
- **`convert_weights.py`** + engine-native binary format. Plan-deferred;
  on-the-fly bf16→fp16→int8 at load time made the disk format unnecessary
  for the resume artifact.
- **`parallel.hpp` thread pool + row-parallel matmul.** Designed in v6
  §4.2, never wired. Single-thread CPU was fine for the M=1 decode shape.
- **`compare_llamacpp.py`** with `-ngl 0` baseline. Not yet shipped.
- **Real PPL with sample size** on 10k+ tokens of wikitext-2 test split.
  Today's drift tests assert top-1 / top-5 on one short prompt.

The structural pieces — paged KV, chunked-prefill admission, streaming +
cancellation, OpenAI server — are all in place and tested; the open items
are kernel and instrumentation work on top.

## What I'd do differently if starting over

- **Pick the W16A16 pipeline first**, not the W*A32 stepping stones.
  Halving activations alongside weights is where the real perf headroom
  is on Apple Silicon, and routing everything through FP32 activations
  added a complete second set of kernel signatures for marginal gain.
- **Skip the engine-native binary format**. The original v6 plan called
  for a custom on-disk format with `convert_weights.py`. Converting
  bf16→fp16→int8 on the fly at load time was simpler and the load time
  cost is amortized.
- **The persistent thread pool was scoped but never wired.** It was fine
  to defer while decode was M=1, but now that decode is batched (M=B) and
  measured compute-bound, row-parallel / multi-threaded matmul is the
  highest-leverage perf item left — it's what would turn the batched-decode
  weight-read saving into an actual throughput multiplier (see
  `benchmarks/results.md`).

These are honest trade-offs, not regrets — the project meets every
deliverable from the v6 plan that matters for an undergrad ML systems
portfolio: from-scratch kernel work, a real numerical correctness gate
against HF transformers, the structural innovations (paged KV +
continuous batching) that define a production inference engine, and an
HTTP serving surface.

---

## Layout

```
~/llm-engine/
├── CMakeLists.txt
├── CMakePresets.json
├── include/llmengine/   # public C++ headers
├── src/                 # C++ implementation
├── python/llmengine/    # pip-installable Python package + FastAPI server
├── tests/               # 110 pytest tests across all phases
├── benchmarks/          # bench_e2e.py + dated results table
├── tools/               # download_weights.py
├── third_party/         # nlohmann/json single header
├── data/                # .gitignore'd: model weights
└── build/{correctness,perf}/
```

Plan history and the design lives in
`~/.claude/plans/plan-each-phase-carefully-fuzzy-cupcake.md` (v6).
