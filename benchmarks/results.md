# Benchmark results

`prefill_tok/s` = `prompt_len / prefill_s` (one-shot prefill). `decode_tok/s`
is measured after a warmup pass so it excludes the one-time
ModelWeightsRef materialization. Single-sequence, M-series Apple Silicon,
perf build, scalar attention path.

## Single-sequence decode (Phase 4–5)

| Mode | Static memory (real 1B, runtime) | KV memory (per 1024 tok) | Decode tok/s | vs FP32 |
|---|---|---|---|---|
| FP32 weights / FP32 KV | ~2.5 GiB (tied lm_head shares embed) | 64 MiB | 9.57  | 1.00× |
| FP16 weights / **FP16 KV** | ~2.7 GiB (linear FP16 + embed FP32 + lm_head FP16 copy) | 32 MiB | 10.61 | 1.11× |
| INT8 weights / **FP16 KV** | ~2.13 GiB (linear INT8 + embed FP32 + lm_head INT8 copy) | 32 MiB | 11.27 | 1.18× |

INT8 cuts the *linear-weight* portion by 4× vs FP32 (~975 MiB vs ~3.9 GiB)
while top-1 accuracy is preserved (top-5 set overlap 4.5/5 on average) and
the model still decodes `"The capital of France is Paris. The"`
identically across all three modes. The static-memory totals above are
higher than the linear-weight portion alone because:
  - `embed_tokens` always lives in FP32 (~525 MiB on real 1B) — its
    consumer is `embed_lookup_f32`, which expects FP32 rows. F16 / I8
    embed quantization is a follow-up.
  - In F16 / I8 modes the `lm_head` matrix is materialized in the chosen
    dtype (cannot alias the FP32 embed). Only the **F32** tied path
    actually shares the buffer (saves ~525 MiB on real 1B vs the
    pre-fix-pass behavior that copied it).

FP16 KV is the v6 §3.1 design; it landed in the post-review fixup
(`ContiguousKVCache::Dtype::F16` selected for engine `dtype=fp16` and
`dtype=int8`). `PagedKVCache` is still FP32 (covers a smaller surface
since it's only used by the scheduler path).

## Multi-sequence throughput (Phase 7)

Workload: 4 prompts, varying lengths (5–12 tokens), `max_new=5` each, real
1B INT8, scalar attention. Reported as aggregate `tok/s` over the whole
batch.

```
  static                         elapsed=  3.747s  gen=  20  agg_tok/s=5.34
  continuous@budget=4            elapsed=  3.739s  gen=  20  agg_tok/s=5.35
  continuous@budget=16           elapsed=  3.757s  gen=  20  agg_tok/s=5.32
  continuous@budget=64           elapsed=  3.751s  gen=  20  agg_tok/s=5.33
  continuous@budget=256          elapsed=  3.786s  gen=  20  agg_tok/s=5.28
```

**Honest read:** continuous batching as implemented is a *scheduler state
machine* (admission, prefill cursor, capacity-aware decode) on top of
per-sequence `forward_step_paged`. It is **not** a batched-matmul
throughput multiplier. The decode loop iterates running sequences and
runs one engine forward each; queries are never stacked into `[batch,
hidden]`. The headline throughput is therefore the single-seq rate plus
overlap between admission/prefill and decode. The v6 plan called for true
batched matmul; that's tracked in the "known gaps" section.

The budget sweep changes the long-prompt admission cadence but doesn't
move aggregate throughput on a workload of mostly-short prompts. A real
p99 inter-token-latency measurement requires per-token wall-clock hooks
inside the scheduler — that wiring is also a known gap.

## Streaming + cancellation (Phase 8)

Engine-native streaming validated through `engine.generate_streaming(prompt,
n, on_token, on_done, cancel_token)`. The HTTP path uses a `janus.Queue`
bridge, `request.is_disconnected()` polling, and the v6 timeout-bug fix
(try/except is **inside** the loop with `continue` on
`asyncio.TimeoutError` so 100 ms quiet windows don't end the stream).

Cancellation regression: `cancel.cancel()` before the engine starts
results in `on_done("cancelled")` with zero `on_token` calls
(`tests/test_server.py::test_generate_streaming_cancellation`).

## Known gaps (open)

- **W16A16 activations** — engine activations and scratch are still FP32
  even in fp16/int8 weight modes. The plan's "W8A16" name was aspirational
  for this implementation; what shipped is W8A32 (and W8A16 only insofar
  as KV is FP16). Closing this requires replumbing every activation
  buffer (`h`, `h_norm`, `q_buf`, etc.) and the kernel signatures.
- **True batched decode** — `ContinuousBatchScheduler::decode_running`
  loops over sequences and calls per-sequence forward; queries are not
  stacked, no batched matmul. The state machine is correct (admission,
  prefill budget, capacity termination) but the perf claim from batching
  isn't established.
- **Paged-KV FP16** — `BlockManager` + `PagedKVCache` still hold FP32. The
  contiguous KV cache (single-seq engine path) is FP16 in fp16/int8 mode;
  paged is the next move.
- **W8A8 with `vdotq_s32`** — real INT8 throughput rather than just
  smaller reads. Needs activation calibration.
- **llama.cpp `-ngl 0` baseline** in this table — comparing only against
  in-tree FP32 right now.
- **Real-PPL columns** with sample size on a 10k+ token wikitext sample —
  the v6 plan's accuracy gate. Drift tests today only assert top-1 / top-5
  on a single short prompt.
- **Persistent thread pool + row-parallel matmul** — designed in v6 §4.2,
  never wired.

The structural pieces (paged KV, chunked-prefill admission, streaming +
cancellation, OpenAI server) are all in place; the open items are
kernel-level optimizations on top.
