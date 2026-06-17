# Benchmark results

`prefill_tok/s` = `prompt_len / prefill_s` (one-shot prefill). `decode_tok/s`
is measured after a warmup pass so it excludes the one-time
ModelWeightsRef materialization. Single-sequence, M-series Apple Silicon,
perf build, scalar attention path.

## Single-sequence decode (Phase 4–5)

| Mode | Static memory (real 1B, runtime) | KV memory (per 1024 tok) | Decode tok/s | vs FP32 |
|---|---|---|---|---|
| FP32 weights / FP32 KV | ~2.5 GiB (tied lm_head shares embed) | 64 MiB | 9.57  | 1.00× |
| FP16 weights / FP16 KV | ~2.7 GiB (linear FP16 + embed FP32 + lm_head FP16 copy) | 32 MiB | 10.61 | 1.11× |
| INT8 weights / FP16 KV | ~2.13 GiB (linear INT8 + embed FP32 + lm_head INT8 copy) | 32 MiB | 11.27 | 1.18× |

INT8 cuts the *linear-weight* portion by 4× vs FP32 (~975 MiB vs ~3.9 GiB)
while top-1 accuracy is preserved (top-5 set overlap 4.5/5 on average) and
the model still decodes `"The capital of France is Paris. The"`
identically across all three modes. The static-memory totals above are
higher than the linear-weight portion alone because:
  - `embed_tokens` always lives in FP32 (~525 MiB on real 1B), its
    consumer is `embed_lookup_f32`, which expects FP32 rows. F16 / I8
    embed quantization is a follow-up.
  - In F16 / I8 modes the `lm_head` matrix is materialized in the chosen
    dtype (cannot alias the FP32 embed). Only the F32 tied path
    actually shares the buffer (saves ~525 MiB on real 1B vs the
    pre-fix-pass behavior that copied it).

FP16 KV is the v6 §3.1 design; it landed in the post-review fixup
(`ContiguousKVCache::Dtype::F16` selected for engine `dtype=fp16` and
`dtype=int8`). `PagedKVCache` is still FP32 (covers a smaller surface
since it's only used by the scheduler path).

## Multi-sequence throughput (Phase 7 + true batched decode)

`ContinuousBatchScheduler::decode_running` now stacks every RUNNING sequence
into one `Engine::forward_decode_batch` call: the projection matmuls (q/k/v/o/
gate/up/down/lm_head) run once at M = batch size against the weight-stationary
kernels (each weight row read once, reused across the batch), while attention
stays per-sequence (each seq has its own paged-KV history and RoPE position).
Aggregate decode `tok/s` over the whole batch, real 1B INT8, scalar attention,
`max_new=32`, single-thread (the Threading section below adds the
multi-core numbers), vs the per-sequence static scheduler:

```
  B=2   static  agg_tok/s=5.77   continuous(batched)  agg_tok/s=9.98   1.73x
  B=4   static  agg_tok/s=6.93   continuous(batched)  agg_tok/s=9.46   1.36x
  B=8   static  agg_tok/s=7.38   continuous(batched)  agg_tok/s=9.19   1.25x
```

Generated tokens are identical in both modes, the matmul reorder is a loop
re-nest that preserves every per-element dot, so batched decode is bit-for-bit
equal to per-seq `generate()` (asserted by `test_*_matches_single_seq` and
`test_batched_decode_ragged_matches_single_seq`).

Honest read: this is a real batched-matmul path now (queries stacked into
`[B, hidden]`, one GEMM per projection), and a strict win, but a modest one
(1.25–1.73x), not the 4–8x a memory-bound server would show. Decode at this
scale is compute-bound, not weight-bandwidth-bound: the per-token FMAs
dominate (the existing single-seq table already notes "the matmul is no longer
the bottleneck"), and batching reuses weight *reads* without cutting *FLOPs*,
B sequences still do B× the multiply-accumulates. So continuous holds a ~flat
~9–10 tok/s single-thread compute ceiling across batch sizes, while the per-seq
static path only climbs toward it as B grows and its per-call overhead
amortizes. The win is largest at small B (overhead + weight-read amortization)
and tapers as compute dominates. The Threading section below confirms the
mechanism: parallelizing the matmul lifts the compute ceiling until weight
bandwidth, which batched decode cuts B×, starts to bind, and the batching win
then grows from 1.14x (1 thread) to 1.65x (8 threads).

A real p99 inter-token-latency measurement requires per-token wall-clock hooks
inside the scheduler, that wiring is still a known gap.

## Threading (row-parallel matmul)

`parallel.hpp` adds a process-global fork-join pool; the matmul kernels split
their output channels [0, N) across it (large N only, tiny matmuls stay serial
via an N*K work threshold, so the tiny-fixture suite never even spawns the
pool). Each `out[m,n]` is still one serial dot, so results are bit-for-bit
identical and every numerical-equality gate holds with threading on.

Default thread count = the performance-core count via
`sysctl hw.perflevel0.logicalcpu`, *not* `hardware_concurrency()`. This matters:
on this 8P+4E machine, handing chunks to the 4 slow E-cores (12 threads) makes
the even-split fork-join block on E-core stragglers and *tanks* the per-seq
path (12-thread lockstep measured slower than single-thread). Override with
`LLMENGINE_NUM_THREADS`.

Real 1B INT8, best-of-3 (to exclude thermal noise from sustained runs):

```
  threads   single-seq decode (generate, max_new=48)
  1          9.7 tok/s
  4         33.8 tok/s  (3.5x)
  8         41.7 tok/s  (4.3x)   <- P-core default

  threads   8-seq batched throughput (continuous, max_new=32)
  1          9.3 tok/s
  8         50.5 tok/s  (5.4x)
```

Threading and batching compound: at 1 thread the batched-vs-per-seq win is
1.14x, but at 8 threads it is 1.65x (50.5 vs 30.6 tok/s), parallelizing the
compute lifts the FLOP ceiling until weight bandwidth (which batched decode
already cuts B×) becomes the binding constraint. That is exactly the win
batched decode was built to deliver, now realized.

The attention kernel is threaded too (per-head): at seq_len=4096 it drops
1.55 -> 0.26 ms/call (5.9x, best-of-50), near-linear on the 8 P-cores, only
slightly sub-linear at very long context where the K/V reads start to bind
memory bandwidth. End-to-end this is still small, at Llama-1B dimensions
attention is <1% of prefill and ~5–10% of long-context decode FLOPs (the
matmuls dominate), so the value is removing the last serial hot loop, not a
headline tok/s jump.

Hand-written NEON attention was tried and *reverted*: under the perf build's
`-O3 -ffast-math` the compiler already auto-vectorizes the scalar q·k and
weighted-V loops, and the explicit-intrinsic version measured ~0.8x (slower).
So the remaining attention lever is fusion, not SIMD, an online-softmax kernel
that walks the paged blocks directly and skips the gather copy, targeting the
memory-bound regime. A lower-overhead pool for the M=1 path is also open.

## Streaming + cancellation (Phase 8)

Engine-native streaming validated through `engine.generate_streaming(prompt,
n, on_token, on_done, cancel_token)`. The HTTP path uses a `janus.Queue`
bridge, `request.is_disconnected()` polling, and the v6 timeout-bug fix
(try/except is inside the loop with `continue` on
`asyncio.TimeoutError` so 100 ms quiet windows don't end the stream).

Cancellation regression: `cancel.cancel()` before the engine starts
results in `on_done("cancelled")` with zero `on_token` calls
(`tests/test_server.py::test_generate_streaming_cancellation`).

## Known gaps (open)

- W16A16 activations, TRIED (scoped: FP16 activation at the matmul
  boundaries via FMLAL, fp16 mode) and reverted. The FMLAL matmul is ~1.2x
  single-thread, but at the default 8 P-cores fp16 decode went 36.2 -> 35.6
  tok/s (flat / slightly negative): once threaded the matmul is
  weight-memory-bandwidth-bound and FMLAL only speeds compute, so the win
  vanishes, not worth the FP16-activation precision shave for a
  single-thread-only gain. The real lever is the int-dot quant path
  (W8A8/sdot), not activation precision (see INT4 below).
- INT4 weights, PROBED (per-row W4A32 NEON matmul, half the weight bytes)
  and not a win: vs int8 at 8 threads it was net neutral-to-slower (q/o
  0.92x, gate/up 0.76x, down 1.03x); only the huge `lm_head` benefited (1.57x,
  the one matmul big enough to be *purely* memory-bound). The nibble-unpack
  compute cancels the byte savings everywhere else, and per-row int4 accuracy is
  ~20x worse than int8 (unusable; group-wise is more accurate but slower). This
  corrects an earlier overclaim that "int4 is the lever", measured with FP32
  activations, it isn't. The genuine quant lever is W8A8 / W4A8: quantize
  *activations* to int8 and use the hardware SDOT / i8mm int-dot (no FP32 widen
, how llama.cpp gets fast int4). That needs a dynamic activation-quant
  pipeline + sdot kernels, the real remaining "v2 stretch", not attempted.
- True batched decode, DONE. `decode_running` now stacks the running
  batch into one `forward_decode_batch` (weight-stationary GEMMs at M=B, with
  per-seq attention) and is bit-for-bit equal to per-seq decode. The
  throughput win is modest (1.25–1.73x) because decode here is compute-bound,
  not bandwidth-bound (see the throughput section above); the large multiplier
  is now gated on the thread-pool item below, not on the batching structure.
- Paged-KV FP16, `BlockManager` + `PagedKVCache` still hold FP32. The
  contiguous KV cache (single-seq engine path) is FP16 in fp16/int8 mode;
  paged is the next move.
- W8A8 with `vdotq_s32`, real INT8 throughput rather than just
  smaller reads. Needs activation calibration.
- llama.cpp `-ngl 0` baseline in this table, comparing only against
  in-tree FP32 right now.
- Real-PPL columns with sample size on a 10k+ token wikitext sample,
  the v6 plan's accuracy gate. Drift tests today only assert top-1 / top-5
  on a single short prompt.
- Persistent thread pool + row-parallel matmul, DONE (`parallel.hpp`).
  Output channels split across a P-core fork-join pool: single-seq decode 4.3x
  and 8-seq batched 5.4x on 8 threads, and the attention head loop is threaded
  too (5.9x at seq_len=4096); see Threading above. Remaining: fused attention
  (online softmax + skip the gather copy, hand-written NEON wasn't a win, the
  compiler already auto-vectorizes the scalar loops), and a lower-overhead pool
  for the M=1 path.

The structural pieces (paged KV, chunked-prefill admission, streaming +
cancellation, OpenAI server) are all in place; the open items are
kernel-level optimizations on top.
