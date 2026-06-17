#pragma once

// Process-global fork-join thread pool for row-parallel kernels.
//
// The pool is created lazily on first use and lives for the process. The
// engine serializes every forward pass through `forward_mu_`, so only one
// caller ever drives the pool at a time; workers are pure C++ (they never
// touch Python / the GIL). Parallelizing over output channels keeps results
// bit-for-bit identical to the serial path, each out[m, n] is still computed
// by exactly one thread as the same serial dot over K, so no reduction order
// changes and the numerical-equality gates hold unchanged.

#include <cstdint>
#include <functional>
#include <utility>

namespace llmengine {

// Number of participating threads (caller + workers) for parallel_for. Read
// once from LLMENGINE_NUM_THREADS, else std::thread::hardware_concurrency(),
// clamped to [1, 64]. Cheap and spawns nothing, only pool_parallel_for
// materializes the worker threads.
int pool_size();

// Type-erased fork-join over [0, N): splits the range into pool_size() chunks,
// runs fn(begin, end) on each (caller takes one chunk, workers the rest), and
// blocks until all complete. Lazily creates the worker pool on first call.
void pool_parallel_for(int N, const std::function<void(int, int)>& fn);

// Work (e.g. N*K MACs) below which fork-join overhead isn't worth it. Tuned so
// real-1B matmuls (N*K in the millions) parallelize while tiny-fixture matmuls
// (tens of thousands) run serially, the tiny test suite then sees no pool and
// no behavior change.
inline constexpr std::int64_t kParallelWorkThreshold = 1 << 18;

// Run fn(begin, end) over the output range [0, N), parallelized across the
// pool when `work` clears the threshold and the pool has >1 thread; otherwise
// serial in the caller. fn must be safe to run concurrently on disjoint
// sub-ranges (each chunk writes disjoint outputs). The serial path constructs
// no std::function, so small matmuls pay exactly zero overhead.
template <class F>
inline void parallel_for_rows(int N, std::int64_t work, F&& fn) {
    if (N < 2 || work < kParallelWorkThreshold || pool_size() <= 1) {
        fn(0, N);
        return;
    }
    pool_parallel_for(N, std::function<void(int, int)>(std::forward<F>(fn)));
}

}  // namespace llmengine
