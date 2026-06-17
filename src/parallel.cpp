#include "llmengine/parallel.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace llmengine {

namespace {

// Even split of [0, total) into nchunks contiguous ranges; chunk c is
// [c*total/nchunks, (c+1)*total/nchunks). int64 math avoids overflow for the
// largest N we see (lm_head N=128256). Empty trailing chunks (when
// total < nchunks) are no-ops.
inline std::pair<int, int> chunk_bounds(int c, int total, int nchunks) {
    const long long b = static_cast<long long>(c)     * total / nchunks;
    const long long e = static_cast<long long>(c + 1) * total / nchunks;
    return {static_cast<int>(b), static_cast<int>(e)};
}

// Blocking fork-join pool. parallel_for posts one job (a function + a range),
// the caller runs chunk 0 itself, the n_-1 worker threads run chunks 1..n_-1,
// and the caller waits for them. A generation counter + predicate waits make
// it robust to missed/early notifications. Only one job runs at a time
// (parallel_for blocks until the previous one fully drains), which matches the
// engine's single-submitter discipline.
class ThreadPool {
public:
    explicit ThreadPool(int nthreads) : n_(std::max(1, nthreads)) {
        for (int i = 1; i < n_; ++i)
            workers_.emplace_back([this, i] { worker_loop(i); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            shutdown_ = true;
        }
        cv_start_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
    }

    int size() const { return n_; }

    void parallel_for(int total, const std::function<void(int, int)>& fn) {
        if (n_ <= 1) { fn(0, total); return; }
        {
            std::lock_guard<std::mutex> lk(m_);
            job_fn_    = &fn;
            job_total_ = total;
            remaining_ = n_ - 1;     // workers; the caller runs its own chunk
            ++generation_;
        }
        cv_start_.notify_all();

        // Caller runs chunk 0 concurrently with the workers.
        const auto [b, e] = chunk_bounds(0, total, n_);
        fn(b, e);

        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [this] { return remaining_ == 0; });
        job_fn_ = nullptr;
    }

private:
    void worker_loop(int idx) {
        unsigned long seen = 0;
        for (;;) {
            const std::function<void(int, int)>* fn;
            int total;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_start_.wait(lk, [this, &seen] {
                    return shutdown_ || generation_ != seen;
                });
                if (shutdown_) return;
                seen  = generation_;
                fn    = job_fn_;
                total = job_total_;
            }
            const auto [b, e] = chunk_bounds(idx, total, n_);
            (*fn)(b, e);
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--remaining_ == 0) cv_done_.notify_one();
            }
        }
    }

    int n_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_start_, cv_done_;
    const std::function<void(int, int)>* job_fn_ = nullptr;
    int job_total_ = 0;
    int remaining_ = 0;
    unsigned long generation_ = 0;
    bool shutdown_ = false;
};

}  // namespace

int pool_size() {
    static const int n = [] {
        if (const char* e = std::getenv("LLMENGINE_NUM_THREADS")) {
            const int v = std::atoi(e);
            if (v >= 1) return std::min(v, 64);
        }
#if defined(__APPLE__)
        // Default to the performance-core count, NOT hardware_concurrency().
        // On Apple Silicon the efficiency cores are much slower, and in an
        // even-split fork-join the caller blocks on the slowest chunk, so
        // handing chunks to E-cores tanks throughput (measured: using all 12
        // cores is *slower* than a single thread for the M=1 per-seq path).
        // perflevel0 is the highest-performance core class (the P-cores).
        int pcores = 0;
        std::size_t sz = sizeof(pcores);
        if (sysctlbyname("hw.perflevel0.logicalcpu", &pcores, &sz, nullptr, 0) == 0
            && pcores >= 1)
            return std::min(pcores, 64);
#endif
        const unsigned hc = std::thread::hardware_concurrency();
        return std::min(hc ? static_cast<int>(hc) : 1, 64);
    }();
    return n;
}

void pool_parallel_for(int N, const std::function<void(int, int)>& fn) {
    static ThreadPool pool(pool_size());
    pool.parallel_for(N, fn);
}

}  // namespace llmengine
