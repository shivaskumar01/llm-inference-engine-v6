#pragma once

#include "llmengine/engine.hpp"
#include "llmengine/paged_kv.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace llmengine {

enum class SeqState { WAITING, PREFILLING, RUNNING, FINISHED };

// One in-flight request, owned by the scheduler. Tests read outcomes via
// SequenceResult after the scheduler drains.
struct Sequence {
    int                       seq_id        = 0;
    std::vector<std::int32_t> token_ids;            // prompt + generated
    int                       prompt_len    = 0;
    int                       prefill_cursor = 0;
    int                       max_new_tokens = 0;
    SeqState                  state         = SeqState::WAITING;
    std::string               finish_reason;        // stop / length / capacity
    std::unique_ptr<PagedKVCache> kv;

    Sequence(int id, std::vector<std::int32_t> prompt, int max_new,
             BlockManager& mgr)
        : seq_id(id),
          token_ids(std::move(prompt)),
          prompt_len(static_cast<int>(token_ids.size())),
          max_new_tokens(max_new),
          kv(std::make_unique<PagedKVCache>(mgr)) {}
};

// Snapshot of a Sequence after it FINISHED, token_ids include the prompt
// plus everything generated up to the terminal transition. Returned in
// enqueue order from the scheduler.
struct SequenceResult {
    int                       seq_id = 0;
    int                       prompt_len = 0;
    std::vector<std::int32_t> token_ids;
    std::string               finish_reason;
};

// Static batching baseline: prefill every seq sequentially, then decode in
// lockstep until every seq is terminal. Cheaper to reason about than
// continuous batching and a useful throughput floor for the comparison
// benchmarks in Phase 7.
class StaticBatchScheduler {
public:
    StaticBatchScheduler(Engine& engine, BlockManager& mgr);

    // Caller hands in (prompt, max_new_tokens). The scheduler creates and
    // owns the Sequence + its PagedKVCache.
    void enqueue(std::vector<std::int32_t> prompt, int max_new_tokens);

    void run_until_done();

    // Returns a copy in enqueue order (sorted by seq_id). The internal
    // results_ vector is appended in completion order, so any C++ caller
    // that needs the documented "enqueue order" contract should go
    // through this accessor rather than a back-channel reference.
    std::vector<SequenceResult> results() const {
        auto out = results_;
        std::sort(out.begin(), out.end(),
                  [](const SequenceResult& a, const SequenceResult& b) {
                      return a.seq_id < b.seq_id;
                  });
        return out;
    }
    void clear_results() { results_.clear(); }
    std::size_t pending() const { return seqs_.size(); }

private:
    Engine&                                engine_;
    BlockManager&                          mgr_;
    int                                    next_id_ = 0;
    std::vector<std::unique_ptr<Sequence>> seqs_;
    std::vector<SequenceResult>            results_;
};

// Continuous batching with chunked-prefill admission and per-step decode.
// New sequences enter PREFILLING; each step admits up to max_concurrent,
// then advances every PREFILLING by at most `max_prefill_tokens_per_step`
// total tokens across the batch. RUNNING seqs take one decode step per
// scheduler iteration. KV pool exhaustion terminates a seq with
// finish_reason="capacity".
class ContinuousBatchScheduler {
public:
    ContinuousBatchScheduler(Engine& engine, BlockManager& mgr,
                             int max_concurrent = 64,
                             int max_prefill_tokens_per_step = 256);

    void enqueue(std::vector<std::int32_t> prompt, int max_new_tokens);

    // One scheduler iteration. Returns true if any work remains.
    bool step();

    // Run step() until idle.
    void run_until_idle();

    bool idle() const {
        return running_.empty() && prefilling_.empty() && waiting_.empty();
    }

    int  max_prefill_tokens_per_step() const { return budget_; }
    void set_max_prefill_tokens_per_step(int v) {
        if (v <= 0)
            throw std::invalid_argument(
                "max_prefill_tokens_per_step must be >= 1");
        budget_ = v;
    }

    // Same contract as StaticBatchScheduler::results(), returns a copy in
    // enqueue order (sorted by seq_id). Internal results_ is completion
    // order so finalize_finished can append cheaply.
    std::vector<SequenceResult> results() const {
        auto out = results_;
        std::sort(out.begin(), out.end(),
                  [](const SequenceResult& a, const SequenceResult& b) {
                      return a.seq_id < b.seq_id;
                  });
        return out;
    }
    void clear_results() { results_.clear(); }

private:
    void advance_prefilling();
    void decode_running();
    // Move any FINISHED seqs out of `list`, release their KV, and append
    // their results to results_.
    void finalize_finished(std::vector<std::unique_ptr<Sequence>>& list);

    Engine&                                engine_;
    BlockManager&                          mgr_;
    std::queue<std::unique_ptr<Sequence>>  waiting_;
    std::vector<std::unique_ptr<Sequence>> prefilling_;
    std::vector<std::unique_ptr<Sequence>> running_;
    int                                    max_concurrent_;
    int                                    budget_;
    int                                    next_id_ = 0;
    std::vector<SequenceResult>            results_;
};

}  // namespace llmengine
