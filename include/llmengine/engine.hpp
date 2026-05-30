#pragma once

#include "llmengine/config.hpp"
#include "llmengine/kv_cache.hpp"
#include "llmengine/model.hpp"
#include "llmengine/paged_kv.hpp"
#include "llmengine/weights.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace llmengine {

// Thread-safe cancellation flag handed in to generate_streaming. The
// streaming worker checks is_cancelled() between forward steps; the
// HTTP layer calls cancel() when a request disconnects so we don't keep
// burning CPU on a stream nobody is listening to.
class CancelToken {
public:
    void cancel()       { flag_.store(true,  std::memory_order_release); }
    bool is_cancelled() const { return flag_.load(std::memory_order_acquire); }
    void reset()        { flag_.store(false, std::memory_order_release); }
private:
    std::atomic<bool> flag_{false};
};

// Phase 0: load config + safetensors metadata.
// Phase 1: scalar FP32 forward.
// Phase 3: ContiguousKVCache + generate.
// Phase 4.2: optional FP16 linear-weight storage path (`dtype="fp16"`).
//
// Thread safety: every public forward entry point —
//   forward_logits, generate, generate_streaming, forward_logits_paged,
//   forward_step_paged
// — serializes through `forward_mu_`. Concurrent callers (FastAPI workers,
// schedulers driven from other threads, direct Python clients) see correct
// per-call output; the mutable scratch buffers are exclusive for the
// duration of each call. generate_streaming uses a unique_lock and
// releases around `on_token`/`on_done` so user callbacks can re-enter the
// engine without deadlocking on std::mutex. The private
// forward_step_paged_locked variant is the lock-free workhorse used
// internally by forward_logits_paged (which already owns the mutex).
class Engine {
public:
    explicit Engine(const std::string& model_dir,
                    ModelWeightsRef::LinearStorage storage =
                        ModelWeightsRef::LinearStorage::F32);

    const ModelConfig& config()  const { return cfg_; }
    const WeightMap&   weights() const { return weights_; }

    ModelWeightsRef::LinearStorage linear_storage() const { return storage_; }

    // Largest valid (pos + 1) accepted by forward_step / forward_step_paged.
    // Pure function of the config — safe to call before the model has been
    // materialized, and what the schedulers use for their enqueue-time check.
    int max_pos() const { return compute_max_pos(cfg_); }

    std::uintptr_t debug_weight_ptr(const std::string& name) const {
        return weights_.debug_ptr(name);
    }

    // Debug-only: pointer to the runtime LM-head buffer in F32 mode (after
    // ModelWeightsRef::load materializes / aliases). Used by tests to
    // verify the tied-embedding share isn't accidentally a copy.
    std::uintptr_t debug_lm_head_f32_ptr() {
        ensure_model_loaded();
        return reinterpret_cast<std::uintptr_t>(model_.lm_head_f32());
    }
    std::uintptr_t debug_embed_tokens_ptr() {
        ensure_model_loaded();
        return reinterpret_cast<std::uintptr_t>(model_.embed_tokens().data());
    }

    void ensure_model_loaded();

    std::vector<float> forward_logits(const std::vector<std::int32_t>& ids);

    std::pair<std::vector<std::int32_t>, std::string>
    generate(const std::vector<std::int32_t>& prompt_ids, int max_new_tokens);

    // Token-by-token streaming with cooperative cancellation. on_token is
    // invoked with each sampled token id; on_done with the terminal reason
    // ("stop" / "length" / "cancelled"). Both callbacks run on this thread
    // (the caller's worker thread, not the engine's). The Python pybind
    // wrapper acquires the GIL inside the wrapped callbacks.
    void generate_streaming(
        const std::vector<std::int32_t>& prompt_ids,
        int max_new_tokens,
        std::function<void(int)> on_token,
        std::function<void(const std::string&)> on_done,
        CancelToken& cancel);

    // Phase 6: same forward, but the per-layer K/V history lives in a
    // BlockManager-backed PagedKVCache. Used to verify paged-vs-contiguous
    // equivalence and as the building block for Phase 7's scheduler.
    std::vector<float> forward_logits_paged(
        const std::vector<std::int32_t>& ids,
        BlockManager& mgr);

    // Single-token paged forward. Public so the Phase 7 scheduler can
    // drive its own loop. Returns false (and leaves logits_out unset) on
    // KV-pool exhaustion — the scheduler should terminate that seq with
    // finish_reason="capacity".
    //
    // Thread safe: takes forward_mu_ for the call, so it can safely be
    // interleaved with engine.generate() / forward_logits() etc. from
    // another thread. The internal forward_logits_paged path uses the
    // private _locked variant since it already holds the mutex.
    [[nodiscard]] bool forward_step_paged(int token_id, int pos,
                                          PagedKVCache& kv,
                                          float* logits_out);

    // Batched single decode step over B sequences — the continuous-batching
    // hot path. tokens[i]/positions[i] advance kvs[i] by one token; logits for
    // seq i land in row i of logits_out (row-major [B, vocab_size]). alive[i]
    // is set to 0 iff seq i's KV pool was exhausted this step (its logits row
    // is then undefined; the caller terminates that seq with "capacity"). The
    // projection matmuls run once at M=B — the weight-stationary kernels reuse
    // each weight row across the batch — while attention stays per-seq since
    // each sequence has its own paged KV history and RoPE position.
    // Thread-safe: serializes on forward_mu_.
    void forward_decode_batch(const std::vector<std::int32_t>& tokens,
                              const std::vector<int>& positions,
                              const std::vector<PagedKVCache*>& kvs,
                              float* logits_out,
                              std::vector<char>& alive);

private:
    void forward_step(int token_id, int pos,
                      ContiguousKVCache& kv,
                      float* logits_out);

    // Same as forward_step_paged but assumes forward_mu_ is already held by
    // the caller. Only used internally from forward_logits_paged.
    [[nodiscard]] bool forward_step_paged_locked(int token_id, int pos,
                                                  PagedKVCache& kv,
                                                  float* logits_out);

    void ensure_paged_scratch(int max_seq_len);
    void ensure_batch_scratch(int batch_size);

    ModelConfig     cfg_;
    WeightMap       weights_;
    ModelWeightsRef model_;
    ModelWeightsRef::LinearStorage storage_;
    bool            model_built_ = false;

    // Forward-pass scratch. Serialized by `forward_mu_` — any public API
    // that mutates these holds the mutex for the full call. See the
    // class-level thread-safety note above.
    mutable std::mutex forward_mu_;
    std::vector<float> h_, h_norm_;
    std::vector<float> q_buf_, k_buf_, v_buf_;
    std::vector<float> attn_out_, o_out_;
    std::vector<float> gate_, up_, ffn_out_;

    // Paged-attention gather scratch (per-call sized).
    std::vector<float> k_gather_, v_gather_;

    // Batched-decode scratch ([batch_size, dim] row-major), grown on demand
    // by ensure_batch_scratch. Used only by forward_decode_batch; the M=1
    // forward_step path keeps its own single-row scratch above.
    std::vector<float> hb_, hb_norm_;
    std::vector<float> qb_, kb_, vb_;
    std::vector<float> attn_b_, ob_;
    std::vector<float> gateb_, upb_, ffnb_;

    void ensure_scratch_loaded();
};

}  // namespace llmengine
