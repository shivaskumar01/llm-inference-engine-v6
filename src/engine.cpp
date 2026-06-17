#include "llmengine/engine.hpp"
#include "llmengine/kernels.hpp"

#include <algorithm>
#include <stdexcept>

namespace llmengine {

namespace {

// Build-mode and storage-mode dispatch wrapper for the hot kernels.
// Overload set: pick the matmul kernel from the type of the weight pointer.
// Within each overload, route to the NEON variant if the perf build was
// compiled. Forward_step doesn't have to know which preset built it or
// which precision the weights are stored in.
inline void fwd_matmul(const float* a, const float* w,
                       int M, int N, int K, float* out) {
#if defined(LLMENGINE_PERF) && defined(__ARM_NEON)
    llmengine::kernels::matmul_f32_neon(a, w, M, N, K, out);
#else
    llmengine::kernels::matmul_f32(a, w, M, N, K, out);
#endif
}

inline void fwd_matmul(const float* a, const __fp16* w,
                       int M, int N, int K, float* out) {
#if defined(LLMENGINE_PERF) && defined(__ARM_NEON)
    llmengine::kernels::matmul_f16w_f32a_neon(a, w, M, N, K, out);
#else
    llmengine::kernels::matmul_f16w_f32a(a, w, M, N, K, out);
#endif
}

inline void fwd_matmul(const float* a, const std::int8_t* w, const __fp16* scales,
                       int M, int N, int K, float* out) {
#if defined(LLMENGINE_PERF) && defined(__ARM_NEON)
    llmengine::kernels::matmul_int8w_f32a_neon(a, w, scales, M, N, K, out);
#else
    llmengine::kernels::matmul_int8w_f32a(a, w, scales, M, N, K, out);
#endif
}

inline void fwd_rmsnorm(const float* x, const float* w,
                        int dim, float eps, float* out) {
#if defined(LLMENGINE_PERF) && defined(__ARM_NEON)
    llmengine::kernels::rmsnorm_f32_neon(x, w, dim, eps, out);
#else
    llmengine::kernels::rmsnorm_f32(x, w, dim, eps, out);
#endif
}

int argmax(const float* logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; ++i)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

inline void validate_token(int t, int vocab_size, const char* where) {
    if (t < 0 || t >= vocab_size)
        throw std::invalid_argument(
            std::string(where) + ": token id " + std::to_string(t)
            + " out of range [0, " + std::to_string(vocab_size) + ")");
}

inline void validate_token_ids(const std::vector<std::int32_t>& ids,
                                int vocab_size, const char* where) {
    for (auto t : ids) validate_token(t, vocab_size, where);
}

}  // namespace

Engine::Engine(const std::string& model_dir,
               ModelWeightsRef::LinearStorage storage)
    : storage_(storage) {
    cfg_ = load_config(model_dir);
    weights_.load_safetensors_dir(model_dir, cfg_);
}

void Engine::ensure_model_loaded() {
    if (model_built_) return;
    model_.load(cfg_, weights_, storage_);
    model_built_ = true;
}

void Engine::ensure_scratch_loaded() {
    if (!h_.empty()) return;
    const int H = cfg_.hidden_size;
    const int FF = cfg_.intermediate_size;
    const int q_dim  = cfg_.num_attention_heads  * cfg_.head_dim;
    const int kv_dim = cfg_.num_key_value_heads  * cfg_.head_dim;
    h_.resize(H);
    h_norm_.resize(H);
    q_buf_.resize(q_dim);
    k_buf_.resize(kv_dim);
    v_buf_.resize(kv_dim);
    attn_out_.resize(q_dim);
    o_out_.resize(H);
    gate_.resize(FF);
    up_.resize(FF);
    ffn_out_.resize(H);
}

void Engine::forward_step(int token_id, int pos,
                          ContiguousKVCache& kv,
                          float* logits_out) {
    const int H        = cfg_.hidden_size;
    const int FF       = cfg_.intermediate_size;
    const int V        = cfg_.vocab_size;
    const int n_q      = cfg_.num_attention_heads;
    const int n_kv     = cfg_.num_key_value_heads;
    const int hd       = cfg_.head_dim;
    const int q_dim    = n_q  * hd;
    const int kv_dim   = n_kv * hd;
    const int n_layers = cfg_.num_hidden_layers;

    kernels::embed_lookup_f32(model_.embed_tokens().data(),
                              token_id, H, h_.data());

    const auto storage = storage_;
    using LS = ModelWeightsRef::LinearStorage;

    for (int l = 0; l < n_layers; ++l) {
        // Resolve per-layer norm pointers once. Linear-weight pointer types
        // differ across storage modes, so each fwd_matmul site is wrapped
        // in a switch on `storage`. Compiler hoists this nicely.
        const float* attn_norm =
            storage == LS::F32 ? model_.layers()[l].attn_norm
          : storage == LS::F16 ? model_.layers_f16()[l].attn_norm
                               : model_.layers_i8()[l].attn_norm;
        const float* ffn_norm  =
            storage == LS::F32 ? model_.layers()[l].ffn_norm
          : storage == LS::F16 ? model_.layers_f16()[l].ffn_norm
                               : model_.layers_i8()[l].ffn_norm;

        fwd_rmsnorm(h_.data(), attn_norm, H, cfg_.rms_norm_eps, h_norm_.data());

        switch (storage) {
        case LS::F32: {
            const auto& L = model_.layers()[l];
            fwd_matmul(h_norm_.data(), L.W_q, 1, q_dim,  H, q_buf_.data());
            fwd_matmul(h_norm_.data(), L.W_k, 1, kv_dim, H, k_buf_.data());
            fwd_matmul(h_norm_.data(), L.W_v, 1, kv_dim, H, v_buf_.data());
            break;
        }
        case LS::F16: {
            const auto& L = model_.layers_f16()[l];
            fwd_matmul(h_norm_.data(), L.W_q, 1, q_dim,  H, q_buf_.data());
            fwd_matmul(h_norm_.data(), L.W_k, 1, kv_dim, H, k_buf_.data());
            fwd_matmul(h_norm_.data(), L.W_v, 1, kv_dim, H, v_buf_.data());
            break;
        }
        case LS::I8: {
            const auto& L = model_.layers_i8()[l];
            fwd_matmul(h_norm_.data(), L.W_q, L.s_q, 1, q_dim,  H, q_buf_.data());
            fwd_matmul(h_norm_.data(), L.W_k, L.s_k, 1, kv_dim, H, k_buf_.data());
            fwd_matmul(h_norm_.data(), L.W_v, L.s_v, 1, kv_dim, H, v_buf_.data());
            break;
        }
        }

        const float* cos_row = model_.cos_row(pos);
        const float* sin_row = model_.sin_row(pos);
        for (int hh = 0; hh < n_q; ++hh)
            kernels::apply_rope_inplace(&q_buf_[hh * hd], cos_row, sin_row, hd);
        for (int hh = 0; hh < n_kv; ++hh)
            kernels::apply_rope_inplace(&k_buf_[hh * hd], cos_row, sin_row, hd);

        kv.write(l, pos, k_buf_.data(), v_buf_.data());

        // F32 KV: attention reads directly from the cache (zero copy).
        // F16 KV: gather + widen into FP32 scratch first since attention_f32
        // expects FP32 inputs.
        const float* k_data;
        const float* v_data;
        if (kv.dtype() == ContiguousKVCache::Dtype::F32) {
            k_data = kv.k_layer_f32(l);
            v_data = kv.v_layer_f32(l);
        } else {
            ensure_paged_scratch(pos + 1);
            kv.gather_layer(l, pos + 1, k_gather_.data(), v_gather_.data());
            k_data = k_gather_.data();
            v_data = v_gather_.data();
        }
        kernels::attention_f32(q_buf_.data(),
                               k_data, v_data,
                               n_q, n_kv, hd, /*seq_len=*/pos + 1,
                               attn_out_.data());

        switch (storage) {
        case LS::F32:
            fwd_matmul(attn_out_.data(), model_.layers()[l].W_o,
                       1, H, q_dim, o_out_.data());
            break;
        case LS::F16:
            fwd_matmul(attn_out_.data(), model_.layers_f16()[l].W_o,
                       1, H, q_dim, o_out_.data());
            break;
        case LS::I8: {
            const auto& L = model_.layers_i8()[l];
            fwd_matmul(attn_out_.data(), L.W_o, L.s_o, 1, H, q_dim, o_out_.data());
            break;
        }
        }
        for (int i = 0; i < H; ++i) h_[i] += o_out_[i];

        fwd_rmsnorm(h_.data(), ffn_norm, H, cfg_.rms_norm_eps, h_norm_.data());

        switch (storage) {
        case LS::F32: {
            const auto& L = model_.layers()[l];
            fwd_matmul(h_norm_.data(), L.W_gate, 1, FF, H, gate_.data());
            fwd_matmul(h_norm_.data(), L.W_up,   1, FF, H, up_.data());
            kernels::silu_f32(gate_.data(), FF, gate_.data());
            for (int i = 0; i < FF; ++i) gate_[i] *= up_[i];
            fwd_matmul(gate_.data(), L.W_down, 1, H, FF, ffn_out_.data());
            break;
        }
        case LS::F16: {
            const auto& L = model_.layers_f16()[l];
            fwd_matmul(h_norm_.data(), L.W_gate, 1, FF, H, gate_.data());
            fwd_matmul(h_norm_.data(), L.W_up,   1, FF, H, up_.data());
            kernels::silu_f32(gate_.data(), FF, gate_.data());
            for (int i = 0; i < FF; ++i) gate_[i] *= up_[i];
            fwd_matmul(gate_.data(), L.W_down, 1, H, FF, ffn_out_.data());
            break;
        }
        case LS::I8: {
            const auto& L = model_.layers_i8()[l];
            fwd_matmul(h_norm_.data(), L.W_gate, L.s_gate, 1, FF, H, gate_.data());
            fwd_matmul(h_norm_.data(), L.W_up,   L.s_up,   1, FF, H, up_.data());
            kernels::silu_f32(gate_.data(), FF, gate_.data());
            for (int i = 0; i < FF; ++i) gate_[i] *= up_[i];
            fwd_matmul(gate_.data(), L.W_down, L.s_down, 1, H, FF, ffn_out_.data());
            break;
        }
        }
        for (int i = 0; i < H; ++i) h_[i] += ffn_out_[i];
    }

    if (logits_out) {
        fwd_rmsnorm(h_.data(), model_.final_norm().data(), H,
                    cfg_.rms_norm_eps, h_norm_.data());
        switch (storage) {
        case LS::F32:
            fwd_matmul(h_norm_.data(), model_.lm_head_f32(),
                       1, V, H, logits_out);
            break;
        case LS::F16:
            fwd_matmul(h_norm_.data(), model_.lm_head_f16().data(),
                       1, V, H, logits_out);
            break;
        case LS::I8:
            fwd_matmul(h_norm_.data(), model_.lm_head_i8().data(),
                       model_.lm_head_i8_scales().data(),
                       1, V, H, logits_out);
            break;
        }
    }
}

void Engine::ensure_paged_scratch(int max_seq_len) {
    const std::size_t per_token = static_cast<std::size_t>(cfg_.num_key_value_heads)
                                * cfg_.head_dim;
    const std::size_t need = static_cast<std::size_t>(max_seq_len) * per_token;
    if (k_gather_.size() < need) {
        k_gather_.assign(need, 0.0f);
        v_gather_.assign(need, 0.0f);
    }
}

bool Engine::forward_step_paged(int token_id, int pos,
                                PagedKVCache& kv,
                                float* logits_out) {
    validate_token(token_id, cfg_.vocab_size, "forward_step_paged");
    // Public entry point: acquire forward_mu_ so callers (the scheduler
    // through pybind, or direct Python use) can't race with engine.generate
    // or each other on the shared scratch.
    std::lock_guard<std::mutex> lk(forward_mu_);
    return forward_step_paged_locked(token_id, pos, kv, logits_out);
}

bool Engine::forward_step_paged_locked(int token_id, int pos,
                                       PagedKVCache& kv,
                                       float* logits_out) {
    ensure_model_loaded();
    ensure_scratch_loaded();
    ensure_paged_scratch(pos + 1);

    const int H        = cfg_.hidden_size;
    const int FF       = cfg_.intermediate_size;
    const int V        = cfg_.vocab_size;
    const int n_q      = cfg_.num_attention_heads;
    const int n_kv     = cfg_.num_key_value_heads;
    const int hd       = cfg_.head_dim;
    const int q_dim    = n_q  * hd;
    const int kv_dim   = n_kv * hd;
    const int n_layers = cfg_.num_hidden_layers;

    kernels::embed_lookup_f32(model_.embed_tokens().data(),
                              token_id, H, h_.data());

    const auto storage = storage_;
    using LS = ModelWeightsRef::LinearStorage;

    for (int l = 0; l < n_layers; ++l) {
        const float* attn_norm =
            storage == LS::F32 ? model_.layers()[l].attn_norm
          : storage == LS::F16 ? model_.layers_f16()[l].attn_norm
                               : model_.layers_i8()[l].attn_norm;
        const float* ffn_norm  =
            storage == LS::F32 ? model_.layers()[l].ffn_norm
          : storage == LS::F16 ? model_.layers_f16()[l].ffn_norm
                               : model_.layers_i8()[l].ffn_norm;

        fwd_rmsnorm(h_.data(), attn_norm, H, cfg_.rms_norm_eps, h_norm_.data());

        switch (storage) {
        case LS::F32: {
            const auto& L = model_.layers()[l];
            fwd_matmul(h_norm_.data(), L.W_q, 1, q_dim,  H, q_buf_.data());
            fwd_matmul(h_norm_.data(), L.W_k, 1, kv_dim, H, k_buf_.data());
            fwd_matmul(h_norm_.data(), L.W_v, 1, kv_dim, H, v_buf_.data());
            break;
        }
        case LS::F16: {
            const auto& L = model_.layers_f16()[l];
            fwd_matmul(h_norm_.data(), L.W_q, 1, q_dim,  H, q_buf_.data());
            fwd_matmul(h_norm_.data(), L.W_k, 1, kv_dim, H, k_buf_.data());
            fwd_matmul(h_norm_.data(), L.W_v, 1, kv_dim, H, v_buf_.data());
            break;
        }
        case LS::I8: {
            const auto& L = model_.layers_i8()[l];
            fwd_matmul(h_norm_.data(), L.W_q, L.s_q, 1, q_dim,  H, q_buf_.data());
            fwd_matmul(h_norm_.data(), L.W_k, L.s_k, 1, kv_dim, H, k_buf_.data());
            fwd_matmul(h_norm_.data(), L.W_v, L.s_v, 1, kv_dim, H, v_buf_.data());
            break;
        }
        }

        const float* cos_row = model_.cos_row(pos);
        const float* sin_row = model_.sin_row(pos);
        for (int hh = 0; hh < n_q; ++hh)
            kernels::apply_rope_inplace(&q_buf_[hh * hd], cos_row, sin_row, hd);
        for (int hh = 0; hh < n_kv; ++hh)
            kernels::apply_rope_inplace(&k_buf_[hh * hd], cos_row, sin_row, hd);

        if (!kv.write(l, pos, k_buf_.data(), v_buf_.data())) {
            // KV pool exhausted. Caller terminates the seq with finish_reason="capacity".
            return false;
        }

        // Paged → contiguous gather into scratch, then attention.
        kv.gather(l, pos + 1, k_gather_.data(), v_gather_.data());

        kernels::attention_f32(q_buf_.data(),
                               k_gather_.data(),
                               v_gather_.data(),
                               n_q, n_kv, hd, /*seq_len=*/pos + 1,
                               attn_out_.data());

        switch (storage) {
        case LS::F32:
            fwd_matmul(attn_out_.data(), model_.layers()[l].W_o,
                       1, H, q_dim, o_out_.data());
            break;
        case LS::F16:
            fwd_matmul(attn_out_.data(), model_.layers_f16()[l].W_o,
                       1, H, q_dim, o_out_.data());
            break;
        case LS::I8: {
            const auto& L = model_.layers_i8()[l];
            fwd_matmul(attn_out_.data(), L.W_o, L.s_o, 1, H, q_dim, o_out_.data());
            break;
        }
        }
        for (int i = 0; i < H; ++i) h_[i] += o_out_[i];

        fwd_rmsnorm(h_.data(), ffn_norm, H, cfg_.rms_norm_eps, h_norm_.data());

        switch (storage) {
        case LS::F32: {
            const auto& L = model_.layers()[l];
            fwd_matmul(h_norm_.data(), L.W_gate, 1, FF, H, gate_.data());
            fwd_matmul(h_norm_.data(), L.W_up,   1, FF, H, up_.data());
            kernels::silu_f32(gate_.data(), FF, gate_.data());
            for (int i = 0; i < FF; ++i) gate_[i] *= up_[i];
            fwd_matmul(gate_.data(), L.W_down, 1, H, FF, ffn_out_.data());
            break;
        }
        case LS::F16: {
            const auto& L = model_.layers_f16()[l];
            fwd_matmul(h_norm_.data(), L.W_gate, 1, FF, H, gate_.data());
            fwd_matmul(h_norm_.data(), L.W_up,   1, FF, H, up_.data());
            kernels::silu_f32(gate_.data(), FF, gate_.data());
            for (int i = 0; i < FF; ++i) gate_[i] *= up_[i];
            fwd_matmul(gate_.data(), L.W_down, 1, H, FF, ffn_out_.data());
            break;
        }
        case LS::I8: {
            const auto& L = model_.layers_i8()[l];
            fwd_matmul(h_norm_.data(), L.W_gate, L.s_gate, 1, FF, H, gate_.data());
            fwd_matmul(h_norm_.data(), L.W_up,   L.s_up,   1, FF, H, up_.data());
            kernels::silu_f32(gate_.data(), FF, gate_.data());
            for (int i = 0; i < FF; ++i) gate_[i] *= up_[i];
            fwd_matmul(gate_.data(), L.W_down, L.s_down, 1, H, FF, ffn_out_.data());
            break;
        }
        }
        for (int i = 0; i < H; ++i) h_[i] += ffn_out_[i];
    }

    if (logits_out) {
        fwd_rmsnorm(h_.data(), model_.final_norm().data(), H,
                    cfg_.rms_norm_eps, h_norm_.data());
        switch (storage) {
        case LS::F32:
            fwd_matmul(h_norm_.data(), model_.lm_head_f32(),
                       1, V, H, logits_out);
            break;
        case LS::F16:
            fwd_matmul(h_norm_.data(), model_.lm_head_f16().data(),
                       1, V, H, logits_out);
            break;
        case LS::I8:
            fwd_matmul(h_norm_.data(), model_.lm_head_i8().data(),
                       model_.lm_head_i8_scales().data(),
                       1, V, H, logits_out);
            break;
        }
    }
    return true;
}

std::vector<float> Engine::forward_logits_paged(
    const std::vector<std::int32_t>& ids, BlockManager& mgr) {
    validate_token_ids(ids, cfg_.vocab_size, "forward_logits_paged");
    std::lock_guard<std::mutex> lk(forward_mu_);

    ensure_model_loaded();
    ensure_scratch_loaded();

    const int T = static_cast<int>(ids.size());
    if (T == 0) return {};
    if (T > model_.max_pos())
        throw std::runtime_error("forward_logits_paged: seq exceeds RoPE max_pos");

    const int V = cfg_.vocab_size;
    PagedKVCache kv(mgr);

    std::vector<float> logits(static_cast<std::size_t>(T) * V);
    for (int t = 0; t < T; ++t) {
        // forward_mu_ already held by this function, use the _locked
        // variant to avoid double-lock on std::mutex.
        bool ok = forward_step_paged_locked(
            ids[t], t, kv, &logits[static_cast<std::size_t>(t) * V]);
        if (!ok) {
            kv.release_all();
            throw std::runtime_error("forward_logits_paged: KV pool exhausted "
                                     "at position " + std::to_string(t));
        }
    }
    kv.release_all();
    return logits;
}

void Engine::ensure_batch_scratch(int batch_size) {
    const std::size_t B      = static_cast<std::size_t>(batch_size);
    const std::size_t H      = cfg_.hidden_size;
    const std::size_t FF     = cfg_.intermediate_size;
    const std::size_t q_dim  = static_cast<std::size_t>(cfg_.num_attention_heads)  * cfg_.head_dim;
    const std::size_t kv_dim = static_cast<std::size_t>(cfg_.num_key_value_heads)  * cfg_.head_dim;
    auto grow = [](std::vector<float>& v, std::size_t n) { if (v.size() < n) v.assign(n, 0.0f); };
    grow(hb_,      B * H);
    grow(hb_norm_, B * H);
    grow(qb_,      B * q_dim);
    grow(kb_,      B * kv_dim);
    grow(vb_,      B * kv_dim);
    grow(attn_b_,  B * q_dim);
    grow(ob_,      B * H);
    grow(gateb_,   B * FF);
    grow(upb_,     B * FF);
    grow(ffnb_,    B * H);
}

void Engine::forward_decode_batch(
    const std::vector<std::int32_t>& tokens,
    const std::vector<int>& positions,
    const std::vector<PagedKVCache*>& kvs,
    float* logits_out,
    std::vector<char>& alive) {
    const int B = static_cast<int>(tokens.size());
    alive.assign(static_cast<std::size_t>(B), 1);
    if (B == 0) return;
    // The scheduler guarantees positions are within RoPE range and each kv is
    // non-null; we still bounds-check token ids like the other entry points.
    for (auto t : tokens) validate_token(t, cfg_.vocab_size, "forward_decode_batch");

    std::lock_guard<std::mutex> lk(forward_mu_);
    ensure_model_loaded();
    ensure_batch_scratch(B);

    const int H        = cfg_.hidden_size;
    const int FF       = cfg_.intermediate_size;
    const int V        = cfg_.vocab_size;
    const int n_q      = cfg_.num_attention_heads;
    const int n_kv     = cfg_.num_key_value_heads;
    const int hd       = cfg_.head_dim;
    const int q_dim    = n_q  * hd;
    const int kv_dim   = n_kv * hd;
    const int n_layers = cfg_.num_hidden_layers;

    int max_pos1 = 0;
    for (int i = 0; i < B; ++i) max_pos1 = std::max(max_pos1, positions[i] + 1);
    ensure_paged_scratch(max_pos1);

    const auto storage = storage_;
    using LS = ModelWeightsRef::LinearStorage;

    // Embed each seq's current token into its row of the batch hidden state.
    for (int i = 0; i < B; ++i)
        kernels::embed_lookup_f32(model_.embed_tokens().data(), tokens[i], H,
                                  &hb_[static_cast<std::size_t>(i) * H]);

    for (int l = 0; l < n_layers; ++l) {
        const float* attn_norm =
            storage == LS::F32 ? model_.layers()[l].attn_norm
          : storage == LS::F16 ? model_.layers_f16()[l].attn_norm
                               : model_.layers_i8()[l].attn_norm;
        const float* ffn_norm  =
            storage == LS::F32 ? model_.layers()[l].ffn_norm
          : storage == LS::F16 ? model_.layers_f16()[l].ffn_norm
                               : model_.layers_i8()[l].ffn_norm;

        for (int i = 0; i < B; ++i)
            fwd_rmsnorm(&hb_[static_cast<std::size_t>(i) * H], attn_norm, H,
                        cfg_.rms_norm_eps, &hb_norm_[static_cast<std::size_t>(i) * H]);

        // Batched Q/K/V projection (M = B): one pass over each weight matrix.
        switch (storage) {
        case LS::F32: {
            const auto& L = model_.layers()[l];
            fwd_matmul(hb_norm_.data(), L.W_q, B, q_dim,  H, qb_.data());
            fwd_matmul(hb_norm_.data(), L.W_k, B, kv_dim, H, kb_.data());
            fwd_matmul(hb_norm_.data(), L.W_v, B, kv_dim, H, vb_.data());
            break;
        }
        case LS::F16: {
            const auto& L = model_.layers_f16()[l];
            fwd_matmul(hb_norm_.data(), L.W_q, B, q_dim,  H, qb_.data());
            fwd_matmul(hb_norm_.data(), L.W_k, B, kv_dim, H, kb_.data());
            fwd_matmul(hb_norm_.data(), L.W_v, B, kv_dim, H, vb_.data());
            break;
        }
        case LS::I8: {
            const auto& L = model_.layers_i8()[l];
            fwd_matmul(hb_norm_.data(), L.W_q, L.s_q, B, q_dim,  H, qb_.data());
            fwd_matmul(hb_norm_.data(), L.W_k, L.s_k, B, kv_dim, H, kb_.data());
            fwd_matmul(hb_norm_.data(), L.W_v, L.s_v, B, kv_dim, H, vb_.data());
            break;
        }
        }

        // Per-seq RoPE + KV write + gather + attention. Each sequence has its
        // own history length / position, so this part can't be one batched op.
        for (int i = 0; i < B; ++i) {
            if (!alive[i]) continue;
            const int pos = positions[i];
            const float* cos_row = model_.cos_row(pos);
            const float* sin_row = model_.sin_row(pos);
            float* qi = &qb_[static_cast<std::size_t>(i) * q_dim];
            float* ki = &kb_[static_cast<std::size_t>(i) * kv_dim];
            float* vi = &vb_[static_cast<std::size_t>(i) * kv_dim];
            for (int hh = 0; hh < n_q;  ++hh)
                kernels::apply_rope_inplace(&qi[hh * hd], cos_row, sin_row, hd);
            for (int hh = 0; hh < n_kv; ++hh)
                kernels::apply_rope_inplace(&ki[hh * hd], cos_row, sin_row, hd);

            if (!kvs[i]->write(l, pos, ki, vi)) {
                // KV pool exhausted for this seq. Zero its hidden row so the
                // remaining batched matmuls can't propagate stale/NaN values
                // through it, then skip its attention. Its row is discarded.
                alive[i] = 0;
                std::fill(&hb_[static_cast<std::size_t>(i) * H],
                          &hb_[static_cast<std::size_t>(i) * H] + H, 0.0f);
                continue;
            }
            kvs[i]->gather(l, pos + 1, k_gather_.data(), v_gather_.data());
            kernels::attention_f32(qi, k_gather_.data(), v_gather_.data(),
                                   n_q, n_kv, hd, /*seq_len=*/pos + 1,
                                   &attn_b_[static_cast<std::size_t>(i) * q_dim]);
        }

        // Batched output projection + per-row residual add.
        switch (storage) {
        case LS::F32:
            fwd_matmul(attn_b_.data(), model_.layers()[l].W_o, B, H, q_dim, ob_.data());
            break;
        case LS::F16:
            fwd_matmul(attn_b_.data(), model_.layers_f16()[l].W_o, B, H, q_dim, ob_.data());
            break;
        case LS::I8: {
            const auto& L = model_.layers_i8()[l];
            fwd_matmul(attn_b_.data(), L.W_o, L.s_o, B, H, q_dim, ob_.data());
            break;
        }
        }
        for (int i = 0; i < B; ++i) {
            float* h = &hb_[static_cast<std::size_t>(i) * H];
            const float* o = &ob_[static_cast<std::size_t>(i) * H];
            for (int j = 0; j < H; ++j) h[j] += o[j];
        }

        // FFN: rmsnorm, batched gate/up, SwiGLU, batched down, residual.
        for (int i = 0; i < B; ++i)
            fwd_rmsnorm(&hb_[static_cast<std::size_t>(i) * H], ffn_norm, H,
                        cfg_.rms_norm_eps, &hb_norm_[static_cast<std::size_t>(i) * H]);
        switch (storage) {
        case LS::F32: {
            const auto& L = model_.layers()[l];
            fwd_matmul(hb_norm_.data(), L.W_gate, B, FF, H, gateb_.data());
            fwd_matmul(hb_norm_.data(), L.W_up,   B, FF, H, upb_.data());
            break;
        }
        case LS::F16: {
            const auto& L = model_.layers_f16()[l];
            fwd_matmul(hb_norm_.data(), L.W_gate, B, FF, H, gateb_.data());
            fwd_matmul(hb_norm_.data(), L.W_up,   B, FF, H, upb_.data());
            break;
        }
        case LS::I8: {
            const auto& L = model_.layers_i8()[l];
            fwd_matmul(hb_norm_.data(), L.W_gate, L.s_gate, B, FF, H, gateb_.data());
            fwd_matmul(hb_norm_.data(), L.W_up,   L.s_up,   B, FF, H, upb_.data());
            break;
        }
        }
        for (int i = 0; i < B; ++i) {
            float* g = &gateb_[static_cast<std::size_t>(i) * FF];
            const float* u = &upb_[static_cast<std::size_t>(i) * FF];
            kernels::silu_f32(g, FF, g);
            for (int j = 0; j < FF; ++j) g[j] *= u[j];
        }
        switch (storage) {
        case LS::F32:
            fwd_matmul(gateb_.data(), model_.layers()[l].W_down, B, H, FF, ffnb_.data());
            break;
        case LS::F16:
            fwd_matmul(gateb_.data(), model_.layers_f16()[l].W_down, B, H, FF, ffnb_.data());
            break;
        case LS::I8: {
            const auto& L = model_.layers_i8()[l];
            fwd_matmul(gateb_.data(), L.W_down, L.s_down, B, H, FF, ffnb_.data());
            break;
        }
        }
        for (int i = 0; i < B; ++i) {
            float* h = &hb_[static_cast<std::size_t>(i) * H];
            const float* f = &ffnb_[static_cast<std::size_t>(i) * H];
            for (int j = 0; j < H; ++j) h[j] += f[j];
        }
    }

    // Final norm (per row) + batched LM head → logits_out [B, V].
    for (int i = 0; i < B; ++i)
        fwd_rmsnorm(&hb_[static_cast<std::size_t>(i) * H], model_.final_norm().data(),
                    H, cfg_.rms_norm_eps, &hb_norm_[static_cast<std::size_t>(i) * H]);
    switch (storage) {
    case LS::F32:
        fwd_matmul(hb_norm_.data(), model_.lm_head_f32(), B, V, H, logits_out);
        break;
    case LS::F16:
        fwd_matmul(hb_norm_.data(), model_.lm_head_f16().data(), B, V, H, logits_out);
        break;
    case LS::I8:
        fwd_matmul(hb_norm_.data(), model_.lm_head_i8().data(),
                   model_.lm_head_i8_scales().data(), B, V, H, logits_out);
        break;
    }
}

// KV dtype follows engine storage (v6 §3.1): FP32 weights → FP32 KV (the
// Phase 1-3 baseline so the hard logit-equality gates keep passing),
// FP16/INT8 weights → FP16 KV (halves the cache's memory footprint at the
// cost of FP16 quantization on the K/V rows).
static ContiguousKVCache::Dtype kv_dtype_for(ModelWeightsRef::LinearStorage s) {
    return (s == ModelWeightsRef::LinearStorage::F32)
               ? ContiguousKVCache::Dtype::F32
               : ContiguousKVCache::Dtype::F16;
}

std::vector<float> Engine::forward_logits(const std::vector<std::int32_t>& ids) {
    validate_token_ids(ids, cfg_.vocab_size, "forward_logits");
    std::lock_guard<std::mutex> lk(forward_mu_);

    ensure_model_loaded();
    ensure_scratch_loaded();

    const int T = static_cast<int>(ids.size());
    if (T == 0) return {};
    if (T > model_.max_pos())
        throw std::runtime_error("forward_logits: sequence length exceeds RoPE max_pos");

    const int V = cfg_.vocab_size;
    ContiguousKVCache kv(cfg_.num_hidden_layers, T,
                         cfg_.num_key_value_heads, cfg_.head_dim,
                         kv_dtype_for(storage_));

    std::vector<float> logits(static_cast<std::size_t>(T) * V);
    for (int t = 0; t < T; ++t) {
        forward_step(ids[t], t, kv, &logits[static_cast<std::size_t>(t) * V]);
    }
    return logits;
}

std::pair<std::vector<std::int32_t>, std::string>
Engine::generate(const std::vector<std::int32_t>& prompt_ids, int max_new_tokens) {
    // All cheap input validation happens BEFORE ensure_model_loaded so a
    // bad request on an incomplete checkpoint reports an input error
    // (ValueError / RuntimeError on the Python side) rather than the
    // weight-loader's "missing weight" message.
    if (prompt_ids.empty())
        throw std::invalid_argument("generate: prompt_ids is empty");
    if (max_new_tokens < 0)
        throw std::invalid_argument("generate: max_new_tokens negative");
    validate_token_ids(prompt_ids, cfg_.vocab_size, "generate");
    const int prompt_len = static_cast<int>(prompt_ids.size());
    // int64 sum so max_new_tokens near INT_MAX can't wrap to a negative
    // total that slips past the RoPE check and then trips
    // ContiguousKVCache: non-positive dim. Uses compute_max_pos(cfg_) so
    // the check doesn't need a materialized model either.
    const std::int64_t total_max64 =
        static_cast<std::int64_t>(prompt_len) + max_new_tokens;
    if (total_max64 > max_pos())
        throw std::runtime_error("generate: prompt + max_new_tokens exceeds RoPE max_pos");

    std::lock_guard<std::mutex> lk(forward_mu_);

    ensure_model_loaded();
    ensure_scratch_loaded();

    const int V = cfg_.vocab_size;
    const int total_max  = static_cast<int>(total_max64);

    ContiguousKVCache kv(cfg_.num_hidden_layers, total_max,
                         cfg_.num_key_value_heads, cfg_.head_dim,
                         kv_dtype_for(storage_));

    std::vector<float> logits(V);

    for (int i = 0; i < prompt_len; ++i) {
        const bool is_last = (i + 1 == prompt_len);
        forward_step(prompt_ids[i], i, kv, is_last ? logits.data() : nullptr);
    }

    std::vector<std::int32_t> out = prompt_ids;
    std::string finish = "length";

    while (static_cast<int>(out.size()) - prompt_len < max_new_tokens) {
        int next = argmax(logits.data(), V);
        if (cfg_.eos_token_ids.count(next)) { finish = "stop"; break; }
        out.push_back(static_cast<std::int32_t>(next));
        // Skip the forward pass after the final token, its logits would
        // never be sampled. Mirrors generate_streaming + the schedulers'
        // "advance only if we'll decode again" guard, so generate() no
        // longer does one wasted forward_step per call.
        if (static_cast<int>(out.size()) - prompt_len < max_new_tokens)
            forward_step(next, static_cast<int>(out.size()) - 1, kv, logits.data());
    }

    return {std::move(out), finish};
}

void Engine::generate_streaming(
    const std::vector<std::int32_t>& prompt_ids,
    int max_new_tokens,
    std::function<void(int)> on_token,
    std::function<void(const std::string&)> on_done,
    CancelToken& cancel) {
    // unique_lock instead of lock_guard so we can release around on_token /
    // on_done calls, a user callback that calls back into engine.generate /
    // forward_logits otherwise deadlocks on std::mutex. The forward state
    // (scratch buffers + the local KV cache) is consistent between forward
    // steps, so it's safe to drop the mutex while invoking the callback.
    std::unique_lock<std::mutex> lk(forward_mu_, std::defer_lock);

    auto finish = [&](const std::string& reason) {
        if (lk.owns_lock()) lk.unlock();   // never hold forward_mu_ across on_done
        if (on_done) on_done(reason);
    };

    // Wrap the whole body so any std::exception (overflow check below, kv
    // allocation, kernel failure, etc.) still produces an on_done, without
    // this, the SSE bridge sits on q.get() forever waiting for a DONE that
    // never arrives.
    try {
        validate_token_ids(prompt_ids, cfg_.vocab_size, "generate_streaming");

        lk.lock();
        ensure_model_loaded();
        ensure_scratch_loaded();

        if (prompt_ids.empty()) { finish("error"); return; }
        if (max_new_tokens < 0) { finish("error"); return; }

        const int V = cfg_.vocab_size;
        const int prompt_len = static_cast<int>(prompt_ids.size());
        // Compute the position cap in int64 so prompt_len + max_new_tokens
        // can't silently wrap when max_new_tokens is near INT_MAX.
        const std::int64_t total_max64 =
            static_cast<std::int64_t>(prompt_len) + max_new_tokens;
        if (total_max64 > model_.max_pos()) { finish("error"); return; }
        const int total_max  = static_cast<int>(total_max64);

        ContiguousKVCache kv(cfg_.num_hidden_layers, total_max,
                             cfg_.num_key_value_heads, cfg_.head_dim,
                             kv_dtype_for(storage_));

        std::vector<float> logits(V);

        // Prefill, capture logits on the last prompt token. Cancellation is
        // checked between forward steps so a long prompt isn't immune to it.
        for (int i = 0; i < prompt_len; ++i) {
            if (cancel.is_cancelled()) { finish("cancelled"); return; }
            const bool is_last = (i + 1 == prompt_len);
            forward_step(prompt_ids[i], i, kv, is_last ? logits.data() : nullptr);
        }

        int generated = 0;
        std::string reason = "length";
        int pos = prompt_len;
        while (generated < max_new_tokens) {
            if (cancel.is_cancelled()) { reason = "cancelled"; break; }
            int next = argmax(logits.data(), V);
            if (cfg_.eos_token_ids.count(next)) { reason = "stop"; break; }

            if (on_token) {
                // Release the lock so a callback that re-enters engine APIs
                // doesn't deadlock; re-acquire before the next forward_step.
                lk.unlock();
                on_token(next);
                lk.lock();
            }
            ++generated;

            // Advance KV / get logits for the next position. Skip on the last
            // iteration since we won't sample again.
            if (generated < max_new_tokens) {
                forward_step(next, pos, kv, logits.data());
                ++pos;
            }
        }

        finish(reason);
    } catch (const std::exception& e) {
        // Any escape from the body (allocation failure, kernel exception,
        // invalid argument from a guard added later, etc.) must still fire
        // on_done so the SSE bridge can complete. The HTTP layer maps
        // internal "error" → an OpenAI-safe finish_reason on egress.
        finish("error");
    } catch (...) {
        finish("error");
    }
}

}  // namespace llmengine
