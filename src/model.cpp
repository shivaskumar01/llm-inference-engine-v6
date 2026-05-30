#include "llmengine/model.hpp"
#include "llmengine/kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace llmengine {

namespace {

inline float bf16_to_fp32(std::uint16_t bits) {
    std::uint32_t v = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

inline float fp16_to_fp32(std::uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
}

void cast_to_fp32(const Tensor& t, float* dst) {
    const std::size_t n = t.numel();
    if (t.dtype == DType::F32) {
        std::memcpy(dst, t.data, n * sizeof(float));
    } else if (t.dtype == DType::BF16) {
        const auto* src = static_cast<const std::uint16_t*>(t.data);
        for (std::size_t i = 0; i < n; ++i) dst[i] = bf16_to_fp32(src[i]);
    } else if (t.dtype == DType::F16) {
        const auto* src = static_cast<const std::uint16_t*>(t.data);
        for (std::size_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(src[i]);
    } else {
        throw std::runtime_error("cast_to_fp32: unsupported dtype");
    }
}

// Cast directly to FP16, going via FP32 only when the source isn't already
// FP16. Saves ~50% memory vs the FP32 buffer for the linear-weight materialization.
void cast_to_fp16(const Tensor& t, __fp16* dst) {
    const std::size_t n = t.numel();
    if (t.dtype == DType::F16) {
        std::memcpy(dst, t.data, n * sizeof(__fp16));
        return;
    }
    if (t.dtype == DType::F32) {
        const auto* src = static_cast<const float*>(t.data);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<__fp16>(src[i]);
        return;
    }
    if (t.dtype == DType::BF16) {
        const auto* src = static_cast<const std::uint16_t*>(t.data);
        for (std::size_t i = 0; i < n; ++i)
            dst[i] = static_cast<__fp16>(bf16_to_fp32(src[i]));
        return;
    }
    throw std::runtime_error("cast_to_fp16: unsupported dtype");
}

// Per-output-channel symmetric INT8 quantization of a 2-D weight matrix
// laid out [N=out, K=in] row-major. Writes N*K int8 weights into `w_out`
// and N fp16 scales into `s_out`. Source can be F32 / F16 / BF16.
void quantize_int8_per_row(const Tensor& t,
                            std::size_t N, std::size_t K,
                            std::int8_t* w_out, __fp16* s_out) {
    if (t.numel() != N * K)
        throw std::runtime_error("quantize_int8_per_row: shape mismatch");

    auto read = [&](std::size_t i) -> float {
        if (t.dtype == DType::F32)
            return static_cast<const float*>(t.data)[i];
        if (t.dtype == DType::F16)
            return fp16_to_fp32(static_cast<const std::uint16_t*>(t.data)[i]);
        if (t.dtype == DType::BF16)
            return bf16_to_fp32(static_cast<const std::uint16_t*>(t.data)[i]);
        throw std::runtime_error("quantize_int8_per_row: unsupported dtype");
    };

    for (std::size_t n = 0; n < N; ++n) {
        // 1. row max.
        float max_abs = 0.0f;
        for (std::size_t k = 0; k < K; ++k) {
            float v = read(n * K + k);
            if (v < 0) v = -v;
            if (v > max_abs) max_abs = v;
        }
        // Guard against all-zero rows.
        float scale = max_abs > 1e-12f ? max_abs / 127.0f : 1.0f;
        float inv_scale = 1.0f / scale;

        // 2. quantize.
        for (std::size_t k = 0; k < K; ++k) {
            float v = read(n * K + k);
            float q = std::nearbyint(v * inv_scale);
            if (q > 127.0f)  q = 127.0f;
            if (q < -128.0f) q = -128.0f;
            w_out[n * K + k] = static_cast<std::int8_t>(q);
        }
        s_out[n] = static_cast<__fp16>(scale);
    }
}

const Tensor& must_get(const WeightMap& wm, const std::string& name) {
    if (!wm.contains(name))
        throw std::runtime_error("missing weight: " + name);
    return wm.at(name);
}

// Enforce the exact tensor shape, not just the element count. cast_to_fp32 /
// cast_to_fp16 / quantize_int8_per_row all copy `numel(shape) * dtype_bytes`
// — a transposed weight (same numel, swapped dims) silently passes them
// otherwise.
std::string fmt_shape(const std::vector<std::int64_t>& s) {
    std::string r = "[";
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) r += ",";
        r += std::to_string(s[i]);
    }
    return r + "]";
}

void require_shape(const Tensor& t,
                   const std::string& name,
                   const std::vector<std::int64_t>& expected) {
    if (t.shape != expected)
        throw std::runtime_error(
            "shape mismatch for " + name
            + ": expected " + fmt_shape(expected)
            + ", got " + fmt_shape(t.shape));
}

const Tensor& must_get_shaped(const WeightMap& wm,
                              const std::string& name,
                              const std::vector<std::int64_t>& expected) {
    const Tensor& t = must_get(wm, name);
    require_shape(t, name, expected);
    return t;
}

void load_into_fp32(const WeightMap& wm,
                    const std::string& name,
                    std::vector<float>& dst,
                    const std::vector<std::int64_t>& expected_shape) {
    const auto& t = must_get_shaped(wm, name, expected_shape);
    std::size_t n = 1;
    for (auto d : expected_shape) n *= static_cast<std::size_t>(d);
    dst.resize(n);
    cast_to_fp32(t, dst.data());
}

}  // namespace

void ModelWeightsRef::load(const ModelConfig& cfg,
                           const WeightMap&   wm,
                           LinearStorage      storage) {
    storage_ = storage;

    const std::size_t hidden       = cfg.hidden_size;
    const std::size_t intermediate = cfg.intermediate_size;
    const std::size_t vocab        = cfg.vocab_size;
    const std::size_t n_q_dim      = static_cast<std::size_t>(cfg.num_attention_heads) * cfg.head_dim;
    const std::size_t n_kv_dim     = static_cast<std::size_t>(cfg.num_key_value_heads) * cfg.head_dim;

    head_dim_ = cfg.head_dim;
    // Single source of truth for the RoPE position cap; the scheduler uses
    // the same helper so its enqueue-time guard can't drift.
    max_pos_  = compute_max_pos(cfg);

    const std::vector<std::int64_t> shape_vocab_hidden = {
        static_cast<std::int64_t>(vocab), static_cast<std::int64_t>(hidden)};
    const std::vector<std::int64_t> shape_hidden = {
        static_cast<std::int64_t>(hidden)};

    // Embeddings + final norm + RoPE — always FP32, regardless of storage mode.
    load_into_fp32(wm, "model.embed_tokens.weight", embed_tokens_, shape_vocab_hidden);
    load_into_fp32(wm, "model.norm.weight",         final_norm_,   shape_hidden);

    // LM head. Tied to embed at the WeightMap level; we materialize a copy
    // here (so the linear-storage path is uniform).
    if (storage_ == LinearStorage::F32) {
        if (cfg.tie_word_embeddings) {
            // Genuine sharing: lm_head IS the embedding matrix in tied
            // models. Avoid the redundant ~vocab*hidden*4 byte copy by
            // aliasing the buffer.
            lm_head_f32_storage_.clear();
            lm_head_f32_ptr_ = embed_tokens_.data();
        } else {
            load_into_fp32(wm, "lm_head.weight", lm_head_f32_storage_,
                           shape_vocab_hidden);
            lm_head_f32_ptr_ = lm_head_f32_storage_.data();
        }
    } else if (storage_ == LinearStorage::F16) {
        lm_head_f16_.resize(vocab * hidden);
        if (cfg.tie_word_embeddings) {
            for (std::size_t i = 0; i < lm_head_f16_.size(); ++i)
                lm_head_f16_[i] = static_cast<__fp16>(embed_tokens_[i]);
        } else {
            cast_to_fp16(must_get_shaped(wm, "lm_head.weight", shape_vocab_hidden),
                         lm_head_f16_.data());
        }
    } else {
        // I8: quantize lm_head per-output-row.
        lm_head_i8_.resize(vocab * hidden);
        lm_head_i8_scales_.resize(vocab);
        if (cfg.tie_word_embeddings) {
            // Build a Tensor view over embed_tokens_ to feed quantizer.
            Tensor t;
            t.dtype = DType::F32;
            t.shape = shape_vocab_hidden;
            t.data  = embed_tokens_.data();
            t.nbytes = embed_tokens_.size() * sizeof(float);
            quantize_int8_per_row(t, vocab, hidden,
                                  lm_head_i8_.data(), lm_head_i8_scales_.data());
        } else {
            const auto& t = must_get_shaped(
                wm, "lm_head.weight", shape_vocab_hidden);
            quantize_int8_per_row(t, vocab, hidden,
                                  lm_head_i8_.data(), lm_head_i8_scales_.data());
        }
    }

    norm_blobs_.assign(cfg.num_hidden_layers, {});
    layer_views_.clear();
    layer_views_f16_.clear();
    layer_views_i8_.clear();
    if (storage_ == LinearStorage::F32) {
        layer_blobs_f32_.assign(cfg.num_hidden_layers, {});
        layer_views_.assign(cfg.num_hidden_layers, {});
    } else if (storage_ == LinearStorage::F16) {
        layer_blobs_f16_.assign(cfg.num_hidden_layers, {});
        layer_views_f16_.assign(cfg.num_hidden_layers, {});
    } else {
        layer_blobs_i8_.assign(cfg.num_hidden_layers, {});
        layer_scale_blobs_.assign(cfg.num_hidden_layers, {});
        layer_views_i8_.assign(cfg.num_hidden_layers, {});
    }

    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        const std::string base = "model.layers." + std::to_string(l) + ".";

        const std::size_t sz_q     = n_q_dim  * hidden;
        const std::size_t sz_k     = n_kv_dim * hidden;
        const std::size_t sz_v     = n_kv_dim * hidden;
        const std::size_t sz_o     = hidden   * n_q_dim;
        const std::size_t sz_gate  = intermediate * hidden;
        const std::size_t sz_up    = intermediate * hidden;
        const std::size_t sz_down  = hidden * intermediate;

        // Norms are FP32 regardless of storage. Pack them into a shared
        // per-layer norm blob.
        std::vector<float>& nb = norm_blobs_[l];
        nb.resize(2 * hidden);
        // Per-layer expected shapes (linear weights are stored as [N=out, K=in]).
        const std::vector<std::int64_t> shp_h     = shape_hidden;
        const std::vector<std::int64_t> shp_q     = {(std::int64_t)n_q_dim,      (std::int64_t)hidden};
        const std::vector<std::int64_t> shp_kv    = {(std::int64_t)n_kv_dim,     (std::int64_t)hidden};
        const std::vector<std::int64_t> shp_o     = {(std::int64_t)hidden,       (std::int64_t)n_q_dim};
        const std::vector<std::int64_t> shp_gate  = {(std::int64_t)intermediate, (std::int64_t)hidden};
        const std::vector<std::int64_t> shp_up    = {(std::int64_t)intermediate, (std::int64_t)hidden};
        const std::vector<std::int64_t> shp_down  = {(std::int64_t)hidden,       (std::int64_t)intermediate};

        cast_to_fp32(must_get_shaped(wm, base + "input_layernorm.weight",          shp_h), &nb[0]);
        cast_to_fp32(must_get_shaped(wm, base + "post_attention_layernorm.weight", shp_h), &nb[hidden]);

        if (storage_ == LinearStorage::F32) {
            std::vector<float>& blob = layer_blobs_f32_[l];
            std::size_t off = 0;
            const std::size_t off_q    = off; off += sz_q;
            const std::size_t off_k    = off; off += sz_k;
            const std::size_t off_v    = off; off += sz_v;
            const std::size_t off_o    = off; off += sz_o;
            const std::size_t off_gate = off; off += sz_gate;
            const std::size_t off_up   = off; off += sz_up;
            const std::size_t off_down = off; off += sz_down;
            blob.resize(off);

            cast_to_fp32(must_get_shaped(wm, base + "self_attn.q_proj.weight", shp_q),    &blob[off_q]);
            cast_to_fp32(must_get_shaped(wm, base + "self_attn.k_proj.weight", shp_kv),   &blob[off_k]);
            cast_to_fp32(must_get_shaped(wm, base + "self_attn.v_proj.weight", shp_kv),   &blob[off_v]);
            cast_to_fp32(must_get_shaped(wm, base + "self_attn.o_proj.weight", shp_o),    &blob[off_o]);
            cast_to_fp32(must_get_shaped(wm, base + "mlp.gate_proj.weight",    shp_gate), &blob[off_gate]);
            cast_to_fp32(must_get_shaped(wm, base + "mlp.up_proj.weight",      shp_up),   &blob[off_up]);
            cast_to_fp32(must_get_shaped(wm, base + "mlp.down_proj.weight",    shp_down), &blob[off_down]);

            LayerWeightsRef& v = layer_views_[l];
            v.attn_norm = &nb[0];
            v.ffn_norm  = &nb[hidden];
            v.W_q    = &blob[off_q];
            v.W_k    = &blob[off_k];
            v.W_v    = &blob[off_v];
            v.W_o    = &blob[off_o];
            v.W_gate = &blob[off_gate];
            v.W_up   = &blob[off_up];
            v.W_down = &blob[off_down];
        } else if (storage_ == LinearStorage::F16) {
            std::vector<__fp16>& blob = layer_blobs_f16_[l];
            std::size_t off = 0;
            const std::size_t off_q    = off; off += sz_q;
            const std::size_t off_k    = off; off += sz_k;
            const std::size_t off_v    = off; off += sz_v;
            const std::size_t off_o    = off; off += sz_o;
            const std::size_t off_gate = off; off += sz_gate;
            const std::size_t off_up   = off; off += sz_up;
            const std::size_t off_down = off; off += sz_down;
            blob.resize(off);

            cast_to_fp16(must_get_shaped(wm, base + "self_attn.q_proj.weight", shp_q),    &blob[off_q]);
            cast_to_fp16(must_get_shaped(wm, base + "self_attn.k_proj.weight", shp_kv),   &blob[off_k]);
            cast_to_fp16(must_get_shaped(wm, base + "self_attn.v_proj.weight", shp_kv),   &blob[off_v]);
            cast_to_fp16(must_get_shaped(wm, base + "self_attn.o_proj.weight", shp_o),    &blob[off_o]);
            cast_to_fp16(must_get_shaped(wm, base + "mlp.gate_proj.weight",    shp_gate), &blob[off_gate]);
            cast_to_fp16(must_get_shaped(wm, base + "mlp.up_proj.weight",      shp_up),   &blob[off_up]);
            cast_to_fp16(must_get_shaped(wm, base + "mlp.down_proj.weight",    shp_down), &blob[off_down]);

            LayerWeightsRefFp16& v = layer_views_f16_[l];
            v.attn_norm = &nb[0];
            v.ffn_norm  = &nb[hidden];
            v.W_q    = &blob[off_q];
            v.W_k    = &blob[off_k];
            v.W_v    = &blob[off_v];
            v.W_o    = &blob[off_o];
            v.W_gate = &blob[off_gate];
            v.W_up   = &blob[off_up];
            v.W_down = &blob[off_down];
        } else {
            // I8: weight bytes in layer_blobs_i8_, fp16 scales in
            // layer_scale_blobs_. Each matrix gets N output-channel scales.
            std::vector<std::int8_t>& wblob = layer_blobs_i8_[l];
            std::vector<__fp16>&      sblob = layer_scale_blobs_[l];

            std::size_t off = 0;
            const std::size_t off_q    = off; off += sz_q;
            const std::size_t off_k    = off; off += sz_k;
            const std::size_t off_v    = off; off += sz_v;
            const std::size_t off_o    = off; off += sz_o;
            const std::size_t off_gate = off; off += sz_gate;
            const std::size_t off_up   = off; off += sz_up;
            const std::size_t off_down = off; off += sz_down;
            wblob.resize(off);

            std::size_t soff = 0;
            const std::size_t s_q    = soff; soff += n_q_dim;
            const std::size_t s_k    = soff; soff += n_kv_dim;
            const std::size_t s_v    = soff; soff += n_kv_dim;
            const std::size_t s_o    = soff; soff += hidden;
            const std::size_t s_gate = soff; soff += intermediate;
            const std::size_t s_up   = soff; soff += intermediate;
            const std::size_t s_down = soff; soff += hidden;
            sblob.resize(soff);

            // [N, K] dims for each matmul. Linear weight layout matches
            // matmul_int8w_f32a: [N=out_dim, K=in_dim] row-major.
            quantize_int8_per_row(must_get_shaped(wm, base + "self_attn.q_proj.weight", shp_q),
                                   n_q_dim, hidden,
                                   &wblob[off_q],    &sblob[s_q]);
            quantize_int8_per_row(must_get_shaped(wm, base + "self_attn.k_proj.weight", shp_kv),
                                   n_kv_dim, hidden,
                                   &wblob[off_k],    &sblob[s_k]);
            quantize_int8_per_row(must_get_shaped(wm, base + "self_attn.v_proj.weight", shp_kv),
                                   n_kv_dim, hidden,
                                   &wblob[off_v],    &sblob[s_v]);
            quantize_int8_per_row(must_get_shaped(wm, base + "self_attn.o_proj.weight", shp_o),
                                   hidden, n_q_dim,
                                   &wblob[off_o],    &sblob[s_o]);
            quantize_int8_per_row(must_get_shaped(wm, base + "mlp.gate_proj.weight",    shp_gate),
                                   intermediate, hidden,
                                   &wblob[off_gate], &sblob[s_gate]);
            quantize_int8_per_row(must_get_shaped(wm, base + "mlp.up_proj.weight",      shp_up),
                                   intermediate, hidden,
                                   &wblob[off_up],   &sblob[s_up]);
            quantize_int8_per_row(must_get_shaped(wm, base + "mlp.down_proj.weight",    shp_down),
                                   hidden, intermediate,
                                   &wblob[off_down], &sblob[s_down]);

            LayerWeightsRefInt8& v = layer_views_i8_[l];
            v.attn_norm = &nb[0];
            v.ffn_norm  = &nb[hidden];
            v.W_q    = &wblob[off_q];    v.s_q    = &sblob[s_q];
            v.W_k    = &wblob[off_k];    v.s_k    = &sblob[s_k];
            v.W_v    = &wblob[off_v];    v.s_v    = &sblob[s_v];
            v.W_o    = &wblob[off_o];    v.s_o    = &sblob[s_o];
            v.W_gate = &wblob[off_gate]; v.s_gate = &sblob[s_gate];
            v.W_up   = &wblob[off_up];   v.s_up   = &sblob[s_up];
            v.W_down = &wblob[off_down]; v.s_down = &sblob[s_down];
        }
    }

    // RoPE tables (always FP32).
    const int half = cfg.head_dim / 2;
    inv_freq_.resize(half);
    kernels::llama3_inv_freq(cfg.head_dim,
                             cfg.rope_theta,
                             cfg.rope_has_scaling,
                             cfg.rope_factor,
                             cfg.rope_low_freq_factor,
                             cfg.rope_high_freq_factor,
                             cfg.rope_original_max_pos,
                             inv_freq_.data());

    cos_table_.assign(static_cast<std::size_t>(max_pos_) * cfg.head_dim, 0.0f);
    sin_table_.assign(static_cast<std::size_t>(max_pos_) * cfg.head_dim, 0.0f);
    kernels::build_rope_tables(inv_freq_.data(),
                               max_pos_,
                               cfg.head_dim,
                               cos_table_.data(),
                               sin_table_.data());
}

}  // namespace llmengine
