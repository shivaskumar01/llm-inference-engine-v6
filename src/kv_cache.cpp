#include "llmengine/kv_cache.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace llmengine {

ContiguousKVCache::ContiguousKVCache(int num_layers,
                                     int max_seq_len,
                                     int num_kv_heads,
                                     int head_dim,
                                     Dtype dtype)
    : num_layers_(num_layers),
      max_seq_len_(max_seq_len),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      dtype_(dtype) {
    if (num_layers <= 0 || max_seq_len <= 0 || num_kv_heads <= 0 || head_dim <= 0)
        throw std::invalid_argument("ContiguousKVCache: non-positive dim");

    per_token_ = static_cast<std::size_t>(num_kv_heads) * head_dim;
    per_layer_ = static_cast<std::size_t>(max_seq_len) * per_token_;

    const std::size_t total = per_layer_ * num_layers;
    if (dtype_ == Dtype::F32) {
        K_f32_.assign(total, 0.0f);
        V_f32_.assign(total, 0.0f);
    } else {
        K_f16_.assign(total, static_cast<__fp16>(0.0f));
        V_f16_.assign(total, static_cast<__fp16>(0.0f));
    }
}

void ContiguousKVCache::reset() {
    if (dtype_ == Dtype::F32) {
        std::fill(K_f32_.begin(), K_f32_.end(), 0.0f);
        std::fill(V_f32_.begin(), V_f32_.end(), 0.0f);
    } else {
        std::fill(K_f16_.begin(), K_f16_.end(), static_cast<__fp16>(0.0f));
        std::fill(V_f16_.begin(), V_f16_.end(), static_cast<__fp16>(0.0f));
    }
}

void ContiguousKVCache::write(int layer, int pos,
                              const float* k_in, const float* v_in) {
    if (layer < 0 || layer >= num_layers_)
        throw std::out_of_range("kv write: layer out of range");
    if (pos < 0 || pos >= max_seq_len_)
        throw std::out_of_range("kv write: pos out of range");

    const std::size_t off = static_cast<std::size_t>(layer) * per_layer_
                          + static_cast<std::size_t>(pos) * per_token_;

    if (dtype_ == Dtype::F32) {
        std::memcpy(&K_f32_[off], k_in, per_token_ * sizeof(float));
        std::memcpy(&V_f32_[off], v_in, per_token_ * sizeof(float));
    } else {
        for (std::size_t i = 0; i < per_token_; ++i) {
            K_f16_[off + i] = static_cast<__fp16>(k_in[i]);
            V_f16_[off + i] = static_cast<__fp16>(v_in[i]);
        }
    }
}

void ContiguousKVCache::gather_layer(int layer, int seq_len,
                                     float* k_out, float* v_out) const {
    if (layer < 0 || layer >= num_layers_)
        throw std::out_of_range("kv gather: layer out of range");
    if (seq_len < 0 || seq_len > max_seq_len_)
        throw std::out_of_range("kv gather: seq_len out of range");

    const std::size_t base = static_cast<std::size_t>(layer) * per_layer_;
    const std::size_t n    = static_cast<std::size_t>(seq_len) * per_token_;

    if (dtype_ == Dtype::F32) {
        std::memcpy(k_out, &K_f32_[base], n * sizeof(float));
        std::memcpy(v_out, &V_f32_[base], n * sizeof(float));
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            k_out[i] = static_cast<float>(K_f16_[base + i]);
            v_out[i] = static_cast<float>(V_f16_[base + i]);
        }
    }
}

const float* ContiguousKVCache::k_layer_f32(int layer) const {
    if (dtype_ != Dtype::F32)
        throw std::runtime_error("k_layer_f32: cache stores FP16; use gather_layer");
    return &K_f32_[static_cast<std::size_t>(layer) * per_layer_];
}

const float* ContiguousKVCache::v_layer_f32(int layer) const {
    if (dtype_ != Dtype::F32)
        throw std::runtime_error("v_layer_f32: cache stores FP16; use gather_layer");
    return &V_f32_[static_cast<std::size_t>(layer) * per_layer_];
}

}  // namespace llmengine
