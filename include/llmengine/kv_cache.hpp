#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llmengine {

// Contiguous (non-paged) KV cache for a single request.
//
// Storage dtype is chosen at construction:
//   Dtype::F32 — k/v stored as float; k_layer()/v_layer() return raw pointers
//                that the FP32 attention kernel consumes directly (no gather).
//   Dtype::F16 — k/v stored as __fp16; engine.cpp gathers into FP32 scratch
//                via view_layer() before calling attention. Halves the cache's
//                memory footprint relative to FP32.
class ContiguousKVCache {
public:
    enum class Dtype { F32, F16 };

    ContiguousKVCache(int  num_layers,
                      int  max_seq_len,
                      int  num_kv_heads,
                      int  head_dim,
                      Dtype dtype = Dtype::F32);

    void reset();

    // Write k/v for a single token at (layer, pos). Inputs are always FP32;
    // the cache narrows to __fp16 in F16 mode.
    void write(int layer, int pos, const float* k_in, const float* v_in);

    // Gather positions [0, seq_len) of `layer` into FP32 scratch buffers.
    // Each buffer has shape [seq_len, num_kv_heads, head_dim].
    void gather_layer(int layer, int seq_len,
                      float* k_out, float* v_out) const;

    // FP32-only direct accessors. Throws std::runtime_error in F16 mode.
    const float* k_layer_f32(int layer) const;
    const float* v_layer_f32(int layer) const;

    Dtype dtype()         const { return dtype_; }
    int   num_layers()    const { return num_layers_; }
    int   max_seq_len()   const { return max_seq_len_; }
    int   num_kv_heads()  const { return num_kv_heads_; }
    int   head_dim()      const { return head_dim_; }

private:
    int num_layers_;
    int max_seq_len_;
    int num_kv_heads_;
    int head_dim_;
    Dtype dtype_;
    std::size_t per_token_;       // num_kv_heads * head_dim elements
    std::size_t per_layer_;       // max_seq_len * per_token elements
    std::vector<float>   K_f32_, V_f32_;
    std::vector<__fp16>  K_f16_, V_f16_;
};

}  // namespace llmengine
