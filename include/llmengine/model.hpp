#pragma once

#include "llmengine/config.hpp"
#include "llmengine/types.hpp"
#include "llmengine/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llmengine {

// Per-layer pointer bundles. The "F32" struct (LayerWeightsRef) is the
// canonical reference path: norms + linear weights all FP32. The "F16"
// struct (LayerWeightsRefFp16) shares the FP32 norms but stores linear
// weights as __fp16, half the memory bandwidth at decode time.
struct LayerWeightsRef {
    const float* attn_norm = nullptr;       // [hidden]
    const float* ffn_norm  = nullptr;       // [hidden]
    const float* W_q       = nullptr;
    const float* W_k       = nullptr;
    const float* W_v       = nullptr;
    const float* W_o       = nullptr;
    const float* W_gate    = nullptr;
    const float* W_up      = nullptr;
    const float* W_down    = nullptr;
};

struct LayerWeightsRefFp16 {
    const float*  attn_norm = nullptr;      // [hidden], still FP32 (tiny)
    const float*  ffn_norm  = nullptr;
    const __fp16* W_q       = nullptr;
    const __fp16* W_k       = nullptr;
    const __fp16* W_v       = nullptr;
    const __fp16* W_o       = nullptr;
    const __fp16* W_gate    = nullptr;
    const __fp16* W_up      = nullptr;
    const __fp16* W_down    = nullptr;
};

// Per-output-channel symmetric INT8. Each weight matrix carries an FP16
// scale per output row. Dequant: `float(w[n,k]) * scales[n]`.
struct LayerWeightsRefInt8 {
    const float*       attn_norm = nullptr;
    const float*       ffn_norm  = nullptr;
    const std::int8_t* W_q       = nullptr; const __fp16* s_q     = nullptr;
    const std::int8_t* W_k       = nullptr; const __fp16* s_k     = nullptr;
    const std::int8_t* W_v       = nullptr; const __fp16* s_v     = nullptr;
    const std::int8_t* W_o       = nullptr; const __fp16* s_o     = nullptr;
    const std::int8_t* W_gate    = nullptr; const __fp16* s_gate  = nullptr;
    const std::int8_t* W_up      = nullptr; const __fp16* s_up    = nullptr;
    const std::int8_t* W_down    = nullptr; const __fp16* s_down  = nullptr;
};

// Materialized model weights. Owns the buffers; views (LayerWeightsRef /
// LayerWeightsRefFp16) point into them. Linear-weight precision is selected
// at load time and is fixed for the engine's lifetime.
class ModelWeightsRef {
public:
    enum class LinearStorage { F32, F16, I8 };

    void load(const ModelConfig& cfg,
              const WeightMap&   wm,
              LinearStorage      storage = LinearStorage::F32);

    LinearStorage linear_storage() const { return storage_; }

    const std::vector<float>& embed_tokens() const { return embed_tokens_; }
    const std::vector<float>& final_norm()   const { return final_norm_; }

    // Valid in LinearStorage::F32 mode. Returns the lm_head matrix as a
    // raw pointer; when the model is tied this points directly into the
    // embed_tokens_ buffer (no copy), otherwise into lm_head_f32_storage_.
    const float* lm_head_f32() const { return lm_head_f32_ptr_; }
    const std::vector<LayerWeightsRef>& layers() const { return layer_views_; }

    // Valid in LinearStorage::F16 mode.
    const std::vector<__fp16>& lm_head_f16() const { return lm_head_f16_; }
    const std::vector<LayerWeightsRefFp16>& layers_f16() const { return layer_views_f16_; }

    // Valid in LinearStorage::I8 mode.
    const std::vector<std::int8_t>& lm_head_i8()       const { return lm_head_i8_; }
    const std::vector<__fp16>&      lm_head_i8_scales() const { return lm_head_i8_scales_; }
    const std::vector<LayerWeightsRefInt8>& layers_i8() const { return layer_views_i8_; }

    const float* cos_row(int pos) const { return &cos_table_[pos * head_dim_]; }
    const float* sin_row(int pos) const { return &sin_table_[pos * head_dim_]; }

    int max_pos() const { return max_pos_; }

private:
    LinearStorage storage_ = LinearStorage::F32;

    // Always-FP32 buffers (norms, embed, RoPE).
    std::vector<float> embed_tokens_;
    std::vector<float> final_norm_;
    std::vector<float> inv_freq_;
    std::vector<float> cos_table_;
    std::vector<float> sin_table_;
    int head_dim_ = 0;
    int max_pos_  = 0;

    // F32 storage. When tie_word_embeddings is true, lm_head_f32_storage_
    // stays empty and lm_head_f32_ptr_ aliases embed_tokens_.data() (saves
    // ~vocab*hidden*4 bytes, ~1 GB on real 1B in FP32 mode). Otherwise
    // lm_head_f32_storage_ holds the materialized matrix and the pointer
    // points into it. The accessor returns the pointer so callers don't
    // need to know which case applies.
    const float* lm_head_f32_ptr_ = nullptr;
    std::vector<float> lm_head_f32_storage_;
    std::vector<std::vector<float>> layer_blobs_f32_;
    std::vector<LayerWeightsRef> layer_views_;

    // F16 storage.
    std::vector<__fp16> lm_head_f16_;
    std::vector<std::vector<__fp16>> layer_blobs_f16_;
    std::vector<LayerWeightsRefFp16> layer_views_f16_;

    // I8 storage. Per-layer blobs are split: one int8 buffer for weights,
    // one fp16 buffer for scales (one per output row of each matrix).
    std::vector<std::int8_t> lm_head_i8_;
    std::vector<__fp16>      lm_head_i8_scales_;
    std::vector<std::vector<std::int8_t>> layer_blobs_i8_;
    std::vector<std::vector<__fp16>>      layer_scale_blobs_;
    std::vector<LayerWeightsRefInt8> layer_views_i8_;

    // Always-FP32 norm slabs (one per layer); shared across storage modes.
    std::vector<std::vector<float>> norm_blobs_;
};

}  // namespace llmengine
