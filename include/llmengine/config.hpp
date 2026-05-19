#pragma once

#include <algorithm>
#include <string>
#include <unordered_set>

namespace llmengine {

struct ModelConfig {
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int max_position_embeddings = 0;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    bool tie_word_embeddings = false;

    // Llama-3 RoPE scaling. Populated from config.json's rope_scaling block
    // when present; left at defaults for older Llama configs.
    bool   rope_has_scaling = false;
    float  rope_factor = 1.0f;
    float  rope_low_freq_factor = 1.0f;
    float  rope_high_freq_factor = 1.0f;
    int    rope_original_max_pos = 0;

    std::unordered_set<int> eos_token_ids;
};

ModelConfig load_config(const std::string& model_dir);

// Maximum position the RoPE table covers for this config. Mirrors the
// computation in ModelWeightsRef::load so the engine and the scheduler
// agree on the cap; keeping it in one place prevents the two paths from
// drifting if the formula changes (e.g. a future override for scaled rope).
inline int compute_max_pos(const ModelConfig& cfg) {
    int mp = cfg.max_position_embeddings;
    if (cfg.rope_has_scaling && cfg.rope_original_max_pos > 0)
        mp = std::min(mp, cfg.rope_original_max_pos);
    if (mp <= 0 || mp > 8192) mp = 8192;
    return mp;
}

}  // namespace llmengine
