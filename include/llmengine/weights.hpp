#pragma once

#include "llmengine/config.hpp"
#include "llmengine/types.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmengine {

// Owns one or more mmap'd safetensors files and exposes their tensors as views.
// After construction it applies post-load fixups (e.g. tied lm_head alias).
class WeightMap {
public:
    WeightMap() = default;
    ~WeightMap();

    WeightMap(const WeightMap&) = delete;
    WeightMap& operator=(const WeightMap&) = delete;
    WeightMap(WeightMap&&) noexcept;
    WeightMap& operator=(WeightMap&&) noexcept;

    // Load every *.safetensors file directly inside `model_dir` and apply tie-LM-head alias.
    void load_safetensors_dir(const std::string& model_dir, const ModelConfig& cfg);

    bool contains(const std::string& name) const;
    const Tensor& at(const std::string& name) const;

    // Test/debug only. Returns the underlying buffer pointer cast to uintptr_t
    // so Python can cheaply compare two names (e.g. lm_head vs embed_tokens).
    std::uintptr_t debug_ptr(const std::string& name) const;

    std::size_t size() const { return tensors_.size(); }

    std::vector<std::string> keys() const;

private:
    struct MappedFile {
        int fd = -1;
        void* addr = nullptr;
        std::size_t length = 0;
    };

    void load_one_file(const std::string& path);
    void apply_tied_lm_head_alias();

    std::vector<MappedFile> files_;
    std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace llmengine
