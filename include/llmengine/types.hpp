#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace llmengine {

enum class DType {
    F32,
    F16,
    BF16,
    I8,
};

inline std::size_t dtype_bytes(DType dt) {
    switch (dt) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::I8:   return 1;
    }
    throw std::invalid_argument("unknown dtype");
}

inline DType parse_safetensors_dtype(const std::string& s) {
    if (s == "F32")  return DType::F32;
    if (s == "F16")  return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I8")   return DType::I8;
    throw std::invalid_argument("unsupported safetensors dtype: " + s);
}

// Non-owning tensor view. Backed by mmap or another buffer the WeightMap owns.
struct Tensor {
    DType dtype = DType::F32;
    std::vector<std::int64_t> shape;
    const void* data = nullptr;     // raw bytes, row-major
    std::size_t nbytes = 0;

    std::size_t numel() const {
        std::size_t n = 1;
        for (auto d : shape) n *= static_cast<std::size_t>(d);
        return n;
    }
};

}  // namespace llmengine
