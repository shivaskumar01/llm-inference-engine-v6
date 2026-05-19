#include "llmengine/weights.hpp"
#include "llmengine/config.hpp"
#include "llmengine/types.hpp"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace llmengine {

namespace {

using json = nlohmann::json;

std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::string> list_safetensors(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (!d) throw std::runtime_error("cannot open dir: " + dir);
    while (auto* e = readdir(d)) {
        std::string name(e->d_name);
        if (name.size() > 12 &&
            name.compare(name.size() - 12, 12, ".safetensors") == 0) {
            out.push_back(dir + "/" + name);
        }
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

ModelConfig load_config(const std::string& model_dir) {
    auto text = slurp(model_dir + "/config.json");
    auto j = json::parse(text);

    ModelConfig c;
    c.hidden_size            = j.value("hidden_size", 0);
    c.intermediate_size      = j.value("intermediate_size", 0);
    c.num_hidden_layers      = j.value("num_hidden_layers", 0);
    c.num_attention_heads    = j.value("num_attention_heads", 0);
    c.num_key_value_heads    = j.value("num_key_value_heads", c.num_attention_heads);
    c.head_dim               = j.value("head_dim",
                                       c.num_attention_heads
                                         ? c.hidden_size / c.num_attention_heads
                                         : 0);
    c.vocab_size             = j.value("vocab_size", 0);
    c.max_position_embeddings = j.value("max_position_embeddings", 0);
    c.rms_norm_eps           = j.value("rms_norm_eps", 1e-5f);
    c.tie_word_embeddings    = j.value("tie_word_embeddings", false);

    // RoPE config has two layouts depending on the transformers version:
    //   - v4.x and earlier: top-level `rope_theta` + optional `rope_scaling`.
    //   - v5.x: a unified `rope_parameters` block that contains rope_theta,
    //           rope_type, factor, low/high_freq_factor, original_max_pos.
    // We accept either; rope_parameters wins if both are present.
    c.rope_theta = j.value("rope_theta", 10000.0f);

    auto parse_scaling = [&](const nlohmann::json& rs) {
        c.rope_has_scaling      = true;
        c.rope_factor           = rs.value("factor", 1.0f);
        c.rope_low_freq_factor  = rs.value("low_freq_factor", 1.0f);
        c.rope_high_freq_factor = rs.value("high_freq_factor", 1.0f);
        c.rope_original_max_pos = rs.value("original_max_position_embeddings", 0);
    };

    if (j.contains("rope_parameters") && !j["rope_parameters"].is_null()) {
        const auto& rp = j["rope_parameters"];
        c.rope_theta = rp.value("rope_theta", c.rope_theta);
        const std::string rope_type = rp.value("rope_type", std::string("default"));
        if (rope_type == "llama3") parse_scaling(rp);
    } else if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
        parse_scaling(j["rope_scaling"]);
    }

    auto absorb_eos = [&](const json& src) {
        if (!src.contains("eos_token_id") || src["eos_token_id"].is_null()) return;
        const auto& eos = src["eos_token_id"];
        if (eos.is_array()) {
            for (const auto& v : eos) c.eos_token_ids.insert(v.get<int>());
        } else if (eos.is_number_integer()) {
            c.eos_token_ids.insert(eos.get<int>());
        }
    };

    absorb_eos(j);

    // HF splits Llama 3.x's full EOS set ({eot, eom, eot_id}) into
    // generation_config.json — config.json keeps only one. Merge both so
    // multi-EOS termination works for Instruct models out of the box.
    std::ifstream gen_f(model_dir + "/generation_config.json");
    if (gen_f) {
        std::ostringstream ss;
        ss << gen_f.rdbuf();
        absorb_eos(json::parse(ss.str()));
    }

    return c;
}

WeightMap::~WeightMap() {
    for (auto& f : files_) {
        if (f.addr) ::munmap(f.addr, f.length);
        if (f.fd >= 0) ::close(f.fd);
    }
}

WeightMap::WeightMap(WeightMap&& other) noexcept {
    files_   = std::move(other.files_);
    tensors_ = std::move(other.tensors_);
    other.files_.clear();
    other.tensors_.clear();
}

WeightMap& WeightMap::operator=(WeightMap&& other) noexcept {
    if (this != &other) {
        for (auto& f : files_) {
            if (f.addr) ::munmap(f.addr, f.length);
            if (f.fd >= 0) ::close(f.fd);
        }
        files_   = std::move(other.files_);
        tensors_ = std::move(other.tensors_);
        other.files_.clear();
        other.tensors_.clear();
    }
    return *this;
}

void WeightMap::load_one_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("open failed: " + path + ": " + std::strerror(errno));

    struct stat st;
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("fstat failed: " + path);
    }

    void* addr = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("mmap failed: " + path + ": " + std::strerror(errno));
    }

    files_.push_back({fd, addr, static_cast<std::size_t>(st.st_size)});

    // safetensors layout:
    //   [8 bytes little-endian header length N]
    //   [N bytes JSON header]
    //   [raw tensor bytes]
    const auto* base = static_cast<const std::uint8_t*>(addr);
    if (st.st_size < 8)
        throw std::runtime_error("safetensors file too small: " + path);

    std::uint64_t header_len;
    std::memcpy(&header_len, base, sizeof(header_len));
    if (header_len + 8 > static_cast<std::uint64_t>(st.st_size))
        throw std::runtime_error("safetensors header overruns file: " + path);

    std::string header_str(reinterpret_cast<const char*>(base + 8), header_len);
    auto header = json::parse(header_str);

    const auto* data_base = base + 8 + header_len;

    for (auto it = header.begin(); it != header.end(); ++it) {
        const std::string& name = it.key();
        if (name == "__metadata__") continue;
        const auto& spec = it.value();

        Tensor t;
        t.dtype = parse_safetensors_dtype(spec.at("dtype").get<std::string>());
        t.shape = spec.at("shape").get<std::vector<std::int64_t>>();

        // Shape sanity + overflow-safe numel. A malformed file with
        // negative dims or a numel that wraps a 64-bit count would
        // otherwise slip past offset checks and trip a much later read.
        std::uint64_t numel = 1;
        for (auto d : t.shape) {
            if (d < 0)
                throw std::runtime_error(
                    "safetensors negative shape dim for " + name);
            const auto du = static_cast<std::uint64_t>(d);
            if (du != 0 && numel > std::numeric_limits<std::uint64_t>::max() / du)
                throw std::runtime_error(
                    "safetensors numel overflow for " + name);
            numel *= du;
        }
        const std::uint64_t dt_bytes = dtype_bytes(t.dtype);
        if (dt_bytes != 0
            && numel > std::numeric_limits<std::uint64_t>::max() / dt_bytes)
            throw std::runtime_error(
                "safetensors byte-count overflow for " + name);
        const std::uint64_t expected_bytes = numel * dt_bytes;

        const auto offsets = spec.at("data_offsets").get<std::array<std::uint64_t, 2>>();
        // safetensors data_offsets are relative to data_base (= base + 8 +
        // header_len), not the whole file. The valid upper bound is the
        // data-section length: file size minus the 8-byte length prefix
        // minus the header. The previous check was too permissive and
        // would let a malformed file point past the mapping.
        const std::uint64_t data_section_bytes =
            static_cast<std::uint64_t>(st.st_size) - 8 - header_len;
        if (offsets[1] < offsets[0] || offsets[1] > data_section_bytes)
            throw std::runtime_error("safetensors offsets out of range for " + name);

        const std::uint64_t actual_bytes = offsets[1] - offsets[0];
        if (actual_bytes != expected_bytes)
            throw std::runtime_error(
                "safetensors byte-count mismatch for " + name
                + ": header says shape*dtype = " + std::to_string(expected_bytes)
                + " bytes but data_offsets cover "
                + std::to_string(actual_bytes));

        t.nbytes = actual_bytes;
        t.data   = data_base + offsets[0];

        tensors_.emplace(name, t);
    }
}

void WeightMap::apply_tied_lm_head_alias() {
    auto embed_it = tensors_.find("model.embed_tokens.weight");
    if (embed_it == tensors_.end()) {
        // No embedding tensor by the conventional name. Nothing to alias against.
        return;
    }
    auto lm_it = tensors_.find("lm_head.weight");
    if (lm_it == tensors_.end()) {
        // Tied checkpoint that omits lm_head.weight — alias it.
        tensors_.emplace("lm_head.weight", embed_it->second);
        return;
    }
    // Both present. Alias to the embedding so the engine treats them as
    // a single buffer and queries are O(1) regardless of save convention.
    if (lm_it->second.data != embed_it->second.data) {
        if (lm_it->second.shape != embed_it->second.shape ||
            lm_it->second.dtype != embed_it->second.dtype) {
            throw std::runtime_error(
                "tie_word_embeddings=true but lm_head.weight shape/dtype "
                "disagrees with model.embed_tokens.weight");
        }
        lm_it->second = embed_it->second;
    }
}

void WeightMap::load_safetensors_dir(const std::string& model_dir,
                                     const ModelConfig& cfg) {
    auto files = list_safetensors(model_dir);
    if (files.empty())
        throw std::runtime_error("no .safetensors files in " + model_dir);

    for (const auto& f : files) load_one_file(f);

    if (cfg.tie_word_embeddings) apply_tied_lm_head_alias();
}

bool WeightMap::contains(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const Tensor& WeightMap::at(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::out_of_range("weight not found: " + name);
    return it->second;
}

std::uintptr_t WeightMap::debug_ptr(const std::string& name) const {
    return reinterpret_cast<std::uintptr_t>(at(name).data);
}

std::vector<std::string> WeightMap::keys() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& [k, _] : tensors_) out.push_back(k);
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace llmengine
