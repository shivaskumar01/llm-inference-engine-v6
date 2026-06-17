#include "llmengine/engine.hpp"
#include "llmengine/config.hpp"
#include "llmengine/kernels.hpp"
#include "llmengine/paged_kv.hpp"
#include "llmengine/scheduler.hpp"
#include "llmengine/types.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace llmengine;

using FArr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IArr = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

namespace {

// ---------- kernel bindings (FP32 reference) ----------

FArr py_rmsnorm(FArr x, FArr w, float eps) {
    if (x.ndim() != 1 || w.ndim() != 1)
        throw std::invalid_argument("rmsnorm: x and w must be 1-D");
    int dim = static_cast<int>(x.shape(0));
    if (w.shape(0) != dim)
        throw std::invalid_argument("rmsnorm: dim mismatch");
    FArr out(dim);
    kernels::rmsnorm_f32(x.data(), w.data(), dim, eps, out.mutable_data());
    return out;
}

FArr py_silu(FArr x) {
    if (x.ndim() != 1) throw std::invalid_argument("silu: x must be 1-D");
    int n = static_cast<int>(x.shape(0));
    FArr out(n);
    kernels::silu_f32(x.data(), n, out.mutable_data());
    return out;
}

FArr py_matmul(FArr a, FArr w) {
    // a: [M, K], w: [N, K], out: [M, N]
    if (a.ndim() != 2 || w.ndim() != 2)
        throw std::invalid_argument("matmul: a and w must be 2-D");
    int M = static_cast<int>(a.shape(0));
    int K = static_cast<int>(a.shape(1));
    int N = static_cast<int>(w.shape(0));
    if (w.shape(1) != K)
        throw std::invalid_argument("matmul: K mismatch");
    FArr out({M, N});
    kernels::matmul_f32(a.data(), w.data(), M, N, K, out.mutable_data());
    return out;
}

FArr py_embed_lookup(FArr embedding, int token_id) {
    if (embedding.ndim() != 2)
        throw std::invalid_argument("embed_lookup: embedding must be 2-D");
    int hidden = static_cast<int>(embedding.shape(1));
    FArr out(hidden);
    kernels::embed_lookup_f32(embedding.data(), token_id, hidden,
                              out.mutable_data());
    return out;
}

FArr py_llama3_inv_freq(int head_dim, float rope_theta, bool has_scaling,
                        float factor, float low_freq, float high_freq,
                        int original_max_pos) {
    int half = head_dim / 2;
    FArr out(half);
    kernels::llama3_inv_freq(head_dim, rope_theta, has_scaling, factor,
                             low_freq, high_freq, original_max_pos,
                             out.mutable_data());
    return out;
}

py::tuple py_build_rope_tables(FArr inv_freq, int max_pos, int head_dim) {
    if (inv_freq.shape(0) != head_dim / 2)
        throw std::invalid_argument("inv_freq must have len head_dim/2");
    FArr cos_t({max_pos, head_dim});
    FArr sin_t({max_pos, head_dim});
    kernels::build_rope_tables(inv_freq.data(), max_pos, head_dim,
                               cos_t.mutable_data(), sin_t.mutable_data());
    return py::make_tuple(cos_t, sin_t);
}

FArr py_apply_rope(FArr q, FArr cos_row, FArr sin_row) {
    if (q.ndim() != 1) throw std::invalid_argument("apply_rope: q must be 1-D");
    int hd = static_cast<int>(q.shape(0));
    if (cos_row.shape(0) != hd || sin_row.shape(0) != hd)
        throw std::invalid_argument("apply_rope: head_dim mismatch");
    FArr out(hd);
    std::memcpy(out.mutable_data(), q.data(), hd * sizeof(float));
    kernels::apply_rope_inplace(out.mutable_data(), cos_row.data(),
                                sin_row.data(), hd);
    return out;
}

FArr py_attention(FArr q, FArr K, FArr V,
                  int num_q_heads, int num_kv_heads, int head_dim) {
    // q: [num_q_heads, head_dim], K/V: [seq_len, num_kv_heads, head_dim]
    if (q.ndim() != 2 || K.ndim() != 3 || V.ndim() != 3)
        throw std::invalid_argument("attention: q must be 2-D, K/V must be 3-D");
    int seq_len = static_cast<int>(K.shape(0));
    if (V.shape(0) != seq_len)
        throw std::invalid_argument("K and V seq_len mismatch");
    FArr out({num_q_heads, head_dim});
    kernels::attention_f32(q.data(), K.data(), V.data(),
                           num_q_heads, num_kv_heads, head_dim,
                           seq_len, out.mutable_data());
    return out;
}

}  // namespace

PYBIND11_MODULE(_llmengine, m) {
    m.doc() = "llm-engine: from-scratch LLM inference engine";

    // ---------- types ----------
    py::enum_<DType>(m, "DType")
        .value("F32",  DType::F32)
        .value("F16",  DType::F16)
        .value("BF16", DType::BF16)
        .value("I8",   DType::I8);

    py::class_<ModelConfig>(m, "ModelConfig")
        .def_readonly("hidden_size",             &ModelConfig::hidden_size)
        .def_readonly("intermediate_size",       &ModelConfig::intermediate_size)
        .def_readonly("num_hidden_layers",       &ModelConfig::num_hidden_layers)
        .def_readonly("num_attention_heads",     &ModelConfig::num_attention_heads)
        .def_readonly("num_key_value_heads",     &ModelConfig::num_key_value_heads)
        .def_readonly("head_dim",                &ModelConfig::head_dim)
        .def_readonly("vocab_size",              &ModelConfig::vocab_size)
        .def_readonly("max_position_embeddings", &ModelConfig::max_position_embeddings)
        .def_readonly("rms_norm_eps",            &ModelConfig::rms_norm_eps)
        .def_readonly("rope_theta",              &ModelConfig::rope_theta)
        .def_readonly("tie_word_embeddings",     &ModelConfig::tie_word_embeddings)
        .def_readonly("rope_has_scaling",        &ModelConfig::rope_has_scaling)
        .def_readonly("rope_factor",             &ModelConfig::rope_factor)
        .def_readonly("rope_low_freq_factor",    &ModelConfig::rope_low_freq_factor)
        .def_readonly("rope_high_freq_factor",   &ModelConfig::rope_high_freq_factor)
        .def_readonly("rope_original_max_pos",   &ModelConfig::rope_original_max_pos)
        .def_readonly("eos_token_ids",           &ModelConfig::eos_token_ids);

    // ---------- engine ----------
    py::class_<Engine>(m, "Engine")
        .def(py::init([](const std::string& model_dir, const std::string& dtype) {
                ModelWeightsRef::LinearStorage s;
                if (dtype == "fp32") s = ModelWeightsRef::LinearStorage::F32;
                else if (dtype == "fp16") s = ModelWeightsRef::LinearStorage::F16;
                else if (dtype == "int8") s = ModelWeightsRef::LinearStorage::I8;
                else throw std::invalid_argument(
                    "dtype must be 'fp32' | 'fp16' | 'int8', got: " + dtype);
                return new Engine(model_dir, s);
             }),
             py::arg("model_dir"), py::arg("dtype") = "fp32")
        .def_property_readonly("dtype",
            [](const Engine& e) {
                switch (e.linear_storage()) {
                    case ModelWeightsRef::LinearStorage::F32: return "fp32";
                    case ModelWeightsRef::LinearStorage::F16: return "fp16";
                    case ModelWeightsRef::LinearStorage::I8:  return "int8";
                }
                return "?";
            })
        .def_property_readonly("cfg",
            [](const Engine& e) { return e.config(); },
            py::return_value_policy::reference_internal)
        .def_property_readonly("max_pos",
            [](const Engine& e) { return e.max_pos(); },
            "Largest valid (pos + 1) accepted by forward_step / generate. "
            "Pure function of the config, useful for clients that want to "
            "validate (prompt_len + max_new_tokens) before submitting.")
        .def("weight_names",
            [](const Engine& e) { return e.weights().keys(); })
        .def("has_weight",
            [](const Engine& e, const std::string& n) { return e.weights().contains(n); })
#if defined(LLMENGINE_DEBUG_BINDINGS)
        // Test/debug only, see plan §2.1. Defaults ON in the correctness
        // build (needed by Phase 0 tied-alias tests), OFF in perf. Override
        // at configure with -DLLMENGINE_DEBUG_BINDINGS=ON/OFF.
        .def("_debug_weight_ptr",
            [](const Engine& e, const std::string& n) { return e.debug_weight_ptr(n); })
        .def("_debug_lm_head_f32_ptr",
            [](Engine& e) { return e.debug_lm_head_f32_ptr(); })
        .def("_debug_embed_tokens_ptr",
            [](Engine& e) { return e.debug_embed_tokens_ptr(); })
#endif
        .def("forward_logits",
            [](Engine& e, const std::vector<std::int32_t>& ids) {
                int V = e.config().vocab_size;
                std::vector<float> out;
                {
                    py::gil_scoped_release release;
                    out = e.forward_logits(ids);
                }
                int T = static_cast<int>(ids.size());
                FArr arr({T, V});
                std::memcpy(arr.mutable_data(), out.data(),
                            out.size() * sizeof(float));
                return arr;
            },
            py::arg("ids"),
            "Run the FP32 reference forward pass over the given token IDs and "
            "return logits of shape [T, vocab_size].")
        .def("forward_logits_paged",
            [](Engine& e, const std::vector<std::int32_t>& ids,
               BlockManager& mgr) {
                int V = e.config().vocab_size;
                std::vector<float> out;
                {
                    py::gil_scoped_release release;
                    out = e.forward_logits_paged(ids, mgr);
                }
                int T = static_cast<int>(ids.size());
                FArr arr({T, V});
                std::memcpy(arr.mutable_data(), out.data(),
                            out.size() * sizeof(float));
                return arr;
            },
            py::arg("ids"), py::arg("block_manager"),
            "Same as forward_logits but the per-layer K/V history is paged.")
        .def("generate_streaming",
            [](Engine& e, const std::vector<std::int32_t>& prompt_ids,
               int max_new_tokens, py::function on_token, py::function on_done,
               CancelToken& cancel) {
                // Wrap the Python callbacks so the engine thread acquires
                // the GIL before invoking them; the actual generate runs
                // with the GIL released so multiple streams can advance
                // in parallel from Python's perspective.
                auto on_tok_cpp = [on_token](int t) {
                    py::gil_scoped_acquire ga;
                    try { on_token(t); }
                    catch (const py::error_already_set&) { /* swallow */ }
                };
                auto on_dn_cpp = [on_done](const std::string& r) {
                    py::gil_scoped_acquire ga;
                    try { on_done(r); }
                    catch (const py::error_already_set&) { /* swallow */ }
                };
                py::gil_scoped_release release;
                e.generate_streaming(prompt_ids, max_new_tokens,
                                      on_tok_cpp, on_dn_cpp, cancel);
            },
            py::arg("prompt_ids"), py::arg("max_new_tokens"),
            py::arg("on_token"), py::arg("on_done"),
            py::arg("cancel_token"),
            "Token-by-token streaming with cooperative cancellation.")
        .def("generate",
            [](Engine& e, const std::vector<std::int32_t>& prompt_ids,
               int max_new_tokens) {
                std::pair<std::vector<std::int32_t>, std::string> out;
                {
                    py::gil_scoped_release release;
                    out = e.generate(prompt_ids, max_new_tokens);
                }
                return py::make_tuple(std::move(out.first), out.second);
            },
            py::arg("prompt_ids"), py::arg("max_new_tokens"),
            "Greedy-decode up to max_new_tokens or until any EOS token is "
            "sampled. Returns (output_ids, finish_reason) where finish_reason "
            "is 'stop' (EOS hit) or 'length' (cap reached).");

    // ---------- cancellation ----------
    py::class_<CancelToken>(m, "CancelToken")
        .def(py::init<>())
        .def("cancel",       &CancelToken::cancel)
        .def("is_cancelled", &CancelToken::is_cancelled)
        .def("reset",        &CancelToken::reset);

    // ---------- paged KV ----------
    py::class_<BlockManager>(m, "BlockManager")
        .def(py::init<int, int, int, int, int>(),
             py::arg("num_blocks"), py::arg("block_size"),
             py::arg("num_layers"), py::arg("num_kv_heads"), py::arg("head_dim"))
        .def_property_readonly("num_blocks",  &BlockManager::num_blocks)
        .def_property_readonly("block_size",  &BlockManager::block_size)
        .def_property_readonly("free_blocks", &BlockManager::free_blocks);

    // ---------- scheduler ----------
    py::class_<SequenceResult>(m, "SequenceResult")
        .def_readonly("seq_id",        &SequenceResult::seq_id)
        .def_readonly("prompt_len",    &SequenceResult::prompt_len)
        .def_readonly("token_ids",     &SequenceResult::token_ids)
        .def_readonly("finish_reason", &SequenceResult::finish_reason)
        .def("__repr__", [](const SequenceResult& r) {
            return "<SequenceResult seq_id=" + std::to_string(r.seq_id)
                 + " finish=" + r.finish_reason
                 + " len=" + std::to_string(r.token_ids.size()) + ">";
        });

    py::class_<StaticBatchScheduler>(m, "StaticBatchScheduler")
        .def(py::init<Engine&, BlockManager&>(),
             py::arg("engine"), py::arg("block_manager"),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("enqueue", &StaticBatchScheduler::enqueue,
             py::arg("prompt"), py::arg("max_new_tokens"))
        .def("run_until_done",
            [](StaticBatchScheduler& sched) {
                py::gil_scoped_release release;
                sched.run_until_done();
            })
        // results() returns enqueue-order (sorted by seq_id) at the C++
        // level, no extra sort needed in the binding.
        .def_property_readonly("results", &StaticBatchScheduler::results)
        .def("clear_results", &StaticBatchScheduler::clear_results);

    py::class_<ContinuousBatchScheduler>(m, "ContinuousBatchScheduler")
        .def(py::init<Engine&, BlockManager&, int, int>(),
             py::arg("engine"), py::arg("block_manager"),
             py::arg("max_concurrent") = 64,
             py::arg("max_prefill_tokens_per_step") = 256,
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("enqueue", &ContinuousBatchScheduler::enqueue,
             py::arg("prompt"), py::arg("max_new_tokens"))
        .def("step",
            [](ContinuousBatchScheduler& sched) {
                py::gil_scoped_release release;
                return sched.step();
            })
        .def("run_until_idle",
            [](ContinuousBatchScheduler& sched) {
                py::gil_scoped_release release;
                sched.run_until_idle();
            })
        .def_property_readonly("idle", &ContinuousBatchScheduler::idle)
        .def_property_readonly("results", &ContinuousBatchScheduler::results)
        .def("clear_results", &ContinuousBatchScheduler::clear_results)
        .def_property("budget",
            &ContinuousBatchScheduler::max_prefill_tokens_per_step,
            &ContinuousBatchScheduler::set_max_prefill_tokens_per_step);

    // ---------- kernels (free-function bindings for unit tests) ----------
    auto k = m.def_submodule("kernels", "FP32 reference kernel bindings (Phase 1).");
    k.def("rmsnorm_f32",     &py_rmsnorm,     py::arg("x"), py::arg("w"), py::arg("eps"));
    k.def("silu_f32",        &py_silu,        py::arg("x"));
    k.def("matmul_f32",      &py_matmul,      py::arg("a"), py::arg("w"));

    // INT8 matmul (scalar reference) for unit tests against torch dequant.
    using I8Arr   = py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>;
    using F16Arr  = py::array_t<std::uint16_t, py::array::c_style | py::array::forcecast>;
    k.def("matmul_int8w_f32a", [](FArr a, I8Arr w, F16Arr scales) -> FArr {
        if (a.ndim() != 2 || w.ndim() != 2 || scales.ndim() != 1)
            throw std::invalid_argument("matmul_int8w_f32a: a/w 2-D, scales 1-D");
        int M = static_cast<int>(a.shape(0));
        int K = static_cast<int>(a.shape(1));
        int N = static_cast<int>(w.shape(0));
        if (w.shape(1) != K) throw std::invalid_argument("K mismatch");
        if (scales.shape(0) != N) throw std::invalid_argument("scales[N] mismatch");
        FArr out({M, N});
        kernels::matmul_int8w_f32a(
            a.data(), w.data(),
            reinterpret_cast<const __fp16*>(scales.data()),
            M, N, K, out.mutable_data());
        return out;
    }, py::arg("a"), py::arg("w"), py::arg("scales"));

#if defined(__ARM_NEON)
    k.def("matmul_int8w_f32a_neon", [](FArr a, I8Arr w, F16Arr scales) -> FArr {
        if (a.ndim() != 2 || w.ndim() != 2 || scales.ndim() != 1)
            throw std::invalid_argument("matmul_int8w_f32a_neon: a/w 2-D, scales 1-D");
        int M = static_cast<int>(a.shape(0));
        int K = static_cast<int>(a.shape(1));
        int N = static_cast<int>(w.shape(0));
        if (w.shape(1) != K) throw std::invalid_argument("K mismatch");
        if (scales.shape(0) != N) throw std::invalid_argument("scales[N] mismatch");
        FArr out({M, N});
        kernels::matmul_int8w_f32a_neon(
            a.data(), w.data(),
            reinterpret_cast<const __fp16*>(scales.data()),
            M, N, K, out.mutable_data());
        return out;
    }, py::arg("a"), py::arg("w"), py::arg("scales"));

    k.def("matmul_f32_neon", [](FArr a, FArr w) -> FArr {
        if (a.ndim() != 2 || w.ndim() != 2)
            throw std::invalid_argument("matmul: a and w must be 2-D");
        int M = static_cast<int>(a.shape(0));
        int K = static_cast<int>(a.shape(1));
        int N = static_cast<int>(w.shape(0));
        if (w.shape(1) != K)
            throw std::invalid_argument("matmul: K mismatch");
        FArr out({M, N});
        kernels::matmul_f32_neon(a.data(), w.data(), M, N, K, out.mutable_data());
        return out;
    }, py::arg("a"), py::arg("w"));
    k.def("rmsnorm_f32_neon", [](FArr x, FArr w, float eps) -> FArr {
        if (x.ndim() != 1 || w.ndim() != 1)
            throw std::invalid_argument("rmsnorm: x and w must be 1-D");
        int dim = static_cast<int>(x.shape(0));
        if (w.shape(0) != dim)
            throw std::invalid_argument("rmsnorm: dim mismatch");
        FArr out(dim);
        kernels::rmsnorm_f32_neon(x.data(), w.data(), dim, eps, out.mutable_data());
        return out;
    }, py::arg("x"), py::arg("w"), py::arg("eps"));
#endif
    k.def("embed_lookup_f32",&py_embed_lookup,py::arg("embedding"), py::arg("token_id"));
    k.def("llama3_inv_freq", &py_llama3_inv_freq,
          py::arg("head_dim"), py::arg("rope_theta"), py::arg("has_scaling"),
          py::arg("factor"), py::arg("low_freq_factor"),
          py::arg("high_freq_factor"), py::arg("original_max_pos"));
    k.def("build_rope_tables", &py_build_rope_tables,
          py::arg("inv_freq"), py::arg("max_pos"), py::arg("head_dim"));
    k.def("apply_rope_f32",  &py_apply_rope,  py::arg("q"), py::arg("cos_row"), py::arg("sin_row"));
    k.def("attention_f32",   &py_attention,
          py::arg("q"), py::arg("K"), py::arg("V"),
          py::arg("num_q_heads"), py::arg("num_kv_heads"), py::arg("head_dim"));
}
