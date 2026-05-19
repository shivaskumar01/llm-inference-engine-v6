#pragma once

// Phase 1: scalar FP32 reference kernels. Used by the CORRECTNESS build and
// as the ground-truth oracle for the perf-build NEON kernels in Phases 4–5.
//
// All linear-weight matrices are stored row-major as [N=out_dim, K=in_dim].
// Matmul iterates K (inner) contiguously across W's last dim.

#include <cstddef>
#include <cstdint>

namespace llmengine::kernels {

// y[i] = w[i] * x[i] / sqrt(mean(x^2) + eps)  (over the last dim of x)
// Matches HF LlamaRMSNorm exactly (FP32 throughout).
void rmsnorm_f32(const float* x,
                 const float* w,
                 int          dim,
                 float        eps,
                 float*       out);

#if defined(__ARM_NEON)
// NEON-accelerated drop-in for rmsnorm_f32. 4-way unrolled sum-of-squares
// + 4-way scaled multiply. May differ from the scalar path by ULPs of FMA
// rounding; results match torch FP32 within atol=1e-5 for typical inputs.
void rmsnorm_f32_neon(const float* x,
                      const float* w,
                      int          dim,
                      float        eps,
                      float*       out);
#endif

// y[i] = x[i] * sigmoid(x[i])  with std::exp
void silu_f32(const float* x, int n, float* out);

// out[m, n] = sum_k a[m, k] * w[n, k]
// W stored row-major as [N, K]. A is [M, K]. Out is [M, N].
void matmul_f32(const float* a,
                const float* w,
                int          M,
                int          N,
                int          K,
                float*       out);

#if defined(__ARM_NEON)
// NEON-accelerated drop-in for matmul_f32 with the same signature/contract.
// Inner-K dot product is 4-way unrolled (16 lanes/iter) with FP32 FMA;
// outer (m, n) loops stay scalar — enough to hit ~4-5x the scalar version
// on Apple Silicon for the matmul shapes used in Llama 3.2 1B forward.
void matmul_f32_neon(const float* a,
                     const float* w,
                     int          M,
                     int          N,
                     int          K,
                     float*       out);
#endif

// W16A32 matmul: FP16 weights, FP32 activations, FP32 output. Halves the
// bytes/element read from the weight matrix — the main bandwidth win for
// decode-time matmuls on memory-bound Apple Silicon.
//
// W layout matches matmul_f32: row-major [N=out, K=in]. Stored as __fp16.
// The NEON path uses vcvt_f32_f16 to widen weights inline and accumulates
// in FP32 (no precision loss beyond the FP16 weight rounding itself).
void matmul_f16w_f32a(const float*  a,
                      const __fp16* w,
                      int           M,
                      int           N,
                      int           K,
                      float*        out);

#if defined(__ARM_NEON)
void matmul_f16w_f32a_neon(const float*  a,
                           const __fp16* w,
                           int           M,
                           int           N,
                           int           K,
                           float*        out);
#endif

// W8A32 matmul: per-output-channel symmetric INT8 weights, FP32 activations,
// FP32 output. `scales[n]` is the FP16 scale for row n of the weight
// matrix; the dequantized value of w[n, k] is `int8_t(w[n, k]) * scales[n]`.
//
// out[m, n] = scales[n] * sum_k (a[m, k] * float(w[n, k]))
//
// The NEON path widens int8 → int16 → int32 → float inline (no sdot/i8mm
// yet — that's W8A8 territory and a v2 stretch).
void matmul_int8w_f32a(const float*       a,
                       const std::int8_t* w,
                       const __fp16*      scales,
                       int                M,
                       int                N,
                       int                K,
                       float*             out);

#if defined(__ARM_NEON)
void matmul_int8w_f32a_neon(const float*       a,
                            const std::int8_t* w,
                            const __fp16*      scales,
                            int                M,
                            int                N,
                            int                K,
                            float*             out);
#endif

// out = embedding[token_id, :], shape [hidden_size]
void embed_lookup_f32(const float* embedding,   // [vocab_size, hidden_size]
                      int          token_id,
                      int          hidden_size,
                      float*       out);

// Llama-3 scaled RoPE. Builds inv_freq with the scaled smoothing rule.
// Caller passes head_dim (must be even), rope_theta, scaling fields, and a
// dest array of length head_dim/2.
void llama3_inv_freq(int    head_dim,
                     float  rope_theta,
                     bool   has_scaling,
                     float  factor,
                     float  low_freq_factor,
                     float  high_freq_factor,
                     int    original_max_pos,
                     float* inv_freq_out);

// Build cos/sin tables for positions [0, max_pos), shape [max_pos, head_dim].
// HF Llama uses the "halved" layout: emb = cat(freqs, freqs) along last dim.
// So cos_table[p, i] = cos_table[p, i + head_dim/2] for i < head_dim/2.
void build_rope_tables(const float* inv_freq,    // [head_dim/2]
                       int          max_pos,
                       int          head_dim,
                       float*       cos_table,   // [max_pos, head_dim]
                       float*       sin_table);  // [max_pos, head_dim]

// Apply RoPE to one head's q (or k) vector in-place at a given position.
// HF's "halved" rotation: q_rot = q*cos + rotate_half(q)*sin
// rotate_half([a, b]) = [-b, a]  (a, b each length head_dim/2).
void apply_rope_inplace(float*       q,
                        const float* cos_row,    // [head_dim]
                        const float* sin_row,    // [head_dim]
                        int          head_dim);

// Multi-head attention with grouped-query attention (GQA).
// q:        [num_q_heads, head_dim]    — current query (single token)
// K_hist:   [seq_len, num_kv_heads, head_dim]   — keys for positions 0..seq_len-1
// V_hist:   [seq_len, num_kv_heads, head_dim]   — values
// out:      [num_q_heads, head_dim]    — concatenated attention output
//
// group_size = num_q_heads / num_kv_heads. Causal handled by seq_len bound.
void attention_f32(const float* q,
                   const float* K_hist,
                   const float* V_hist,
                   int          num_q_heads,
                   int          num_kv_heads,
                   int          head_dim,
                   int          seq_len,
                   float*       out);

}  // namespace llmengine::kernels
