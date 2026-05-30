#include "llmengine/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define LLMENGINE_HAS_NEON 1
#else
#define LLMENGINE_HAS_NEON 0
#endif

namespace llmengine::kernels {

void rmsnorm_f32(const float* x, const float* w, int dim, float eps, float* out) {
    // Match HF LlamaRMSNorm: variance = mean(x^2), out = w * x * rsqrt(variance + eps).
    float sumsq = 0.0f;
    for (int i = 0; i < dim; ++i) sumsq += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sumsq / static_cast<float>(dim) + eps);
    for (int i = 0; i < dim; ++i) out[i] = w[i] * x[i] * inv_rms;
}

void silu_f32(const float* x, int n, float* out) {
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        out[i] = v / (1.0f + std::exp(-v));
    }
}

void matmul_f32(const float* a,
                const float* w,
                int          M,
                int          N,
                int          K,
                float*       out) {
    // out[m, n] = sum_k a[m, k] * w[n, k]; W stored row-major as [N, K].
    // Weight-stationary order (n outer, m inner): each weight row is read once
    // and reused across all M activation rows, so a batched decode step
    // (M = #running seqs) reads the weights ~M x fewer times. The per-element
    // dot is unchanged, so out[m,n] is bit-identical to the m-outer order.
    for (int n = 0; n < N; ++n) {
        const float* w_row = &w[static_cast<std::size_t>(n) * K];
        for (int m = 0; m < M; ++m) {
            const float* a_row = &a[static_cast<std::size_t>(m) * K];
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += a_row[k] * w_row[k];
            out[static_cast<std::size_t>(m) * N + n] = acc;
        }
    }
}

void embed_lookup_f32(const float* embedding,
                      int          token_id,
                      int          hidden_size,
                      float*       out) {
    std::memcpy(out,
                embedding + static_cast<std::size_t>(token_id) * hidden_size,
                static_cast<std::size_t>(hidden_size) * sizeof(float));
}

void matmul_f16w_f32a(const float*  a,
                      const __fp16* w,
                      int           M,
                      int           N,
                      int           K,
                      float*        out) {
    for (int n = 0; n < N; ++n) {
        const __fp16* w_row = &w[static_cast<std::size_t>(n) * K];
        for (int m = 0; m < M; ++m) {
            const float* a_row = &a[static_cast<std::size_t>(m) * K];
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += a_row[k] * static_cast<float>(w_row[k]);
            out[static_cast<std::size_t>(m) * N + n] = acc;
        }
    }
}

void matmul_int8w_f32a(const float*       a,
                       const std::int8_t* w,
                       const __fp16*      scales,
                       int                M,
                       int                N,
                       int                K,
                       float*             out) {
    for (int n = 0; n < N; ++n) {
        const std::int8_t* w_row = &w[static_cast<std::size_t>(n) * K];
        const float scale_n = static_cast<float>(scales[n]);
        for (int m = 0; m < M; ++m) {
            const float* a_row = &a[static_cast<std::size_t>(m) * K];
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += a_row[k] * static_cast<float>(w_row[k]);
            out[static_cast<std::size_t>(m) * N + n] = acc * scale_n;
        }
    }
}

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

void llama3_inv_freq(int    head_dim,
                     float  rope_theta,
                     bool   has_scaling,
                     float  factor,
                     float  low_freq_factor,
                     float  high_freq_factor,
                     int    original_max_pos,
                     float* inv_freq_out) {
    // We compute inv_freq in FP64 throughout. Large position × inv_freq
    // amplifies any ULP error in inv_freq into many ULPs of angle, which is
    // very visible in cos/sin at positions >=1000. Storing FP32 at the end.
    const int half = head_dim / 2;
    const double theta = static_cast<double>(rope_theta);

    std::vector<double> inv_freq(half);
    for (int i = 0; i < half; ++i) {
        double exponent = static_cast<double>(2 * i) / static_cast<double>(head_dim);
        inv_freq[i] = 1.0 / std::pow(theta, exponent);
    }

    if (has_scaling) {
        const double low_freq_wavelen  = static_cast<double>(original_max_pos)
                                       / low_freq_factor;
        const double high_freq_wavelen = static_cast<double>(original_max_pos)
                                       / high_freq_factor;
        const double band              = static_cast<double>(high_freq_factor)
                                       - low_freq_factor;
        const double two_pi            = 2.0 * static_cast<double>(kPi);

        for (int i = 0; i < half; ++i) {
            double fr = inv_freq[i];
            double wavelen = two_pi / fr;
            if (wavelen > low_freq_wavelen) {
                inv_freq[i] = fr / factor;
            } else if (wavelen < high_freq_wavelen) {
                // (no-op)
            } else {
                double smooth = (static_cast<double>(original_max_pos) / wavelen
                                 - low_freq_factor) / band;
                inv_freq[i] = (1.0 - smooth) * fr / factor + smooth * fr;
            }
        }
    }

    for (int i = 0; i < half; ++i)
        inv_freq_out[i] = static_cast<float>(inv_freq[i]);
}

void build_rope_tables(const float* inv_freq,
                       int          max_pos,
                       int          head_dim,
                       float*       cos_table,
                       float*       sin_table) {
    // Build angles + cos/sin in FP64 then cast at the end. cos(p * inv_freq)
    // for p ~ 1e4 and inv_freq ~ 1.0 has rapid oscillation; FP32 angle
    // truncation visibly shifts cos by ~1e-5 even when inv_freq is exact.
    const int half = head_dim / 2;
    std::vector<double> ifd(half);
    for (int i = 0; i < half; ++i)
        ifd[i] = static_cast<double>(inv_freq[i]);

    for (int p = 0; p < max_pos; ++p) {
        const double pd = static_cast<double>(p);
        for (int i = 0; i < half; ++i) {
            double angle = pd * ifd[i];
            float c = static_cast<float>(std::cos(angle));
            float s = static_cast<float>(std::sin(angle));
            cos_table[p * head_dim + i]        = c;
            cos_table[p * head_dim + i + half] = c;     // halved layout
            sin_table[p * head_dim + i]        = s;
            sin_table[p * head_dim + i + half] = s;
        }
    }
}

void apply_rope_inplace(float*       q,
                        const float* cos_row,
                        const float* sin_row,
                        int          head_dim) {
    // rotate_half([a, b]) = [-b, a] where a, b each have length head_dim/2.
    // q_new[i]       = q[i]       * cos[i]       - q[i + h] * sin[i]   for i < h
    // q_new[i + h]   = q[i + h]   * cos[i + h]   + q[i]     * sin[i + h]
    const int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        float a = q[i];
        float b = q[i + half];
        q[i]        = a * cos_row[i]        - b * sin_row[i];
        q[i + half] = b * cos_row[i + half] + a * sin_row[i + half];
    }
}

#if LLMENGINE_HAS_NEON

// 4-way unrolled NEON dot product over `d` floats; pattern from
// vectordb/src/distance.cpp:59-91. Returns the scalar dot.
static inline float neon_dot_f32(const float* a, const float* b, int d) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        acc2 = vfmaq_f32(acc2, a2, b2);
        acc3 = vfmaq_f32(acc3, a3, b3);
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc0 = vfmaq_f32(acc0, va, vb);
    }
    float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float sum = vaddvq_f32(acc);
    for (; i < d; ++i) sum += a[i] * b[i];
    return sum;
}

void matmul_f32_neon(const float* a,
                     const float* w,
                     int          M,
                     int          N,
                     int          K,
                     float*       out) {
    // out[m, n] = sum_k a[m, k] * w[n, k]; W row-major [N, K].
    // Inner loop is a NEON-accelerated dot over K. Weight-stationary order
    // (n outer, m inner) reuses each w_row across the M activation rows.
    for (int n = 0; n < N; ++n) {
        const float* w_row = &w[static_cast<std::size_t>(n) * K];
        for (int m = 0; m < M; ++m) {
            const float* a_row = &a[static_cast<std::size_t>(m) * K];
            out[static_cast<std::size_t>(m) * N + n] =
                neon_dot_f32(a_row, w_row, K);
        }
    }
}

void matmul_f16w_f32a_neon(const float*  a,
                           const __fp16* w,
                           int           M,
                           int           N,
                           int           K,
                           float*        out) {
    // Weight-stationary order (n outer, m inner): widen each w_row once and
    // reuse it across the M activation rows.
    for (int n = 0; n < N; ++n) {
        const __fp16* w_row = &w[static_cast<std::size_t>(n) * K];
        for (int m = 0; m < M; ++m) {
            const float* a_row = &a[static_cast<std::size_t>(m) * K];

            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            int k = 0;
            for (; k + 16 <= K; k += 16) {
                float16x8_t wlo = vld1q_f16(w_row + k);
                float16x8_t whi = vld1q_f16(w_row + k + 8);

                float32x4_t w0 = vcvt_f32_f16(vget_low_f16(wlo));
                float32x4_t w1 = vcvt_high_f32_f16(wlo);
                float32x4_t w2 = vcvt_f32_f16(vget_low_f16(whi));
                float32x4_t w3 = vcvt_high_f32_f16(whi);

                float32x4_t a0 = vld1q_f32(a_row + k);
                float32x4_t a1 = vld1q_f32(a_row + k + 4);
                float32x4_t a2 = vld1q_f32(a_row + k + 8);
                float32x4_t a3 = vld1q_f32(a_row + k + 12);

                acc0 = vfmaq_f32(acc0, a0, w0);
                acc1 = vfmaq_f32(acc1, a1, w1);
                acc2 = vfmaq_f32(acc2, a2, w2);
                acc3 = vfmaq_f32(acc3, a3, w3);
            }
            for (; k + 4 <= K; k += 4) {
                float16x4_t w4 = vld1_f16(w_row + k);
                float32x4_t w_f32 = vcvt_f32_f16(w4);
                float32x4_t a_f32 = vld1q_f32(a_row + k);
                acc0 = vfmaq_f32(acc0, a_f32, w_f32);
            }
            float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float sum = vaddvq_f32(acc);
            for (; k < K; ++k) sum += a_row[k] * static_cast<float>(w_row[k]);

            out[static_cast<std::size_t>(m) * N + n] = sum;
        }
    }
}

void matmul_int8w_f32a_neon(const float*       a,
                            const std::int8_t* w,
                            const __fp16*      scales,
                            int                M,
                            int                N,
                            int                K,
                            float*             out) {
    // Weight-stationary order (n outer, m inner): load each int8 w_row once
    // and reuse it across the M activation rows.
    for (int n = 0; n < N; ++n) {
        const std::int8_t* w_row = &w[static_cast<std::size_t>(n) * K];
        for (int m = 0; m < M; ++m) {
            const float* a_row = &a[static_cast<std::size_t>(m) * K];

            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            int k = 0;
            for (; k + 16 <= K; k += 16) {
                int8x16_t w8 = vld1q_s8(w_row + k);
                int16x8_t w16_lo = vmovl_s8(vget_low_s8(w8));
                int16x8_t w16_hi = vmovl_high_s8(w8);
                int32x4_t w32_0 = vmovl_s16(vget_low_s16(w16_lo));
                int32x4_t w32_1 = vmovl_high_s16(w16_lo);
                int32x4_t w32_2 = vmovl_s16(vget_low_s16(w16_hi));
                int32x4_t w32_3 = vmovl_high_s16(w16_hi);
                float32x4_t wf0 = vcvtq_f32_s32(w32_0);
                float32x4_t wf1 = vcvtq_f32_s32(w32_1);
                float32x4_t wf2 = vcvtq_f32_s32(w32_2);
                float32x4_t wf3 = vcvtq_f32_s32(w32_3);

                float32x4_t a0 = vld1q_f32(a_row + k);
                float32x4_t a1 = vld1q_f32(a_row + k + 4);
                float32x4_t a2 = vld1q_f32(a_row + k + 8);
                float32x4_t a3 = vld1q_f32(a_row + k + 12);

                acc0 = vfmaq_f32(acc0, a0, wf0);
                acc1 = vfmaq_f32(acc1, a1, wf1);
                acc2 = vfmaq_f32(acc2, a2, wf2);
                acc3 = vfmaq_f32(acc3, a3, wf3);
            }
            float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float sum = vaddvq_f32(acc);
            for (; k < K; ++k) sum += a_row[k] * static_cast<float>(w_row[k]);

            out[static_cast<std::size_t>(m) * N + n] = sum * static_cast<float>(scales[n]);
        }
    }
}

void rmsnorm_f32_neon(const float* x, const float* w, int dim, float eps, float* out) {
    // Sum of squares in 4 accumulators, then scaled write.
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);
        s0 = vfmaq_f32(s0, x0, x0);
        s1 = vfmaq_f32(s1, x1, x1);
        s2 = vfmaq_f32(s2, x2, x2);
        s3 = vfmaq_f32(s3, x3, x3);
    }
    for (; i + 4 <= dim; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        s0 = vfmaq_f32(s0, xv, xv);
    }
    float32x4_t s = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    float sumsq = vaddvq_f32(s);
    for (; i < dim; ++i) sumsq += x[i] * x[i];

    const float inv_rms = 1.0f / std::sqrt(sumsq / static_cast<float>(dim) + eps);
    const float32x4_t scale = vdupq_n_f32(inv_rms);

    i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(w + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(xv, scale), wv));
    }
    for (; i < dim; ++i) out[i] = w[i] * x[i] * inv_rms;
}

#endif  // LLMENGINE_HAS_NEON

void attention_f32(const float* q,
                   const float* K_hist,
                   const float* V_hist,
                   int          num_q_heads,
                   int          num_kv_heads,
                   int          head_dim,
                   int          seq_len,
                   float*       out) {
    const int group_size = num_q_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Per-thread reused scratch for the score row. attention_f32 runs once
    // per layer per token during decode; a fresh heap allocation each call
    // was churn in the hot path. thread_local keeps it race-free — the engine
    // serializes forward passes, but this kernel is also reachable from the
    // pybind test harness and from independent Engine instances.
    thread_local std::vector<float> scores;
    scores.resize(static_cast<std::size_t>(seq_len));

    for (int h = 0; h < num_q_heads; ++h) {
        const int kv_h = h / group_size;
        const float* qh = &q[static_cast<std::size_t>(h) * head_dim];

        // 1. scores[t] = (q . K[t, kv_h]) * scale
        float max_score = -INFINITY;
        for (int t = 0; t < seq_len; ++t) {
            const float* kt = &K_hist[(static_cast<std::size_t>(t) * num_kv_heads + kv_h)
                                      * head_dim];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += qh[d] * kt[d];
            float s = dot * scale;
            scores[t] = s;
            if (s > max_score) max_score = s;
        }

        // 2. stable softmax
        float denom = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            denom += scores[t];
        }
        float inv_denom = 1.0f / denom;
        for (int t = 0; t < seq_len; ++t) scores[t] *= inv_denom;

        // 3. out[h] = sum_t scores[t] * V[t, kv_h]
        float* oh = &out[static_cast<std::size_t>(h) * head_dim];
        std::fill(oh, oh + head_dim, 0.0f);
        for (int t = 0; t < seq_len; ++t) {
            const float* vt = &V_hist[(static_cast<std::size_t>(t) * num_kv_heads + kv_h)
                                      * head_dim];
            float s = scores[t];
            for (int d = 0; d < head_dim; ++d) oh[d] += s * vt[d];
        }
    }
}

}  // namespace llmengine::kernels
