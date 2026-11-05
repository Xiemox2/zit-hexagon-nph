// NPH 0.30: combined ("concatenated") LoRA for HMX.
//
// For every target Linear, all standard LoRAs (lora_down/lora_up, optional per-part index for fused
// qkv) are packed into ONE down matrix A [K, Rp] and ONE up matrix B [Rp, N] (FP16, HMX operand
// type) plus an FP32 per-rank scale vector s [Rp]:
//
//     delta = B * diag(s) * (A * x)
//
// - Ranks of all LoRAs/parts are concatenated; fused-qkv parts become block rows of B (zeros
//   elsewhere), so no ggml_concat is needed.
// - The strength, alpha/rank (or .scale) live in s (FP32), never folded into FP16 data.
// - Rank is zero-padded to a multiple of 32 (HMX tile); padded columns have A=0, B=0, s=0 (exact).
// - FP16 exactness: every file value must round-trip. Each rank vector may get a power-of-two shift
//   (compensated exactly in s). Values that still do not fit FP16 (F32 files, extreme BF16) are split
//   into up to `max_split` FP16 parts per vector (hi/lo...), expanding the rank with cross terms, so
//   the stored weights reproduce the file values exactly (stats report any residual).
//
// Pure host code (ggml.h only) so it can be unit-tested on the box.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

namespace nph_lora {

struct PartSrc {
    int K       = 0;         // in features
    int r       = 0;         // rank
    int n       = 0;         // output rows covered by this part
    int row_off = 0;         // first output row of this part in the target
    const float* down = nullptr;  // ggml [K, r]  -> down[j*K + k]
    const float* up   = nullptr;  // ggml [r, n]  -> up[i*r + j]
    float scale       = 1.0f;     // alpha/rank (or .scale) * strength
};

struct Stats {
    int64_t values          = 0;  // file values packed
    int64_t inexact_values  = 0;  // values not reproduced exactly (should be 0)
    double max_abs_err      = 0;  // max |packed - file|
    int shifted_vectors     = 0;  // rank vectors that needed a power-of-two shift
    int split_vectors       = 0;  // rank vectors that needed >1 FP16 part
};

struct Combined {
    int K = 0, N = 0, R = 0, Rp = 0;
    std::vector<ggml_fp16_t> A;  // ggml [K, Rp]: A[j*K + k]
    std::vector<ggml_fp16_t> B;  // ggml [Rp, N]: B[n*Rp + j]
    std::vector<float> s;        // [Rp]
    // Power-of-two gain for scaled Linears (Z-Image w2, lin_scale 1/128): the delta is computed as
    // B*(diag(s)*2^gain*A*x_s) and unscaled by 2^-gain/lin_scale in FP32 afterwards (exact), so the
    // FP16 HMX activation/output of the B matmul sits in the same range as the per-LoRA path's
    // (which applied strength*alpha/rank after the up matmul) instead of near the FP16 subnormal range.
    int gain = 0;
    Stats st;
};

struct VecPart {
    int k = 0;                        // value = h * 2^-k
    std::vector<ggml_fp16_t> h;
};

inline int ilog2_floor(double m) {
    int e = 0;
    std::frexp(m, &e);  // m = f * 2^e, f in [0.5, 1)
    return e - 1;
}

// Split a strided vector (L values) into FP16 parts with power-of-two shifts.
// first_minimal: first part uses the smallest non-negative shift that is exact (keeps activations
// in the same range as the unshifted data); later parts are aligned to the first part's magnitude.
inline void split_vec(const float* v, int L, int stride, int max_parts, std::vector<VecPart>& parts, Stats& st) {
    parts.clear();
    std::vector<double> res(L);
    double m = 0;
    for (int i = 0; i < L; i++) {
        res[i] = (double)v[(size_t)i * stride];
        m      = std::max(m, std::fabs(res[i]));
    }
    st.values += L;
    if (m == 0) {
        return;
    }
    auto exact_at = [&](int k) {
        for (int i = 0; i < L; i++) {
            const float t = (float)std::ldexp(res[i], k);
            if (ggml_fp16_to_fp32(ggml_fp32_to_fp16(t)) != t || std::fabs(t) > 65504.0f) {
                return false;
            }
        }
        return true;
    };
    double ref_m = 0;  // scaled max of the first part
    for (int p = 0; p < std::max(1, max_parts); p++) {
        double mr = 0;
        for (int i = 0; i < L; i++) {
            mr = std::max(mr, std::fabs(res[i]));
        }
        if (mr == 0) {
            break;
        }
        const int e    = ilog2_floor(mr);
        const int kmax = 14 - e;  // mr * 2^kmax < 2^15 (no FP16 overflow)
        int k0, kcap;
        if (p == 0) {
            // Smallest shift first; never push the vector above |2| by shifting (keeps the HMX
            // activations of A*x and diag(s)*y in the same range as unshifted LoRA data).
            k0   = std::min(0, kmax);
            kcap = std::max(k0, std::min(kmax, -e));
        } else {
            // Residual parts: align their max to the first part's scaled max.
            k0   = std::min(kmax, ilog2_floor(ref_m) - e);
            kcap = k0;
        }
        int k      = kcap;
        bool exact = false;
        for (int kk = k0; kk <= kcap; kk++) {
            if (exact_at(kk)) {
                k     = kk;
                exact = true;
                break;
            }
        }
        if (p == 0) {
            ref_m = std::ldexp(mr, k);
        }
        VecPart vp;
        vp.k = k;
        vp.h.resize(L);
        for (int i = 0; i < L; i++) {
            const float t = (float)std::ldexp(res[i], k);
            vp.h[i]       = ggml_fp32_to_fp16(t);
            res[i] -= std::ldexp((double)ggml_fp16_to_fp32(vp.h[i]), -k);
        }
        parts.push_back(std::move(vp));
        if (exact) {
            break;
        }
    }
    for (const auto& vp : parts) {
        if (vp.k != 0) {
            st.shifted_vectors++;
            break;
        }
    }
    if (parts.size() > 1) {
        st.split_vectors++;
    }
    for (int i = 0; i < L; i++) {
        const double err = std::fabs(res[i]);
        if (err != 0) {
            st.inexact_values++;
            st.max_abs_err = std::max(st.max_abs_err, err);
        }
    }
}

// Pack all parts of one target into A/B/s. K = in features, N = out features of the target.
inline bool combine(const std::vector<PartSrc>& srcs, int K, int N, int max_split, Combined& out, std::string* err = nullptr) {
    out = Combined();
    out.K = K;
    out.N = N;
    struct Term {
        const PartSrc* src;
        const VecPart* a;
        const VecPart* b;
    };
    std::vector<std::vector<VecPart>> a_parts, b_parts;  // per (src, j)
    std::vector<Term> terms;
    for (const auto& ps : srcs) {
        if (ps.K != K || ps.row_off < 0 || ps.row_off + ps.n > N || ps.r <= 0) {
            if (err) {
                *err = "incompatible part";
            }
            return false;
        }
    }
    // Two passes: first split (stable storage), then build terms.
    size_t total_vecs = 0;
    for (const auto& ps : srcs) {
        total_vecs += (size_t)ps.r;
    }
    a_parts.resize(total_vecs);
    b_parts.resize(total_vecs);
    size_t vi = 0;
    for (const auto& ps : srcs) {
        for (int j = 0; j < ps.r; j++, vi++) {
            split_vec(ps.down + (size_t)j * K, K, 1, max_split, a_parts[vi], out.st);
            split_vec(ps.up + j, ps.n, ps.r, max_split, b_parts[vi], out.st);
            for (const auto& a : a_parts[vi]) {
                for (const auto& b : b_parts[vi]) {
                    terms.push_back({&ps, &a, &b});
                }
            }
        }
    }
    double max_scale = 0;
    for (const auto& ps : srcs) {
        max_scale = std::max(max_scale, std::fabs((double)ps.scale));
    }
    out.gain = max_scale > 0 ? std::max(0, std::min(10, -ilog2_floor(max_scale))) : 0;
    out.R  = (int)terms.size();
    out.Rp = std::max(32, (out.R + 31) / 32 * 32);
    out.A.assign((size_t)K * out.Rp, ggml_fp32_to_fp16(0.0f));
    out.B.assign((size_t)out.Rp * N, ggml_fp32_to_fp16(0.0f));
    out.s.assign(out.Rp, 0.0f);
    for (int t = 0; t < out.R; t++) {
        const Term& tm = terms[t];
        std::copy(tm.a->h.begin(), tm.a->h.end(), out.A.begin() + (size_t)t * K);
        for (int i = 0; i < tm.src->n; i++) {
            out.B[(size_t)(tm.src->row_off + i) * out.Rp + t] = tm.b->h[i];
        }
        // scale * 2^-(ka+kb), exact in FP32 unless the product leaves the FP32 range
        out.s[t] = (float)std::ldexp((double)tm.src->scale, -(tm.a->k + tm.b->k));
    }
    return true;
}

// Reference (double) of the original per-LoRA math for tests: delta[n, m] for x[m, k].
inline void reference_delta(const std::vector<PartSrc>& srcs, const float* x, int M, int K, int N, std::vector<double>& out) {
    out.assign((size_t)M * N, 0.0);
    for (const auto& ps : srcs) {
        std::vector<double> y((size_t)ps.r);
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < ps.r; j++) {
                double acc = 0;
                for (int k = 0; k < K; k++) {
                    acc += (double)ps.down[(size_t)j * K + k] * x[(size_t)m * K + k];
                }
                y[j] = acc;
            }
            for (int i = 0; i < ps.n; i++) {
                double acc = 0;
                for (int j = 0; j < ps.r; j++) {
                    acc += (double)ps.up[(size_t)i * ps.r + j] * y[j];
                }
                out[(size_t)m * N + ps.row_off + i] += acc * ps.scale;
            }
        }
    }
}

// On-load safety check: delta*x for a random x from the packed FP16 A/B + FP32 s (evaluated in
// double, i.e. exactly what the packing stores) vs the per-LoRA math on the file values.
// Returns max|combined - per-LoRA| / max|per-LoRA| (0 when the delta is all zero).
inline double check_packed(const std::vector<PartSrc>& srcs, const Combined& c, int M, uint32_t seed) {
    const int K = c.K, N = c.N;
    std::vector<float> x((size_t)M * K);
    uint64_t st = 0x9E3779B97F4A7C15ull ^ seed;
    for (auto& v : x) {  // deterministic uniform(-1, 1) (xorshift), no <random> dependency
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        v = (float)((double)(st >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0);
    }
    std::vector<double> ref;
    reference_delta(srcs, x.data(), M, K, N, ref);
    std::vector<double> y((size_t)c.Rp);
    double em = 0, rm = 0;
    for (int m = 0; m < M; m++) {
        for (int t = 0; t < c.Rp; t++) {
            double acc = 0;
            if (c.s[t] != 0) {
                const ggml_fp16_t* a = &c.A[(size_t)t * K];
                for (int k = 0; k < K; k++) acc += (double)ggml_fp16_to_fp32(a[k]) * x[(size_t)m * K + k];
            }
            y[t] = acc * (double)c.s[t];
        }
        for (int n = 0; n < N; n++) {
            double acc = 0;
            const ggml_fp16_t* b = &c.B[(size_t)n * c.Rp];
            for (int t = 0; t < c.Rp; t++) {
                if (y[t] != 0) acc += (double)ggml_fp16_to_fp32(b[t]) * y[t];
            }
            const double r = ref[(size_t)m * N + n];
            em = std::max(em, std::fabs(acc - r));
            rm = std::max(rm, std::fabs(r));
        }
    }
    return rm > 0 ? em / rm : em;
}

// Graph: delta added to `out`. x_s is the Linear input already multiplied by lin_scale (the base
// path's own pre-scaled input is reused, so no extra pass). Returns the new output tensor.
// The B matmul is the last node before the ADD so ggml-hexagon fuses MUL_MAT+ADD when lin_scale==1.
// Node names "nphL_*" let the ggml-hexagon profiler account LoRA ops separately (0.30.1).
inline ggml_tensor* graph_linear(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, bool prec_f32, const char* name = nullptr) {
    if (x->ne[2] * x->ne[3] > 1024) {
        const int64_t ne2 = x->ne[2], ne3 = x->ne[3];
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]);
        x = ggml_mul_mat(ctx, w, x);
        if (prec_f32) {
            ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        }
        if (name) {
            ggml_set_name(x, name);
        }
        return ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / ne2 / ne3, ne2, ne3);
    }
    x = ggml_mul_mat(ctx, w, x);
    if (prec_f32) {
        ggml_mul_mat_set_prec(x, GGML_PREC_F32);
    }
    if (name) {
        ggml_set_name(x, name);
    }
    return x;
}

inline ggml_tensor* graph_apply(ggml_context* ctx,
                                ggml_tensor* x_s,
                                ggml_tensor* A,
                                ggml_tensor* B,
                                ggml_tensor* s,
                                float lin_scale,
                                bool prec_f32,
                                ggml_tensor* out,
                                int gain = 0) {
    ggml_tensor* y = graph_linear(ctx, x_s, A, prec_f32, "nphL_A");  // [Rp, M]
    y              = ggml_mul(ctx, y, s);                  // FP32 per-rank strength*alpha/rank*2^-shift
    ggml_set_name(y, "nphL_s");
    if (lin_scale == 1.0f) {
        gain = 0;  // keep MUL_MAT+ADD fusable (no op between the B matmul and the ADD)
    }
    if (gain != 0) {
        y = ggml_scale(ctx, y, std::ldexp(1.0f, gain));  // small [Rp, M] tensor, exact pow2
        ggml_set_name(y, "nphL_g");
    }
    ggml_tensor* d = graph_linear(ctx, y, B, prec_f32, "nphL_B");    // [N, M]
    if (lin_scale != 1.0f) {
        d = ggml_scale(ctx, d, std::ldexp(1.0f, -gain) / lin_scale);  // exact for power-of-two scales (w2: 128)
        ggml_set_name(d, "nphL_c");
        ggml_tensor* r = ggml_add(ctx, out, d);
        ggml_set_name(r, "nphL_d");
        return r;
    }
    // 0.30.2: ADD(d, out) (commutative, bit-identical). The graph then orders the LoRA chain
    // (A, s, B) before the base matmul group, so ggml-hexagon can fuse B + base + ADD into one
    // HTP_OP_MUL_MAT_LORA, and the ADD output is placed in-place on d (allocated while x_s and y are
    // still alive, so it never aliases the fused op's inputs).
    ggml_tensor* r = ggml_add(ctx, d, out);
    ggml_set_name(r, "nphL_d");
    return r;
}

// 0.30.1 deferred variant for scaled Linears: `out_scaled` = base * lin_scale * 2^gain (not yet
// unscaled). The delta is produced in the same units and the B matmul is the last op before the
// ADD (MUL_MAT+ADD fuses on ggml-hexagon); the caller applies scale(2^-gain / lin_scale) once.
inline ggml_tensor* graph_apply_deferred(ggml_context* ctx,
                                         ggml_tensor* x_s,
                                         ggml_tensor* A,
                                         ggml_tensor* B,
                                         ggml_tensor* s,
                                         bool prec_f32,
                                         ggml_tensor* out_scaled,
                                         int gain) {
    ggml_tensor* y = graph_linear(ctx, x_s, A, prec_f32, "nphL_A");
    y              = ggml_mul(ctx, y, s);
    ggml_set_name(y, "nphL_s");
    if (gain != 0) {
        y = ggml_scale(ctx, y, std::ldexp(1.0f, gain));
        ggml_set_name(y, "nphL_g");
    }
    ggml_tensor* d = graph_linear(ctx, y, B, prec_f32, "nphL_B");
    ggml_tensor* r = ggml_add(ctx, d, out_scaled);  // 0.30.2: operand order, see graph_apply
    ggml_set_name(r, "nphL_d");
    return r;
}

}  // namespace nph_lora
