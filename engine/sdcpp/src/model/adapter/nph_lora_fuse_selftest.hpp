// NPH 0.30.2: on-device self-test of the fused LoRA matmul (HTP_OP_MUL_MAT_LORA, ggml-hexagon).
//
// Runs once per process, before the first LoRA DiT compute on an HTP backend. Synthetic data only
// (no model/LoRA file): it builds the exact graph shapes the Linear + combined-LoRA code emits
// (plain and deferred/scaled variants), computes them on the NPU with the fusion OFF (0.30.1 path)
// and ON, and compares both against a double-precision host reference. If the fused result is
// not within tolerance of the unfused one (or is not finite), the fusion is switched off for the
// rest of the process (fallback = 0.30.1 ops, same precision). Log lines start with "nph-lora-fuse".
// MIT (ZIT-HEXAGON project code).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/util.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "model/adapter/nph_lora_core.h"
#include "nph_stats.h"

namespace nph_lora {

struct FuseSelftestCase {
    int K, N, M, R;
    float lin_scale;  // 1: plain path (graph_apply); else deferred path (graph_apply_deferred)
    int gain;         // deferred gain (2^gain on base and LoRA), pow2 only
};

struct FuseSelftestRun {
    std::vector<float> out;
    uint64_t fused = 0;  // MUL_MAT_LORA ops planned for this graph
    bool ok        = false;
    std::string err;
};

inline float fuse_st_f8_to_f32(uint8_t v) {
    const int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float r;
    if (e == 0) {
        r = std::ldexp((float)m, -9);
    } else {
        r = std::ldexp(1.0f + m / 8.0f, e - 7);
    }
    return s ? -r : r;
}

// One NPU run of the case. Data is identical for both runs (same seed).
inline FuseSelftestRun fuse_selftest_run(ggml_backend_t be, const FuseSelftestCase& c, bool fuse,
                                         const std::vector<uint8_t>& w8, float ws_v,
                                         const std::vector<ggml_fp16_t>& a16, const std::vector<ggml_fp16_t>& b16,
                                         const std::vector<float>& s_v, const std::vector<float>& x_v,
                                         const nph_hex_hooks* hk, ggml_type wtype = GGML_TYPE_F8_E4M3) {
    FuseSelftestRun res;
    ggml_init_params wp = {ggml_tensor_overhead() * 8, nullptr, true};
    ggml_context* cw    = ggml_init(wp);
    ggml_init_params gp = {ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true};
    ggml_context* cg    = ggml_init(gp);
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_gallocr_t galloc      = nullptr;
    do {
        if (!cw || !cg) { res.err = "ggml_init"; break; }
        // creation order = buffer order: B is followed by a pad (the DSP may read up to 128 bytes past B)
        // wtype F16 only for the box test (CPU has no F8 matmul); the F8 values are exact in F16.
        ggml_tensor* W   = ggml_new_tensor_2d(cw, wtype, c.K, c.N);
        ggml_tensor* A   = ggml_new_tensor_2d(cw, GGML_TYPE_F16, c.K, c.R);
        ggml_tensor* B   = ggml_new_tensor_2d(cw, GGML_TYPE_F16, c.R, c.N);
        ggml_tensor* pad = ggml_new_tensor_1d(cw, GGML_TYPE_F32, 256);
        ggml_tensor* S   = ggml_new_tensor_1d(cw, GGML_TYPE_F32, c.R);
        ggml_tensor* WS  = ggml_new_tensor_1d(cw, GGML_TYPE_F32, 1);
        ggml_set_name(W, "nph_fst.weight");
        wbuf = ggml_backend_alloc_ctx_tensors(cw, be);
        if (!wbuf) { res.err = "weight alloc"; break; }
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (wtype == GGML_TYPE_F8_E4M3) {
            ggml_backend_tensor_set(W, w8.data(), 0, w8.size());
        } else {
            std::vector<ggml_fp16_t> w16(w8.size());
            for (size_t i = 0; i < w8.size(); ++i) w16[i] = ggml_fp32_to_fp16(fuse_st_f8_to_f32(w8[i] == 0x80 ? 0 : w8[i]));
            ggml_backend_tensor_set(W, w16.data(), 0, w16.size() * sizeof(ggml_fp16_t));
        }
        ggml_backend_tensor_set(A, a16.data(), 0, a16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(B, b16.data(), 0, b16.size() * sizeof(ggml_fp16_t));
        std::vector<float> zeros(256, 0.0f);
        ggml_backend_tensor_set(pad, zeros.data(), 0, zeros.size() * sizeof(float));
        ggml_backend_tensor_set(S, s_v.data(), 0, s_v.size() * sizeof(float));
        ggml_backend_tensor_set(WS, &ws_v, 0, sizeof(float));

        ggml_tensor* x = ggml_new_tensor_2d(cg, GGML_TYPE_F32, c.K, c.M);
        ggml_set_input(x);
        ggml_tensor* r = nullptr;
        if (c.lin_scale == 1.0f) {
            // Linear::forward, nph_prescaled, scale == 1 (x_s = x): mm + mul(ws) + graph_apply
            ggml_tensor* out = ggml_mul_mat(cg, W, x);
            out              = ggml_mul(cg, out, WS);
            r                = graph_apply(cg, x, A, B, S, 1.0f, false, out);
        } else {
            // deferred: x_s = x*scale ; mm ; scale(2^g) ; mul(ws) ; apply_deferred ; scale(2^-g/scale)
            ggml_tensor* xs  = ggml_scale(cg, x, c.lin_scale);
            ggml_tensor* out = ggml_mul_mat(cg, W, xs);
            if (c.gain != 0) out = ggml_scale(cg, out, std::ldexp(1.0f, c.gain));
            out = ggml_mul(cg, out, WS);
            r   = graph_apply_deferred(cg, xs, A, B, S, false, out, c.gain);
            r   = ggml_scale(cg, r, std::ldexp(1.0f, -c.gain) / c.lin_scale);
        }
        ggml_set_output(r);
        ggml_cgraph* gf = ggml_new_graph(cg);
        ggml_build_forward_expand(gf, r);
        bool supported = true;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i))) { supported = false; break; }
        }
        if (!supported) { res.err = "op not supported by backend"; break; }
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) { res.err = "graph alloc"; break; }
        ggml_backend_tensor_set(x, x_v.data(), 0, x_v.size() * sizeof(float));
        if (hk && hk->set_lora_fuse) hk->set_lora_fuse(fuse ? 1 : 0);
        const uint64_t n0 = (hk && hk->lora_fused_count) ? hk->lora_fused_count() : 0;
        const ggml_status st = ggml_backend_graph_compute(be, gf);
        res.fused = ((hk && hk->lora_fused_count) ? hk->lora_fused_count() : 0) - n0;
        if (st != GGML_STATUS_SUCCESS) { res.err = "compute status " + std::to_string((int)st); break; }
        res.out.resize((size_t)c.N * c.M);
        ggml_backend_tensor_get(r, res.out.data(), 0, res.out.size() * sizeof(float));
        res.ok = true;
    } while (false);
    if (galloc) ggml_gallocr_free(galloc);
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (cg) ggml_free(cg);
    if (cw) ggml_free(cw);
    return res;
}

// rel-l2 over rows m with m % m_step == 0 (all rows: m_step 1); `a` and `ref` are [N, M] row-major by m.
inline double fuse_st_rel_l2(const std::vector<float>& a, const std::vector<double>& ref, bool* finite,
                             size_t N = 0, int m_step = 1) {
    double num = 0, den = 0;
    *finite    = true;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) { *finite = false; return INFINITY; }
        if (m_step > 1 && N > 0 && ((i / N) % (size_t)m_step) != 0) continue;
        const double d = a[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1.0));
}

// Runs the cases on `be` (no backend-name check; used by fuse_selftest_once and the box test).
// Returns true when every case passes. Leaves the fusion toggle at the result.
inline bool fuse_selftest_exec(ggml_backend_t be, const nph_hex_hooks* hk, ggml_type wtype = GGML_TYPE_F8_E4M3) {
    // Real ZIT shapes (K=3840 attention/w1/w3, K=10240 w2 with the deferred 1/128 path); M reduced.
    const FuseSelftestCase cases[] = {
        {3840, 3840, 1024, 96, 1.0f, 0},
        {10240, 3840, 512, 64, 1.0f / 128.0f, 3},
    };
    bool pass      = true;
    uint64_t fused = 0;
    for (const auto& c : cases) {
        std::mt19937 rng(0x5eed1234u + c.K);
        std::uniform_int_distribution<int> ue(3, 9), um(0, 7), us(0, 1);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<uint8_t> w8((size_t)c.K * c.N);
        for (auto& v : w8) v = (uint8_t)((us(rng) << 7) | (ue(rng) << 3) | um(rng));
        const float ws = 0.0123f;
        std::vector<ggml_fp16_t> a16((size_t)c.K * c.R), b16((size_t)c.R * c.N);
        std::vector<float> af(a16.size()), bf(b16.size());
        for (size_t i = 0; i < a16.size(); ++i) { a16[i] = ggml_fp32_to_fp16(0.05f * nd(rng)); af[i] = ggml_fp16_to_fp32(a16[i]); }
        for (size_t i = 0; i < b16.size(); ++i) { b16[i] = ggml_fp32_to_fp16(0.05f * nd(rng)); bf[i] = ggml_fp16_to_fp32(b16[i]); }
        std::vector<float> s_v(c.R);
        for (int i = 0; i < c.R; ++i) s_v[i] = (i % 32 < 28) ? std::ldexp(0.45f, -(i % 5)) : 0.0f;  // padded ranks 0
        std::vector<float> x_v((size_t)c.K * c.M);
        for (auto& v : x_v) v = nd(rng);

        const int m_step = std::max(1, c.M / 16);  // host reference on 16 token rows (keeps it < ~1 s on a phone)
        // double reference: (W·xs·2^g·ws + B·(s·2^g·(A·xs))) · 2^-g / scale, xs = x·scale
        std::vector<double> ref((size_t)c.N * c.M);
        {
            std::vector<double> wd((size_t)c.K * c.N);
            for (size_t i = 0; i < wd.size(); ++i) wd[i] = fuse_st_f8_to_f32(w8[i] == 0x80 ? 0 : w8[i]);
            std::vector<double> xs((size_t)c.K), y(c.R);
            for (int m = 0; m < c.M; m += m_step) {
                for (int k = 0; k < c.K; ++k) xs[k] = (double)(float)(x_v[(size_t)m * c.K + k] * c.lin_scale);
                for (int rr = 0; rr < c.R; ++rr) {
                    double acc = 0;
                    const float* ar = &af[(size_t)rr * c.K];
                    for (int k = 0; k < c.K; ++k) acc += ar[k] * xs[k];
                    y[rr] = acc * s_v[rr];
                }
                for (int n = 0; n < c.N; ++n) {
                    double acc = 0;
                    const double* wr = &wd[(size_t)n * c.K];
                    for (int k = 0; k < c.K; ++k) acc += wr[k] * xs[k];
                    double d = 0;
                    const float* br = &bf[(size_t)n * c.R];
                    for (int rr = 0; rr < c.R; ++rr) d += br[rr] * y[rr];
                    ref[(size_t)m * c.N + n] = (acc * ws + d) / c.lin_scale;
                }
            }
        }
        FuseSelftestRun r0 = fuse_selftest_run(be, c, false, w8, ws, a16, b16, s_v, x_v, hk, wtype);
        FuseSelftestRun r1 = fuse_selftest_run(be, c, true, w8, ws, a16, b16, s_v, x_v, hk, wtype);
        char buf[320];
        if (!r0.ok || !r1.ok) {
            snprintf(buf, sizeof(buf), "K%d: error (0.30.1 run: %s, fused run: %s)", c.K, r0.ok ? "ok" : r0.err.c_str(), r1.ok ? "ok" : r1.err.c_str());
            LOG_WARN("nph-lora-fuse: selftest case %s", buf);
            pass = false;
            continue;
        }
        bool f0, f1;
        const double e0 = fuse_st_rel_l2(r0.out, ref, &f0, (size_t)c.N, m_step);
        const double e1 = fuse_st_rel_l2(r1.out, ref, &f1, (size_t)c.N, m_step);
        std::vector<double> r0d(r0.out.begin(), r0.out.end());
        bool f01;
        const double e01 = fuse_st_rel_l2(r1.out, r0d, &f01);
        size_t nbit = 0;
        for (size_t i = 0; i < r0.out.size(); ++i) nbit += memcmp(&r0.out[i], &r1.out[i], 4) != 0;
        // 0.30.1 vs ref: FP16 conversion of activations on HMX (~1e-3). Fused vs 0.30.1: same products,
        // only the FP32 epilogue order differs (a few ulp) -> 1e-4
        // (a wrong LoRA term would be ~1e-1: the synthetic delta is about as large as the base output).
        const bool ok_case = f0 && f1 && f01 && e0 < 2e-2 && e01 < 1e-4 && e1 < 2e-2;
        snprintf(buf, sizeof(buf), "K%d/M%d/R%d%s: fused_ops=%llu rel(fused,0.30.1)=%.2e diff_elems=%zu rel(0.30.1,ref)=%.2e rel(fused,ref)=%.2e %s",
                 c.K, c.M, c.R, c.lin_scale != 1.0f ? "/deferred" : "", (unsigned long long)r1.fused, e01, nbit, e0, e1,
                 ok_case ? (r1.fused ? "OK" : "OK(no-fuse)") : "FAIL");
        LOG_INFO("nph-lora-fuse: selftest case %s", buf);
        if (!ok_case) pass = false;
        fused += r1.fused;
    }
    if (pass && fused == 0 && hk && hk->lora_fused_count) {
        // nothing was fused, so nothing was validated: keep the 0.30.1 ops
        LOG_WARN("nph-lora-fuse: selftest planned no fused op (shape/VTCM) -> not validated, fusion off");
        pass = false;
    }
    if (hk && hk->set_lora_fuse) hk->set_lora_fuse(pass ? 1 : 0);
    return pass;
}

// Returns true when the fusion stays enabled. Runs once per process (later calls return the cached result).
inline bool fuse_selftest_once(ggml_backend_t be) {
    // Called before every combined-LoRA DiT use. NPH_LORA_FUSE is re-read each time (the app sets it
    // before each generation), so the Ajustes switch works without restarting; the self-test runs once.
    static int cached   = -1;  // self-test verdict (-1: not run yet)
    static int last_off = -1;
    const nph_hex_hooks* hk = nph_get_hex_hooks();
    const char* bn          = be ? ggml_backend_name(be) : nullptr;
    if (!be || !bn || strncmp(bn, "HTP", 3) != 0 || !hk || !hk->set_lora_fuse || !hk->get_lora_fuse) {
        return false;  // not an HTP backend: nothing to test (not cached: backend may change)
    }
    if (!nph_env_flag("NPH_LORA_FUSE", false)) {
        hk->set_lora_fuse(0);
        if (last_off != 1) LOG_INFO("nph-lora-fuse: off (NPH_LORA_FUSE=0); 0.30.1 ops (B matmul+ADD, SCALE)");
        last_off = 1;
        return false;
    }
    last_off = 0;
    if (cached >= 0) {
        hk->set_lora_fuse(cached);
        return cached != 0;
    }
    if (!nph_env_flag("NPH_LORA_FUSE_SELFTEST", true)) {
        LOG_INFO("nph-lora-fuse: on (self-test skipped by NPH_LORA_FUSE_SELFTEST=0)");
        hk->set_lora_fuse(1);
        cached = 1;
        return true;
    }
    const int64_t t0 = ggml_time_us();
    const bool pass  = fuse_selftest_exec(be, hk);
    cached = pass ? 1 : 0;
    const double ms = (ggml_time_us() - t0) / 1000.0;
    if (pass) {
        LOG_INFO("nph-lora-fuse: selftest PASS (%.0f ms) -> fused HMX LoRA on", ms);
    } else {
        LOG_WARN("nph-lora-fuse: selftest FAIL (%.0f ms) -> fallback 0.30.1 ops (same precision)", ms);
    }
    return pass;
}

}  // namespace nph_lora
