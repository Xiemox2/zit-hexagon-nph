// NPH 0.29 NPG self-test (see npg_selftest.h). Weights are FP16 (as in the VAE), activations FP32;
// the reference uses the same FP16-rounded weights and accumulates in double, so the error printed
// is the backend's own activation/accumulation error (never "hidden" by a lower-precision reference).
#include "npg_selftest.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"

namespace {

struct In {
    ggml_type type;
    int64_t ne[4];
    std::vector<float> data;  // already rounded to f16 when type == F16
};

static float f16r(float v) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); }

static In mk(ggml_type t, int64_t a, int64_t b, int64_t c, int64_t d, std::mt19937& rng, float scale) {
    In in{t, {a, b, c, d}, {}};
    std::normal_distribution<float> nd(0.f, scale);
    in.data.resize((size_t)(a * b * c * d));
    for (auto& v : in.data) v = t == GGML_TYPE_F16 ? f16r(nd(rng)) : nd(rng);
    return in;
}

// 0 = ok, 1 = unsupported, 2 = error
static int run(ggml_backend_t be, std::vector<In>& ins,
               const std::function<ggml_tensor*(ggml_context*, std::vector<ggml_tensor*>&)>& build,
               std::vector<float>& out, std::string& why) {
    ggml_init_params ip{ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) { why = "ggml_init"; return 2; }
    std::vector<ggml_tensor*> ts;
    for (auto& in : ins) ts.push_back(ggml_new_tensor_4d(ctx, in.type, in.ne[0], in.ne[1], in.ne[2], in.ne[3]));
    ggml_tensor* o = build(ctx, ts);
    if (!ggml_backend_supports_op(be, o)) { ggml_free(ctx); return 1; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); why = "sin memoria de backend"; return 2; }
    for (size_t i = 0; i < ins.size(); ++i) {
        if (ins[i].type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h(ins[i].data.size());
            ggml_fp32_to_fp16_row(ins[i].data.data(), h.data(), (int64_t)h.size());
            ggml_backend_tensor_set(ts[i], h.data(), 0, h.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(ts[i], ins[i].data.data(), 0, ins[i].data.size() * sizeof(float));
        }
    }
    ggml_cgraph* g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, o);
    int rc = 0;
    if (ggml_backend_graph_compute(be, g) != GGML_STATUS_SUCCESS) {
        why = "graph_compute falló";
        rc = 2;
    } else {
        out.resize((size_t)ggml_nelements(o));
        ggml_backend_tensor_get(o, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return rc;
}

struct Cmp {
    double max_abs = 0, ref_max = 0;
    size_t nonfinite = 0;
};

static Cmp compare(const std::vector<float>& got, const std::vector<double>& ref) {
    Cmp c;
    for (size_t i = 0; i < ref.size(); ++i) {
        c.ref_max = std::max(c.ref_max, std::fabs(ref[i]));
        if (i >= got.size() || !std::isfinite(got[i])) { c.nonfinite++; continue; }
        c.max_abs = std::max(c.max_abs, std::fabs((double)got[i] - ref[i]));
    }
    return c;
}

}  // namespace

NpgSelftestResult npg_selftest(ggml_backend_t be) {
    NpgSelftestResult R;
    auto t0 = std::chrono::steady_clock::now();
    std::mt19937 rng(1234);
    bool all_ok = true;
    char line[256];

    auto report = [&](const char* name, int rc, const std::string& why, const Cmp& c, double tol_rel) {
        const bool is_mm = std::strncmp(name, "mul_mat", 7) == 0 || std::strncmp(name, "flash_attn", 10) == 0;
        const int bad_before = R.bad_mm + R.bad_other;
        auto mark_bad = [&] { if (is_mm) R.bad_mm++; else R.bad_other++; };
        if (rc == 1) {
            R.unsupported++;
            snprintf(line, sizeof line, "  %-14s no soportado en GPU → iría a CPU (FP32)\n", name);
        } else if (rc == 2) {
            all_ok = false;
            mark_bad();
            snprintf(line, sizeof line, "  %-14s ERROR: %s\n", name, why.c_str());
        } else {
            R.supported++;
            const double rel = c.max_abs / std::max(c.ref_max, 1e-12);
            const bool ok    = c.nonfinite == 0 && rel <= tol_rel;
            if (!ok) { all_ok = false; mark_bad(); }
            // Precision class of the result: FP32-level (rel < 3e-5) or FP16-activation level (rel < 3e-3,
            // what ggml-cpu and HTP themselves do for F16-weight convs/matmuls).
            const char* cls = rel < 3e-5 ? "nivel FP32" : rel < 3e-3 ? "nivel FP16" : "bajo FP16";
            snprintf(line, sizeof line, "  %-14s %s err.máx %.2e (rel %.2e, %s, tol %.0e)%s\n", name, ok ? "OK " : "MAL",
                     c.max_abs, rel, cls, tol_rel, c.nonfinite ? " ¡no finitos!" : "");
        }
        (void)bad_before;
        R.text += line;
    };

    const int W = 16, H = 16, C = 64, OC = 64;
    // 1) conv2d 3x3 direct, F16 weights, F32 input/output
    {
        std::vector<In> ins = {mk(GGML_TYPE_F16, 3, 3, C, OC, rng, 0.05f), mk(GGML_TYPE_F32, W, H, C, 1, rng, 1.f)};
        std::vector<float> got;
        std::string why;
        int rc = run(be, ins, [](ggml_context* c, std::vector<ggml_tensor*>& t) {
            return ggml_conv_2d_direct(c, t[0], t[1], 1, 1, 1, 1, 1, 1);
        }, got, why);
        std::vector<double> ref((size_t)W * H * OC, 0.0);
        const auto& w = ins[0].data;
        const auto& x = ins[1].data;
        for (int oc = 0; oc < OC; ++oc)
            for (int oy = 0; oy < H; ++oy)
                for (int ox = 0; ox < W; ++ox) {
                    double s = 0;
                    for (int ic = 0; ic < C; ++ic)
                        for (int ky = 0; ky < 3; ++ky)
                            for (int kx = 0; kx < 3; ++kx) {
                                int ix = ox + kx - 1, iy = oy + ky - 1;
                                if (ix < 0 || iy < 0 || ix >= W || iy >= H) continue;
                                s += (double)w[((size_t)(oc * C + ic) * 3 + ky) * 3 + kx] * x[((size_t)ic * H + iy) * W + ix];
                            }
                    ref[((size_t)oc * H + oy) * W + ox] = s;
                }
        report("conv2d 3x3", rc, why, rc == 0 ? compare(got, ref) : Cmp{}, 2e-3);
    }
    // 2) add with channel-broadcast bias
    {
        std::vector<In> ins = {mk(GGML_TYPE_F32, W, H, C, 1, rng, 1.f), mk(GGML_TYPE_F32, 1, 1, C, 1, rng, 1.f)};
        std::vector<float> got;
        std::string why;
        int rc = run(be, ins, [](ggml_context* c, std::vector<ggml_tensor*>& t) { return ggml_add(c, t[0], t[1]); }, got, why);
        std::vector<double> ref(ins[0].data.size());
        for (size_t i = 0; i < ref.size(); ++i) ref[i] = (double)ins[0].data[i] + ins[1].data[i / (W * H)];
        report("add (bias)", rc, why, rc == 0 ? compare(got, ref) : Cmp{}, 1e-6);
    }
    // 3) group_norm (32 groups) and 4) silu
    {
        std::vector<In> ins = {mk(GGML_TYPE_F32, W, H, C, 1, rng, 2.f)};
        for (auto& v : ins[0].data) v += 0.5f;
        std::vector<float> got;
        std::string why;
        int rc = run(be, ins, [](ggml_context* c, std::vector<ggml_tensor*>& t) {
            return ggml_group_norm(c, t[0], 32, 1e-6f);
        }, got, why);
        const int G = 32, cpg = C / G;
        const size_t gsz = (size_t)W * H * cpg;
        std::vector<double> ref(ins[0].data.size());
        for (int gi = 0; gi < G; ++gi) {
            double m = 0, v = 0;
            for (size_t i = 0; i < gsz; ++i) m += ins[0].data[gi * gsz + i];
            m /= gsz;
            for (size_t i = 0; i < gsz; ++i) { double d = ins[0].data[gi * gsz + i] - m; v += d * d; }
            v /= gsz;
            for (size_t i = 0; i < gsz; ++i) ref[gi * gsz + i] = (ins[0].data[gi * gsz + i] - m) / std::sqrt(v + 1e-6);
        }
        report("group_norm", rc, why, rc == 0 ? compare(got, ref) : Cmp{}, 1e-4);

        got.clear();
        rc = run(be, ins, [](ggml_context* c, std::vector<ggml_tensor*>& t) { return ggml_silu(c, t[0]); }, got, why);
        for (size_t i = 0; i < ref.size(); ++i) { double x = ins[0].data[i]; ref[i] = x / (1.0 + std::exp(-x)); }
        report("silu", rc, why, rc == 0 ? compare(got, ref) : Cmp{}, 1e-5);
    }
    // 5) nearest upscale x2
    {
        std::vector<In> ins = {mk(GGML_TYPE_F32, W, H, C, 1, rng, 1.f)};
        std::vector<float> got;
        std::string why;
        int rc = run(be, ins, [](ggml_context* c, std::vector<ggml_tensor*>& t) {
            return ggml_upscale(c, t[0], 2, GGML_SCALE_MODE_NEAREST);
        }, got, why);
        std::vector<double> ref((size_t)W * H * C * 4);
        for (int ch = 0; ch < C; ++ch)
            for (int y = 0; y < 2 * H; ++y)
                for (int x = 0; x < 2 * W; ++x)
                    ref[((size_t)ch * 2 * H + y) * 2 * W + x] = ins[0].data[((size_t)ch * H + y / 2) * W + x / 2];
        report("upscale x2", rc, why, rc == 0 ? compare(got, ref) : Cmp{}, 1e-7);
    }
    // 6) mul_mat F16 weight x F32 activations (1x1 conv / attention projections), K=512
    const int K = 512, M = 128, N = 64;
    {
        std::vector<In> ins = {mk(GGML_TYPE_F16, K, M, 1, 1, rng, 0.05f), mk(GGML_TYPE_F32, K, N, 1, 1, rng, 1.f)};
        std::vector<float> got;
        std::string why;
        int rc = run(be, ins, [](ggml_context* c, std::vector<ggml_tensor*>& t) { return ggml_mul_mat(c, t[0], t[1]); }, got, why);
        std::vector<double> ref((size_t)M * N);
        for (int n = 0; n < N; ++n)
            for (int m = 0; m < M; ++m) {
                double s = 0;
                for (int k = 0; k < K; ++k) s += (double)ins[0].data[(size_t)m * K + k] * ins[1].data[(size_t)n * K + k];
                ref[(size_t)n * M + m] = s;
            }
        report("mul_mat f16", rc, why, rc == 0 ? compare(got, ref) : Cmp{}, 2e-3);
    }
    // 6b) 0.30: VAE-attention-sized mul_mat (K=512, M=512, N=256) with larger activations. The Adreno
    // "xmem" F16xF32 GEMM (ggml-opencl default on Adreno) converts activations to half and accumulates in
    // half; this shape is eligible for it, so it shows the FP16-accumulation error / overflow directly.
    {
        const int K2 = 512, M2 = 512, N2 = 256;
        std::vector<In> ins = {mk(GGML_TYPE_F16, K2, M2, 1, 1, rng, 0.05f), mk(GGML_TYPE_F32, K2, N2, 1, 1, rng, 8.f)};
        std::vector<float> got;
        std::string why;
        int rc = run(be, ins, [](ggml_context* c, std::vector<ggml_tensor*>& t) { return ggml_mul_mat(c, t[0], t[1]); }, got, why);
        std::vector<double> ref((size_t)M2 * N2);
        for (int n = 0; n < N2; ++n)
            for (int m = 0; m < M2; ++m) {
                double s = 0;
                for (int k = 0; k < K2; ++k) s += (double)ins[0].data[(size_t)m * K2 + k] * ins[1].data[(size_t)n * K2 + k];
                ref[(size_t)n * M2 + m] = s;
            }
        report("mul_mat 512x256", rc, why, rc == 0 ? compare(got, ref) : Cmp{}, 2e-3);
    }
    // 7) flash attention, single head d=512 (VAE mid-block), F16 K/V, F32 Q, FP32 precision requested
    {
        const int D = 512, NQ = 64, NK = 64;
        std::vector<In> ins = {mk(GGML_TYPE_F32, D, NQ, 1, 1, rng, 1.f), mk(GGML_TYPE_F16, D, NK, 1, 1, rng, 1.f),
                               mk(GGML_TYPE_F16, D, NK, 1, 1, rng, 1.f)};
        const float scale = 1.0f / std::sqrt((float)D);
        std::vector<float> got;
        std::string why;
        int rc = run(be, ins, [scale](ggml_context* c, std::vector<ggml_tensor*>& t) {
            ggml_tensor* o = ggml_flash_attn_ext(c, t[0], t[1], t[2], nullptr, scale, 0.f, 0.f);
            ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
            return o;
        }, got, why);
        std::vector<double> ref((size_t)D * NQ);
        std::vector<double> p(NK);
        for (int i = 0; i < NQ; ++i) {
            double mx = -1e300;
            for (int j = 0; j < NK; ++j) {
                double s = 0;
                for (int d = 0; d < D; ++d) s += (double)ins[0].data[(size_t)i * D + d] * ins[1].data[(size_t)j * D + d];
                p[j] = s * scale;
                mx   = std::max(mx, p[j]);
            }
            double z = 0;
            for (int j = 0; j < NK; ++j) { p[j] = std::exp(p[j] - mx); z += p[j]; }
            for (int d = 0; d < D; ++d) {
                double s = 0;
                for (int j = 0; j < NK; ++j) s += p[j] * ins[2].data[(size_t)j * D + d];
                ref[(size_t)i * D + d] = s / z;
            }
        }
        // FA kernels may hold Q·K in FP16 internally on some backends; tolerance reflects that and the
        // printed error makes it visible. The HTP path uses the same FP16 K/V.
        report("flash_attn 512", rc, why, rc == 0 ? compare(got, ref) : Cmp{}, 2e-2);
    }

    R.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    R.ok = all_ok && R.supported > 0;
    return R;
}
