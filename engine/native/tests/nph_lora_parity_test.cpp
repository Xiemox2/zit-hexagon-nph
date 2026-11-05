// NPH 0.30.1 exact-shape LoRA parity test (synthetic tensors only, no model files).
// One Z-Image layer at real dims: fused qkv 3840->3*3840 (LoRA keys T, T.1, T.2), out 3840->3840,
// w1/w3 3840->10240, w2 10240->3840 (Linear scale 1/128). Two rank-32 LoRAs: f16 x0.9, bf16 x0.7,
// separately and together. Checks:
//  (1) ggml CPU backend: combined graph (nph_lora::graph_apply) vs the old per-LoRA graph
//      (ext_linear(down) -> ext_linear(up) -> scale -> concat parts -> add), final layer output.
//  (2) HMX numeric emulation (FP16 operands/outputs, FP32 accumulate, optional flush-to-zero of
//      FP16 subnormals as on HMX/HVX qf->hf) of both paths vs an FP64 reference.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "model/adapter/nph_lora_core.h"

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static float bf16_round(float v) { uint32_t u; memcpy(&u, &v, 4); u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u; float r; memcpy(&r, &u, 4); return r; }
static float f16_round(float v) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); }
static bool g_ftz = true;
static int g_ftz_mask = 7;  // 1 = weights, 2 = activations, 4 = outputs
static inline float hm(float v, int what) {  // value as HMX sees an FP16 operand / writes an FP16 output
    float r = f16_round(v);
    if (g_ftz && (g_ftz_mask & what) && std::fabs(r) < 6.103515625e-05f) r = 0.f;
    return r;
}

// ---- synthetic LoRA "files": name -> data, like LoraModel::lora_tensors after name conversion ----
struct T2 { int ne0 = 0, ne1 = 0; std::vector<float> v; };
struct LoraFile { std::string name; int dtype; float mult; std::map<std::string, T2> t; std::map<std::string, float> scalars; };

struct Target { std::string name; int K, N, parts; float lin_scale; };

// Real-ish magnitude spread: normal * lognormal (so a few values are tiny or large).
static float sample(std::mt19937& rng, float sd) {
    std::normal_distribution<float> nd(0.f, 1.f);
    return nd(rng) * sd * std::exp(0.9f * nd(rng));
}

static LoraFile make_lora(std::mt19937& rng, const std::string& fname, int dtype, float mult, float alpha_over_r,
                          const std::vector<Target>& tg, int r) {
    LoraFile L; L.name = fname; L.dtype = dtype; L.mult = mult;
    auto q = [&](float v) { return dtype == 0 ? bf16_round(v) : f16_round(v); };
    for (auto& t : tg) {
        const int n = t.N / t.parts;
        for (int p = 0; p < t.parts; p++) {
            std::string k = p == 0 ? t.name : t.name + "." + std::to_string(p);
            T2 d, u; d.ne0 = t.K; d.ne1 = r; u.ne0 = r; u.ne1 = n;
            d.v.resize((size_t)t.K * r); u.v.resize((size_t)r * n);
            for (auto& v : d.v) v = q(sample(rng, 0.015f));
            for (auto& v : u.v) v = q(sample(rng, 0.002f));
            L.t["lora." + k + ".lora_down"] = std::move(d);
            L.t["lora." + k + ".lora_up"]   = std::move(u);
            L.scalars["lora." + k + ".alpha"] = alpha_over_r * r;
        }
    }
    return L;
}

// Same part walk as CombinedLoraAdapter::build (T, T.1, T.2 ...; row offsets restart per file;
// scale = (.scale | alpha/rank) * multiplier).
static std::vector<nph_lora::PartSrc> plan(const std::vector<LoraFile*>& files, const Target& tg) {
    std::vector<nph_lora::PartSrc> srcs;
    for (auto* f : files) {
        int off = 0;
        for (int i = 0;; i++) {
            std::string k = i == 0 ? tg.name : tg.name + "." + std::to_string(i);
            auto d = f->t.find("lora." + k + ".lora_down"), u = f->t.find("lora." + k + ".lora_up");
            if (d == f->t.end() || u == f->t.end()) break;
            nph_lora::PartSrc ps; ps.K = tg.K; ps.r = d->second.ne1; ps.n = u->second.ne1; ps.row_off = off; off += ps.n;
            ps.down = d->second.v.data(); ps.up = u->second.v.data();
            float sv = 1.f;
            auto sc = f->scalars.find("lora." + k + ".scale");
            if (sc != f->scalars.end()) sv = sc->second;
            else { auto al = f->scalars.find("lora." + k + ".alpha"); if (al != f->scalars.end()) sv = al->second / (float)ps.r; }
            ps.scale = sv * f->mult;
            srcs.push_back(ps);
        }
        CHECK(off == tg.N, "%s %s rows %d != %d", f->name.c_str(), tg.name.c_str(), off, tg.N);
    }
    return srcs;
}

// ---------------- HMX emulation ----------------
// out[m][n] = h( sum_k h(w[n][k]) * h(x[m][k]) )   (FP32 accumulate)
static void emu_mm(const float* w, int N, int K, const float* x, int M, std::vector<float>& out) {
    std::vector<float> wh((size_t)N * K), xh((size_t)M * K);
    for (size_t i = 0; i < wh.size(); i++) wh[i] = hm(w[i], 1);
    for (size_t i = 0; i < xh.size(); i++) xh[i] = hm(x[i], 2);
    out.assign((size_t)M * N, 0.f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            const float* a = &wh[(size_t)n * K]; const float* b = &xh[(size_t)m * K];
            for (int k = 0; k < K; k++) acc += a[k] * b[k];
            out[(size_t)m * N + n] = hm(acc, 4);
        }
}
// delta via the per-LoRA path (0.29): per part ext_linear(down, ls) -> ext_linear(up, ls) -> *scale
static void emu_old(const std::vector<nph_lora::PartSrc>& srcs, const float* x, int M, int K, int N, float ls, std::vector<float>& d) {
    d.assign((size_t)M * N, 0.f);
    std::vector<float> xs((size_t)M * K); for (size_t i = 0; i < xs.size(); i++) xs[i] = x[i] * ls;
    for (auto& ps : srcs) {
        std::vector<float> y, z;
        emu_mm(ps.down, ps.r, K, xs.data(), M, y);              // [M][r]
        for (auto& v : y) v = v * (1.f / ls) * ls;              // scale(1/ls) then the up linear's x*ls
        std::vector<float> up((size_t)ps.n * ps.r);             // ggml [r, n] -> rows n
        for (int i = 0; i < ps.n; i++) for (int j = 0; j < ps.r; j++) up[(size_t)i * ps.r + j] = ps.up[(size_t)i * ps.r + j];
        emu_mm(up.data(), ps.n, ps.r, y.data(), M, z);
        for (int m = 0; m < M; m++) for (int i = 0; i < ps.n; i++)
            d[(size_t)m * N + ps.row_off + i] += z[(size_t)m * ps.n + i] * (1.f / ls) * ps.scale;
    }
}
static void emu_new(const nph_lora::Combined& c, const float* x, int M, float ls, std::vector<float>& d) {
    const int K = c.K, N = c.N, Rp = c.Rp;
    std::vector<float> xs((size_t)M * K); for (size_t i = 0; i < xs.size(); i++) xs[i] = x[i] * ls;
    std::vector<float> A((size_t)Rp * K), B((size_t)N * Rp);
    for (size_t i = 0; i < A.size(); i++) A[i] = ggml_fp16_to_fp32(c.A[i]);
    for (size_t i = 0; i < B.size(); i++) B[i] = ggml_fp16_to_fp32(c.B[i]);
    std::vector<float> y, z;
    emu_mm(A.data(), Rp, K, xs.data(), M, y);
    const int gain = (ls != 1.f && !getenv("NOGAIN")) ? c.gain : 0;
    for (int m = 0; m < M; m++) for (int t = 0; t < Rp; t++) y[(size_t)m * Rp + t] = y[(size_t)m * Rp + t] * c.s[t] * std::ldexp(1.f, gain);
    emu_mm(B.data(), N, Rp, y.data(), M, z);
    d.resize(z.size());
    for (size_t i = 0; i < z.size(); i++) d[i] = z[i] * (std::ldexp(1.f, -gain) / ls);
}

struct Err { double rel_l2 = 0, rel_max = 0; };
static Err err_vs(const std::vector<float>& v, const std::vector<double>& ref) {
    double e2 = 0, r2 = 0, em = 0, rm = 0;
    for (size_t i = 0; i < v.size(); i++) { double e = v[i] - ref[i]; e2 += e * e; r2 += ref[i] * ref[i]; em = std::max(em, std::fabs(e)); rm = std::max(rm, std::fabs(ref[i])); }
    Err E; E.rel_l2 = r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2); E.rel_max = rm > 0 ? em / rm : em; return E;
}

// ---------------- ggml CPU graph check ----------------
static void cpu_graph_check(const char* label, const Target& tg, const std::vector<nph_lora::PartSrc>& srcs, const nph_lora::Combined& c,
                            const std::vector<float>& x, int M, std::mt19937& rng) {
    const int K = tg.K, N = tg.N; const float ls = tg.lin_scale;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w((size_t)N * K); for (auto& v : w) v = f16_round(nd(rng) * 0.02f);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_init_params ip = {(srcs.size() * 2 + 16) * ggml_tensor_overhead(), nullptr, true};
    ggml_context* wctx = ggml_init(ip);
    ggml_tensor* tx = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, M);
    ggml_tensor* tw = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, K, N);
    ggml_tensor* tA = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, K, c.Rp);
    ggml_tensor* tB = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, c.Rp, N);
    ggml_tensor* ts = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, c.Rp);
    std::vector<ggml_tensor*> td, tu;
    for (auto& ps : srcs) { td.push_back(ggml_new_tensor_2d(wctx, GGML_TYPE_F16, K, ps.r)); tu.push_back(ggml_new_tensor_2d(wctx, GGML_TYPE_F16, ps.r, ps.n)); }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(wctx, be);
    auto set16 = [](ggml_tensor* t, const float* v) { std::vector<ggml_fp16_t> hh((size_t)ggml_nelements(t)); for (size_t i = 0; i < hh.size(); i++) hh[i] = ggml_fp32_to_fp16(v[i]); ggml_backend_tensor_set(t, hh.data(), 0, ggml_nbytes(t)); };
    ggml_backend_tensor_set(tx, x.data(), 0, ggml_nbytes(tx));
    set16(tw, w.data());
    ggml_backend_tensor_set(tA, c.A.data(), 0, ggml_nbytes(tA));
    ggml_backend_tensor_set(tB, c.B.data(), 0, ggml_nbytes(tB));
    ggml_backend_tensor_set(ts, c.s.data(), 0, ggml_nbytes(ts));
    for (size_t i = 0; i < srcs.size(); i++) { set16(td[i], srcs[i].down); set16(tu[i], srcs[i].up); }
    ggml_init_params gp = {4096 * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true};
    ggml_context* g = ggml_init(gp);
    ggml_tensor* x_s = ls != 1.f ? ggml_scale(g, tx, ls) : tx;
    ggml_tensor* base = ggml_mul_mat(g, tw, x_s);
    if (ls != 1.f) base = ggml_scale(g, base, 1.f / ls);
    ggml_tensor* out_new = nph_lora::graph_apply(g, x_s, tA, tB, ts, ls, false, base, c.gain);
    ggml_tensor* out_def = nullptr;  // 0.30.1 deferred-unscale graph (scaled Linears only)
    if (ls != 1.f) {
        ggml_tensor* bs = ggml_mul_mat(g, tw, x_s);
        if (c.gain) bs = ggml_scale(g, bs, std::ldexp(1.f, c.gain));
        out_def = nph_lora::graph_apply_deferred(g, x_s, tA, tB, ts, false, bs, c.gain);
        out_def = ggml_scale(g, out_def, std::ldexp(1.f, -c.gain) / ls);
    }
    ggml_tensor* out_old = base;
    // per file: concat its parts, add (LoraModel::get_out_diff + add_lora_to_output)
    size_t i = 0;
    while (i < srcs.size()) {
        ggml_tensor* diff = nullptr; int rows = 0;
        for (; i < srcs.size() && (rows < N); i++) {
            auto& ps = srcs[i];
            ggml_tensor* xi = ls != 1.f ? ggml_scale(g, tx, ls) : tx;
            ggml_tensor* lx = ggml_mul_mat(g, td[i], xi);
            if (ls != 1.f) lx = ggml_scale(g, lx, 1.f / ls);
            ggml_tensor* li = ls != 1.f ? ggml_scale(g, lx, ls) : lx;
            lx = ggml_mul_mat(g, tu[i], li);
            if (ls != 1.f) lx = ggml_scale(g, lx, 1.f / ls);
            lx = ggml_scale(g, lx, ps.scale);
            diff = diff ? ggml_concat(g, diff, lx, 0) : lx;
            rows += ps.n;
        }
        out_old = ggml_add(g, out_old, diff);
    }
    ggml_cgraph* gf = ggml_new_graph_custom(g, 4096, false);
    ggml_build_forward_expand(gf, out_new);
    ggml_build_forward_expand(gf, out_old);
    if (out_def) ggml_build_forward_expand(gf, out_def);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    CHECK(ggml_gallocr_alloc_graph(ga, gf), "alloc");
    ggml_backend_cpu_set_n_threads(be, 8);
    CHECK(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS, "compute");
    std::vector<float> on((size_t)M * N), oo((size_t)M * N);
    ggml_backend_tensor_get(out_new, on.data(), 0, ggml_nbytes(out_new));
    ggml_backend_tensor_get(out_old, oo.data(), 0, ggml_nbytes(out_old));
    if (out_def) {
        std::vector<float> od((size_t)M * N);
        ggml_backend_tensor_get(out_def, od.data(), 0, ggml_nbytes(out_def));
        size_t ndiff = 0;
        for (size_t k = 0; k < od.size(); k++) ndiff += memcmp(&od[k], &on[k], 4) != 0;
        printf("  CPU  %-34s deferred-unscale graph vs combined: %zu / %zu values differ (bit-exact expected)\n", label, ndiff, od.size());
        CHECK(ndiff == 0, "%s deferred graph not bit-identical (%zu values)", label, ndiff);
    }
    std::vector<double> dref; nph_lora::reference_delta(srcs, x.data(), M, K, N, dref);
    std::vector<double> ref((size_t)M * N), dnew((size_t)M * N), dold((size_t)M * N);
    std::vector<float> dn((size_t)M * N), dol((size_t)M * N);
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double b = 0; for (int k = 0; k < K; k++) b += (double)w[(size_t)n * K + k] * x[(size_t)m * K + k];
        ref[(size_t)m * N + n] = b + dref[(size_t)m * N + n];
        dn[(size_t)m * N + n]  = (float)(on[(size_t)m * N + n] - b);
        dol[(size_t)m * N + n] = (float)(oo[(size_t)m * N + n] - b);
    }
    Err en = err_vs(on, ref), eo = err_vs(oo, ref), dne = err_vs(dn, dref), doe = err_vs(dol, dref);
    double diff = 0, mref = 0;
    for (size_t k = 0; k < on.size(); k++) { diff = std::max(diff, (double)std::fabs(on[k] - oo[k])); mref = std::max(mref, std::fabs(ref[k])); }
    printf("  CPU  %-34s out rel-l2 new=%.3g old=%.3g | delta rel-l2 new=%.3g old=%.3g | max|new-old|/max|out|=%.3g\n",
           label, en.rel_l2, eo.rel_l2, dne.rel_l2, doe.rel_l2, diff / mref);
    CHECK(en.rel_l2 <= eo.rel_l2 * 1.25 + 1e-7, "%s CPU combined less accurate than per-LoRA (%g vs %g)", label, en.rel_l2, eo.rel_l2);
    CHECK(dne.rel_l2 <= doe.rel_l2 * 1.25 + 1e-6, "%s CPU combined delta less accurate (%g vs %g)", label, dne.rel_l2, doe.rel_l2);
    ggml_gallocr_free(ga); ggml_free(g); ggml_backend_buffer_free(buf); ggml_free(wctx); ggml_backend_free(be);
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 48;
    const char* only = getenv("ONLY");
    if (const char* fm = getenv("FTZMASK")) g_ftz_mask = atoi(fm);
    const bool nocpu = getenv("NOCPU") != nullptr;
    std::mt19937 rng(20261030);
    const std::string L = "diffusion_model.layers.7.";
    const std::vector<Target> tg = {
        {L + "attention.qkv.weight", 3840, 3 * 3840, 3, 1.f},
        {L + "attention.out.weight", 3840, 3840, 1, 1.f},
        {L + "feed_forward.w1.weight", 3840, 10240, 1, 1.f},
        {L + "feed_forward.w3.weight", 3840, 10240, 1, 1.f},
        {L + "feed_forward.w2.weight", 10240, 3840, 1, 1.f / 128.f},
    };
    // alpha/rank: f16 file alpha=rank (1.0), bf16 file alpha=16 for rank 32 (0.5)
    LoraFile f16 = make_lora(rng, "style_f16.safetensors", 1, 0.9f, 1.0f, tg, 32);
    LoraFile b16 = make_lora(rng, "skin_bf16.safetensors", 0, 0.7f, 0.5f, tg, 32);
    struct Combo { const char* name; std::vector<LoraFile*> files; };
    std::vector<Combo> combos = {{"f16 x0.9", {&f16}}, {"bf16 x0.7", {&b16}}, {"f16 x0.9 + bf16 x0.7", {&f16, &b16}}};
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& t : tg) {
        if (only && t.name.find(only) == std::string::npos) continue;
        // input like a modulated RMS-normed activation (w2 input: SwiGLU output, larger spread)
        std::vector<float> x((size_t)M * t.K);
        for (auto& v : x) v = nd(rng) * (t.lin_scale != 1.f ? 4.f : 1.f);
        for (int o = 0; o < 8; o++) x[rng() % x.size()] *= 40.f;  // outlier channels
        for (auto& cb : combos) {
            auto srcs = plan(cb.files, t);
            nph_lora::Combined c; std::string err;
            CHECK(nph_lora::combine(srcs, t.K, t.N, 3, c, &err), "combine %s", err.c_str());
            char label[160]; snprintf(label, sizeof label, "%s [%s]", t.name.c_str() + L.size(), cb.name);
            printf("%s: R=%d Rp=%d shifted=%d split=%d inexact=%lld\n", label, c.R, c.Rp, c.st.shifted_vectors, c.st.split_vectors, (long long)c.st.inexact_values);
            CHECK(c.st.inexact_values == 0, "%s inexact pack", label);
            if (!nocpu) cpu_graph_check(label, t, srcs, c, x, M, rng);
            std::vector<double> dref; nph_lora::reference_delta(srcs, x.data(), M, t.K, t.N, dref);
            for (int ftz = 0; ftz < 2; ftz++) {
                g_ftz = ftz;
                std::vector<float> dn, dol;
                emu_new(c, x.data(), M, t.lin_scale, dn);
                emu_old(srcs, x.data(), M, t.K, t.N, t.lin_scale, dol);
                Err en = err_vs(dn, dref), eo = err_vs(dol, dref);
                printf("  HMX-emu %-4s delta rel-l2 combined=%.3g per-LoRA=%.3g | rel-max combined=%.3g per-LoRA=%.3g\n",
                       ftz ? "FTZ" : "IEEE", en.rel_l2, eo.rel_l2, en.rel_max, eo.rel_max);
                CHECK(en.rel_l2 <= eo.rel_l2 * 1.25 + 1e-6, "%s HMX-emu(%s) combined less accurate than per-LoRA (%g vs %g)", label, ftz ? "FTZ" : "IEEE", en.rel_l2, eo.rel_l2);
            }
        }
    }
    printf(g_fail ? "RESULT: %d FAIL\n" : "RESULT: ALL PASS%.0d\n", g_fail);
    return g_fail ? 1 : 0;
}
