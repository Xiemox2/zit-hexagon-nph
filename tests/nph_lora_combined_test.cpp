// NPH 0.30 synthetic test: combined LoRA (concat over rank, block-diag qkv, pow2 shifts, rank pad)
// vs a double reference of the original per-LoRA math, and vs the old per-LoRA ggml graph on the
// CPU backend. No model files are touched.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "model/adapter/nph_lora_core.h"

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static float bf16_round(float v) {
    uint32_t u; memcpy(&u, &v, 4);
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u;
    float r; memcpy(&r, &u, 4); return r;
}
static float f16_round(float v) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); }

struct Lora { int r; float scale; std::vector<std::vector<float>> down, up; };  // per part

// exact double reconstruction of B*diag(s)*A from the packed data
static double packed_weight_err(const nph_lora::Combined& c, const std::vector<nph_lora::PartSrc>& srcs) {
    std::vector<double> W((size_t)c.N * c.K, 0.0), R((size_t)c.N * c.K, 0.0);
    for (int t = 0; t < c.Rp; t++)
        for (int n = 0; n < c.N; n++) {
            double b = ggml_fp16_to_fp32(c.B[(size_t)n * c.Rp + t]) * (double)c.s[t];
            if (b == 0) continue;
            for (int k = 0; k < c.K; k++) W[(size_t)n * c.K + k] += b * ggml_fp16_to_fp32(c.A[(size_t)t * c.K + k]);
        }
    for (const auto& ps : srcs)
        for (int i = 0; i < ps.n; i++)
            for (int j = 0; j < ps.r; j++)
                for (int k = 0; k < c.K; k++)
                    R[(size_t)(ps.row_off + i) * c.K + k] += (double)ps.up[(size_t)i * ps.r + j] * ps.down[(size_t)j * c.K + k] * ps.scale;
    double mx = 0, ref = 0;
    for (size_t i = 0; i < W.size(); i++) { mx = std::max(mx, std::fabs(W[i] - R[i])); ref = std::max(ref, std::fabs(R[i])); }
    return ref > 0 ? mx / ref : mx;
}

struct Case { const char* name; int K, N, M, parts; std::vector<int> ranks; std::vector<float> strengths; float lin_scale; int dtype; /*0=bf16 1=f16 2=f32*/ };

static void run_case(const Case& cs, std::mt19937& rng) {
    const int K = cs.K, N = cs.N, M = cs.M, P = cs.parts;
    const int n_part = N / P;
    std::normal_distribution<float> nd(0.f, 1.f);
    auto q = [&](float v) { return cs.dtype == 0 ? bf16_round(v) : cs.dtype == 1 ? f16_round(v) : v; };
    std::vector<Lora> loras;
    for (size_t l = 0; l < cs.ranks.size(); l++) {
        Lora L; L.r = cs.ranks[l];
        const float alpha = (float)L.r * (l == 0 ? 1.0f : 0.5f);
        L.scale = alpha / L.r * cs.strengths[l];
        for (int p = 0; p < P; p++) {
            std::vector<float> d((size_t)L.r * K), u((size_t)n_part * L.r);
            for (auto& v : d) v = q(nd(rng) * 0.02f);
            for (auto& v : u) v = q(nd(rng) * 0.01f);
            // tiny / huge values to exercise shifts and splits
            d[1] = q(3.1e-7f); d[2] = q(-1.7e-9f); u[3] = q(2.5e-8f);
            if (cs.dtype == 2) { d[5] = 1.0f / 3.0f; u[7] = 70000.0f * 0.0f + 0.1234567f; }
            L.down.push_back(d); L.up.push_back(u);
        }
        loras.push_back(L);
    }
    std::vector<nph_lora::PartSrc> srcs;
    for (auto& L : loras)
        for (int p = 0; p < P; p++) {
            nph_lora::PartSrc ps; ps.K = K; ps.r = L.r; ps.n = n_part; ps.row_off = p * n_part;
            ps.down = L.down[p].data(); ps.up = L.up[p].data(); ps.scale = L.scale; srcs.push_back(ps);
        }
    nph_lora::Combined c;
    std::string err;
    CHECK(nph_lora::combine(srcs, K, N, 3, c, &err), "%s combine: %s", cs.name, err.c_str());
    CHECK(c.Rp % 32 == 0 && c.Rp >= c.R, "%s Rp=%d R=%d", cs.name, c.Rp, c.R);
    for (int t = c.R; t < c.Rp; t++) CHECK(c.s[t] == 0.f, "%s pad s", cs.name);
    if (cs.dtype == 2) {
        CHECK(c.st.max_abs_err <= std::ldexp(1.0, -24) * 0.2, "%s f32 residual %g", cs.name, c.st.max_abs_err);
    } else {
        CHECK(c.st.inexact_values == 0, "%s inexact=%lld maxerr=%g", cs.name, (long long)c.st.inexact_values, c.st.max_abs_err);
    }
    const double werr = packed_weight_err(c, srcs);
    CHECK(werr < (cs.dtype == 2 ? 1e-9 : 1e-12), "%s packed weight rel err %g", cs.name, werr);

    // ---- ggml CPU graph: base F16 linear + combined delta vs old per-LoRA path ----
    std::vector<float> x((size_t)M * K), w((size_t)N * K);
    for (auto& v : x) v = nd(rng) * (cs.lin_scale != 1.f ? 300.f : 1.f);
    for (auto& v : w) v = f16_round(nd(rng) * 0.02f);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_init_params ip = {64 * ggml_tensor_overhead() + 16 * 1024 * 1024, nullptr, true};
    ggml_context* wctx = ggml_init(ip);
    ggml_tensor* tx = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, M);
    ggml_tensor* tw = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, K, N);
    ggml_tensor* tA = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, K, c.Rp);
    ggml_tensor* tB = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, c.Rp, N);
    ggml_tensor* ts = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, c.Rp);
    std::vector<ggml_tensor*> tdown, tup;
    for (auto& L : loras) for (int p = 0; p < P; p++) {
        tdown.push_back(ggml_new_tensor_2d(wctx, GGML_TYPE_F16, K, L.r));
        tup.push_back(ggml_new_tensor_2d(wctx, GGML_TYPE_F16, L.r, n_part));
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(wctx, be);
    ggml_backend_tensor_set(tx, x.data(), 0, ggml_nbytes(tx));
    std::vector<ggml_fp16_t> wh(w.size()); for (size_t i = 0; i < w.size(); i++) wh[i] = ggml_fp32_to_fp16(w[i]);
    ggml_backend_tensor_set(tw, wh.data(), 0, ggml_nbytes(tw));
    ggml_backend_tensor_set(tA, c.A.data(), 0, ggml_nbytes(tA));
    ggml_backend_tensor_set(tB, c.B.data(), 0, ggml_nbytes(tB));
    ggml_backend_tensor_set(ts, c.s.data(), 0, ggml_nbytes(ts));
    {
        size_t i = 0;
        for (auto& L : loras) for (int p = 0; p < P; p++, i++) {
            std::vector<ggml_fp16_t> dh(L.down[p].size()), uh(L.up[p].size());
            for (size_t k = 0; k < dh.size(); k++) dh[k] = ggml_fp32_to_fp16(L.down[p][k]);
            for (size_t k = 0; k < uh.size(); k++) uh[k] = ggml_fp32_to_fp16(L.up[p][k]);
            ggml_backend_tensor_set(tdown[i], dh.data(), 0, ggml_nbytes(tdown[i]));
            ggml_backend_tensor_set(tup[i], uh.data(), 0, ggml_nbytes(tup[i]));
        }
    }
    ggml_init_params gp = {4096 * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true};
    ggml_context* g = ggml_init(gp);
    const float ls = cs.lin_scale;
    // base (as ggml_ext_linear): x_s = x*ls ; mm ; *1/ls
    ggml_tensor* x_s = ls != 1.f ? ggml_scale(g, tx, ls) : tx;
    ggml_tensor* base = ggml_mul_mat(g, tw, x_s);
    if (ls != 1.f) base = ggml_scale(g, base, 1.f / ls);
    ggml_tensor* out_new = nph_lora::graph_apply(g, x_s, tA, tB, ts, ls, false, base);
    // old path (per LoRA: ext_linear(down) -> ext_linear(up) -> scale -> concat parts -> add)
    ggml_tensor* out_old = base;
    {
        size_t i = 0;
        for (auto& L : loras) {
            ggml_tensor* diff = nullptr;
            for (int p = 0; p < P; p++, i++) {
                ggml_tensor* xi = ls != 1.f ? ggml_scale(g, tx, ls) : tx;
                ggml_tensor* lx = ggml_mul_mat(g, tdown[i], xi);
                if (ls != 1.f) lx = ggml_scale(g, lx, 1.f / ls);
                ggml_tensor* li = ls != 1.f ? ggml_scale(g, lx, ls) : lx;
                lx = ggml_mul_mat(g, tup[i], li);
                if (ls != 1.f) lx = ggml_scale(g, lx, 1.f / ls);
                lx = ggml_scale(g, lx, L.scale);
                diff = diff ? ggml_concat(g, diff, lx, 0) : lx;
            }
            out_old = ggml_add(g, out_old, diff);
        }
    }
    ggml_cgraph* gf = ggml_new_graph_custom(g, 4096, false);
    ggml_build_forward_expand(gf, out_new);
    ggml_build_forward_expand(gf, out_old);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    CHECK(ggml_gallocr_alloc_graph(ga, gf), "%s alloc", cs.name);
    ggml_backend_cpu_set_n_threads(be, 4);
    CHECK(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS, "%s compute", cs.name);
    std::vector<float> on((size_t)M * N), oo((size_t)M * N);
    ggml_backend_tensor_get(out_new, on.data(), 0, ggml_nbytes(out_new));
    ggml_backend_tensor_get(out_old, oo.data(), 0, ggml_nbytes(out_old));
    // double reference: base (with F16 weights, exact x) + delta
    std::vector<double> dref; nph_lora::reference_delta(srcs, x.data(), M, K, N, dref);
    double e_new = 0, e_old = 0, e_no = 0, mref = 0, dmax = 0;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double b = 0; for (int k = 0; k < K; k++) b += (double)w[(size_t)n * K + k] * x[(size_t)m * K + k];
        const double r = b + dref[(size_t)m * N + n];
        mref = std::max(mref, std::fabs(r)); dmax = std::max(dmax, std::fabs(dref[(size_t)m * N + n]));
        e_new = std::max(e_new, std::fabs((double)on[(size_t)m * N + n] - r));
        e_old = std::max(e_old, std::fabs((double)oo[(size_t)m * N + n] - r));
        e_no  = std::max(e_no, (double)std::fabs(on[(size_t)m * N + n] - oo[(size_t)m * N + n]));
    }
    printf("%-28s K=%d N=%d M=%d parts=%d ranks=", cs.name, K, N, M, P);
    for (int r : cs.ranks) printf("%d,", r);
    printf(" R=%d Rp=%d shifted=%d split=%d inexact=%lld | rel err new=%.3g old=%.3g new-vs-old=%.3g (|delta|max=%.3g)\n",
           c.R, c.Rp, c.st.shifted_vectors, c.st.split_vectors, (long long)c.st.inexact_values,
           e_new / mref, e_old / mref, e_no / mref, dmax);
    // the new path must be at least as accurate as the old one (CPU converts x to F16 for F16 weights on both)
    CHECK(e_new <= e_old * 1.5 + 1e-6 * mref, "%s new path less accurate: %g vs %g", cs.name, e_new, e_old);
    CHECK(e_new / mref < 5e-3, "%s rel err %g", cs.name, e_new / mref);
    // count graph nodes (HVX-type full-size ops) new vs old
    int n_new = 0, n_old = 0;
    {
        ggml_cgraph* g1 = ggml_new_graph_custom(g, 4096, false); ggml_build_forward_expand(g1, out_new);
        ggml_cgraph* g2 = ggml_new_graph_custom(g, 4096, false); ggml_build_forward_expand(g2, out_old);
        auto count = [](ggml_cgraph* gg, int Nout) { int c2 = 0; for (int i = 0; i < ggml_graph_n_nodes(gg); i++) { ggml_tensor* t = ggml_graph_node(gg, i);
            if (t->op != GGML_OP_MUL_MAT && t->op != GGML_OP_NONE && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_VIEW && t->ne[0] >= Nout / 3) c2++; } return c2; };
        n_new = count(g1, N) - (ls != 1.f ? 2 : 0); n_old = count(g2, N) - (ls != 1.f ? 2 : 0);
    }
    printf("%-28s full-size elementwise ops beyond base: new=%d old=%d\n", "", n_new, n_old);
    ggml_gallocr_free(ga); ggml_free(g); ggml_backend_buffer_free(buf); ggml_free(wctx); ggml_backend_free(be);
}

static void exactness_unit() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    // F32 values: must be reproduced exactly by <=3 FP16 parts
    std::vector<float> v(4096);
    for (auto& t : v) t = nd(rng) * 0.01f;
    v[0] = 1.0f / 3.0f; v[1] = 1e-9f; v[2] = 123456.7f; v[3] = -7.7e-6f;
    std::vector<nph_lora::VecPart> parts; nph_lora::Stats st;
    nph_lora::split_vec(v.data(), (int)v.size(), 1, 3, parts, st);
    // F32 files: residual must be below FP32 resolution of the vector (2^-24 * max|v|)
    CHECK(st.max_abs_err <= std::ldexp(123456.7, -24), "f32 split err=%g parts=%zu", st.max_abs_err, parts.size());
    printf("exactness f32: parts=%zu shifts=", parts.size()); for (auto& p : parts) printf("%d,", p.k); printf(" inexact=%lld\n", (long long)st.inexact_values);
    // BF16 with tiny values: shifts or split, exact
    for (auto& t : v) t = bf16_round(nd(rng) * 1e-4f);
    v[9] = bf16_round(2.3e-9f);
    nph_lora::Stats st2; nph_lora::split_vec(v.data(), (int)v.size(), 1, 3, parts, st2);
    CHECK(st2.inexact_values == 0, "bf16 split inexact=%lld", (long long)st2.inexact_values);
    printf("exactness bf16 tiny: parts=%zu k0=%d inexact=%lld\n", parts.size(), parts.empty() ? 0 : parts[0].k, (long long)st2.inexact_values);
    // plain F16 data: 1 part, no shift
    for (auto& t : v) t = f16_round(nd(rng) * 0.02f);
    nph_lora::Stats st3; nph_lora::split_vec(v.data(), (int)v.size(), 1, 3, parts, st3);
    CHECK(parts.size() == 1 && parts[0].k == 0 && st3.inexact_values == 0, "f16 passthrough parts=%zu k=%d", parts.size(), parts.empty() ? -99 : parts[0].k);
    printf("exactness f16: parts=%zu k=%d (passthrough)\n", parts.size(), parts.empty() ? 0 : parts[0].k);
}

int main() {
    exactness_unit();
    std::mt19937 rng(1234);
    const Case cases[] = {
        {"qkv 2 LoRAs bf16 r16+r8", 256, 3 * 96, 40, 3, {16, 8}, {0.8f, 1.2f}, 1.0f, 0},
        {"out 2 LoRAs f16 r32+r32", 256, 256, 33, 1, {32, 32}, {1.0f, 0.6f}, 1.0f, 1},
        {"w2 scale1/128 bf16 r16+r4", 512, 128, 17, 1, {16, 4}, {1.0f, 0.9f}, 1.0f / 128.0f, 0},
        {"f32 LoRA r8 (split)", 128, 64, 9, 1, {8}, {1.0f}, 1.0f, 2},
        {"3 LoRAs qkv bf16 r4+r12+r20", 192, 3 * 64, 21, 3, {4, 12, 20}, {0.5f, 1.0f, -0.7f}, 1.0f, 0},
    };
    for (const auto& cs : cases) run_case(cs, rng);
    printf(g_fail ? "RESULT: %d FAIL\n" : "RESULT: ALL PASS%.0d\n", g_fail);
    return g_fail ? 1 : 0;
}
