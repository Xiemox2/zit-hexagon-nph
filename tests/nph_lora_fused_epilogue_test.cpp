// NPH 0.30.2: bit-level emulation of the HTP_OP_MUL_MAT_LORA output epilogue vs the 0.30.1 op chain.
// Synthetic values only. v79/v81 HVX f32 ops are IEEE (Q6_Vsf_vmpy/vadd_VsfVsf), no FMA, so plain
// float arithmetic with -ffp-contract=off reproduces them.
//   h1 = FP16 HMX tile of the FP8 base product (scaled 1/256), h2 = FP16 HMX tile of B·y.
//   0.30.1: base = fl(h1*S) [MUL_MAT + fused SCALE(2^g) + MUL(ws), S = 256*2^g*ws]
//           t    = fl(h2 + base) [MUL_MAT(B)+ADD fused]   ; out = fl(t*P) [SCALE op, only if P != 1]
//   0.30.2: out  = fl(fl(h1*alpha) + (P != 1 ? fl(h2*P) : h2)), alpha = fl(S*P)  (one op)
// Exact equality requires P = power of two (the host fuser only folds pow2 SCALEs) and no
// overflow/underflow in the intermediate products.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

static float h16(float x) { return (float)(_Float16)x; }

int main() {
    std::mt19937_64 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ex(-14, 8);
    struct Cfg { const char* name; int g; float ls; float ws; };
    const Cfg cfgs[] = {
        {"plain (lin 1, attn/w1/w3)", 0, 1.0f, 0.0123f},
        {"deferred w2 (g=3, 1/128)", 3, 1.0f / 128.0f, 0.0123f},
        {"deferred w2 (g=0, 1/128)", 0, 1.0f / 128.0f, 0.00731f},
        {"deferred w2 (g=7, 1/128)", 7, 1.0f / 128.0f, 0.0217f},
    };
    long long total_bad = 0, total = 0;
    for (const auto& c : cfgs) {
        const float output_scale = (256.0f * std::ldexp(1.0f, c.g)) * c.ws;  // DSP: 256, *= scale, *= ws
        const float P            = c.ls == 1.0f ? 1.0f : std::ldexp(1.0f, -c.g) / c.ls;
        const float alpha        = output_scale * P;
        long long bad = 0, n = 0;
        for (long long i = 0; i < 20000000; ++i) {
            const float h1 = h16(nd(rng) * std::ldexp(1.0f, ex(rng) / 2));
            const float h2 = h16(nd(rng) * std::ldexp(1.0f, ex(rng) / 2));
            float base = h1 * output_scale;
            float t    = h2 + base;
            float o301 = (P != 1.0f) ? t * P : t;
            float b    = (P != 1.0f) ? h2 * P : h2;
            float o302 = h1 * alpha + b;
            ++n;
            if (memcmp(&o301, &o302, 4) != 0 && !(o301 == 0 && o302 == 0)) {
                if (bad < 3) printf("  mismatch h1=%a h2=%a 0.30.1=%a 0.30.2=%a\n", h1, h2, o301, o302);
                ++bad;
            }
        }
        printf("%-30s S=%.6g P=%g: %lld / %lld bit mismatches\n", c.name, output_scale, P, bad, n);
        total_bad += bad;
        total += n;
    }
    // non-pow2 P is NOT bit-exact -> the fuser must refuse it (it does: frexp check)
    {
        const float S = 256.0f * 0.0123f, P = 0.3f, alpha = S * P;
        long long bad = 0;
        for (int i = 0; i < 1000000; ++i) {
            const float h1 = h16(nd(rng)), h2 = h16(nd(rng));
            float o301 = (h2 + h1 * S) * P, o302 = h1 * alpha + h2 * P;
            bad += memcmp(&o301, &o302, 4) != 0;
        }
        printf("control: non-pow2 P=0.3 (refused by the fuser): %lld / 1000000 mismatches (expected > 0)\n", bad);
    }
    printf("RESULT: %s (%lld mismatches in %lld)\n", total_bad == 0 ? "ALL PASS (bit-identical)" : "FAIL", total_bad, total);
    return total_bad == 0 ? 0 : 1;
}
