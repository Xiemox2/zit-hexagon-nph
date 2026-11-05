// NPH 0.30.2 box test of the on-device fused-LoRA self-test logic (synthetic data only).
// Runs nph_lora::fuse_selftest_exec on the ggml CPU backend (no HTP on the box), with fake hooks.
// Checks the graph build (same graph as Linear + combined LoRA), the FP64 reference, the F8 decoder
// (vs ggml's own F8_E4M3 to_float) and the tolerance logic. The NPU fused kernel itself can only be
// checked on the phone (that is what the self-test does there).
#include <cstdio>
#include <cstring>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "stable-diffusion.h"
#include "model/adapter/nph_lora_fuse_selftest.hpp"

static int g_fuse = 1, g_sets = 0;
static void f_set(int on) { g_fuse = on; ++g_sets; }
static int f_get() { return g_fuse; }

static void logcb(sd_log_level_t, const char* text, void*) { fputs(text, stdout); }

int main() {
    sd_set_log_callback(logcb, nullptr);
    setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_time_init();
    // (1) F8 E4M3 decoder == ggml's
    const auto* tr = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
    int bad = 0;
    if (tr && tr->to_float) {
        for (int v = 0; v < 256; ++v) {
            if ((v & 0x7f) == 0x7f) continue;  // NaN codes (never generated)
            uint8_t b = (uint8_t)v;
            float f;
            tr->to_float(&b, &f, 1);
            const float g = nph_lora::fuse_st_f8_to_f32(b);
            if (!(f == g)) { if (bad < 5) printf("f8 mismatch 0x%02x ggml %g mine %g\n", v, f, g); ++bad; }
        }
        printf("f8 decoder vs ggml to_float: %s (%d mismatches / 254)\n", bad ? "FAIL" : "OK", bad);
    } else {
        printf("f8 decoder: ggml has no to_float for F8_E4M3 (skipped)\n");
    }
    // (2) full self-test on CPU (no fused op exists there -> both runs = same graph)
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 8);
    nph_hex_hooks hk{};
    hk.set_lora_fuse = f_set;
    hk.get_lora_fuse = f_get;
    const int64_t t0 = ggml_time_us();
    const bool pass  = nph_lora::fuse_selftest_exec(cpu, &hk, GGML_TYPE_F16);
    printf("fuse_selftest_exec on CPU: %s (%.0f ms), toggle sets=%d final=%d\n", pass ? "PASS" : "FAIL",
           (ggml_time_us() - t0) / 1000.0, g_sets, g_fuse);
    ggml_backend_free(cpu);
    const bool ok = pass && bad == 0 && g_fuse == 1;
    printf("RESULT: %s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
