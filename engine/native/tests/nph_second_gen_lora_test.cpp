// NPH 0.30.2 regression test for the 0.30.1 phone failure "model_manager_files.cpp:60 cannot update
// model sources during execution" on the 2nd image with LoRAs (resident DiT, Extremo soft runner_end).
// Synthetic only: two tiny generated .safetensors LoRA files (no real model/LoRA).
// Sequence = what the engine does across generations with a resident runner:
//   gen 1: runner_start registers the workspace reclaimer -> compute -> soft runner_end (idle).
//   gen 2: set_loras (same set / changed strength / changed set) -> must be accepted while idle,
//          and must be rejected while the runner is executing.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model.h"
#include "model_manager.h"
#include "stable-diffusion.h"

static void logcb(sd_log_level_t, const char* text, void*) { fputs(text, stdout); }

static bool write_lora(const std::string& path, float v) {
    // one F32 tensor [4] = v
    const std::string hdr = "{\"lora_unet_layers_0_attention_out.lora_down.weight\":{\"dtype\":\"F32\",\"shape\":[1,4],\"data_offsets\":[0,16]}}";
    std::string h = hdr;
    while ((h.size() + 8) % 8) h += ' ';
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    uint64_t n = h.size();
    fwrite(&n, 8, 1, f);
    fwrite(h.data(), 1, h.size(), f);
    float d[4] = {v, v, v, v};
    fwrite(d, 4, 4, f);
    fclose(f);
    return true;
}

static int fails = 0;
static void expect(bool got, bool want, const char* what) {
    printf("%-74s %s (got %s)\n", what, got == want ? "OK" : "FAIL", got ? "accepted" : "rejected");
    if (got != want) ++fails;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    sd_set_log_callback(logcb, nullptr);
    const std::string a = "/tmp/nph_sg_a.safetensors", b = "/tmp/nph_sg_b.safetensors";
    if (!write_lora(a, 0.5f) || !write_lora(b, 0.25f)) { printf("cannot write test files\n"); return 1; }
    ModelManager mm;
    auto spec = [](const std::string& p, float m) { ModelManager::LoraSpec s; s.path = p; s.multiplier = m; s.required = true; return s; };
    const uintptr_t dit = 0x1234;

    expect(mm.set_loras({spec(a, 0.9f), spec(b, 0.7f)}, VERSION_Z_IMAGE), true, "gen1: set_loras before any runner");
    // gen 1 runner_start (cold) registers the reclaimer: executing
    mm.set_workspace_reclaimer(dit, [] { return true; });
    expect(mm.set_loras({spec(a, 0.9f)}, VERSION_Z_IMAGE), false, "gen1 executing: change LoRA set (must be rejected)");
    // soft runner_end (Extremo keeps workspace + reclaimer): idle
    mm.set_runtime_owner_idle(dit, true);
    expect(mm.set_loras({spec(a, 0.9f), spec(b, 0.7f)}, VERSION_Z_IMAGE), true, "gen2 idle: same LoRAs (0.30.1 failed here)");
    expect(mm.set_loras({spec(a, 0.8f), spec(b, 0.7f)}, VERSION_Z_IMAGE), true, "gen2 idle: changed strength");
    expect(mm.set_loras({spec(b, 0.7f)}, VERSION_Z_IMAGE), true, "gen2 idle: changed LoRA set (one file removed)");
    expect(mm.set_loras({spec(a, 0.9f), spec(b, 0.7f)}, VERSION_Z_IMAGE), true, "gen2 idle: back to both");
    // gen 2 runner_start on the resident runner (resume): executing again
    mm.set_runtime_owner_idle(dit, false);
    expect(mm.set_loras({spec(a, 0.6f)}, VERSION_Z_IMAGE), false, "gen2 executing: change LoRA set (must be rejected)");
    expect(mm.set_loras({spec(a, 0.9f), spec(b, 0.7f)}, VERSION_Z_IMAGE), true, "gen2 executing: identical set (no-op, accepted)");
    mm.set_runtime_owner_idle(dit, true);
    expect(mm.set_loras({}, VERSION_Z_IMAGE), true, "gen3 idle: no LoRAs");
    // a second, non-idle owner blocks updates even if the DiT is idle
    mm.set_workspace_reclaimer(0x5678, [] { return true; });
    expect(mm.set_loras({spec(a, 0.9f)}, VERSION_Z_IMAGE), false, "other runner executing (TE/VAE): rejected");
    mm.remove_runtime_owner(0x5678);
    expect(mm.set_loras({spec(a, 0.9f)}, VERSION_Z_IMAGE), true, "after it is removed: accepted");
    // idle flag must not outlive a fresh registration (hard runner_end + new runner_start)
    mm.remove_runtime_owner(dit);
    mm.set_workspace_reclaimer(dit, [] { return true; });
    expect(mm.set_loras({spec(b, 0.7f)}, VERSION_Z_IMAGE), false, "fresh runner_start after hard end: rejected while executing");
    mm.set_runtime_owner_idle(0x9999, true);  // unknown owner: ignored
    expect(mm.set_loras({spec(b, 0.7f)}, VERSION_Z_IMAGE), false, "idle mark of an unknown owner does not unlock");
    printf("RESULT: %s (%d failures)\n", fails ? "FAIL" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
