// NPH 0.29 NPG: tiny synthetic self-test of the VAE ops on a ggml backend (e.g. GPUOpenCL)
// against a double-precision reference. Never touches model weights. MIT (part of zit-hexagon).
#pragma once
#include <string>
#include "ggml-backend.h"

struct NpgSelftestResult {
    bool ok = false;           // every supported op within tolerance
    int  supported = 0;        // ops run on the backend
    int  unsupported = 0;      // ops the backend refused (would run on CPU via the scheduler)
    double ms = 0.0;           // wall time of the whole test
    std::string text;          // one line per op, Spanish, for the in-app report
    int bad_mm = 0;            // 0.30: failing mul_mat / flash_attn checks (can be routed to the NPU)
    int bad_other = 0;         // 0.30: failing checks of every other op
};

NpgSelftestResult npg_selftest(ggml_backend_t backend);
