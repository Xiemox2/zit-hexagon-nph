#pragma once
// NPH 0.30 NPG: per-part health check of a GPU VAE decode result (tile or whole image).
// Returns nullptr when the part is healthy, otherwise a Spanish reason written into buf.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

inline const char* nph_npg_part_problem(const float* d, int64_t n, float limit, char* buf, size_t cap) {
    if (d == nullptr || n <= 0) return "error de cómputo (OpenCL/grafo)";
    size_t nonfinite = 0, over = 0;
    float mx = 0.f;
    for (int64_t i = 0; i < n; ++i) {
        const float v = d[i];
        if (!std::isfinite(v)) { ++nonfinite; continue; }
        const float a = std::fabs(v);
        if (a > mx) mx = a;
        if (a > limit) ++over;
    }
    if (nonfinite) { snprintf(buf, cap, "%zu valores NaN/Inf", nonfinite); return buf; }
    if (over) { snprintf(buf, cap, "desbordamiento: %zu valores con |v|>%.0f (máx %.3g)", over, (double)limit, (double)mx); return buf; }
    return nullptr;
}
