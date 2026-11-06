// NPG bench — common helpers. Copyright (c) 2026 ZIT-HEXAGON authors. MIT License.
#pragma once
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace npg {

double now_ms();
void   P(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void   progress(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
extern std::atomic<bool> g_stop;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    inline uint64_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 0x2545F4914F6CDD1Dull; }
    inline float uni() { return (next() >> 40) * (1.0f / 16777216.0f); }
    float gauss() { float u = std::max(uni(), 1e-7f), v = uni(); return std::sqrt(-2.f * std::log(u)) * std::cos(6.2831853f * v); }
};

// FP8 E4M3 (fn) helpers
float   e4m3_to_f32(uint8_t b);
uint8_t f32_to_e4m3(float x);
extern float g_e4m3_lut[256];
void    init_e4m3();
uint16_t f32_to_f16(float f);
float    f16_to_f32(uint16_t h);

// fill helpers (fast, table-based)
void fill_fp8_weights(uint8_t * dst, size_t n, uint64_t seed);   // q = e4m3(N(0, 0.06)) clipped to |q| <= 1
void fill_gauss(float * dst, size_t n, uint64_t seed, float sd = 1.0f);

// ---- OpenCL (dlopen) ----
struct CLApi {
    bool ok = false;
    std::string lib;
#define NPG_CL_FN(name) decltype(&::name) name = nullptr;
#include "npg_cl_fns.inc"
#undef NPG_CL_FN
};
extern CLApi cl;
bool cl_load(std::string & err);
const char * cl_errstr(cl_int e);

struct Gpu {
    bool              ok = false;
    cl_platform_id    plat = nullptr;
    cl_device_id      dev  = nullptr;
    cl_context        ctx  = nullptr;
    cl_command_queue  q    = nullptr;   // in-order, profiling
    std::string       name, version, driver, exts;
    bool              fp16 = false, dmabuf = false, ion = false, ext_host = false, iocoh = false, ahb = false;
    cl_uint           page_size = 4096, ext_padding = 0;
    cl_ulong          local_mem = 0, global_mem = 0;
    size_t            max_wg = 0;
    std::string       build_opts_base;
    cl_program build(const std::string & defs, std::string & log);
};

}  // namespace npg
