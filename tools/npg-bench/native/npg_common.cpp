// NPG bench — common helpers. Copyright (c) 2026 ZIT-HEXAGON authors. MIT License.
#include "npg_common.h"
#ifndef __ANDROID__
#include <CL/cl_icd.h>
#endif
#include <dlfcn.h>

namespace npg {

CLApi cl;
float g_e4m3_lut[256];

double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

float e4m3_to_f32(uint8_t b) {
    int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 15 && m == 7) v = NAN;
    else if (e == 0) v = std::ldexp((float) m / 8.f, -6);
    else v = std::ldexp(1.f + (float) m / 8.f, e - 7);
    return s ? -v : v;
}

void init_e4m3() {
    for (int i = 0; i < 256; ++i) g_e4m3_lut[i] = e4m3_to_f32((uint8_t) i);
}

uint8_t f32_to_e4m3(float x) {
    uint8_t s = std::signbit(x) ? 0x80 : 0;
    float a = std::fabs(x);
    if (!(a == a)) return 0x7F;
    if (a >= 448.f) return s | 0x7E;
    // binary search over positive codes 0..126 (monotonic)
    int lo = 0, hi = 126;
    while (hi - lo > 1) { int mid = (lo + hi) / 2; if (g_e4m3_lut[mid] <= a) lo = mid; else hi = mid; }
    float dl = a - g_e4m3_lut[lo], dh = g_e4m3_lut[hi] - a;
    int c = (dl < dh) ? lo : (dh < dl) ? hi : ((lo & 1) ? hi : lo);
    return s | (uint8_t) c;
}

uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  e = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
    if (e >= 31) return sign | 0x7c00;
    if (e <= 0) {
        if (e < -10) return sign;
        m |= 0x800000;
        int shift = 14 - e;
        uint32_t hm = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (hm & 1))) hm++;
        return sign | hm;
    }
    uint32_t hm = m >> 13, rem = m & 0x1fff;
    uint32_t r = sign | (e << 10) | hm;
    if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) r++;
    return r;
}

float f16_to_f32(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023, x;
    if (e == 0) {
        if (!m) x = s;
        else { float v = std::ldexp((float) m, -24); return s ? -v : v; }
    } else if (e == 31) x = s | 0x7f800000 | (m << 13);
    else x = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}

void fill_fp8_weights(uint8_t * dst, size_t n, uint64_t seed) {
    static std::vector<uint8_t> tab;
    static std::once_flag once;
    std::call_once(once, [] {
        tab.resize(1 << 16);
        Rng r(12345);
        for (auto & t : tab) { float v = r.gauss() * 0.06f; v = std::max(-1.f, std::min(1.f, v)); t = f32_to_e4m3(v); }
    });
    Rng r(seed);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint64_t v = r.next();
        dst[i] = tab[v & 0xffff]; dst[i + 1] = tab[(v >> 16) & 0xffff]; dst[i + 2] = tab[(v >> 32) & 0xffff]; dst[i + 3] = tab[v >> 48];
    }
    for (; i < n; ++i) dst[i] = tab[r.next() & 0xffff];
}

void fill_gauss(float * dst, size_t n, uint64_t seed, float sd) {
    static std::vector<float> tab;
    static std::once_flag once;
    std::call_once(once, [] { tab.resize(1 << 16); Rng r(777); for (auto & t : tab) t = r.gauss(); });
    Rng r(seed);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint64_t v = r.next();
        dst[i] = sd * tab[v & 0xffff]; dst[i + 1] = sd * tab[(v >> 16) & 0xffff];
        dst[i + 2] = sd * tab[(v >> 32) & 0xffff]; dst[i + 3] = sd * tab[v >> 48];
    }
    for (; i < n; ++i) dst[i] = sd * tab[r.next() & 0xffff];
}

const char * cl_errstr(cl_int e) {
    static char buf[32];
    switch (e) {
        case 0: return "CL_SUCCESS";
        case -4: return "MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "OUT_OF_RESOURCES";
        case -6: return "OUT_OF_HOST_MEMORY";
        case -11: return "BUILD_PROGRAM_FAILURE";
        case -30: return "INVALID_VALUE";
        case -37: return "INVALID_HOST_PTR";
        case -38: return "INVALID_MEM_OBJECT";
        case -48: return "INVALID_KERNEL";
        case -52: return "INVALID_KERNEL_ARGS";
        case -54: return "INVALID_WORK_GROUP_SIZE";
        case -61: return "INVALID_BUFFER_SIZE";
        case -64: return "INVALID_PROPERTY";
    }
    snprintf(buf, sizeof buf, "CL_ERR_%d", e);
    return buf;
}

bool cl_load(std::string & err) {
    if (cl.ok) return true;
    std::vector<std::string> cands;
    if (const char * e = getenv("NPG_OPENCL_LIB")) cands.push_back(e);
    cands.insert(cands.end(), { "libOpenCL.so", "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so",
                                "libOpenCL_adreno.so", "/vendor/lib64/libOpenCL_adreno.so", "libOpenCL.so.1" });
    void * h = nullptr;
    for (auto & c : cands) {
        h = dlopen(c.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (h) { cl.lib = c; break; }
        err += c + ": " + (dlerror() ?: "?") + "; ";
    }
    if (!h) return false;
#ifndef __ANDROID__
    // Host test path (pocl as a bare ICD, no loader): resolve through the ICD dispatch table.
    if (!dlsym(h, "clGetPlatformIDs")) {
        typedef void * (*gefa_t)(const char *);
        typedef cl_int (*icdgp_t)(cl_uint, cl_platform_id *, cl_uint *);
        auto gefa = (gefa_t) dlsym(h, "clGetExtensionFunctionAddress");
        auto icdgp = gefa ? (icdgp_t) gefa("clIcdGetPlatformIDsKHR") : nullptr;
        cl_platform_id pl = nullptr; cl_uint n = 0;
        if (!icdgp || icdgp(1, &pl, &n) || !pl) { err += "ICD sin plataforma; "; return false; }
        const cl_icd_dispatch * d = *(const cl_icd_dispatch * const *) pl;
        static icdgp_t s_icdgp = icdgp;
        cl.clGetPlatformIDs = [](cl_uint ne, cl_platform_id * ps, cl_uint * np) { return s_icdgp(ne, ps, np); };
#define NPG_CL_FN(name) if (!cl.name) cl.name = (decltype(cl.name)) d->name;
#include "npg_cl_fns.inc"
#undef NPG_CL_FN
        cl.ok = true;
        return true;
    }
#endif
    bool all = true;
#define NPG_CL_FN(name) cl.name = (decltype(cl.name)) dlsym(h, #name); if (!cl.name) { err += "falta " #name "; "; all = false; }
#include "npg_cl_fns.inc"
#undef NPG_CL_FN
    cl.ok = all;
    return all;
}

cl_program Gpu::build(const std::string & defs, std::string & log) {
    extern const char * npg_kernel_src;
    const char * src = npg_kernel_src;
    cl_int err;
    cl_program p = cl.clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    if (err) { log = std::string("create: ") + cl_errstr(err); return nullptr; }
    std::string opts = build_opts_base + " " + defs;
    err = cl.clBuildProgram(p, 1, &dev, opts.c_str(), nullptr, nullptr);
    size_t n = 0;
    cl.clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
    log.resize(n);
    if (n) cl.clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
    if (err) { log = std::string("build ") + cl_errstr(err) + ": " + log; cl.clReleaseProgram(p); return nullptr; }
    return p;
}

}  // namespace npg
