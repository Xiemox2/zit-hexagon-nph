// NPG bench — Phase-2 measurements for the hybrid NPU (Hexagon HMX via ggml-hexagon) + GPU (Adreno OpenCL)
// engine of ZIT-HEXAGON. Copyright (c) 2026 ZIT-HEXAGON authors. MIT License.
// Uses ggml (MIT) from the happyyzy stable-diffusion.cpp fork. Synthetic weights only.
#include "npg_common.h"
#include "npg_split.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <map>
#include <memory>
#include <thread>
#include <unistd.h>
#ifdef __ANDROID__
#include <android/log.h>
#include <jni.h>
#endif

#ifdef NPG_HEXAGON
extern "C" {
void * rpcmem_alloc2(int heapid, uint32_t flags, size_t size);
void   rpcmem_free(void * po);
int    rpcmem_to_fd(void * po);
}
#endif

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

// Qualcomm OpenCL extension constants (public cl_ext_qcom.h values)
#define NPG_CL_MEM_EXT_HOST_PTR_QCOM          (1u << 29)
#define NPG_CL_MEM_HOST_UNCACHED_QCOM         0x40A4
#define NPG_CL_MEM_HOST_WRITEBACK_QCOM        0x40A5
#define NPG_CL_MEM_HOST_IOCOHERENT_QCOM       0x40A9
#define NPG_CL_MEM_ION_HOST_PTR_QCOM          0x40A8
#define NPG_CL_MEM_DMABUF_HOST_PTR_QCOM       0x411D
#define NPG_CL_DEVICE_EXT_MEM_PADDING_QCOM    0x40A0
#define NPG_CL_DEVICE_PAGE_SIZE_QCOM          0x40A1
#define NPG_CL_CONTEXT_PERF_HINT_QCOM         0x40C2
#define NPG_CL_PERF_HINT_HIGH_QCOM            0x40C3

struct npg_cl_mem_ext_host_ptr { cl_uint allocation_type; cl_uint host_cache_policy; };
struct npg_cl_mem_dmabuf_host_ptr { npg_cl_mem_ext_host_ptr ext_host_ptr; int dmabuf_filedesc; void * dmabuf_hostptr; };
struct npg_cl_mem_ion_host_ptr { npg_cl_mem_ext_host_ptr ext_host_ptr; int ion_filedesc; void * ion_hostptr; };

namespace npg {

std::atomic<bool> g_stop{ false };
static std::mutex g_mu;
static std::string g_rep, g_file;
static std::function<void(const std::string &, bool)> g_sink;
static double g_t0 = 0;

static void emit(const std::string & s, bool prog) {
    if (!prog) {
        std::lock_guard<std::mutex> lk(g_mu);
        g_rep += s; g_rep += '\n';
    }
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_file.empty()) { FILE * f = fopen(g_file.c_str(), "a"); if (f) { if (prog) fputs("  .. ", f); fputs(s.c_str(), f); fputc('\n', f); fclose(f); } }
    }
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "npg", "%s", s.c_str());
#else
    fprintf(stderr, "%s%s\n", prog ? "  .. " : "", s.c_str());
#endif
    if (g_sink) g_sink(s, prog);
}
void P(const char * fmt, ...) {
    char buf[2048]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    emit(buf, false);
}
void progress(const char * fmt, ...) {
    char buf[1024]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    emit(buf, true);
}
static double tsec() { return (now_ms() - g_t0) / 1e3; }

// T8: ggml-hexagon "profile-batch" INFO lines (GGML_HEXAGON_PROFILE=1) -> per-submission host gaps
struct ProfAcc {
    std::mutex mu;
    int    batches = 0, ops = 0;
    double dsp_us = 0, wait_us = 0, max_wait_us = 0, max_dsp_us = 0;
    void reset() { std::lock_guard<std::mutex> lk(mu); batches = ops = 0; dsp_us = wait_us = max_wait_us = max_dsp_us = 0; }
};
static ProfAcc g_prof;
static std::atomic<bool> g_prof_on{ false };
static void prof_parse(const char * text) {
    const char * p = strstr(text, "profile-batch n=");
    if (!p) return;
    unsigned n = 0, dsp = 0; long long wait = 0;
    if (sscanf(p, "profile-batch n=%u dsp_us=%u host_wait_us=%lld", &n, &dsp, &wait) != 3) return;
    std::lock_guard<std::mutex> lk(g_prof.mu);
    g_prof.batches++; g_prof.ops += (int) n; g_prof.dsp_us += dsp; g_prof.wait_us += (double) wait;
    g_prof.max_wait_us = std::max(g_prof.max_wait_us, (double) wait); g_prof.max_dsp_us = std::max(g_prof.max_dsp_us, (double) dsp);
}

static void ggml_log_cb(enum ggml_log_level level, const char * text, void *) {
    if (g_prof_on && text) prof_parse(text);
    if (level >= GGML_LOG_LEVEL_WARN) {
        std::string s(text);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (!s.empty()) P("  [ggml] %s", s.c_str());
    }
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_DEBUG, "npg-ggml", "%s", text);
#endif
}

// ---------------------------------------------------------------- options / shapes
struct Opts {
    bool quick = false;
    int  sustain_s = 60;
    int  fs_small = 8;
    int  fs_big = 30;
    bool fs_big_on = true;
    int  cpu_threads = 0;
};

static const int M_TOK = 4608;   // 1024x1024 -> 64x64 latent patches (4096) + 512 text tokens
static const int HID = 3840, FFN = 10240, NH = 30, HD = 128;
struct LinShape { const char * name; int K, N, per_layer; };
static const LinShape kLin[] = { { "qkv", HID, 3 * HID, 1 }, { "out", HID, HID, 1 }, { "w1|w3", HID, FFN, 2 }, { "w2", FFN, HID, 1 } };

static double gflop_mm(double M, double N, double K) { return 2.0 * M * N * K * 1e-9; }

static double median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// time f(): one warm-up (not counted unless warm=false), then until both >= min_iter and >= min_ms
static double time_med(const std::function<bool()> & f, int min_iter = 3, double min_ms = 1500, int max_iter = 40, bool warm = true,
                       double * out_min = nullptr) {
    if (warm && !f()) return NAN;
    std::vector<double> ts;
    double t_start = now_ms();
    while (((int) ts.size() < min_iter || now_ms() - t_start < min_ms) && (int) ts.size() < max_iter && !g_stop) {
        double t = now_ms();
        if (!f()) return NAN;
        ts.push_back(now_ms() - t);
    }
    if (out_min) *out_min = ts.empty() ? NAN : *std::min_element(ts.begin(), ts.end());
    return median(ts);
}

// ---------------------------------------------------------------- system readers
static std::string read_line(const std::string & path) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) return "";
    char buf[256] = { 0 };
    if (!fgets(buf, sizeof buf, f)) buf[0] = 0;
    fclose(f);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}
static long mem_available_mb() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256]; long v = -1;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %ld kB", &v) == 1) break;
    fclose(f);
    return v < 0 ? -1 : v / 1024;
}

// ---------------------------------------------------------------- crash-proof stage log (0.1.2)
// Every stage is appended to the report file with fsync before the stage runs, and remembered for the
// SIGSEGV/SIGBUS/SIGABRT handler, so a crash shows exactly where the process died.
static char g_stage[192] = "-";
static int  g_crash_fd = -1;
static void stage(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
static void stage(const char * fmt, ...) {
    char buf[192]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    for (size_t i = 0; i < sizeof g_stage; ++i) { g_stage[i] = buf[i]; if (!buf[i]) break; }
    g_stage[sizeof g_stage - 1] = 0;
    char line[320];
    int n = snprintf(line, sizeof line, "  >> etapa: %s (t=%.1f s, MemAvailable %ld MB)\n", buf, tsec(), mem_available_mb());
    if (n < 0) return;
    if (n >= (int) sizeof line) n = (int) sizeof line - 1;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_file.empty()) {
            int fd = open(g_file.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
            if (fd >= 0) { ssize_t w = write(fd, line, (size_t) n); (void) w; fsync(fd); close(fd); }
        }
    }
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "npg", "%s", line);
#else
    fputs(line, stderr);
#endif
    if (g_sink) g_sink(std::string("etapa: ") + buf, true);
}

static void sa_put(char *& p, char * e, const char * s) { while (s && *s && p < e) *p++ = *s++; }
static void sa_hex(char *& p, char * e, uintptr_t v) {
    char t[2 + 16]; int n = 0;
    do { int d = (int) (v & 15); t[n++] = (char) (d < 10 ? '0' + d : 'a' + d - 10); v >>= 4; } while (v && n < 16);
    sa_put(p, e, "0x"); while (n > 0 && p < e) *p++ = t[--n];
}
static const char * sig_name(int s) {
    switch (s) { case SIGSEGV: return "SIGSEGV"; case SIGBUS: return "SIGBUS"; case SIGABRT: return "SIGABRT";
                 case SIGFPE: return "SIGFPE"; case SIGILL: return "SIGILL"; default: return "senal"; }
}
static struct sigaction g_old_sa[65];
static void crash_handler(int sig, siginfo_t * si, void *) {
    if (g_crash_fd >= 0) {
        char buf[400]; char * p = buf; char * e = buf + sizeof buf - 2;
        sa_put(p, e, "** CIERRE por "); sa_put(p, e, sig_name(sig));
        if (si && (sig == SIGSEGV || sig == SIGBUS)) { sa_put(p, e, " (dir "); sa_hex(p, e, (uintptr_t) si->si_addr); sa_put(p, e, ")"); }
        sa_put(p, e, " durante la etapa: "); sa_put(p, e, g_stage); sa_put(p, e, " -- la app se cierra\n");
        ssize_t w = write(g_crash_fd, buf, (size_t) (p - buf)); (void) w;
        fsync(g_crash_fd);
    }
    // hand over to the previous handler (debuggerd tombstone): restore it and re-deliver
    sigaction(sig, &g_old_sa[sig], nullptr);
    if (sig == SIGABRT || !si || si->si_code <= 0) raise(sig);   // a returning fault re-triggers by itself
}
static void install_crash_handlers(const std::string & path) {
    if (g_crash_fd >= 0) { close(g_crash_fd); g_crash_fd = -1; }
    if (!path.empty()) g_crash_fd = open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
    static bool installed = false;
    if (installed) return;
    installed = true;
    static std::vector<uint8_t> alt(64 * 1024);
    stack_t ss = {}; ss.ss_sp = alt.data(); ss.ss_size = alt.size(); ss.ss_flags = 0;
    sigaltstack(&ss, nullptr);   // this (run) thread
    for (int sig : { SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL }) {
        struct sigaction sa = {};
        sa.sa_sigaction = crash_handler;
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&sa.sa_mask);
        sigaction(sig, &sa, &g_old_sa[sig]);
    }
}

struct Zone { std::string type, path; };
static std::vector<Zone> g_zones;
static bool g_zones_scanned = false;
static void scan_zones() {
    if (g_zones_scanned) return;
    g_zones_scanned = true;
    DIR * d = opendir("/sys/class/thermal");
    if (!d) return;
    struct dirent * e;
    std::vector<Zone> all;
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, "thermal_zone", 12)) continue;
        std::string base = std::string("/sys/class/thermal/") + e->d_name;
        std::string t = read_line(base + "/type");
        if (t.empty()) continue;
        if (read_line(base + "/temp").empty()) continue;
        all.push_back({ t, base + "/temp" });
    }
    closedir(d);
    const char * keys[] = { "cpu", "gpu", "nsp", "cdsp", "q6", "ddr", "skin", "battery", "therm", "soc" };
    for (auto & z : all) {
        std::string lt = z.type;
        std::transform(lt.begin(), lt.end(), lt.begin(), ::tolower);
        for (auto k : keys) if (lt.find(k) != std::string::npos) { g_zones.push_back(z); break; }
    }
    if (g_zones.empty()) g_zones = all;
}
// summary: max temp per group
static std::string thermal_summary() {
    scan_zones();
    if (g_zones.empty()) return "temps: no legibles";
    std::map<std::string, double> mx;
    for (auto & z : g_zones) {
        std::string v = read_line(z.path);
        if (v.empty()) continue;
        double t = atof(v.c_str());
        if (std::fabs(t) > 1000) t /= 1000.0;
        std::string lt = z.type, g;
        std::transform(lt.begin(), lt.end(), lt.begin(), ::tolower);
        if (lt.find("gpu") != std::string::npos) g = "gpu";
        else if (lt.find("nsp") != std::string::npos || lt.find("cdsp") != std::string::npos || lt.find("q6") != std::string::npos) g = "npu";
        else if (lt.find("cpu") != std::string::npos) g = "cpu";
        else if (lt.find("ddr") != std::string::npos) g = "ddr";
        else if (lt.find("skin") != std::string::npos) g = "skin";
        else if (lt.find("batt") != std::string::npos) g = "bat";
        else g = "otro";
        if (!mx.count(g) || t > mx[g]) mx[g] = t;
    }
    std::string s;
    char b[64];
    for (auto & kv : mx) { snprintf(b, sizeof b, "%s=%.1fC ", kv.first.c_str(), kv.second); s += b; }
    return s.empty() ? "temps: no legibles" : s;
}
// max temperature of the NPU zones (nsp/cdsp/q6), NAN if none readable
static double npu_temp_c() {
    scan_zones();
    double mx = NAN;
    for (auto & z : g_zones) {
        std::string lt = z.type;
        std::transform(lt.begin(), lt.end(), lt.begin(), ::tolower);
        if (lt.find("nsp") == std::string::npos && lt.find("cdsp") == std::string::npos && lt.find("q6") == std::string::npos) continue;
        std::string v = read_line(z.path);
        if (v.empty()) continue;
        double t = atof(v.c_str());
        if (std::fabs(t) > 1000) t /= 1000.0;
        if (std::isnan(mx) || t > mx) mx = t;
    }
    return mx;
}
static std::string clocks_summary() {
    std::string s;
    std::string g = read_line("/sys/class/kgsl/kgsl-3d0/gpuclk");
    if (g.empty()) g = read_line("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq");
    s += "gpu=" + (g.empty() ? std::string("?") : std::to_string(atol(g.c_str()) / 1000000) + "MHz");
    for (int p : { 0, 2, 5, 6, 7 }) {
        std::string c = read_line("/sys/devices/system/cpu/cpufreq/policy" + std::to_string(p) + "/scaling_cur_freq");
        if (!c.empty()) s += " cpu" + std::to_string(p) + "=" + std::to_string(atol(c.c_str()) / 1000) + "MHz";
    }
    return s;
}

}  // namespace npg

// ================================================================ GPU (OpenCL)
namespace npg {

static Gpu g_gpu;

static bool gpu_init() {
    std::string err;
    if (!cl_load(err)) { P("GPU: no se pudo cargar OpenCL: %s", err.c_str()); return false; }
    cl_uint np = 0;
    if (cl.clGetPlatformIDs(0, nullptr, &np) || !np) { P("GPU: sin plataformas OpenCL"); return false; }
    std::vector<cl_platform_id> ps(np);
    cl.clGetPlatformIDs(np, ps.data(), nullptr);
    for (auto p : ps) {
        cl_uint nd = 0;
        if (cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) == 0 && nd) {
            g_gpu.plat = p; cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &g_gpu.dev, nullptr); break;
        }
    }
    if (!g_gpu.dev) {  // host test (pocl) — accept any device
        for (auto p : ps) {
            cl_uint nd = 0;
            if (cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) == 0 && nd) {
                g_gpu.plat = p; cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 1, &g_gpu.dev, nullptr); break;
            }
        }
    }
    if (!g_gpu.dev) { P("GPU: sin dispositivo"); return false; }
    auto sinfo = [](cl_device_info i) {
        size_t n = 0; cl.clGetDeviceInfo(g_gpu.dev, i, 0, nullptr, &n);
        std::string s(n, 0); if (n) cl.clGetDeviceInfo(g_gpu.dev, i, n, &s[0], nullptr);
        while (!s.empty() && s.back() == 0) s.pop_back();
        return s;
    };
    g_gpu.name = sinfo(CL_DEVICE_NAME); g_gpu.version = sinfo(CL_DEVICE_VERSION); g_gpu.driver = sinfo(CL_DRIVER_VERSION);
    g_gpu.exts = sinfo(CL_DEVICE_EXTENSIONS);
    cl_uint cu = 0, clk = 0;
    cl.clGetDeviceInfo(g_gpu.dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, nullptr);
    cl.clGetDeviceInfo(g_gpu.dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof clk, &clk, nullptr);
    cl.clGetDeviceInfo(g_gpu.dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof g_gpu.local_mem, &g_gpu.local_mem, nullptr);
    cl.clGetDeviceInfo(g_gpu.dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof g_gpu.global_mem, &g_gpu.global_mem, nullptr);
    cl.clGetDeviceInfo(g_gpu.dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof g_gpu.max_wg, &g_gpu.max_wg, nullptr);
    auto has = [](const char * e) { return g_gpu.exts.find(e) != std::string::npos; };
    g_gpu.fp16 = has("cl_khr_fp16");
    g_gpu.dmabuf = has("cl_qcom_dmabuf_host_ptr");
    g_gpu.ion = has("cl_qcom_ion_host_ptr");
    g_gpu.ext_host = has("cl_qcom_ext_host_ptr");
    g_gpu.iocoh = has("cl_qcom_ext_host_ptr_iocoherent");
    g_gpu.ahb = has("cl_qcom_android_ahardwarebuffer_host_ptr");
    if (g_gpu.ext_host) {
        cl.clGetDeviceInfo(g_gpu.dev, NPG_CL_DEVICE_PAGE_SIZE_QCOM, sizeof g_gpu.page_size, &g_gpu.page_size, nullptr);
        cl.clGetDeviceInfo(g_gpu.dev, NPG_CL_DEVICE_EXT_MEM_PADDING_QCOM, sizeof g_gpu.ext_padding, &g_gpu.ext_padding, nullptr);
    }
    P("GPU: %s | %s | driver %s | CU %u | max clk %u MHz | local %llu KB | global %llu MB | max WG %zu",
      g_gpu.name.c_str(), g_gpu.version.c_str(), g_gpu.driver.c_str(), cu, clk, (unsigned long long) g_gpu.local_mem / 1024,
      (unsigned long long) g_gpu.global_mem >> 20, g_gpu.max_wg);
    P("GPU ext: fp16=%d dmabuf_host_ptr=%d ion_host_ptr=%d ext_host_ptr=%d iocoherent=%d ahb_host_ptr=%d perf_hint=%d recordable_q=%d subgroups=%d | page %u pad %u | lib %s",
      g_gpu.fp16, g_gpu.dmabuf, g_gpu.ion, g_gpu.ext_host, g_gpu.iocoh, g_gpu.ahb, has("cl_qcom_perf_hint"),
      has("cl_qcom_recordable_queues"), has("cl_khr_subgroups") || has("cl_qcom_subgroup"), g_gpu.page_size, g_gpu.ext_padding, cl.lib.c_str());
    {
        // list qcom extensions compactly
        std::string q;
        size_t p = 0;
        while (p < g_gpu.exts.size()) {
            size_t e = g_gpu.exts.find(' ', p); if (e == std::string::npos) e = g_gpu.exts.size();
            std::string x = g_gpu.exts.substr(p, e - p);
            if (x.find("qcom") != std::string::npos) q += x + " ";
            p = e + 1;
        }
        P("GPU ext qcom: %s", q.c_str());
    }
    cl_int err2;
    std::vector<cl_context_properties> props = { CL_CONTEXT_PLATFORM, (cl_context_properties) g_gpu.plat };
    if (has("cl_qcom_perf_hint")) { props.push_back(NPG_CL_CONTEXT_PERF_HINT_QCOM); props.push_back(NPG_CL_PERF_HINT_HIGH_QCOM); }
    props.push_back(0);
    g_gpu.ctx = cl.clCreateContext(props.data(), 1, &g_gpu.dev, nullptr, nullptr, &err2);
    if (!g_gpu.ctx) {
        cl_context_properties p2[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) g_gpu.plat, 0 };
        g_gpu.ctx = cl.clCreateContext(p2, 1, &g_gpu.dev, nullptr, nullptr, &err2);
    }
    if (!g_gpu.ctx) { P("GPU: clCreateContext %s", cl_errstr(err2)); return false; }
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    g_gpu.q = cl.clCreateCommandQueueWithProperties(g_gpu.ctx, g_gpu.dev, qp, &err2);
    if (!g_gpu.q) { P("GPU: queue %s", cl_errstr(err2)); return false; }
    g_gpu.build_opts_base = g_gpu.fp16 ? "-DHAS_FP16" : "";
    g_gpu.ok = true;
    return true;
}

static cl_mem gbuf(size_t bytes, cl_mem_flags fl = CL_MEM_READ_WRITE) {
    cl_int e;
    cl_mem m = cl.clCreateBuffer(g_gpu.ctx, fl, bytes, nullptr, &e);
    if (!m) P("  GPU alloc %zu MB fallo: %s", bytes >> 20, cl_errstr(e));
    return m;
}
static bool gwrite(cl_mem m, const void * src, size_t bytes, size_t off = 0) {
    return cl.clEnqueueWriteBuffer(g_gpu.q, m, CL_TRUE, off, bytes, src, 0, nullptr, nullptr) == 0;
}
static bool gread(cl_mem m, void * dst, size_t bytes, size_t off = 0) {
    return cl.clEnqueueReadBuffer(g_gpu.q, m, CL_TRUE, off, bytes, dst, 0, nullptr, nullptr) == 0;
}
struct MemGuard { std::vector<cl_mem> v; ~MemGuard() { for (auto m : v) if (m) cl.clReleaseMemObject(m); } cl_mem add(cl_mem m) { v.push_back(m); return m; } };

template <class T> static void setarg(cl_kernel k, int i, const T & v) { cl.clSetKernelArg(k, i, sizeof(T), &v); }

struct GemmVar {
    std::string name;
    cl_program  prog = nullptr;
    cl_kernel   k = nullptr;
    bool tiled = false;
    int tm = 4, tn = 4;
    size_t lx = 0, ly = 0;  // 0 = driver chooses
    double ms_qkv = NAN;
};
static std::map<std::string, cl_program> g_progs;
static cl_program get_prog(const std::string & defs) {
    auto it = g_progs.find(defs);
    if (it != g_progs.end()) return it->second;
    std::string log;
    double t = now_ms();
    cl_program p = g_gpu.build(defs, log);
    if (!p) P("  GPU build [%s] fallo: %.600s", defs.c_str(), log.c_str());
    else progress("compilado [%s] %.0f ms", defs.c_str(), now_ms() - t);
    g_progs[defs] = p;
    return p;
}
static cl_kernel get_kernel(const std::string & defs, const char * name) {
    cl_program p = get_prog(defs);
    if (!p) return nullptr;
    cl_int e;
    cl_kernel k = cl.clCreateKernel(p, name, &e);
    if (!k) P("  GPU kernel %s: %s", name, cl_errstr(e));
    return k;
}

// enqueue Y = alpha * X W^T (+Y) in row chunks; returns false on error
static int g_gemm_chunk = 512;
static bool gemm_enqueue(GemmVar & v, cl_mem X, cl_mem W, cl_mem Y, int M, int N, int K, float alpha, int acc) {
    setarg(v.k, 0, X); setarg(v.k, 1, W); setarg(v.k, 2, Y);
    setarg(v.k, 3, M); setarg(v.k, 4, N); setarg(v.k, 5, K); setarg(v.k, 6, alpha); setarg(v.k, 7, acc);
    const int rows_per_unit = v.tiled ? 16 * v.tm : v.tm;
    int chunk = std::max(rows_per_unit, (g_gemm_chunk / rows_per_unit) * rows_per_unit);
    for (int mb = 0; mb < M; mb += chunk) {
        int rows = std::min(chunk, M - mb);
        setarg(v.k, 8, mb);
        size_t g[2], l[2];
        cl_int e;
        if (v.tiled) {
            g[0] = (size_t) ((N + 16 * v.tn - 1) / (16 * v.tn)) * 16;
            g[1] = (size_t) ((rows + 16 * v.tm - 1) / (16 * v.tm)) * 16;
            l[0] = 16; l[1] = 16;
            e = cl.clEnqueueNDRangeKernel(g_gpu.q, v.k, 2, nullptr, g, l, 0, nullptr, nullptr);
        } else {
            g[0] = (size_t) (N / v.tn); g[1] = (size_t) (rows / v.tm);
            bool use_l = v.lx && (g[0] % v.lx == 0) && (g[1] % v.ly == 0);
            l[0] = v.lx; l[1] = v.ly;
            e = cl.clEnqueueNDRangeKernel(g_gpu.q, v.k, 2, nullptr, g, use_l ? l : nullptr, 0, nullptr, nullptr);
        }
        if (e) { P("  GPU gemm %s enqueue: %s", v.name.c_str(), cl_errstr(e)); return false; }
    }
    return true;
}
static bool gemm_run(GemmVar & v, cl_mem X, cl_mem W, cl_mem Y, int M, int N, int K, float alpha, int acc) {
    if (!gemm_enqueue(v, X, W, Y, M, N, K, alpha, acc)) return false;
    return cl.clFinish(g_gpu.q) == 0;
}

static GemmVar make_var(const char * wt, bool tiled, int tm, int tn, size_t lx = 0, size_t ly = 0) {
    GemmVar v;
    char defs[128];
    snprintf(defs, sizeof defs, "-D%s -DTM=%d -DTN=%d", wt, tm, tn);
    char nm[128];
    snprintf(nm, sizeof nm, "%s %s %dx%d%s", wt + 3, tiled ? "tiled" : "reg", tm, tn,
             lx ? (" wg" + std::to_string(lx) + "x" + std::to_string(ly)).c_str() : "");
    v.name = nm; v.tiled = tiled; v.tm = tm; v.tn = tn; v.lx = lx; v.ly = ly;
    v.k = get_kernel(defs, tiled ? "gemm_tiled" : "gemm");
    return v;
}

}  // namespace npg

// ================================================================ ggml backends
namespace npg {

struct GB {
    ggml_backend_t             be = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    std::string                name;
    bool                       htp = false;
    bool ok() const { return be != nullptr; }
};
static GB g_htp, g_cpu;
static double g_htp_init_ms = NAN;

static bool htp_init() {
#ifdef NPG_HEXAGON
    double t = now_ms();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HTP0");
    double t_reg = now_ms() - t;
    if (!dev) { P("HTP: dispositivo HTP0 no encontrado (registro %.0f ms)", t_reg); return false; }
    double t1 = now_ms();
    g_htp.be = ggml_backend_dev_init(dev, nullptr);
    double t_init = now_ms() - t1;
    if (!g_htp.be) { P("HTP: ggml_backend_dev_init fallo"); return false; }
    g_htp.buft = ggml_backend_get_default_buffer_type(g_htp.be);
    g_htp.name = "HTP0"; g_htp.htp = true;
    size_t fr = 0, tot = 0;
    ggml_backend_dev_memory(dev, &fr, &tot);
    g_htp_init_ms = t_reg + t_init;
    P("HTP: %s | registro+carga skel %.0f ms | sesion FastRPC/dev_init %.0f ms | mem %zu/%zu MB | buft %s max %zu MB",
      ggml_backend_dev_description(dev), t_reg, t_init, fr >> 20, tot >> 20, ggml_backend_buft_name(g_htp.buft),
      ggml_backend_buft_get_max_size(g_htp.buft) >> 20);
    return true;
#else
    P("HTP: no compilado (build host)");
    return false;
#endif
}

static bool cpu_init(int threads) {
    g_cpu.be = ggml_backend_cpu_init();
    if (!g_cpu.be) return false;
    ggml_backend_cpu_set_n_threads(g_cpu.be, threads);
    g_cpu.buft = ggml_backend_get_default_buffer_type(g_cpu.be);
    g_cpu.name = "CPU";
    return true;
}

// a context of tensors + its backend buffer
struct TSet {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    explicit TSet(int n = 64) {
        ggml_init_params p = { (size_t) n * ggml_tensor_overhead() + 1024, nullptr, true };
        ctx = ggml_init(p);
    }
    ~TSet() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); }
    bool alloc(GB & b, bool weights = false) {
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, b.buft);
        if (!buf) { P("  alloc tensores en %s fallo", b.name.c_str()); return false; }
        if (weights) ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        return true;
    }
    TSet(const TSet &) = delete;
};

struct GGraph {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf = nullptr;
    ggml_gallocr_t  ga = nullptr;
    ggml_tensor *   out = nullptr;
    explicit GGraph(int nodes = 2048) {
        ggml_init_params p = { (size_t) nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false) + 4096, nullptr, true };
        ctx = ggml_init(p);
        gf = ggml_new_graph_custom(ctx, nodes, false);
    }
    ~GGraph() { if (ga) ggml_gallocr_free(ga); if (ctx) ggml_free(ctx); }
    GGraph(const GGraph &) = delete;
};

static bool op_is_view(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: return true;
        default: return false;
    }
}
static bool graph_supported(GB & b, ggml_cgraph * gf, std::string & bad) {
    bool ok = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * n = ggml_graph_node(gf, i);
        if (op_is_view(n)) continue;
        if (!ggml_backend_supports_op(b.be, n)) {
            ok = false;
            char tmp[160];
            snprintf(tmp, sizeof tmp, "%s(%s %s [%lld,%lld,%lld]) ", ggml_op_desc(n), ggml_type_name(n->src[0] ? n->src[0]->type : n->type),
                     n->src[1] ? ggml_type_name(n->src[1]->type) : "-", (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2]);
            if (bad.find(tmp) == std::string::npos) bad += tmp;
        }
    }
    return ok;
}
// finalize graph: expand, check support, allocate compute buffers
static bool graph_prepare(GB & b, GGraph & g, ggml_tensor * out, const char * what, double * t_alloc = nullptr) {
    g.out = out;
    ggml_build_forward_expand(g.gf, out);
    std::string bad;
    if (!graph_supported(b, g.gf, bad)) { P("  %s: ops no soportadas en %s: %s", what, b.name.c_str(), bad.c_str()); return false; }
    double t = now_ms();
    g.ga = ggml_gallocr_new(b.buft);
    if (!ggml_gallocr_alloc_graph(g.ga, g.gf)) { P("  %s: gallocr fallo en %s", what, b.name.c_str()); return false; }
    if (t_alloc) *t_alloc = now_ms() - t;
    return true;
}
static bool graph_run(GB & b, GGraph & g) {
    if (ggml_backend_graph_compute(b.be, g.gf) != GGML_STATUS_SUCCESS) { P("  compute fallo en %s", b.name.c_str()); return false; }
    ggml_backend_synchronize(b.be);
    return true;
}

static void set_rand_f32(ggml_tensor * t, uint64_t seed, float sd = 1.0f, std::vector<float> * keep = nullptr) {
    std::vector<float> h(ggml_nelements(t));
    fill_gauss(h.data(), h.size(), seed, sd);
    ggml_backend_tensor_set(t, h.data(), 0, h.size() * 4);
    if (keep) *keep = std::move(h);
}
static void set_const_f32(ggml_tensor * t, float v) {
    std::vector<float> h(ggml_nelements(t), v);
    ggml_backend_tensor_set(t, h.data(), 0, h.size() * 4);
}
static void set_f16_from(ggml_tensor * t, const std::vector<float> & src) {
    std::vector<uint16_t> h(src.size());
    for (size_t i = 0; i < src.size(); ++i) h[i] = f32_to_f16(src[i]);
    ggml_backend_tensor_set(t, h.data(), 0, h.size() * 2);
}

// error stats of sampled outputs vs double reference
struct ErrAcc {
    double se = 0, sr = 0, maxrel = 0; long n = 0; bool nan = false;
    void add(double y, double ref) {
        if (!(y == y) || std::isinf(y)) nan = true;
        se += (y - ref) * (y - ref); sr += ref * ref; ++n;
    }
    double rel() const { return sr > 0 ? std::sqrt(se / sr) : NAN; }
};

}  // namespace npg

// ================================================================ T1 matmul
namespace npg {

struct HostData {
    std::vector<float>   X;       // M_TOK x HID gaussian
    std::vector<float>   X2;      // M_TOK x FFN gaussian (input of w2)
    std::vector<uint8_t> W[4];    // FP8 weights per kLin shape (N x K)
    float scale[4] = { 0.25f, 0.2f, 0.3f, 0.15f };
};
static HostData g_hd;

struct MMGraph {
    std::unique_ptr<TSet>   w, in;
    std::unique_ptr<GGraph> g;
    ggml_tensor *W = nullptr, *X = nullptr, *s = nullptr;
    int M = 0, N = 0, K = 0;
};

// y[N, M] = (W[K,N] x X[K,M]) * s ; W FP8 on HTP, F32 (exact dequant) on CPU
static std::unique_ptr<MMGraph> build_mm(GB & b, int si, int M) {
    auto r = std::make_unique<MMGraph>();
    const LinShape & sh = kLin[si];
    r->M = M; r->N = sh.N; r->K = sh.K;
    r->w = std::make_unique<TSet>(4);
    r->in = std::make_unique<TSet>(4);
    ggml_type wt = b.htp ? GGML_TYPE_F8_E4M3 : GGML_TYPE_F32;
    r->W = ggml_new_tensor_2d(r->w->ctx, wt, sh.K, sh.N);
    r->X = ggml_new_tensor_2d(r->in->ctx, GGML_TYPE_F32, sh.K, M);
    r->s = ggml_new_tensor_1d(r->in->ctx, GGML_TYPE_F32, 1);
    if (!r->w->alloc(b, true) || !r->in->alloc(b)) return nullptr;
    const auto & W8 = g_hd.W[si];
    if (b.htp) ggml_backend_tensor_set(r->W, W8.data(), 0, W8.size());
    else {
        std::vector<float> f(W8.size());
        for (size_t i = 0; i < f.size(); ++i) f[i] = g_e4m3_lut[W8[i]];
        ggml_backend_tensor_set(r->W, f.data(), 0, f.size() * 4);
    }
    const auto & X = (sh.K == HID) ? g_hd.X : g_hd.X2;
    ggml_backend_tensor_set(r->X, X.data(), 0, (size_t) M * sh.K * 4);
    ggml_backend_tensor_set(r->s, &g_hd.scale[si], 0, 4);
    r->g = std::make_unique<GGraph>(16);
    ggml_tensor * y = ggml_mul(r->g->ctx, ggml_mul_mat(r->g->ctx, r->W, r->X), r->s);
    if (!graph_prepare(b, *r->g, y, sh.name)) return nullptr;
    return r;
}

// check sampled outputs: y is row-major M x N floats (from fetch(m, row))
static ErrAcc check_mm(int si, int M, const std::function<bool(int, std::vector<float> &)> & fetch, double extra_scale = 1.0) {
    const LinShape & sh = kLin[si];
    const auto & X = (sh.K == HID) ? g_hd.X : g_hd.X2;
    const auto & W8 = g_hd.W[si];
    ErrAcc acc;
    std::vector<float> row(sh.N);
    int rows[] = { 0, 1, M / 3, M / 2, M - 1 };
    Rng r(99 + si);
    for (int m : rows) {
        if (m >= M || !fetch(m, row)) continue;
        for (int c = 0; c < 48; ++c) {
            int n = (int) (r.next() % sh.N);
            double ref = 0;
            const float * x = &X[(size_t) m * sh.K];
            const uint8_t * w = &W8[(size_t) n * sh.K];
            for (int k = 0; k < sh.K; ++k) ref += (double) x[k] * g_e4m3_lut[w[k]];
            ref *= g_hd.scale[si];
            acc.add(row[n] * extra_scale, ref);
        }
    }
    return acc;
}

struct T1Res {
    double htp_ms[4] = { NAN, NAN, NAN, NAN }, gpu_ms[4] = { NAN, NAN, NAN, NAN }, gpu16_ms[4] = { NAN, NAN, NAN, NAN }, cpu_ms[4] = { NAN, NAN, NAN, NAN };
    double htp_layer = NAN, gpu_layer = NAN, cpu_layer = NAN;
};
static T1Res g_t1;
static std::unique_ptr<MMGraph> g_htp_qkv;     // kept for concurrency tests
static GemmVar g_best8, g_best16;
static cl_mem g_gX = nullptr, g_gX2 = nullptr, g_gY = nullptr, g_gW8[4] = { nullptr, nullptr, nullptr, nullptr };

static void prepare_host_data() {
    if (!g_hd.X.empty()) return;
    double t = now_ms();
    g_hd.X.resize((size_t) M_TOK * HID);
    g_hd.X2.resize((size_t) M_TOK * FFN);
    fill_gauss(g_hd.X.data(), g_hd.X.size(), 1);
    fill_gauss(g_hd.X2.data(), g_hd.X2.size(), 2);
    for (int i = 0; i < 4; ++i) {
        g_hd.W[i].resize((size_t) kLin[i].K * kLin[i].N);
        fill_fp8_weights(g_hd.W[i].data(), g_hd.W[i].size(), 10 + i);
    }
    progress("datos sinteticos listos %.0f ms", now_ms() - t);
}

static void t1_htp() {
    P("-- T1a matmul FP8 en HTP/HMX (ggml-hexagon, como NPH): y = (W8 x X) * escala, M=%d", M_TOK);
    double layer = 0;
    for (int si = 0; si < 4 && !g_stop; ++si) {
        const LinShape & sh = kLin[si];
        progress("HTP matmul %s", sh.name);
        auto mm = build_mm(g_htp, si, M_TOK);
        if (!mm) { layer = NAN; continue; }
        double t_first = now_ms();
        bool ok = graph_run(g_htp, *mm->g);
        t_first = now_ms() - t_first;
        double tmin;
        double ms = ok ? time_med([&] { return graph_run(g_htp, *mm->g); }, 4, 1500, 30, false, &tmin) : NAN;
        ErrAcc e = check_mm(si, M_TOK, [&](int m, std::vector<float> & row) {
            ggml_backend_tensor_get(mm->g->out, row.data(), (size_t) m * sh.N * 4, (size_t) sh.N * 4); return true; });
        double gf = gflop_mm(M_TOK, sh.N, sh.K);
        P("  HTP %-6s K=%5d N=%5d: %7.1f ms (min %.1f, 1a %.1f) = %6.0f GFLOP/s | err rel %.2e%s", sh.name, sh.K, sh.N, ms, tmin, t_first,
          gf / ms * 1e3, e.rel(), e.nan ? " NaN!" : "");
        g_t1.htp_ms[si] = ms;
        layer += ms * sh.per_layer;
        if (si == 0) g_htp_qkv = std::move(mm);
    }
    g_t1.htp_layer = layer;
    P("  HTP capa (qkv+out+w1+w3+w2) = %.0f ms -> x30 capas = %.2f s/paso (solo linears)", layer, layer * 30 / 1e3);
}

static void t1_cpu(int threads) {
    const int Mc = 512;
    P("-- T1c matmul en CPU (ggml CPU, %d hilos, pesos F32 exactos, M=%d extrapolado x%.0f)", threads, Mc, (double) M_TOK / Mc);
    double layer = 0;
    for (int si = 0; si < 4 && !g_stop; ++si) {
        const LinShape & sh = kLin[si];
        progress("CPU matmul %s", sh.name);
        auto mm = build_mm(g_cpu, si, Mc);
        if (!mm) { layer = NAN; continue; }
        double ms = time_med([&] { return graph_run(g_cpu, *mm->g); }, 2, 800, 10);
        ErrAcc e = check_mm(si, Mc, [&](int m, std::vector<float> & row) {
            ggml_backend_tensor_get(mm->g->out, row.data(), (size_t) m * sh.N * 4, (size_t) sh.N * 4); return true; });
        double full = ms * M_TOK / Mc;
        P("  CPU %-6s: %7.1f ms (M=%d) -> %7.0f ms a M=%d = %5.0f GFLOP/s | err rel %.2e", sh.name, ms, Mc, full, M_TOK,
          gflop_mm(Mc, sh.N, sh.K) / ms * 1e3, e.rel());
        g_t1.cpu_ms[si] = full;
        layer += full * sh.per_layer;
    }
    g_t1.cpu_layer = layer;
    P("  CPU capa = %.0f ms (extrapolado)", layer);
}

static void t1_gpu() {
    P("-- T1b matmul en GPU (OpenCL propio: FP8->FP16 exacto en registro, acumulacion FP32), M=%d", M_TOK);
    MemGuard mg;
    g_gX = gbuf((size_t) M_TOK * HID * 4);
    g_gX2 = gbuf((size_t) M_TOK * FFN * 4);
    g_gY = gbuf((size_t) M_TOK * 3 * HID * 4);
    if (!g_gX || !g_gX2 || !g_gY) return;
    gwrite(g_gX, g_hd.X.data(), g_hd.X.size() * 4);
    gwrite(g_gX2, g_hd.X2.data(), g_hd.X2.size() * 4);
    for (int si = 0; si < 4; ++si) {
        g_gW8[si] = gbuf(g_hd.W[si].size(), CL_MEM_READ_ONLY);
        if (!g_gW8[si]) return;
        gwrite(g_gW8[si], g_hd.W[si].data(), g_hd.W[si].size());
    }
    // F16 pre-dequantised copy of qkv for the "dequant-on-load" variant
    std::vector<uint16_t> w16(g_hd.W[0].size());
    for (size_t i = 0; i < w16.size(); ++i) w16[i] = f32_to_f16(g_e4m3_lut[g_hd.W[0][i]]);
    cl_mem gW16 = mg.add(gbuf(w16.size() * 2, CL_MEM_READ_ONLY));
    if (!gW16) return;
    gwrite(gW16, w16.data(), w16.size() * 2);
    w16.clear(); w16.shrink_to_fit();

    const LinShape & q = kLin[0];
    const int Mt = 1152;
    // safety probe: 128 rows with the simplest kernel
    GemmVar probe = make_var("WT_F8", false, 4, 4);
    if (!probe.k) { P("  GPU: kernel gemm no compila; se omite T1b"); return; }
    {
        int saved = g_gemm_chunk; g_gemm_chunk = 128;
        double t = now_ms();
        bool ok = gemm_run(probe, g_gX, g_gW8[0], g_gY, 128, q.N, q.K, g_hd.scale[0] * 256.f, 0);
        t = now_ms() - t;
        g_gemm_chunk = saved;
        P("  sonda 128 filas qkv: %.1f ms (%s)", t, ok ? "ok" : "FALLO");
        if (!ok) return;
        if (t > 400) g_gemm_chunk = 256;   // keep each dispatch short (GPU watchdog)
    }
    std::vector<GemmVar> vars;
    vars.push_back(make_var("WT_F8", false, 4, 4));
    vars.push_back(make_var("WT_F8", false, 4, 4, 32, 4));
    vars.push_back(make_var("WT_F8", false, 4, 4, 64, 2));
    vars.push_back(make_var("WT_F8", false, 8, 4));
    vars.push_back(make_var("WT_F8", false, 4, 8));
    vars.push_back(make_var("WT_F8", false, 8, 8));
    vars.push_back(make_var("WT_F8", true, 4, 4));
    vars.push_back(make_var("WT_F8", true, 8, 4));
    vars.push_back(make_var("WT_F16", false, 4, 4));
    vars.push_back(make_var("WT_F16", false, 8, 4));
    vars.push_back(make_var("WT_F16", false, 4, 8));
    vars.push_back(make_var("WT_F16", true, 4, 4));
    vars.push_back(make_var("WT_F16", true, 8, 4));
    P("  autotune en qkv con M=%d (min de 2):", Mt);
    for (auto & v : vars) {
        if (!v.k || g_stop) continue;
        bool f16 = v.name.rfind("F16", 0) == 0;
        cl_mem W = f16 ? gW16 : g_gW8[0];
        float alpha = g_hd.scale[0] * (f16 ? 1.f : 256.f);
        double tmin = NAN;
        if (!gemm_run(v, g_gX, W, g_gY, Mt, q.N, q.K, alpha, 0)) continue;
        for (int r = 0; r < 2; ++r) {
            double t = now_ms();
            if (!gemm_run(v, g_gX, W, g_gY, Mt, q.N, q.K, alpha, 0)) break;
            t = now_ms() - t;
            tmin = std::isnan(tmin) ? t : std::min(tmin, t);
        }
        v.ms_qkv = tmin;
        P("    %-22s %8.1f ms = %6.0f GFLOP/s", v.name.c_str(), tmin, gflop_mm(Mt, q.N, q.K) / tmin * 1e3);
        if (!f16 && (std::isnan(g_best8.ms_qkv) || tmin < g_best8.ms_qkv)) g_best8 = v;
        if (f16 && (std::isnan(g_best16.ms_qkv) || tmin < g_best16.ms_qkv)) g_best16 = v;
    }
    if (!g_best8.k) { P("  GPU: ninguna variante FP8 funciono"); return; }
    P("  mejor FP8: %s | mejor FP16 (pre-dequant): %s", g_best8.name.c_str(), g_best16.k ? g_best16.name.c_str() : "-");
    double layer = 0;
    for (int si = 0; si < 4 && !g_stop; ++si) {
        const LinShape & sh = kLin[si];
        cl_mem X = sh.K == HID ? g_gX : g_gX2;
        float alpha = g_hd.scale[si] * 256.f;
        double ms = time_med([&] { return gemm_run(g_best8, X, g_gW8[si], g_gY, M_TOK, sh.N, sh.K, alpha, 0); }, 2, 1000, 6);
        ErrAcc e = check_mm(si, M_TOK, [&](int m, std::vector<float> & row) { return gread(g_gY, row.data(), (size_t) sh.N * 4, (size_t) m * sh.N * 4); });
        P("  GPU %-6s FP8: %7.1f ms = %5.0f GFLOP/s | err rel %.2e%s | ratio GPU/HTP = %.1f%%", sh.name, ms, gflop_mm(M_TOK, sh.N, sh.K) / ms * 1e3,
          e.rel(), e.nan ? " NaN!" : "", g_t1.htp_ms[si] / ms * 100);
        g_t1.gpu_ms[si] = ms;
        layer += ms * sh.per_layer;
    }
    if (g_best16.k && !g_stop) {
        double ms = time_med([&] { return gemm_run(g_best16, g_gX, gW16, g_gY, M_TOK, q.N, q.K, g_hd.scale[0], 0); }, 2, 1000, 6);
        ErrAcc e = check_mm(0, M_TOK, [&](int m, std::vector<float> & row) { return gread(g_gY, row.data(), (size_t) q.N * 4, (size_t) m * q.N * 4); });
        P("  GPU qkv FP16 pre-dequant (2x memoria): %7.1f ms = %5.0f GFLOP/s | err rel %.2e", ms, gflop_mm(M_TOK, q.N, q.K) / ms * 1e3, e.rel());
        g_t1.gpu16_ms[0] = ms;
    }
    g_t1.gpu_layer = layer;
    P("  GPU capa (FP8) = %.0f ms | throughput GPU/HMX = %.1f%% (criterio go: >= 8%%)", layer, g_t1.htp_layer / layer * 100);
}

}  // namespace npg

// ================================================================ T2 flash attention
namespace npg {

struct FAData { std::vector<float> Q; std::vector<uint16_t> K, V; };  // [H][S][D]
static FAData g_fa;
struct FAGraph { std::unique_ptr<TSet> in; std::unique_ptr<GGraph> g; int H = 0; };
static std::unique_ptr<FAGraph> g_htp_fa;
static cl_mem g_gQ = nullptr, g_gK = nullptr, g_gV = nullptr, g_gO = nullptr;
static cl_kernel g_kfa = nullptr;
static double g_fa_htp_ms = NAN, g_fa_gpu_ms = NAN;

static void fa_ref(int h, int qi, int S, std::vector<double> & o) {
    const float * q = &g_fa.Q[((size_t) h * S + qi) * HD];
    std::vector<double> s(S);
    double mx = -1e300, scale = 1.0 / std::sqrt((double) HD);
    for (int j = 0; j < S; ++j) {
        const uint16_t * k = &g_fa.K[((size_t) h * S + j) * HD];
        double d = 0;
        for (int c = 0; c < HD; ++c) d += (double) q[c] * f16_to_f32(k[c]);
        s[j] = d * scale; mx = std::max(mx, s[j]);
    }
    double l = 0;
    for (auto & v : s) { v = std::exp(v - mx); l += v; }
    o.assign(HD, 0.0);
    for (int j = 0; j < S; ++j) {
        const uint16_t * v = &g_fa.V[((size_t) h * S + j) * HD];
        for (int c = 0; c < HD; ++c) o[c] += s[j] * f16_to_f32(v[c]);
    }
    for (auto & v : o) v /= l;
}

static std::unique_ptr<FAGraph> build_fa(GB & b, int H) {
    const int S = M_TOK;
    auto r = std::make_unique<FAGraph>();
    r->H = H;
    r->in = std::make_unique<TSet>(4);
    ggml_tensor * q = ggml_new_tensor_3d(r->in->ctx, GGML_TYPE_F32, HD, S, H);
    ggml_tensor * k = ggml_new_tensor_3d(r->in->ctx, GGML_TYPE_F16, HD, S, H);
    ggml_tensor * v = ggml_new_tensor_3d(r->in->ctx, GGML_TYPE_F16, HD, S, H);
    if (!r->in->alloc(b)) return nullptr;
    ggml_backend_tensor_set(q, g_fa.Q.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, g_fa.K.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, g_fa.V.data(), 0, ggml_nbytes(v));
    r->g = std::make_unique<GGraph>(16);
    ggml_tensor * o = ggml_flash_attn_ext(r->g->ctx, q, k, v, nullptr, 1.0f / std::sqrt((float) HD), 0, 0);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    if (!graph_prepare(b, *r->g, o, "flash_attn_ext")) return nullptr;
    return r;
}
static ErrAcc check_fa_ggml(FAGraph & f) {
    ErrAcc e;
    std::vector<double> ref;
    std::vector<float> got(HD);
    int qs[] = { 0, 777, M_TOK - 1 };
    for (int h : { 0, f.H - 1 }) for (int qi : qs) {
        fa_ref(h, qi, M_TOK, ref);
        // output [D, H, S]
        ggml_backend_tensor_get(f.g->out, got.data(), ((size_t) qi * f.H + h) * HD * 4, HD * 4);
        for (int c = 0; c < HD; ++c) e.add(got[c], ref[c]);
    }
    return e;
}

static bool gpu_fa_run(int H) {
    const int S = M_TOK;
    float sc = 1.0f / std::sqrt((float) HD);
    setarg(g_kfa, 0, g_gQ); setarg(g_kfa, 1, g_gK); setarg(g_kfa, 2, g_gV); setarg(g_kfa, 3, g_gO);
    setarg(g_kfa, 4, S); setarg(g_kfa, 5, sc);
    for (int h0 = 0; h0 < H; h0 += 2) {
        size_t off[2] = { 0, (size_t) h0 }, g[2] = { (size_t) S / 16 * 64, (size_t) std::min(2, H - h0) }, l[2] = { 64, 1 };
        cl_int e = cl.clEnqueueNDRangeKernel(g_gpu.q, g_kfa, 2, off, g, l, 0, nullptr, nullptr);
        if (e) { P("  GPU FA enqueue: %s", cl_errstr(e)); return false; }
    }
    return cl.clFinish(g_gpu.q) == 0;
}

static void t2_fa(bool do_cpu) {
    const int S = M_TOK;
    double gf = 4.0 * S * (double) S * HD * NH * 1e-9;
    P("-- T2 flash-attention S=%d, %d cabezas x %d, Q F32, K/V F16, acumulacion F32 (%.0f GFLOP)", S, NH, HD, gf);
    if (g_fa.Q.empty()) {
        size_t n = (size_t) NH * S * HD;
        g_fa.Q.resize(n); g_fa.K.resize(n); g_fa.V.resize(n);
        fill_gauss(g_fa.Q.data(), n, 21, 1.0f);
        std::vector<float> tmp(n);
        fill_gauss(tmp.data(), n, 22, 1.0f);
        for (size_t i = 0; i < n; ++i) g_fa.K[i] = f32_to_f16(tmp[i]);
        fill_gauss(tmp.data(), n, 23, 1.0f);
        for (size_t i = 0; i < n; ++i) g_fa.V[i] = f32_to_f16(tmp[i]);
    }
    if (g_htp.ok() && !g_stop) {
        progress("HTP flash-attn");
        g_htp_fa = build_fa(g_htp, NH);
        if (g_htp_fa) {
            double first = now_ms();
            graph_run(g_htp, *g_htp_fa->g);
            first = now_ms() - first;
            g_fa_htp_ms = time_med([&] { return graph_run(g_htp, *g_htp_fa->g); }, 3, 1500, 10, false);
            ErrAcc e = check_fa_ggml(*g_htp_fa);
            P("  HTP FA: %7.1f ms (1a %.0f) = %5.0f GFLOP/s | err rel %.2e%s", g_fa_htp_ms, first, gf / g_fa_htp_ms * 1e3, e.rel(), e.nan ? " NaN!" : "");
        }
    }
    if (g_gpu.ok && !g_stop) {
        progress("GPU flash-attn");
        size_t n = (size_t) NH * S * HD;
        g_gQ = gbuf(n * 4); g_gK = gbuf(n * 2); g_gV = gbuf(n * 2); g_gO = gbuf(n * 4);
        g_kfa = get_kernel("-DWT_F16", "flash_attn");
        if (g_gQ && g_gK && g_gV && g_gO && g_kfa) {
            gwrite(g_gQ, g_fa.Q.data(), n * 4); gwrite(g_gK, g_fa.K.data(), n * 2); gwrite(g_gV, g_fa.V.data(), n * 2);
            double t = now_ms();
            bool ok = gpu_fa_run(2);
            t = now_ms() - t;
            P("  GPU FA sonda 2 cabezas: %.0f ms", t);
            if (ok) {
                g_fa_gpu_ms = time_med([&] { return gpu_fa_run(NH); }, 2, 1000, 4);
                ErrAcc e;
                std::vector<double> ref;
                std::vector<float> got(HD);
                for (int h : { 0, NH - 1 }) for (int qi : { 0, 777, S - 1 }) {
                    fa_ref(h, qi, S, ref);
                    gread(g_gO, got.data(), HD * 4, ((size_t) h * S + qi) * HD * 4);
                    for (int c = 0; c < HD; ++c) e.add(got[c], ref[c]);
                }
                P("  GPU FA (kernel propio simple): %7.1f ms = %5.0f GFLOP/s | err rel %.2e | GPU/HTP = %.1f%%", g_fa_gpu_ms, gf / g_fa_gpu_ms * 1e3,
                  e.rel(), g_fa_htp_ms / g_fa_gpu_ms * 100);
            }
        }
    }
    if (do_cpu && g_cpu.ok() && !g_stop) {
        progress("CPU flash-attn (2 cabezas)");
        auto f = build_fa(g_cpu, 2);
        if (f) {
            double ms = time_med([&] { return graph_run(g_cpu, *f->g); }, 1, 500, 3);
            ErrAcc e = check_fa_ggml(*f);
            P("  CPU FA 2 cabezas: %.0f ms -> 30 cabezas ~%.0f ms | err rel %.2e", ms, ms * NH / 2, e.rel());
        }
    }
}

// ================================================================ T3 elementwise / norms
static void t3_elem(bool do_cpu) {
    P("-- T3 ops elementwise/norm (F32) a [%d x %d] (FFN %d)", HID, M_TOK, FFN);
    auto run_ggml = [&](GB & b, const char * tag) {
        // E1 rms_norm*w -> modulate ; E2 gated_residual ; E3 swiglu_split
        TSet in(16);
        ggml_tensor * x = ggml_new_tensor_2d(in.ctx, GGML_TYPE_F32, HID, M_TOK);
        ggml_tensor * br = ggml_new_tensor_2d(in.ctx, GGML_TYPE_F32, HID, M_TOK);
        ggml_tensor * w = ggml_new_tensor_1d(in.ctx, GGML_TYPE_F32, HID);
        ggml_tensor * sc = ggml_new_tensor_3d(in.ctx, GGML_TYPE_F32, HID, 1, 1);
        ggml_tensor * sh = ggml_new_tensor_3d(in.ctx, GGML_TYPE_F32, HID, 1, 1);
        ggml_tensor * gt = ggml_new_tensor_3d(in.ctx, GGML_TYPE_F32, HID, 1, 1);
        ggml_tensor * a = ggml_new_tensor_2d(in.ctx, GGML_TYPE_F32, FFN, M_TOK);
        ggml_tensor * bb = ggml_new_tensor_2d(in.ctx, GGML_TYPE_F32, FFN, M_TOK);
        if (!in.alloc(b)) return;
        int seed = 40;
        for (auto t : { x, br, w, sc, sh, gt, a, bb }) set_rand_f32(t, seed++, 0.5f);
        struct E { const char * name; std::function<ggml_tensor *(ggml_context *)> f; double bytes; };
        E es[] = {
            { "rms_norm*w+modulate", [&](ggml_context * c) { return ggml_modulate(c, ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), w), sc, sh, M_TOK, nullptr, nullptr); }, 2.0 * HID * M_TOK * 4 },
            { "gated_residual", [&](ggml_context * c) { return ggml_gated_residual(c, x, br, gt, M_TOK, nullptr); }, 3.0 * HID * M_TOK * 4 },
            { "swiglu_split", [&](ggml_context * c) { return ggml_swiglu_split(c, a, bb); }, 3.0 * FFN * M_TOK * 4 },
            { "rms_norm*w", [&](ggml_context * c) { return ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), w); }, 2.0 * HID * M_TOK * 4 },
        };
        double layer = 0;
        for (auto & e : es) {
            if (g_stop) break;
            GGraph g(16);
            if (!graph_prepare(b, g, e.f(g.ctx), e.name)) continue;
            double ms = time_med([&] { return graph_run(b, g); }, 3, 600, 20);
            P("  %s %-22s %7.2f ms (%5.1f GB/s efectivos)", tag, e.name, ms, e.bytes / ms * 1e-6);
            layer += ms * 2;  // each appears ~2x per layer
        }
        P("  %s ~elementwise por capa (2x cada op) = %.1f ms", tag, layer - 0);
    };
    if (g_htp.ok() && !g_stop) { progress("HTP elementwise"); run_ggml(g_htp, "HTP"); }
    if (do_cpu && g_cpu.ok() && !g_stop) { progress("CPU elementwise"); run_ggml(g_cpu, "CPU"); }
    if (g_gpu.ok && !g_stop) {
        progress("GPU elementwise");
        MemGuard mg;
        size_t n = (size_t) HID * M_TOK, nf = (size_t) FFN * M_TOK;
        cl_mem x = mg.add(gbuf(n * 4)), r = mg.add(gbuf(n * 4)), y = mg.add(gbuf(n * 4)), w = mg.add(gbuf(HID * 4)), sc = mg.add(gbuf(HID * 4)),
               sh = mg.add(gbuf(HID * 4)), a = mg.add(gbuf(nf * 4)), b = mg.add(gbuf(nf * 4)), o = mg.add(gbuf(nf * 4));
        if (!x || !r || !y || !a || !b || !o) return;
        std::vector<float> h(nf);
        fill_gauss(h.data(), nf, 50, 0.5f);
        gwrite(x, h.data(), n * 4); gwrite(r, h.data(), n * 4); gwrite(a, h.data(), nf * 4); gwrite(b, h.data(), nf * 4);
        gwrite(w, h.data(), HID * 4); gwrite(sc, h.data(), HID * 4); gwrite(sh, h.data(), HID * 4);
        cl_kernel krm = get_kernel("-DWT_F16", "rmsnorm_mod"), kgr = get_kernel("-DWT_F16", "gated_res"), ksw = get_kernel("-DWT_F16", "swiglu"),
                  kcp = get_kernel("-DWT_F16", "copy4");
        if (!krm || !kgr || !ksw || !kcp) return;
        int C = HID; float eps = 1e-6f;
        setarg(krm, 0, x); setarg(krm, 1, w); setarg(krm, 2, sc); setarg(krm, 3, sh); setarg(krm, 4, y); setarg(krm, 5, C); setarg(krm, 6, eps);
        setarg(kgr, 0, x); setarg(kgr, 1, r); setarg(kgr, 2, w); setarg(kgr, 3, y); setarg(kgr, 4, C);
        setarg(ksw, 0, a); setarg(ksw, 1, b); setarg(ksw, 2, o);
        setarg(kcp, 0, a); setarg(kcp, 1, o);
        auto launch = [&](cl_kernel k, size_t g, size_t l) {
            cl_int e = cl.clEnqueueNDRangeKernel(g_gpu.q, k, 1, nullptr, &g, l ? &l : nullptr, 0, nullptr, nullptr);
            return e == 0 && cl.clFinish(g_gpu.q) == 0;
        };
        double t1 = time_med([&] { return launch(krm, (size_t) M_TOK * 256, 256); }, 3, 600, 20);
        double t2 = time_med([&] { return launch(kgr, n, 0); }, 3, 600, 20);
        double t3 = time_med([&] { return launch(ksw, nf, 0); }, 3, 600, 20);
        double t4 = time_med([&] { return launch(kcp, nf / 4, 0); }, 3, 600, 20);
        P("  GPU rms_norm*w+modulate %7.2f ms (%5.1f GB/s)", t1, 2.0 * n * 4 / t1 * 1e-6);
        P("  GPU gated_residual      %7.2f ms (%5.1f GB/s)", t2, 3.0 * n * 4 / t2 * 1e-6);
        P("  GPU swiglu              %7.2f ms (%5.1f GB/s)", t3, 3.0 * nf * 4 / t3 * 1e-6);
        P("  GPU copia %zu MB          %7.2f ms (%5.1f GB/s lectura+escritura)", nf * 4 >> 20, t4, 2.0 * nf * 4 / t4 * 1e-6);
    }
}

}  // namespace npg

// ================================================================ T4 LoRA B(Ax)
namespace npg {

struct LoraPair { int in, out; const char * name; };
static const LoraPair kLora[] = { { HID, HID, "q" }, { HID, HID, "k" }, { HID, HID, "v" }, { HID, HID, "out" },
                                  { HID, FFN, "w1" }, { HID, FFN, "w3" }, { FFN, HID, "w2" } };
struct GpuLora {
    int r = 0;
    MemGuard mg;
    cl_mem A[7] = {}, B[7] = {}, T = nullptr;
    GemmVar ka, kb;
};
static std::unique_ptr<GpuLora> g_glora64;
static double g_lora_gpu_ms[3] = { NAN, NAN, NAN }, g_lora_htp_ms[3] = { NAN, NAN, NAN };

static std::unique_ptr<GpuLora> gpu_lora_make(int r) {
    auto L = std::make_unique<GpuLora>();
    L->r = r;
    for (int i = 0; i < 7; ++i) {
        std::vector<float> f((size_t) r * kLora[i].in);
        fill_gauss(f.data(), f.size(), 60 + i, 0.02f);
        std::vector<uint16_t> h(f.size());
        for (size_t j = 0; j < f.size(); ++j) h[j] = f32_to_f16(f[j]);
        L->A[i] = L->mg.add(gbuf(h.size() * 2, CL_MEM_READ_ONLY));
        if (!L->A[i]) return nullptr;
        gwrite(L->A[i], h.data(), h.size() * 2);
        f.assign((size_t) kLora[i].out * r, 0.f);
        fill_gauss(f.data(), f.size(), 70 + i, 0.02f);
        h.resize(f.size());
        for (size_t j = 0; j < f.size(); ++j) h[j] = f32_to_f16(f[j]);
        L->B[i] = L->mg.add(gbuf(h.size() * 2, CL_MEM_READ_ONLY));
        if (!L->B[i]) return nullptr;
        gwrite(L->B[i], h.data(), h.size() * 2);
    }
    L->T = L->mg.add(gbuf((size_t) M_TOK * r * 4));
    L->ka = make_var("WT_F16", false, 4, 4);
    L->kb = g_best16.k ? g_best16 : make_var("WT_F16", false, 4, 4);
    if (!L->T || !L->ka.k || !L->kb.k) return nullptr;
    return L;
}
// one DiT layer worth of LoRA branches: Y += B (A x) for the 7 linears (enqueue only)
static bool gpu_lora_enqueue(GpuLora & L) {
    for (int i = 0; i < 7; ++i) {
        cl_mem X = kLora[i].in == HID ? g_gX : g_gX2;
        if (!gemm_enqueue(L.ka, X, L.A[i], L.T, M_TOK, L.r, kLora[i].in, 1.f, 0)) return false;
        if (!gemm_enqueue(L.kb, L.T, L.B[i], g_gY, M_TOK, kLora[i].out, L.r, 1.f, 1)) return false;
    }
    return true;
}
static bool gpu_lora_run(GpuLora & L) { return gpu_lora_enqueue(L) && cl.clFinish(g_gpu.q) == 0; }

static void t4_lora() {
    P("-- T4 LoRA en tiempo de ejecucion: por capa 7 ramas (q,k,v,out,w1,w3,w2) Y += B(A x), A/B FP16, acumulacion F32, M=%d", M_TOK);
    const int ranks[3] = { 16, 64, 128 };
    for (int ri = 0; ri < 3 && !g_stop; ++ri) {
        int r = ranks[ri];
        double flop = 0;
        for (auto & p : kLora) flop += gflop_mm(M_TOK, r, p.in) + gflop_mm(M_TOK, p.out, r);
        if (g_gpu.ok && g_gX && g_gY) {
            progress("GPU LoRA r=%d", r);
            auto L = gpu_lora_make(r);
            if (L) {
                g_lora_gpu_ms[ri] = time_med([&] { return gpu_lora_run(*L); }, 3, 800, 10);
                if (r == 64) g_glora64 = std::move(L);
            }
        }
        if (g_htp.ok()) {
            progress("HTP LoRA r=%d", r);
            TSet w(16), in(4);
            ggml_tensor *A[7], *B[7];
            for (int i = 0; i < 7; ++i) {
                A[i] = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F16, kLora[i].in, r);
                B[i] = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F16, r, kLora[i].out);
            }
            ggml_tensor * X = ggml_new_tensor_2d(in.ctx, GGML_TYPE_F32, HID, M_TOK);
            ggml_tensor * X2 = ggml_new_tensor_2d(in.ctx, GGML_TYPE_F32, FFN, M_TOK);
            if (w.alloc(g_htp, true) && in.alloc(g_htp)) {
                for (int i = 0; i < 7; ++i) {
                    std::vector<float> f(ggml_nelements(A[i]));
                    fill_gauss(f.data(), f.size(), 60 + i, 0.02f); set_f16_from(A[i], f);
                    f.resize(ggml_nelements(B[i]));
                    fill_gauss(f.data(), f.size(), 70 + i, 0.02f); set_f16_from(B[i], f);
                }
                ggml_backend_tensor_set(X, g_hd.X.data(), 0, ggml_nbytes(X));
                ggml_backend_tensor_set(X2, g_hd.X2.data(), 0, ggml_nbytes(X2));
                GGraph g(64);
                ggml_tensor * last = nullptr;
                for (int i = 0; i < 7; ++i) {
                    ggml_tensor * t = ggml_mul_mat(g.ctx, A[i], kLora[i].in == HID ? X : X2);
                    ggml_tensor * y = ggml_mul_mat(g.ctx, B[i], t);
                    ggml_build_forward_expand(g.gf, y);
                    last = y;
                }
                if (graph_prepare(g_htp, g, last, "LoRA HTP")) {
                    g_lora_htp_ms[ri] = time_med([&] { return graph_run(g_htp, g); }, 3, 800, 10);
                }
            }
        }
        double base = g_t1.htp_layer;
        P("  r=%3d (%.1f GFLOP/capa): GPU %7.1f ms/capa (%4.1f%% de linears HMX) | HTP %7.1f ms/capa (%4.1f%%) | x30 capas: GPU %.2f s, HTP %.2f s",
          r, flop, g_lora_gpu_ms[ri], g_lora_gpu_ms[ri] / base * 100, g_lora_htp_ms[ri], g_lora_htp_ms[ri] / base * 100,
          g_lora_gpu_ms[ri] * 30 / 1e3, g_lora_htp_ms[ri] * 30 / 1e3);
    }
}

// ================================================================ T5 concurrency NPU || GPU
struct RunStat { double mean_ms = NAN; int n = 0; };
static RunStat run_solo(const std::function<bool()> & f, double window_ms) {
    RunStat r;
    if (!f()) return r;  // warm
    double t0 = now_ms(), sum = 0;
    while (now_ms() - t0 < window_ms && !g_stop) {
        double t = now_ms();
        if (!f()) return r;
        sum += now_ms() - t; r.n++;
    }
    r.mean_ms = r.n ? sum / r.n : NAN;
    return r;
}
static void run_pair(const std::function<bool()> & fa, const std::function<bool()> & fb, double window_ms, RunStat & ra, RunStat & rb) {
    std::atomic<bool> stop{ false };
    auto worker = [&](const std::function<bool()> & f, RunStat & r) {
        double sum = 0; int n = 0; bool first = true;
        while (!stop && !g_stop) {
            double t = now_ms();
            if (!f()) break;
            double d = now_ms() - t;
            if (first) { first = false; continue; }   // drop ramp-up iteration
            sum += d; ++n;
        }
        r.n = n; r.mean_ms = n ? sum / n : NAN;
    };
    std::thread ta(worker, std::cref(fa), std::ref(ra)), tb(worker, std::cref(fb), std::ref(rb));
    double t0 = now_ms();
    while (now_ms() - t0 < window_ms && !g_stop) usleep(20000);
    stop = true;
    ta.join(); tb.join();
}

static std::vector<uint8_t> g_mc_src, g_mc_dst;
static bool cpu_memcpy_job() {
    const size_t n = g_mc_src.size(), part = n / 4;
    std::thread th[4];
    for (int i = 0; i < 4; ++i) th[i] = std::thread([&, i] { memcpy(g_mc_dst.data() + i * part, g_mc_src.data() + i * part, part); });
    for (auto & t : th) t.join();
    return true;
}

static double g_s3_gain = NAN, g_contention_h = NAN, g_contention_g = NAN;

static void t5_concurrency(double win) {
    P("-- T5 concurrencia NPU(HMX) || GPU (ventanas de %.0f s; factor = t_concurrente / t_solo, 1.00 = sin contencion)", win / 1e3);
    if (!g_htp_qkv) { P("  sin grafo HTP qkv: omitido"); return; }
    auto h_qkv = [&] { return graph_run(g_htp, *g_htp_qkv->g); };
    const LinShape & q = kLin[0];
    auto g_qkv = [&] { return gemm_run(g_best8, g_gX, g_gW8[0], g_gY, M_TOK, q.N, q.K, g_hd.scale[0] * 256.f, 0); };
    MemGuard mg;
    cl_mem ca = nullptr, cb = nullptr;
    cl_kernel kcp = nullptr;
    const size_t cbytes = 192u << 20;
    if (g_gpu.ok) {
        ca = mg.add(gbuf(cbytes)); cb = mg.add(gbuf(cbytes));
        kcp = get_kernel("-DWT_F16", "copy4");
        if (kcp) { setarg(kcp, 0, ca); setarg(kcp, 1, cb); }
    }
    auto g_copy = [&] {
        size_t g = cbytes / 16;
        return cl.clEnqueueNDRangeKernel(g_gpu.q, kcp, 1, nullptr, &g, nullptr, 0, nullptr, nullptr) == 0 && cl.clFinish(g_gpu.q) == 0;
    };
    g_mc_src.assign(256u << 20, 1); g_mc_dst.assign(256u << 20, 0);

    progress("solo HTP qkv");
    RunStat hs = run_solo(h_qkv, win * 0.6);
    P("  solo: HTP qkv %.1f ms", hs.mean_ms);
    struct Case { const char * name; std::function<bool()> f; bool is_gemm; };
    std::vector<Case> cases;
    if (g_gpu.ok && g_best8.k) cases.push_back({ "GPU GEMM FP8 qkv", g_qkv, true });
    if (g_gpu.ok && g_glora64) cases.push_back({ "GPU LoRA r64 (capa)", [&] { return gpu_lora_run(*g_glora64); }, false });
    if (g_gpu.ok && kcp) cases.push_back({ "GPU copia 192MB (ancho banda)", g_copy, false });
    cases.push_back({ "CPU memcpy 4 hilos 256MB", cpu_memcpy_job, false });
    for (auto & c : cases) {
        if (g_stop) break;
        progress("par HTP qkv || %s", c.name);
        RunStat bs = run_solo(c.f, win * 0.6);
        RunStat ha, ba;
        run_pair(h_qkv, c.f, win, ha, ba);
        double fh = ha.mean_ms / hs.mean_ms, fb = ba.mean_ms / bs.mean_ms;
        P("  HTP qkv || %-28s: HTP %.1f->%.1f ms (x%.2f) | otro %.1f->%.1f ms (x%.2f) | n=%d/%d", c.name, hs.mean_ms, ha.mean_ms, fh, bs.mean_ms,
          ba.mean_ms, fb, ha.n, ba.n);
        if (c.is_gemm) {
            g_contention_h = fh; g_contention_g = fb;
            double Hc = ha.mean_ms, Gc = ba.mean_ms;
            double Tstar = Hc * Gc / (Hc + Gc);
            double f = Hc / (Hc + Gc);
            g_s3_gain = hs.mean_ms / Tstar - 1;
            double Tideal = hs.mean_ms * bs.mean_ms / (hs.mean_ms + bs.mean_ms);
            P("    S3 (repartir N de cada GEMM): fraccion GPU optima f=%.3f; t=%.1f ms vs HTP solo %.1f -> ganancia %+.1f%% (sin contencion seria %+.1f%%)",
              f, Tstar, hs.mean_ms, g_s3_gain * 100, (hs.mean_ms / Tideal - 1) * 100);
        }
    }
    if (g_htp_fa && g_kfa && !g_stop) {
        progress("par HTP FA || GPU FA");
        auto h_fa = [&] { return graph_run(g_htp, *g_htp_fa->g); };
        auto gfa = [&] { return gpu_fa_run(NH); };
        RunStat a = run_solo(h_fa, win * 0.6), b = run_solo(gfa, win * 0.6), a2, b2;
        run_pair(h_fa, gfa, win * 1.5, a2, b2);
        P("  HTP FA || GPU FA: HTP %.1f->%.1f ms (x%.2f) | GPU %.1f->%.1f ms (x%.2f)", a.mean_ms, a2.mean_ms, a2.mean_ms / a.mean_ms, b.mean_ms,
          b2.mean_ms, b2.mean_ms / b.mean_ms);
    }
    g_mc_src.clear(); g_mc_src.shrink_to_fit(); g_mc_dst.clear(); g_mc_dst.shrink_to_fit();
}

}  // namespace npg

// ================================================================ T6 sync latency + dmabuf zero-copy
namespace npg {

static std::string pct(std::vector<double> v) {
    if (v.empty()) return "-";
    std::sort(v.begin(), v.end());
    char b[96];
    snprintf(b, sizeof b, "p50 %.3f / p90 %.3f / min %.3f ms", v[v.size() / 2], v[v.size() * 9 / 10], v[0]);
    return b;
}

static bool g_dmabuf_ok = false;
static cl_uint g_best_pol = 0;
static bool g_best_ion = false;
// HTP element-wise ops stage whole rows (src0+src1+dst, per thread) in VTCM; a too-large row makes the DSP
// return VTCM-TOO-SMALL and ggml-hexagon then aborts the process on synchronize (not recoverable).
// Pre-check: keep rows <= 64 KB (real ZIT rows are 15-40 KB).
static bool htp_rows_fit(const ggml_tensor * t, const char * what) {
    size_t row = ggml_row_size(t->type, t->ne[0]);
    if (row > (64u << 10)) { P("  %s: fila de %zu KB > 64 KB, no cabe en VTCM -> se omite (evita abortar)", what, row >> 10); return false; }
    return true;
}
static std::string g_dmabuf_note = "no probado";

#ifdef NPG_HEXAGON
static cl_mem import_fd(int fd, void * host, size_t size, cl_uint policy, bool use_ion, cl_int & err) {
    cl_mem_flags fl = CL_MEM_USE_HOST_PTR | NPG_CL_MEM_EXT_HOST_PTR_QCOM | CL_MEM_READ_WRITE;
    if (use_ion) {
        npg_cl_mem_ion_host_ptr d = {};
        d.ext_host_ptr.allocation_type = NPG_CL_MEM_ION_HOST_PTR_QCOM; d.ext_host_ptr.host_cache_policy = policy;
        d.ion_filedesc = fd; d.ion_hostptr = host;
        return cl.clCreateBuffer(g_gpu.ctx, fl, size, &d, &err);
    }
    npg_cl_mem_dmabuf_host_ptr d = {};
    d.ext_host_ptr.allocation_type = NPG_CL_MEM_DMABUF_HOST_PTR_QCOM; d.ext_host_ptr.host_cache_policy = policy;
    d.dmabuf_filedesc = fd; d.dmabuf_hostptr = host;
    return cl.clCreateBuffer(g_gpu.ctx, fl, size, &d, &err);
}
#endif

static void t6_sync_dmabuf() {
    P("-- T6 latencias de sincronizacion y zero-copy (rpcmem/dmabuf -> OpenCL)");
    cl_kernel kempty = nullptr;
    MemGuard mg;
    if (g_gpu.ok) {
        cl_mem tiny = mg.add(gbuf(4096));
        kempty = get_kernel("-DWT_F16", "empty_k");
        if (tiny && kempty) {
            setarg(kempty, 0, tiny);
            size_t g = 64;
            std::vector<double> a, b, c;
            for (int i = 0; i < 220; ++i) {
                double t = now_ms();
                cl.clEnqueueNDRangeKernel(g_gpu.q, kempty, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
                cl.clFinish(g_gpu.q);
                if (i >= 20) a.push_back(now_ms() - t);
            }
            for (int i = 0; i < 220; ++i) {
                cl_event ev = nullptr;
                double t = now_ms();
                cl.clEnqueueNDRangeKernel(g_gpu.q, kempty, 1, nullptr, &g, nullptr, 0, nullptr, &ev);
                cl.clFlush(g_gpu.q);
                cl.clWaitForEvents(1, &ev);
                double d = now_ms() - t;
                cl_ulong qd = 0, st = 0, en = 0;
                cl.clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_QUEUED, sizeof qd, &qd, nullptr);
                cl.clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof st, &st, nullptr);
                cl.clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof en, &en, nullptr);
                cl.clReleaseEvent(ev);
                if (i >= 20) { b.push_back(d); c.push_back((st - qd) * 1e-6); }
            }
            P("  GPU kernel vacio + clFinish: %s", pct(a).c_str());
            P("  GPU kernel vacio + flush + waitForEvents: %s | cola->inicio (perfilado) %s", pct(b).c_str(), pct(c).c_str());
        }
    }
    if (g_htp.ok()) {
        TSet in(4);
        ggml_tensor * x = ggml_new_tensor_1d(in.ctx, GGML_TYPE_F32, 32);
        if (in.alloc(g_htp)) {
            set_const_f32(x, 1.f);
            GGraph g(8);
            if (graph_prepare(g_htp, g, ggml_add(g.ctx, x, x), "tiny add")) {
                std::vector<double> a;
                for (int i = 0; i < 120; ++i) { double t = now_ms(); graph_run(g_htp, g); if (i >= 20) a.push_back(now_ms() - t); }
                P("  HTP grafo minimo (1 op) compute+sync: %s", pct(a).c_str());
            }
        }
    }
#ifdef NPG_HEXAGON
    if (!g_gpu.ok || !g_htp.ok()) { g_dmabuf_note = "sin GPU o HTP"; return; }
    if (!g_gpu.ext_host) { P("  cl_qcom_ext_host_ptr no disponible: sin zero-copy"); g_dmabuf_note = "sin cl_qcom_ext_host_ptr"; return; }
    const size_t page = g_gpu.page_size ? g_gpu.page_size : 4096;
    const size_t size = 64u << 20;
    const size_t alloc_sz = size + ((g_gpu.ext_padding + page - 1) / page) * page;
    double t = now_ms();
    void * p = rpcmem_alloc2(25 /*SYSTEM*/, 1 /*default flags*/, alloc_sz);
    double t_alloc = now_ms() - t;
    int fd = p ? rpcmem_to_fd(p) : -1;
    P("  rpcmem_alloc2 %zu MB: %.1f ms, fd=%d, ptr align %s", alloc_sz >> 20, t_alloc, fd, p && ((uintptr_t) p % page == 0) ? "ok" : "NO");
    if (!p || fd < 0) { g_dmabuf_note = "rpcmem_alloc2 fallo"; if (p) rpcmem_free(p); return; }
    struct Pol { const char * n; cl_uint v; };
    std::vector<Pol> pols = { { "writeback", NPG_CL_MEM_HOST_WRITEBACK_QCOM }, { "uncached", NPG_CL_MEM_HOST_UNCACHED_QCOM } };
    if (g_gpu.iocoh) pols.insert(pols.begin(), { "iocoherent", NPG_CL_MEM_HOST_IOCOHERENT_QCOM });
    cl_kernel kadd = get_kernel("-DWT_F16", "add_one_u32"), kcp = get_kernel("-DWT_F16", "copy4");
    cl_mem dst = mg.add(gbuf(32u << 20)), src2 = mg.add(gbuf(32u << 20));
    double base_bw = NAN;
    if (kcp && dst && src2) {
        setarg(kcp, 0, src2); setarg(kcp, 1, dst);
        size_t g = (32u << 20) / 16;
        double ms = time_med([&] { return cl.clEnqueueNDRangeKernel(g_gpu.q, kcp, 1, nullptr, &g, nullptr, 0, nullptr, nullptr) == 0 && cl.clFinish(g_gpu.q) == 0; }, 5, 300, 30);
        base_bw = 2.0 * (32u << 20) / ms * 1e-6;
        P("  copia GPU buffer normal->normal 32MB: %.2f ms (%.1f GB/s)", ms, base_bw);
    }
    cl_uint best_pol = 0;
    bool best_ion = false;
    for (int use_ion = 0; use_ion < 2; ++use_ion) {
        if (use_ion && (best_pol || !g_gpu.ion)) break;
        if (!use_ion && !g_gpu.dmabuf) { P("  cl_qcom_dmabuf_host_ptr NO anunciado; se intenta igualmente"); }
        for (auto & pol : pols) {
            cl_int err = 0;
            t = now_ms();
            cl_mem m = import_fd(fd, p, size, pol.v, use_ion, err);
            double t_imp = now_ms() - t;
            if (!m) { P("  import %s %s: FALLO %s", use_ion ? "ION" : "DMABUF", pol.n, cl_errstr(err)); continue; }
            // correctness: CPU writes -> GPU +1 -> CPU reads
            uint32_t * u = (uint32_t *) p;
            size_t n = size / 4;
            bool ok = kadd != nullptr;
            double t_k = NAN;
            if (ok) {
                // CPU writes inside map/unmap so the driver does the cache maintenance
                void * mp = cl.clEnqueueMapBuffer(g_gpu.q, m, CL_TRUE, CL_MAP_WRITE, 0, size, 0, nullptr, nullptr, &err);
                uint32_t * w = mp ? (uint32_t *) mp : u;
                for (size_t i = 0; i < n; ++i) w[i] = (uint32_t) (i * 3 + 1);
                if (mp) cl.clEnqueueUnmapMemObject(g_gpu.q, m, mp, 0, nullptr, nullptr);
                setarg(kadd, 0, m);
                size_t g = n;
                t = now_ms();
                ok = cl.clEnqueueNDRangeKernel(g_gpu.q, kadd, 1, nullptr, &g, nullptr, 0, nullptr, nullptr) == 0 && cl.clFinish(g_gpu.q) == 0;
                t_k = now_ms() - t;
                mp = cl.clEnqueueMapBuffer(g_gpu.q, m, CL_TRUE, CL_MAP_READ, 0, size, 0, nullptr, nullptr, &err);
                size_t bad = 0;
                const uint32_t * r = mp ? (const uint32_t *) mp : u;
                for (size_t i = 0; i < n; ++i) if (r[i] != (uint32_t) (i * 3 + 2)) ++bad;
                size_t bad_raw = 0;
                for (size_t i = 0; i < n; i += 97) if (u[i] != (uint32_t) (i * 3 + 2)) ++bad_raw;
                if (mp) cl.clEnqueueUnmapMemObject(g_gpu.q, m, mp, 0, nullptr, nullptr);
                cl.clFinish(g_gpu.q);
                ok = ok && bad == 0;
                P("  import %s %-10s: %.2f ms | map=%s | CPU->GPU(+1)->CPU %s (malos %zu, via puntero rpcmem directo %zu/%zu) | kernel 64MB %.2f ms",
                  use_ion ? "ION" : "DMABUF", pol.n, t_imp, mp == p ? "mismo ptr" : (mp ? "otro ptr" : "fallo"), ok ? "OK" : "FALLO", bad, bad_raw, n / 97 + 1, t_k);
            }
            if (ok && kcp && dst) {
                setarg(kcp, 0, m); setarg(kcp, 1, dst);
                size_t g = (32u << 20) / 16;
                double ms = time_med([&] { return cl.clEnqueueNDRangeKernel(g_gpu.q, kcp, 1, nullptr, &g, nullptr, 0, nullptr, nullptr) == 0 && cl.clFinish(g_gpu.q) == 0; }, 5, 300, 30);
                P("    copia GPU desde buffer importado 32MB: %.2f ms (%.1f GB/s, %.0f%% de normal)", ms, 2.0 * (32u << 20) / ms * 1e-6,
                  2.0 * (32u << 20) / ms * 1e-6 / base_bw * 100);
            }
            cl.clReleaseMemObject(m);
            if (ok && !best_pol) { best_pol = pol.v; best_ion = use_ion; }
        }
    }
    rpcmem_free(p);
    if (!best_pol) { g_dmabuf_note = "import fallo con todas las politicas"; P("  zero-copy: NO disponible"); return; }

    g_best_pol = best_pol; g_best_ion = best_ion;
    g_dmabuf_note = "import OK (traspaso NPU<->GPU real: ver T6b al final)";
    P("  (el traspaso real NPU->GPU->NPU con buffers ggml-hexagon se mide en T6b, al final, para que T7/T8 corran siempre)");
#else
    P("  dmabuf: solo en Android/HTP");
#endif
}


// T6b: real NPU -> GPU -> NPU handoff through ggml-hexagon's own (rpcmem) buffers imported into OpenCL.
static void t6b_handoff() {
    P("-- T6b traspaso real GPU->NPU->GPU con buffers ggml-hexagon importados (zero-copy)");
#ifdef NPG_HEXAGON
    if (!g_gpu.ok || !g_htp.ok()) { P("  sin GPU o HTP: omitido"); return; }
    if (!g_best_pol) { P("  sin politica de import valida (T6 no corrio o fallo): omitido"); return; }
    const size_t page = g_gpu.page_size ? g_gpu.page_size : 4096;
    const cl_uint best_pol = g_best_pol;
    const bool best_ion = g_best_ion;
    MemGuard mg;
    // NPU <-> GPU through the real ggml-hexagon buffers
    {
        TSet in(4);
        // 0.1.0 used one 1M-float row (4 MB): the HTP binary op stages whole rows in VTCM -> VTCM-TOO-SMALL and
        // ggml-hexagon aborts on synchronize. 0.1.1: 1024 rows x 1024 floats (4 KB rows, like real activations).
        const int ne0 = 1024, ne1 = 1024, n = ne0 * ne1;
        ggml_tensor * x = ggml_new_tensor_2d(in.ctx, GGML_TYPE_F32, ne0, ne1);
        if (!htp_rows_fit(x, "T6b add")) { g_dmabuf_note = "T6b omitido (fila demasiado grande para VTCM)"; return; }
        if (!in.alloc(g_htp)) return;
        set_const_f32(x, 0.f);
        GGraph g(8);
        if (!graph_prepare(g_htp, g, ggml_add(g.ctx, x, x), "add 1024x1024")) return;
        auto import_tensor_buf = [&](ggml_tensor * t, int & off_elems) -> cl_mem {
            ggml_backend_buffer_t b = t->buffer;
            uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(b);
            size_t bsz = ggml_backend_buffer_get_size(b);
            off_elems = (int) (((uint8_t *) t->data - base) / 4);
            int bfd = rpcmem_to_fd(base);
            cl_int err = 0;
            size_t sz = bsz;
            cl_mem m = import_fd(bfd, base, sz, best_pol, best_ion, err);
            if (!m && g_gpu.ext_padding) {
                sz = ((bsz - g_gpu.ext_padding) / page) * page;
                m = import_fd(bfd, base, sz, best_pol, best_ion, err);
            }
            if (!m) P("  import buffer ggml-hexagon (fd %d, %zu B): FALLO %s", bfd, bsz, cl_errstr(err));
            return m;
        };
        int offx = 0, offz = 0;
        double ti = now_ms();
        cl_mem mx = import_tensor_buf(x, offx);
        cl_mem mz = mx ? import_tensor_buf(g.out, offz) : nullptr;
        ti = now_ms() - ti;
        cl_kernel kfill = get_kernel("-DWT_F16", "fill_f32"), kcpf = get_kernel("-DWT_F16", "copy_f32");
        cl_mem out = mg.add(gbuf((size_t) n * 4));
        if (mx && mz && kfill && kcpf && out) {
            float b0 = 1.0f, st = 0.5f;
            setarg(kfill, 0, mx); setarg(kfill, 1, offx); setarg(kfill, 2, b0); setarg(kfill, 3, st);
            size_t gsz = n;
            std::vector<double> hand;
            size_t bad1 = 0, bad2 = 0;
            for (int it = 0; it < 12; ++it) {
                double t0 = now_ms();
                cl.clEnqueueNDRangeKernel(g_gpu.q, kfill, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);
                cl.clFinish(g_gpu.q);                       // GPU -> NPU handoff
                graph_run(g_htp, g);                        // NPU consumes GPU output, produces z = 2x
                setarg(kcpf, 0, mz); setarg(kcpf, 1, offz); setarg(kcpf, 2, out);
                cl.clEnqueueNDRangeKernel(g_gpu.q, kcpf, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr);
                cl.clFinish(g_gpu.q);                       // NPU -> GPU handoff
                if (it >= 2) hand.push_back(now_ms() - t0);
            }
            std::vector<float> zt(n), zg(n);
            ggml_backend_tensor_get(g.out, zt.data(), 0, (size_t) n * 4);
            gread(out, zg.data(), (size_t) n * 4);
            for (int i = 0; i < n; ++i) {
                float e = 2.f * (b0 + st * (float) (i % 1024));
                if (zt[i] != e) ++bad1;
                if (zg[i] != e) ++bad2;
            }
            g_dmabuf_ok = bad1 == 0 && bad2 == 0;
            P("  NPU<->GPU con buffers ggml-hexagon importados (%.1f ms import): GPU escribe x -> HTP z=x+x: %s (malos %zu) | GPU lee z del HTP: %s (malos %zu)",
              ti, bad1 ? "FALLO" : "OK", bad1, bad2 ? "FALLO" : "OK", bad2);
            P("  ciclo GPU(fill 1024x1024)->HTP(add)->GPU(copia) con 3 sincronizaciones: %s", pct(hand).c_str());
            g_dmabuf_note = g_dmabuf_ok ? "OK (zero-copy NPU<->GPU verificado)" : "import OK pero datos incoherentes";
        } else if (!mx || !mz) {
            g_dmabuf_note = "import de buffers ggml-hexagon fallo";
        }
        if (mx) cl.clReleaseMemObject(mx);
        if (mz) cl.clReleaseMemObject(mz);
    }
#else
    P("  solo en Android/HTP");
#endif
}

// ================================================================ T7 sustained thermal
// When T1 was skipped ("Solo T6-T8"), T7 builds its own HTP qkv graph + cold baseline, and uploads the GPU
// qkv buffers using the FP8 kernel that won the 0.1.0 autotune on SM8850 (F8 tiled 8x4) instead of re-tuning.
static void t7_ensure_resources() {
    if (!g_htp_qkv && g_htp.ok()) {
        progress("T7: preparando grafo HTP qkv (T1 omitida)");
        g_htp_qkv = build_mm(g_htp, 0, M_TOK);
        if (g_htp_qkv && std::isnan(g_t1.htp_ms[0])) {
            graph_run(g_htp, *g_htp_qkv->g);
            g_t1.htp_ms[0] = time_med([&] { return graph_run(g_htp, *g_htp_qkv->g); }, 4, 1500, 30, false);
            P("  HTP qkv en frio (referencia, T1 omitida): %.1f ms", g_t1.htp_ms[0]);
        }
    }
    if (g_gpu.ok && !g_best8.k) {
        progress("T7: preparando GEMM GPU qkv (T1 omitida)");
        if (!g_gX) { g_gX = gbuf((size_t) M_TOK * HID * 4); if (g_gX) gwrite(g_gX, g_hd.X.data(), g_hd.X.size() * 4); }
        if (!g_gY) g_gY = gbuf((size_t) M_TOK * 3 * HID * 4);
        if (!g_gW8[0]) { g_gW8[0] = gbuf(g_hd.W[0].size(), CL_MEM_READ_ONLY); if (g_gW8[0]) gwrite(g_gW8[0], g_hd.W[0].data(), g_hd.W[0].size()); }
        if (g_gX && g_gY && g_gW8[0]) {
            g_gemm_chunk = 256;
            GemmVar v = make_var("WT_F8", true, 8, 4);
            if (v.k) { g_best8 = v; P("  GPU qkv con %s (ganador del autotune 0.1.0; sin re-tuning)", v.name.c_str()); }
        }
    }
}

static void t7_sustained(int secs) {
    P("-- T7 sostenido %d s: HTP qkv || GPU qkv FP8 en paralelo (media por ventana de 5 s)", secs);
    t7_ensure_resources();
    if (!g_htp_qkv) { P("  sin grafo HTP: omitido"); return; }
    P("  antes: %s | %s", thermal_summary().c_str(), clocks_summary().c_str());
    std::atomic<bool> stop{ false };
    std::mutex mu;
    double hsum = 0, gsum = 0; int hn = 0, gn = 0;
    const LinShape & q = kLin[0];
    std::thread th([&] {
        while (!stop && !g_stop) { double t = now_ms(); if (!graph_run(g_htp, *g_htp_qkv->g)) break; std::lock_guard<std::mutex> lk(mu); hsum += now_ms() - t; ++hn; }
    });
    std::thread tg;
    bool gpu = g_gpu.ok && g_best8.k;
    if (gpu) tg = std::thread([&] {
        while (!stop && !g_stop) {
            double t = now_ms();
            if (!gemm_run(g_best8, g_gX, g_gW8[0], g_gY, M_TOK, q.N, q.K, g_hd.scale[0] * 256.f, 0)) break;
            std::lock_guard<std::mutex> lk(mu); gsum += now_ms() - t; ++gn;
        }
    });
    double t0 = now_ms();
    double first_h = NAN, first_g = NAN, last_h = NAN, last_g = NAN;
    for (int w = 5; w <= secs && !g_stop; w += 5) {
        while (now_ms() - t0 < w * 1000.0 && !g_stop) usleep(50000);
        double h, g; int nh, ng;
        { std::lock_guard<std::mutex> lk(mu); h = hn ? hsum / hn : NAN; g = gn ? gsum / gn : NAN; nh = hn; ng = gn; hsum = gsum = 0; hn = gn = 0; }
        if (w == 5) { first_h = h; first_g = g; }
        last_h = h; last_g = g;
        P("  t=%3ds HTP %6.1f ms (n=%2d) GPU %7.1f ms (n=%2d) | %s | %s", w, h, nh, g, ng, thermal_summary().c_str(), clocks_summary().c_str());
        progress("sostenido %d/%d s", w, secs);
    }
    stop = true;
    th.join();
    if (tg.joinable()) tg.join();
    P("  degradacion 1a->ultima ventana: HTP %+.1f%%, GPU %+.1f%%", (last_h / first_h - 1) * 100, (last_g / first_g - 1) * 100);
    if (!g_stop) {
        RunStat hot = run_solo([&] { return graph_run(g_htp, *g_htp_qkv->g); }, 10000);
        P("  HTP solo justo despues (caliente, 10 s): %.1f ms vs solo en frio (T1) %.1f ms -> %+.1f%% | %s | %s", hot.mean_ms, g_t1.htp_ms[0],
          (hot.mean_ms / g_t1.htp_ms[0] - 1) * 100, thermal_summary().c_str(), clocks_summary().c_str());
    }
}

}  // namespace npg

// ================================================================ T8 first-step host breakdown (synthetic DiT)
// 0.1.2: fixes the 0.1.1 crash at the start of T8. Root cause: the host staging buffer for the FP8 upload was
// sized HID*FFN (39.3 MB) but the first tensor written into it is qkv, HID*3*HID (44.2 MB): fill_fp8_weights
// overran the heap block by 4.9 MB -> SIGSEGV before any ggml/HTP call (hence no "ABORTO de ggml" line).
// Also: weights are now split explicitly into backend buffers <= ggml_backend_buft_get_max_size (1 GiB on HTP),
// every alloc is checked, memory / buffer / VTCM-row pre-checks skip instead of aborting, and each sub-step is
// logged with fsync (stage()).
namespace npg {

struct LW { ggml_tensor *qkv, *out, *w1, *w3, *w2, *s[5], *n1, *n2, *n3, *n4, *qn, *kn; };

static ggml_tensor * lin(ggml_context * c, ggml_tensor * W, ggml_tensor * s, ggml_tensor * x) { return ggml_mul(c, ggml_mul_mat(c, W, x), s); }

static ggml_tensor * build_layer(ggml_context * c, ggml_tensor * x, LW & w, ggml_tensor * mods, int l, ggml_tensor * zero, ggml_tensor * theta,
                                 bool use_qkr) {
    const int S = (int) x->ne[1];
    auto mv = [&](int j) { return ggml_view_1d(c, mods, HID, ((size_t) l * 4 + j) * HID * 4); };
    ggml_tensor *sc_msa = mv(0), *g_msa = mv(1), *sc_mlp = mv(2), *g_mlp = mv(3);
    ggml_tensor * h = ggml_mul(c, ggml_rms_norm(c, x, 1e-5f), w.n1);
    h = ggml_modulate(c, h, sc_msa, zero, S, nullptr, nullptr);
    ggml_tensor * qkv = lin(c, w.qkv, w.s[0], h);
    qkv = ggml_reshape_4d(c, qkv, HD, 3 * NH, S, 1);
    ggml_tensor * q = ggml_view_4d(c, qkv, HD, NH, S, 1, qkv->nb[1], qkv->nb[2], qkv->nb[3], 0);
    ggml_tensor * k = ggml_view_4d(c, qkv, HD, NH, S, 1, qkv->nb[1], qkv->nb[2], qkv->nb[3], (size_t) NH * qkv->nb[1]);
    ggml_tensor * v = ggml_view_4d(c, qkv, HD, NH, S, 1, qkv->nb[1], qkv->nb[2], qkv->nb[3], (size_t) 2 * NH * qkv->nb[1]);
    if (use_qkr) {
        q = ggml_qknorm_rope(c, q, w.qn, theta, 1e-5f);   // -> [HD, S, NH]
        k = ggml_qknorm_rope(c, k, w.kn, theta, 1e-5f);
    } else {
        q = ggml_cont(c, ggml_permute(c, ggml_mul(c, ggml_rms_norm(c, q, 1e-5f), w.qn), 0, 2, 1, 3));
        k = ggml_cont(c, ggml_permute(c, ggml_mul(c, ggml_rms_norm(c, k, 1e-5f), w.kn), 0, 2, 1, 3));
        q = ggml_reshape_3d(c, q, HD, S, NH);
        k = ggml_reshape_3d(c, k, HD, S, NH);
    }
    const float kv_scale = 1.f / 128.f;
    k = ggml_cast(c, ggml_scale(c, k, kv_scale), GGML_TYPE_F16);
    v = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));
    v = ggml_reshape_3d(c, v, HD, S, NH);
    v = ggml_cast(c, ggml_scale(c, v, kv_scale), GGML_TYPE_F16);
    ggml_tensor * a = ggml_flash_attn_ext(c, q, k, v, nullptr, 1.f / std::sqrt((float) HD) / kv_scale, 0, 0);
    ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
    a = ggml_scale(c, a, 1.f / kv_scale);
    a = ggml_reshape_2d(c, a, HID, S);
    ggml_tensor * o = lin(c, w.out, w.s[1], a);
    o = ggml_mul(c, ggml_rms_norm(c, o, 1e-5f), w.n2);
    x = ggml_gated_residual(c, x, o, g_msa, S, nullptr);
    h = ggml_mul(c, ggml_rms_norm(c, x, 1e-5f), w.n3);
    h = ggml_modulate(c, h, sc_mlp, zero, S, nullptr, nullptr);
    ggml_tensor * f = ggml_swiglu_split(c, lin(c, w.w1, w.s[2], h), lin(c, w.w3, w.s[3], h));
    f = lin(c, w.w2, w.s[4], f);
    f = ggml_mul(c, ggml_rms_norm(c, f, 1e-5f), w.n4);
    return ggml_gated_residual(c, x, f, g_mlp, S, nullptr);
}

// per-layer tensor list, in creation order: 5 FP8 matrices, 5 F32 scales, 6 F32 norm weights
struct TSpec { ggml_type type; int64_t ne0, ne1; };
// FP8 E4M3 always on the device. Host test only (NPG_T8_CPU=2, the CPU backend has no FP8 matmul): the same FP8
// values expanded exactly to F32, to exercise the graph / alloc / timing code paths on the box.
static ggml_type g_t8_wtype = GGML_TYPE_F8_E4M3;
static std::vector<TSpec> layer_specs() {
    const ggml_type wt = g_t8_wtype;
    return { { wt, HID, 3 * HID }, { wt, HID, HID }, { wt, HID, FFN },
             { wt, HID, FFN },     { wt, FFN, HID },
             { GGML_TYPE_F32, 1, 1 }, { GGML_TYPE_F32, 1, 1 }, { GGML_TYPE_F32, 1, 1 }, { GGML_TYPE_F32, 1, 1 }, { GGML_TYPE_F32, 1, 1 },
             { GGML_TYPE_F32, HID, 1 }, { GGML_TYPE_F32, HID, 1 }, { GGML_TYPE_F32, HID, 1 }, { GGML_TYPE_F32, HID, 1 },
             { GGML_TYPE_F32, HD, 1 }, { GGML_TYPE_F32, HD, 1 } };
}
static ggml_tensor * new_spec(ggml_context * c, const TSpec & sp) {
    return sp.ne1 == 1 ? ggml_new_tensor_1d(c, sp.type, sp.ne0) : ggml_new_tensor_2d(c, sp.type, sp.ne0, sp.ne1);
}

struct T8Env {
    GB *   b = nullptr;
    int    S = M_TOK;
    size_t maxbuf = 0;   // weight buffer cap (bytes) = buft max size (or a smaller test override)
};

struct FsWeights {
    std::vector<std::unique_ptr<TSet>> sets;   // one context + backend buffer per chunk (each <= maxbuf)
    std::vector<LW> lw;
    double t_ctx = 0, t_alloc = 0, t_upload = 0, gb = 0;
    size_t nbuf = 0, max_chunk = 0;
};

static std::unique_ptr<FsWeights> fs_make_weights(const T8Env & E, int L, uint64_t seed) {
    auto W = std::make_unique<FsWeights>();
    GB & b = *E.b;
    double t = now_ms();
    stage("T8 L=%d: plan de buffers de pesos", L);
    const auto specs = layer_specs();
    const size_t align = ggml_backend_buft_get_alignment(b.buft);
    // exact per-tensor allocation sizes as the backend computes them (HTP repack pads FP8 to 32x32 tiles)
    std::vector<size_t> spec_size(specs.size());
    {
        ggml_init_params p = { specs.size() * ggml_tensor_overhead() + 256, nullptr, true };
        ggml_context * sc = ggml_init(p);
        if (!sc) { P("  T8: ggml_init (contexto de medida) fallo"); return nullptr; }
        for (size_t i = 0; i < specs.size(); ++i) spec_size[i] = pad_to(ggml_backend_buft_get_alloc_size(b.buft, new_spec(sc, specs[i])), align);
        ggml_free(sc);
    }
    std::vector<size_t> sizes;
    sizes.reserve(specs.size() * L);
    for (int l = 0; l < L; ++l) for (size_t i = 0; i < specs.size(); ++i) sizes.push_back(spec_size[i]);
    std::vector<Chunk> chunks;
    size_t bad = 0;
    if (!plan_chunks(sizes, E.maxbuf, chunks, &bad)) {
        P("  T8: el tensor #%zu (%zu MB) no cabe en un buffer de %s (max %zu MB) -> se omite", bad, sizes[bad] >> 20, b.name.c_str(), E.maxbuf >> 20);
        return nullptr;
    }
    // create tensors chunk by chunk
    W->lw.resize(L);
    size_t idx = 0;
    for (auto & ch : chunks) {
        auto ts = std::make_unique<TSet>((int) (ch.end - ch.begin) + 4);
        if (!ts->ctx) { P("  T8: ggml_init fallo"); return nullptr; }
        for (size_t k = ch.begin; k < ch.end; ++k, ++idx) {
            int l = (int) (k / specs.size()), j = (int) (k % specs.size());
            ggml_tensor * T = new_spec(ts->ctx, specs[j]);
            if (!T) { P("  T8: ggml_new_tensor fallo"); return nullptr; }
            LW & w = W->lw[l];
            ggml_tensor ** slot[16] = { &w.qkv, &w.out, &w.w1, &w.w3, &w.w2, &w.s[0], &w.s[1], &w.s[2], &w.s[3], &w.s[4],
                                        &w.n1, &w.n2, &w.n3, &w.n4, &w.qn, &w.kn };
            *slot[j] = T;
        }
        W->sets.push_back(std::move(ts));
    }
    W->t_ctx = now_ms() - t;
    // allocate one backend buffer per chunk (each <= max size), check every return value
    t = now_ms();
    for (size_t c = 0; c < chunks.size(); ++c) {
        stage("T8 L=%d: reservar buffer de pesos %zu/%zu (%.0f MB)", L, c + 1, chunks.size(), chunks[c].bytes / 1048576.0);
        if (!W->sets[c]->alloc(b, true)) {
            P("  T8: fallo reservando buffer %zu/%zu de %.0f MB (MemAvailable %ld MB) -> se omite", c + 1, chunks.size(), chunks[c].bytes / 1048576.0,
              mem_available_mb());
            return nullptr;
        }
        size_t sz = ggml_backend_buffer_get_size(W->sets[c]->buf);
        W->nbuf += sz;
        W->max_chunk = std::max(W->max_chunk, sz);
        if (sz > E.maxbuf) { P("  T8: buffer %zu de %zu MB > max %zu MB (inesperado) -> se omite", c + 1, sz >> 20, E.maxbuf >> 20); return nullptr; }
    }
    W->t_alloc = now_ms() - t;
    for (auto & w : W->lw)
        for (ggml_tensor * T : { w.qkv, w.out, w.w1, w.w3, w.w2, w.n1, w.qn })
            if (!T || !T->buffer || !T->data) { P("  T8: tensor sin memoria tras alloc -> se omite"); return nullptr; }
    // upload: staging buffer sized for the LARGEST tensor (0.1.1 bug: HID*FFN < qkv HID*3*HID -> heap overflow)
    size_t max_n = 0;
    for (auto & w : W->lw) for (ggml_tensor * T : { w.qkv, w.out, w.w1, w.w3, w.w2 }) max_n = std::max(max_n, ggml_nbytes(T));
    std::vector<uint8_t> h;
    try { h.resize(max_n); } catch (const std::exception &) { P("  T8: sin memoria para el buffer de subida de %zu MB", max_n >> 20); return nullptr; }
    double up = 0, bytes = 0;
    for (int l = 0; l < L && !g_stop; ++l) {
        stage("T8 L=%d: subir+reempaquetar pesos FP8 capa %d/%d", L, l + 1, L);
        LW & w = W->lw[l];
        int ti = 0;
        for (ggml_tensor * T : { w.qkv, w.out, w.w1, w.w3, w.w2 }) {
            size_t n = ggml_nbytes(T);
            if (n > h.size()) { P("  T8: tensor de %zu B > buffer de subida %zu B (bug) -> se omite", n, h.size()); return nullptr; }
            if (T->type == GGML_TYPE_F32) {   // host test only: exact FP8 -> F32 expansion
                size_t ne = (size_t) ggml_nelements(T);
                std::vector<uint8_t> h8(ne);
                fill_fp8_weights(h8.data(), ne, seed + l * 7 + ti++);
                float * f = (float *) h.data();
                for (size_t i = 0; i < ne; ++i) f[i] = g_e4m3_lut[h8[i]];
            } else {
                fill_fp8_weights(h.data(), n, seed + l * 7 + ti++);
            }
            double t1 = now_ms();
            ggml_backend_tensor_set(T, h.data(), 0, n);
            up += now_ms() - t1;
            bytes += n;
        }
        float sv[5] = { 0.02f, 0.02f, 0.02f, 0.02f, 0.02f };
        for (int i = 0; i < 5; ++i) ggml_backend_tensor_set(w.s[i], &sv[i], 0, 4);
        for (auto T : { w.n1, w.n2, w.n3, w.n4, w.qn, w.kn }) set_const_f32(T, 1.0f);
    }
    W->t_upload = up;
    W->gb = bytes / 1e9;
    return W;
}

// rows staged in VTCM by HTP element-wise ops must be <= 64 KB (see htp_rows_fit). theta of QKNORM_ROPE and
// FA K/V are streamed from DDR, so only dst/src0/src1 of non-view nodes are checked.
static bool graph_rows_fit(ggml_cgraph * gf, std::string & bad) {
    bool ok = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * n = ggml_graph_node(gf, i);
        if (op_is_view(n) || n->op == GGML_OP_FLASH_ATTN_EXT) continue;
        const ggml_tensor * ts[3] = { n, n->src[0], n->op == GGML_OP_MUL_MAT ? nullptr : n->src[1] };
        for (auto t : ts) {
            if (!t) continue;
            size_t row = ggml_row_size(t->type, t->ne[0]);
            if (row > (64u << 10)) {
                ok = false;
                char tmp[128];
                snprintf(tmp, sizeof tmp, "%s fila %zu KB; ", ggml_op_desc(n), row >> 10);
                if (bad.find(tmp) == std::string::npos) bad += tmp;
            }
        }
    }
    return ok;
}

struct StepT { double submit = NAN, sync = NAN, wall = NAN; int batches = 0, ops = 0; double dsp_ms = 0, wait_ms = 0, max_wait_ms = 0; bool ok = false; };
// one step: host submit (graph_compute: per-node param precompute + enqueue) and wait (synchronize), and the
// per-batch host gaps reported by ggml-hexagon (profile-batch host_wait_us = batch wall - DSP time)
static StepT step_timed(GB & b, ggml_cgraph * gf, const char * what) {
    StepT r;
    stage("T8 %s: compute (enviar)", what);
    g_prof.reset();
    double t0 = now_ms();
    if (ggml_backend_graph_compute_async(b.be, gf) != GGML_STATUS_SUCCESS) { P("  T8 %s: compute fallo en %s", what, b.name.c_str()); return r; }
    double t1 = now_ms();
    stage("T8 %s: sync", what);
    ggml_backend_synchronize(b.be);
    double t2 = now_ms();
    r.submit = t1 - t0; r.sync = t2 - t1; r.wall = t2 - t0; r.ok = true;
    std::lock_guard<std::mutex> lk(g_prof.mu);
    r.batches = g_prof.batches; r.ops = g_prof.ops; r.dsp_ms = g_prof.dsp_us / 1e3; r.wait_ms = g_prof.wait_us / 1e3; r.max_wait_ms = g_prof.max_wait_us / 1e3;
    return r;
}
static std::string step_str(const StepT & s) {
    char b[256];
    if (!s.ok) return "fallo";
    if (s.batches > 0)
        snprintf(b, sizeof b, "%.0f ms (enviar %.1f + esperar %.0f) | %d lotes/%d ops, DSP %.0f ms, huecos host %.1f ms (max %.1f)", s.wall, s.submit, s.sync,
                 s.batches, s.ops, s.dsp_ms, s.wait_ms, s.max_wait_ms);
    else snprintf(b, sizeof b, "%.0f ms (enviar %.1f + esperar %.0f)", s.wall, s.submit, s.sync);
    return b;
}

static void fs_experiment(const T8Env & E, int L) {
    GB & b = *E.b;
    const int S = E.S;
    const double gb_est = L * 0.176947;
    P("-- T8 primer paso (DiT sintetico de %d capas ZIT, pesos FP8 %.2f GB, S=%d) — desglose de tiempos de host", L, gb_est, S);
    long mem0 = mem_available_mb();
    // memory pre-check: weights + ~compute (S x FFN F32 x ~5) + inputs + 1 GB margin for the rest of the system
    const double need_mb = gb_est * 1e3 / 1.048576 + (double) S * FFN * 4 * 5 / 1048576.0 + 1024;
    if (mem0 > 0 && mem0 < need_mb) { P("  T8: MemAvailable %ld MB < ~%.0f MB necesarios -> se omite (evita que Android cierre la app)", mem0, need_mb); return; }
    double t_total0 = now_ms();
    auto W = fs_make_weights(E, L, 1000);
    if (!W || g_stop) { P("  T8: no se pudieron crear los pesos (ver arriba); se omite este experimento"); return; }
    P("  crear ctx/tensores %.1f ms | reservar %zu buffers (%.2f GB, max %zu MB c/u, mayor %.0f MB) %.1f ms | subir+reempaquetar FP8 %.0f ms (%.2f GB/s)",
      W->t_ctx, W->sets.size(), W->nbuf / 1e9, E.maxbuf >> 20, W->max_chunk / 1048576.0, W->t_alloc, W->t_upload, W->gb / (W->t_upload / 1e3));
    // inputs
    stage("T8 L=%d: entradas (alloc)", L);
    double t = now_ms();
    TSet in(8);
    ggml_tensor * x = ggml_new_tensor_2d(in.ctx, GGML_TYPE_F32, HID, S);
    ggml_tensor * mods = ggml_new_tensor_2d(in.ctx, GGML_TYPE_F32, HID * 4, L);
    ggml_tensor * zero = ggml_new_tensor_1d(in.ctx, GGML_TYPE_F32, HID);
    ggml_tensor * theta = ggml_new_tensor_1d(in.ctx, GGML_TYPE_F32, (int64_t) S * 2 * HD);
    if (!in.alloc(b)) return;
    size_t in_bytes = ggml_backend_buffer_get_size(in.buf);
    double t_in_alloc = now_ms() - t;
    stage("T8 L=%d: entradas (set)", L);
    t = now_ms();
    if (g_hd.X.size() < (size_t) S * HID) { P("  T8: datos X no preparados"); return; }
    ggml_backend_tensor_set(x, g_hd.X.data(), 0, ggml_nbytes(x));
    set_rand_f32(mods, 5, 0.05f);
    set_const_f32(zero, 0.f);
    {
        std::vector<float> th((size_t) S * 2 * HD);
        for (size_t i = 0; i < th.size(); i += 2) { float a = 0.001f * (float) (i % 977); th[i] = std::cos(a); th[i + 1] = std::sin(a); }
        ggml_backend_tensor_set(theta, th.data(), 0, th.size() * 4);
    }
    double t_in_set = now_ms() - t;
    // graph build + support / VTCM-row pre-checks (unsupported -> skip, never let ggml-hexagon abort)
    std::unique_ptr<GGraph> g;
    double t_build = 0;
    bool use_qkr = true;
    for (int attempt = 0; attempt < 2; ++attempt) {
        stage("T8 L=%d: construir grafo (%s)", L, use_qkr ? "qknorm_rope" : "sin qknorm_rope");
        t = now_ms();
        g = std::make_unique<GGraph>(L * 96 + 64);
        if (!g->ctx || !g->gf) { P("  T8: ggml_init del grafo fallo"); return; }
        ggml_tensor * h = x;
        for (int l = 0; l < L; ++l) h = build_layer(g->ctx, h, W->lw[l], mods, l, zero, theta, use_qkr);
        ggml_build_forward_expand(g->gf, h);
        g->out = h;
        std::string bad;
        bool ok = graph_supported(b, g->gf, bad);
        t_build = now_ms() - t;
        if (ok) break;
        P("  grafo %s: ops no soportadas: %s", use_qkr ? "con qknorm_rope" : "sin qknorm_rope", bad.c_str());
        if (!use_qkr) { P("  T8: se omite (ops no soportadas en %s)", b.name.c_str()); return; }
        use_qkr = false;
    }
    if (b.htp) {
        std::string bad;
        if (!graph_rows_fit(g->gf, bad)) { P("  T8: filas > 64 KB no caben en VTCM (%s) -> se omite (evita abortar)", bad.c_str()); return; }
    }
    stage("T8 L=%d: gallocr reserve", L);
    t = now_ms();
    g->ga = ggml_gallocr_new(b.buft);
    if (!g->ga || !ggml_gallocr_reserve(g->ga, g->gf)) { P("  T8: gallocr reserve fallo (MemAvailable %ld MB) -> se omite", mem_available_mb()); return; }
    double t_reserve = now_ms() - t;
    size_t cbuf = ggml_gallocr_get_buffer_size(g->ga, 0);
    // DSP virtual-memory pre-check: one op may touch a weight buffer + compute buffer + inputs; ggml-hexagon maps
    // at most ~3.2 GB per batch (HTP_OP_MAX_VMEM_DEFAULT minus queue); beyond that it would fail on the DSP.
    if (b.htp && (double) W->max_chunk + cbuf + in_bytes > 3.0e9) {
        P("  T8: pesos %.0f MB + compute %.0f MB + entradas %.0f MB > 3.0 GB de VA del DSP por lote -> se omite", W->max_chunk / 1e6, cbuf / 1e6, in_bytes / 1e6);
        return;
    }
    stage("T8 L=%d: gallocr alloc (compute %.0f MB)", L, cbuf / 1e6);
    t = now_ms();
    if (!ggml_gallocr_alloc_graph(g->ga, g->gf)) { P("  T8: gallocr alloc fallo -> se omite"); return; }
    double t_ga = now_ms() - t;
    P("  entradas: alloc %.1f ms (%.0f MB), set %.1f ms | grafo (%d nodos%s): construir+check %.1f ms | gallocr reserve %.1f ms + alloc %.1f ms (compute %.0f MB)",
      t_in_alloc, in_bytes / 1e6, t_in_set, ggml_graph_n_nodes(g->gf), use_qkr ? ", qknorm_rope" : ", SIN qknorm_rope", t_build, t_reserve, t_ga, cbuf / 1e6);
    // A: step 1 vs later steps (same graph, same buffers)
    StepT st[3];
    for (int i = 0; i < 3 && !g_stop; ++i) {
        char what[48]; snprintf(what, sizeof what, "L=%d paso %d/3", L, i + 1);
        progress("primer paso: compute %d/3 (%d capas)", i + 1, L);
        st[i] = step_timed(b, g->gf, what);
        if (!st[i].ok) return;
        P("  paso #%d: %s", i + 1, step_str(st[i]).c_str());
    }
    if (g_stop) return;
    stage("T8 L=%d: leer salida", L);
    std::vector<float> chk(64);
    ggml_backend_tensor_get(g->out, chk.data(), 0, 256);
    bool finite = true;
    for (float v : chk) if (!std::isfinite(v)) finite = false;
    double c0 = st[0].wall, c2 = st[2].wall;
    P("  compute #1 %.0f ms | #2 %.0f ms | #3 %.0f ms -> extra 1er paso %.0f ms (%.0f ms/capa, %.0f ms/GB pesos) | estable %.0f ms/capa -> x30 %.2f s | salida %s",
      c0, st[1].wall, c2, c0 - c2, (c0 - c2) / L, (c0 - c2) / W->gb, c2 / L, c2 / L * 30 / 1e3, finite ? "finita" : "NaN/Inf!");
    P("  desglose extra 1er paso: enviar %+.1f ms | esperar %+.0f ms (de ello DSP %+.0f ms, huecos host %+.1f ms)", st[0].submit - st[2].submit,
      st[0].sync - st[2].sync, st[0].dsp_ms - st[2].dsp_ms, st[0].wait_ms - st[2].wait_ms);
    // B: new compute allocation (as when a new image re-plans the graph)
    if (!g_stop) {
        stage("T8 L=%d: B) gallocr nuevo", L);
        // Rebuild the graph in a fresh context: re-running gallocr on the old graph would keep the old tensors'
        // data pointers (ggml-alloc treats tensors with data as pre-allocated) -> use-after-free of the freed buffer.
        g.reset();
        t = now_ms();
        auto gb2 = std::make_unique<GGraph>(L * 96 + 64);
        bool ok = gb2->ctx && gb2->gf;
        if (ok) {
            ggml_tensor * h = x;
            for (int l = 0; l < L; ++l) h = build_layer(gb2->ctx, h, W->lw[l], mods, l, zero, theta, use_qkr);
            ggml_build_forward_expand(gb2->gf, h);
            gb2->ga = ggml_gallocr_new(b.buft);
            ok = gb2->ga && ggml_gallocr_alloc_graph(gb2->ga, gb2->gf);
        }
        double ta = now_ms() - t;
        StepT b1, b2;
        if (ok) { b1 = step_timed(b, gb2->gf, "B #1"); if (b1.ok) b2 = step_timed(b, gb2->gf, "B #2"); }
        else P("  B) gallocr alloc fallo");
        P("  B) grafo + buffer de compute NUEVOS (pesos ya tocados): construir+alloc %.1f ms, compute #1 %s | #2 %s -> extra %.0f ms", ta, step_str(b1).c_str(), step_str(b2).c_str(),
          b1.wall - b2.wall);
    }
    // C: one fresh layer of weights (never touched by the DSP) on a new compute buffer
    if (!g_stop) {
        stage("T8 L=%d: C) 1 capa nueva", L);
        auto W1 = fs_make_weights(E, 1, 5000);
        if (W1) {
            GGraph g1(160);
            ggml_tensor * h = build_layer(g1.ctx, x, W1->lw[0], mods, 0, zero, theta, use_qkr);
            ggml_build_forward_expand(g1.gf, h);
            g1.ga = ggml_gallocr_new(b.buft);
            if (g1.ga && ggml_gallocr_alloc_graph(g1.ga, g1.gf)) {
                StepT c1 = step_timed(b, g1.gf, "C #1");
                StepT c2 = c1.ok ? step_timed(b, g1.gf, "C #2") : StepT();
                StepT c3 = c2.ok ? step_timed(b, g1.gf, "C #3") : StepT();
                P("  C) 1 capa con pesos NUEVOS + buffer compute nuevo (%.0f MB, subida %.0f ms): compute #1 %s | #2 %.0f | #3 %.0f -> extra %.0f ms (restar B/L para aislar pesos)",
                  W1->gb * 1e3, W1->t_upload, step_str(c1).c_str(), c2.wall, c3.wall, c1.wall - std::min(c2.wall, c3.wall));
            } else P("  C) gallocr alloc fallo");
        }
    }
    stage("T8 L=%d: liberar", L);
    long mem1 = mem_available_mb();
    P("  total experimento %.1f s | MemAvailable %ld -> %ld MB", (now_ms() - t_total0) / 1e3, mem0, mem1);
}

// wait up to max_s for the NPU to cool below target_c (T7 leaves it at ~100 C)
static void t8_cooldown(int max_s, double target_c) {
    double t0 = now_ms();
    double n0 = npu_temp_c();
    P("  enfriamiento previo (max %d s o NPU < %.0f C): %s", max_s, target_c, thermal_summary().c_str());
    if (std::isnan(n0)) P("  (temperatura de la NPU no legible: espera fija de %d s)", max_s);
    double n = n0;
    while (!g_stop && now_ms() - t0 < max_s * 1000.0) {
        n = npu_temp_c();
        if (!std::isnan(n) && n < target_c) break;
        int el = (int) ((now_ms() - t0) / 1000);
        progress("enfriando antes de T8: %d/%d s, NPU %.1f C", el, max_s, n);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    P("  enfriamiento: NPU %.1f -> %.1f C en %.0f s | %s | %s", n0, npu_temp_c(), (now_ms() - t0) / 1e3, thermal_summary().c_str(), clocks_summary().c_str());
}

static void t8_first_step(const Opts & o) {
    T8Env E;
    bool cpu_test = getenv("NPG_T8_CPU") && atoi(getenv("NPG_T8_CPU")) > 0;   // host test only
    if (g_htp.ok()) E.b = &g_htp;
    else if (cpu_test && g_cpu.ok()) { E.b = &g_cpu; if (atoi(getenv("NPG_T8_CPU")) == 2) g_t8_wtype = GGML_TYPE_F32; }
    else { P("-- T8 omitido (sin HTP)"); return; }
    if (getenv("NPG_T8_S")) E.S = std::max(32, std::min(M_TOK, atoi(getenv("NPG_T8_S"))));
    E.maxbuf = ggml_backend_buft_get_max_size(E.b->buft);
    if (getenv("NPG_T8_MAXBUF_MB")) { size_t m = (size_t) atol(getenv("NPG_T8_MAXBUF_MB")) << 20; if (m > 0 && m < E.maxbuf) E.maxbuf = m; }
    int L_small = o.fs_small;
    if (getenv("NPG_T8_L")) L_small = std::max(1, atoi(getenv("NPG_T8_L")));
    stage("T8: enfriamiento");
    if (E.b->htp) t8_cooldown(60, 60.0);
    P("  T8 backend %s | buffer max %zu MB | S=%d | %s", E.b->name.c_str(), E.maxbuf >> 20, E.S, thermal_summary().c_str());
    // per-submission host gaps: ggml-hexagon prints one "profile-batch" line per batch when GGML_HEXAGON_PROFILE=1
    const char * prev = getenv("GGML_HEXAGON_PROFILE");
    std::string prev_s = prev ? prev : "";
    if (E.b->htp) setenv("GGML_HEXAGON_PROFILE", "1", 1);
    g_prof_on = true;
    fs_experiment(E, L_small);
    if (!g_stop && o.fs_big_on && o.fs_big > L_small) {
        long mem = mem_available_mb();
        // ~177 MB FP8 per layer x1.15 + compute/inputs ~1.2 GB + 1.5 GB margin for Android
        int need_mb = (int) (o.fs_big * 177 * 1.15 + 2700);
        if (mem > 0 && mem < need_mb) {
            int Lfit = (int) ((mem - 2700) / (177 * 1.15));
            P("-- T8 grande: MemAvailable %ld MB < %d MB necesarios para %d capas; caben %d capas", mem, need_mb, o.fs_big, std::max(0, Lfit));
            if (Lfit > L_small) fs_experiment(E, std::min(Lfit, o.fs_big));
            else P("  T8 grande omitido (no aporta sobre %d capas)", L_small);
        } else {
            fs_experiment(E, o.fs_big);
        }
    }
    g_prof_on = false;
    if (E.b->htp) { if (prev) setenv("GGML_HEXAGON_PROFILE", prev_s.c_str(), 1); else unsetenv("GGML_HEXAGON_PROFILE"); }
    stage("T8: fin");
}

}  // namespace npg

// ================================================================ driver
namespace npg {

static void free_all() {
    g_htp_qkv.reset(); g_htp_fa.reset(); g_glora64.reset();
    for (auto & m : { &g_gX, &g_gX2, &g_gY, &g_gQ, &g_gK, &g_gV, &g_gO }) if (*m) { cl.clReleaseMemObject(*m); *m = nullptr; }
    for (auto & m : g_gW8) if (m) { cl.clReleaseMemObject(m); m = nullptr; }
    g_fa = FAData();
    g_hd.X2.clear(); g_hd.X2.shrink_to_fit();
    for (auto & w : g_hd.W) { w.clear(); w.shrink_to_fit(); }
}

static void summary() {
    P("==== RESUMEN / criterios go-no-go (plan §7) ====");
    double ratio = g_t1.htp_layer / g_t1.gpu_layer;
    P("  linears por capa: HMX %.0f ms | GPU FP8 %.0f ms | CPU ~%.0f ms -> GPU = %.1f%% del HMX", g_t1.htp_layer, g_t1.gpu_layer, g_t1.cpu_layer, ratio * 100);
    P("  FA por capa: HTP %.0f ms | GPU %.0f ms", g_fa_htp_ms, g_fa_gpu_ms);
    P("  contencion (HTP qkv || GPU qkv): HTP x%.2f, GPU x%.2f | S3 ganancia estimada %+.1f%%", g_contention_h, g_contention_g, g_s3_gain * 100);
    bool c1_gpu = ratio >= 0.08, c1_cont = g_contention_h <= 1.05;
    P("  (1) GPU >= 8%% del HMX: %s ; contencion t_HMX <= +5%%: %s -> S3/S5 %s", std::isnan(ratio) ? "?" : (c1_gpu ? "SI" : "NO"),
      std::isnan(g_contention_h) ? "?" : (c1_cont ? "SI" : "NO"), (std::isnan(ratio) || std::isnan(g_contention_h)) ? "SIN DATOS" : (c1_gpu && c1_cont) ? "SIGUEN" : "DESCARTADOS (seguir S1/S2/S4)");
    P("  (2) zero-copy dmabuf: %s", g_dmabuf_note.c_str());
    P("  LoRA por capa r=16/64/128: GPU %.1f/%.1f/%.1f ms | HTP %.1f/%.1f/%.1f ms", g_lora_gpu_ms[0], g_lora_gpu_ms[1], g_lora_gpu_ms[2], g_lora_htp_ms[0],
      g_lora_htp_ms[1], g_lora_htp_ms[2]);
    P("  (3) merge FP8 por canal: evaluado en la caja (Fase 2b), no en este APK");
}

int run(const Opts & o, const std::string & report_path) {
    g_stop = false;
    g_t0 = now_ms();
    { std::lock_guard<std::mutex> lk(g_mu); g_rep.clear(); g_file = report_path; }
    if (!report_path.empty()) { FILE * f = fopen(report_path.c_str(), "w"); if (f) fclose(f); }
    install_crash_handlers(report_path);
    init_e4m3();
    ggml_log_set(ggml_log_cb, nullptr);
    ggml_set_abort_callback([](const char * msg) {
        std::string m = std::string("** ABORTO de ggml (la app se cerrara) en la etapa '") + g_stage + "': " + (msg ? msg : "?");
        while (!m.empty() && (m.back() == '\n' || m.back() == ' ')) m.pop_back();
        emit(m, false);
    });
    int threads = o.cpu_threads > 0 ? o.cpu_threads : (int) std::max(1u, std::thread::hardware_concurrency());
    P("NPG bench 0.1.2 (NPU+GPU, Fase 2) | modo %s | hilos CPU %d | MemAvailable %ld MB", o.quick ? "rapido" : "completo", threads, mem_available_mb());
    P("inicio: %s | %s", thermal_summary().c_str(), clocks_summary().c_str());
    {
        std::string sk = getenv("NPG_SKIP") ? getenv("NPG_SKIP") : "";
        if (sk.find("HTP") == std::string::npos) htp_init(); else P("HTP omitido por el usuario");
        if (sk.find("GPU") == std::string::npos) gpu_init(); else P("GPU omitida por el usuario");
    }
    cpu_init(threads);
    prepare_host_data();
    struct Step { const char * n; std::function<void()> f; };
    std::vector<Step> steps = {
        { "T1a", [&] { if (g_htp.ok()) t1_htp(); } },
        { "T1b", [&] { if (g_gpu.ok) t1_gpu(); } },
        { "T1c", [&] { if (!o.quick) t1_cpu(threads); } },
        { "T2", [&] { t2_fa(!o.quick); } },
        { "T3", [&] { t3_elem(!o.quick); } },
        { "T4", [&] { t4_lora(); } },
        { "T5", [&] { if (g_htp.ok()) t5_concurrency(o.quick ? 3000 : 6000); } },
        { "T6", [&] { t6_sync_dmabuf(); } },
        { "T7", [&] { if (g_htp.ok()) t7_sustained(o.quick ? 20 : o.sustain_s); } },
        { "free", [&] { free_all(); } },
        { "T8", [&] { Opts oo = o; if (o.quick) oo.fs_big_on = false; t8_first_step(oo); } },
        { "T6b", [&] { t6b_handoff(); } },
    };
    std::string skip = getenv("NPG_SKIP") ? getenv("NPG_SKIP") : "";
    if (!skip.empty()) P("omitidas por el usuario: %s", skip.c_str());
    for (auto & s : steps) {
        if (g_stop) { P("** cancelado por el usuario **"); break; }
        {
            std::string n = s.n;
            std::string key = n.substr(0, 2);  // "T1".."T8" (or "fr")
            bool sk = false;
            for (size_t p = 0; p <= skip.size();) {
                size_t e = skip.find_first_of(", ", p); if (e == std::string::npos) e = skip.size();
                std::string t = skip.substr(p, e - p);
                for (auto & c : t) c = (char) toupper((unsigned char) c);
                std::string nu = n; for (auto & c : nu) c = (char) toupper((unsigned char) c);
                if (!t.empty() && (t == nu || t == key)) sk = true;
                p = e + 1;
            }
            if (sk && n != "free") { P("-- %s omitida", s.n); continue; }
        }
        progress("inicio %s (total %.0f s)", s.n, tsec());
        stage("%s inicio", s.n);
        double t = now_ms();
        try { s.f(); } catch (const std::exception & e) { P("  %s excepcion: %s", s.n, e.what()); }
        progress("%s listo en %.1f s (total %.0f s)", s.n, (now_ms() - t) / 1e3, tsec());
    }
    stage("resumen");
    summary();
    P("fin: %.0f s | %s | %s", tsec(), thermal_summary().c_str(), clocks_summary().c_str());
    return 0;
}

std::string report() { std::lock_guard<std::mutex> lk(g_mu); return g_rep; }

}  // namespace npg

#ifdef __ANDROID__
// ---------------------------------------------------------------- JNI
static JavaVM * g_vm = nullptr;
static jobject g_cb_obj = nullptr;
static jmethodID g_cb_mid = nullptr;

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM * vm, void *) { g_vm = vm; return JNI_VERSION_1_6; }

extern "C" JNIEXPORT jstring JNICALL Java_com_zithexagon_npgbench_NpgNative_run(JNIEnv * env, jclass, jint quick, jint sustain_s, jint fs_big,
                                                                                jstring path, jobject cb) {
    const char * p = env->GetStringUTFChars(path, nullptr);
    std::string sp(p);
    env->ReleaseStringUTFChars(path, p);
    g_cb_obj = env->NewGlobalRef(cb);
    jclass cls = env->GetObjectClass(cb);
    g_cb_mid = env->GetMethodID(cls, "onLine", "(Ljava/lang/String;Z)V");
    npg::g_sink = [](const std::string & s, bool prog) {
        JNIEnv * e = nullptr;
        bool attached = false;
        if (g_vm->GetEnv((void **) &e, JNI_VERSION_1_6) != JNI_OK) { g_vm->AttachCurrentThread(&e, nullptr); attached = true; }
        // sanitize to modified UTF-8 safe ASCII
        std::string a; a.reserve(s.size());
        for (unsigned char ch : s) a.push_back(ch < 0x80 ? (char) ch : '?');
        jstring js = e->NewStringUTF(a.c_str());
        e->CallVoidMethod(g_cb_obj, g_cb_mid, js, (jboolean) prog);
        e->DeleteLocalRef(js);
        if (e->ExceptionCheck()) e->ExceptionClear();
        if (attached) g_vm->DetachCurrentThread();
    };
    npg::Opts o;
    o.quick = quick != 0;
    o.sustain_s = sustain_s > 0 ? sustain_s : 60;
    o.fs_big = fs_big > 0 ? fs_big : 30;
    npg::run(o, sp);
    npg::g_sink = nullptr;
    env->DeleteGlobalRef(g_cb_obj);
    g_cb_obj = nullptr;
    std::string r = npg::report();
    for (auto & ch : r) if ((unsigned char) ch >= 0x80) ch = '?';
    return env->NewStringUTF(r.c_str());
}

extern "C" JNIEXPORT void JNICALL Java_com_zithexagon_npgbench_NpgNative_cancel(JNIEnv *, jclass) { npg::g_stop = true; }
#else
#include <execinfo.h>
#include <signal.h>
static void segv(int sig) {
    void * b[64]; int n = backtrace(b, 64);
    fprintf(stderr, "signal %d\n", sig); backtrace_symbols_fd(b, n, 2); _exit(1);
}
int main(int argc, char ** argv) {
    signal(SIGSEGV, segv);
    npg::Opts o;
    o.quick = argc > 1 && !strcmp(argv[1], "quick");
    o.sustain_s = 10;
    o.fs_big_on = false;
    return npg::run(o, "/tmp/npg_report.txt");
}
#endif
