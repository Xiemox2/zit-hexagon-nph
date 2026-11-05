// NPH JNI: Hexagon sd.cpp engine for ZIT-HEXAGON (branded NPH = NPU + Hexagon).
// Based on sdbench_jni; adds create/free residency + LoRA bindings. MIT.
#include <jni.h>
#include <android/log.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <mutex>
#include <string>
#include <vector>
#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include "stable-diffusion.h"
#include "ggml-hexagon.h"
#include "../sdcpp/src/nph_stats.h"
#include "../sdcpp/src/nph_cache.h"
#include "../sdcpp/src/nph_te_cache.h"
#include "ggml-backend.h"
#include "npg_selftest.h"
#ifdef NPH_HAVE_OPENCL
#include "ggml-opencl.h"
extern "C" int nph_ocl_shim_load(const char** info);
#endif
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <sys/stat.h>

#define TAG "nph"

static JavaVM* g_vm = nullptr;
static std::mutex g_cb_mutex;
static jobject g_cb = nullptr;
static jmethodID g_on_log = nullptr;
static jmethodID g_on_progress = nullptr;
static std::atomic<sd_ctx_t*> g_cancel_target{nullptr};

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static JNIEnv* get_env() {
    if (!g_vm) return nullptr;
    JNIEnv* env = nullptr;
    if (g_vm->GetEnv((void**)&env, JNI_VERSION_1_6) == JNI_OK) return env;
    if (g_vm->AttachCurrentThreadAsDaemon(&env, nullptr) == JNI_OK) return env;
    return nullptr;
}

static jstring new_jstring_safe(JNIEnv* env, const char* s) {
    std::string out;
    const unsigned char* p = (const unsigned char*)s;
    while (*p) {
        unsigned char c = *p;
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 0;
        bool ok = len > 0;
        for (int i = 1; ok && i < len; i++) ok = (p[i] & 0xC0) == 0x80;
        if (!ok) { out.push_back('?'); p++; continue; }
        out.append((const char*)p, len); p += len;
    }
    return env->NewStringUTF(out.c_str());
}

static void npg_capture(const char* text);
static void log_cb(enum sd_log_level_t level, const char* text, void*) {
    if (!text) return;
    npg_capture(text);
    __android_log_print(level >= SD_LOG_ERROR ? ANDROID_LOG_ERROR : level == SD_LOG_WARN ? ANDROID_LOG_WARN : ANDROID_LOG_INFO, TAG, "%s", text);
    std::lock_guard<std::mutex> lk(g_cb_mutex);
    if (!g_cb || !g_on_log) return;
    JNIEnv* env = get_env(); if (!env) return;
    jstring js = new_jstring_safe(env, text);
    env->CallVoidMethod(g_cb, g_on_log, (jint)level, js);
    if (env->ExceptionCheck()) env->ExceptionClear();
    env->DeleteLocalRef(js);
}

// ---------------- NPH 0.28 per-phase timing ----------------
struct NphSnap {
    std::string label;
    int step = -1;
    double t = 0;
    uint64_t sd[NPH_SD_COUNT] = {};
    uint64_t hx[GGML_HEXAGON_NPH_CTR_COUNT] = {};
};
static std::mutex g_snap_mtx;
static std::vector<NphSnap> g_snaps;
static std::atomic<bool> g_recording{false};
static std::atomic<int> g_sampler_steps{0};
static std::mutex g_report_mtx;
static std::string g_last_report;

static void take_snap(const char* label, int step) {
    if (!g_recording.load()) return;
    NphSnap s;
    s.label = label;
    s.step = step;
    s.t = now_s();
    nph_sd_counters(s.sd, NPH_SD_COUNT);
    ggml_backend_hexagon_nph_counters(s.hx, GGML_HEXAGON_NPH_CTR_COUNT);
    std::lock_guard<std::mutex> lk(g_snap_mtx);
    g_snaps.push_back(std::move(s));
}

// background page-cache prefetch (DiT/cache file during TE; VAE during sampling)
struct NphPrefetch {
    std::thread th;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> bytes{0};
    double secs = 0;
    std::string what;
    void start(const std::string& path, const std::string& label) {
        join();
        stop = false; bytes = 0; secs = 0; what = label;
        th = std::thread([this, path]() {
            setpriority(PRIO_PROCESS, 0, 10);
            const double t0 = now_s();
            int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) return;
            posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
            std::vector<char> buf(16u << 20);
            off_t off = 0;
            while (!stop.load()) {
                ssize_t r = pread(fd, buf.data(), buf.size(), off);
                if (r <= 0) break;
                off += r;
                bytes += (uint64_t)r;
            }
            close(fd);
            secs = now_s() - t0;
        });
    }
    void join() { if (th.joinable()) { stop = true; th.join(); } }
};
static NphPrefetch g_pf_dit, g_pf_vae;
static std::string g_pf_vae_path;

static void mark_cb(const char* name) {
    take_snap(name, -1);
    if (strcmp(name, "sample_begin") == 0 && !g_pf_vae_path.empty()) {
        g_pf_vae.start(g_pf_vae_path, "VAE");
        g_pf_vae_path.clear();
    }
}

static size_t hk_f8_size(ggml_tensor* t) { return ggml_backend_hexagon_nph_f8_weight_size(t); }
static bool hk_view(ggml_tensor* t, void** d, size_t* n) { return ggml_backend_hexagon_nph_tiled_view(t, d, n); }
static bool hk_mark(ggml_tensor* t) { return ggml_backend_hexagon_nph_mark_prepacked(t); }
static bool hk_par(ggml_tensor* t) { return ggml_backend_hexagon_nph_parallel_set_ok(t); }
static void hk_set_fuse(int on) { ggml_backend_hexagon_nph_set_lora_fuse(on); }
static int hk_get_fuse() { return ggml_backend_hexagon_nph_get_lora_fuse(); }
static uint64_t hk_fused_n() { return ggml_backend_hexagon_nph_lora_fused_count(); }
static const nph_hex_hooks g_hex_hooks = {hk_f8_size, hk_view, hk_mark, hk_par, hk_set_fuse, hk_get_fuse, hk_fused_n};

static std::string fmt_phase(const NphSnap& a, const NphSnap& b, bool detail) {
    auto sd = [&](int i) { return (b.sd[i] - a.sd[i]) / 1e9; };
    auto sdb = [&](int i) { return (double)(b.sd[i] - a.sd[i]); };
    auto hx = [&](int i) { return (b.hx[i] - a.hx[i]) / 1e9; };
    auto hxn = [&](int i) { return (unsigned long long)(b.hx[i] - a.hx[i]); };
    const double wall = b.t - a.t;
    const double start = sd(NPH_SD_RUNNER_START_NS), build = sd(NPH_SD_GRAPH_BUILD_NS), meas = sd(NPH_SD_MEASURE_NS),
                 wts = sd(NPH_SD_WEIGHTS_NS), ws = sd(NPH_SD_WS_ALLOC_NS), in = sd(NPH_SD_INPUT_NS),
                 ex = sd(NPH_SD_EXEC_NS), out = sd(NPH_SD_OUTPUT_NS);
    const double accounted = start + build + meas + wts + ws + in + ex + out;
    char buf[1400];
    int n = snprintf(buf, sizeof buf, "%.2f s", wall);
    const double load = sd(NPH_SD_LOAD_NS);
    if (load > 0.0005 || detail) {
        const double rp = hx(GGML_HEXAGON_NPH_REPACK_NS), rpb = (double)(b.hx[GGML_HEXAGON_NPH_REPACK_BYTES] - a.hx[GGML_HEXAGON_NPH_REPACK_BYTES]);
        n += snprintf(buf + n, sizeof buf - n,
                      " | carga pesos %.2f (%.0f MB; lectura Σ%.2f, set Σ%.2f, repack Σ%.2f%s%s, caché hit %.0f MB/%.2f, escrito %.0f MB/%.2f)",
                      load, sdb(NPH_SD_LOAD_BYTES) / 1048576.0, sd(NPH_SD_READ_NS), sd(NPH_SD_SETT_NS), rp,
                      rp > 0 ? " @" : "", rp > 0 ? (std::to_string((int)(rpb / rp / 1e6)) + " MB/s").c_str() : "",
                      sdb(NPH_SD_CACHE_HIT_BYTES) / 1048576.0, sd(NPH_SD_CACHE_HIT_NS),
                      sdb(NPH_SD_CACHE_WRITE_BYTES) / 1048576.0, sd(NPH_SD_CACHE_WRITE_NS));
    }
    n += snprintf(buf + n, sizeof buf - n,
                  " | inicio runner %.2f · prep pesos %.2f · grafo %.2f · medida %.2f · reserva %.2f (rpcmem %.3f ×%llu, mmap NPU %.3f ×%llu, free %.3f, pool hit %llu/miss %llu)"
                  " · entrada %.3f · cómputo %.2f (envío %.2f ×%llu, espera NPU %.2f) · salida %.3f · huecos host %.2f",
                  start, wts, build, meas, ws, hx(GGML_HEXAGON_NPH_RPCMEM_NS), hxn(GGML_HEXAGON_NPH_RPCMEM_N),
                  hx(GGML_HEXAGON_NPH_MMAP_NS), hxn(GGML_HEXAGON_NPH_MMAP_N), hx(GGML_HEXAGON_NPH_FREE_NS),
                  hxn(GGML_HEXAGON_NPH_POOL_HIT), hxn(GGML_HEXAGON_NPH_POOL_MISS), in, ex,
                  hx(GGML_HEXAGON_NPH_SUBMIT_NS), hxn(GGML_HEXAGON_NPH_SUBMIT_N), hx(GGML_HEXAGON_NPH_WAIT_NS), out,
                  wall - accounted > 0 ? wall - accounted : 0.0);
    return std::string(buf, std::min<int>(n, (int)sizeof buf - 1));
}

static std::string build_phase_report(double gen_s) {
    std::vector<NphSnap> s;
    {
        std::lock_guard<std::mutex> lk(g_snap_mtx);
        s = g_snaps;
    }
    std::string r = "— Fases NPH 0.29 (por imagen) —\n";
    if (s.size() < 2) return r + "(sin datos)\n";
    auto find = [&](const char* l) -> int {
        for (size_t i = 0; i < s.size(); ++i) if (s[i].label == l) return (int)i;
        return -1;
    };
    const int g0 = 0, te0 = find("te_begin"), te1 = find("te_end"), sb = find("sample_begin"), se = find("sample_end"),
              v0 = find("vae_begin"), v1 = find("vae_end"), gend = (int)s.size() - 1;
    if (te0 > g0) r += "Pre-TE (LoRA/latentes): " + fmt_phase(s[g0], s[te0], false) + "\n";
    if (te0 >= 0 && te1 > te0) r += "TE: " + fmt_phase(s[te0], s[te1], false) + "\n";
    if (sb >= 0) {
        int prev = sb;
        for (size_t i = sb + 1; i < s.size(); ++i) {
            if (s[i].label != "step") continue;
            char h[48];
            snprintf(h, sizeof h, "Paso %d: ", s[i].step);
            r += h + fmt_phase(s[prev], s[i], s[i].step == 1) + "\n";
            prev = (int)i;
        }
        if (se > prev) r += "Tras pasos: " + fmt_phase(s[prev], s[se], false) + "\n";
    }
    if (v0 >= 0 && v1 > v0) r += "VAE: " + fmt_phase(s[v0], s[v1], false) + "\n";
    r += "Total generate: " + fmt_phase(s[g0], s[gend], true) + "\n";
    {
        const auto& a = s[g0]; const auto& b = s[gend];
        char buf[512];
        const double rp = (b.hx[GGML_HEXAGON_NPH_REPACK_NS] - a.hx[GGML_HEXAGON_NPH_REPACK_NS]) / 1e9;
        const double rpb = (double)(b.hx[GGML_HEXAGON_NPH_REPACK_BYTES] - a.hx[GGML_HEXAGON_NPH_REPACK_BYTES]);
        snprintf(buf, sizeof buf,
                 "Repack FP8: %.0f MB en Σ%.2f s (%s, hilos≤%s) · pool reutilizado %.0f MB · rechazos caché %llu\n",
                 rpb / 1048576.0, rp, getenv("NPH_FAST_REPACK") && strcmp(getenv("NPH_FAST_REPACK"), "0") == 0 ? "lento/ref" : "rápido",
                 getenv("NPH_REPACK_THREADS") ? getenv("NPH_REPACK_THREADS") : "4",
                 (b.hx[GGML_HEXAGON_NPH_POOL_HIT_BYTES] - a.hx[GGML_HEXAGON_NPH_POOL_HIT_BYTES]) / 1048576.0,
                 (unsigned long long)(b.sd[NPH_SD_CACHE_REJECT] - a.sd[NPH_SD_CACHE_REJECT]));
        r += buf;
    }
    if (g_pf_dit.what.size()) {
        char buf[200];
        snprintf(buf, sizeof buf, "Prefetch %s: %.0f MB en %.2f s\n", g_pf_dit.what.c_str(), g_pf_dit.bytes.load() / 1048576.0, g_pf_dit.secs);
        r += buf;
    }
    if (g_pf_vae.what.size()) {
        char buf[200];
        snprintf(buf, sizeof buf, "Prefetch VAE (exp): %.0f MB en %.2f s\n", g_pf_vae.bytes.load() / 1048576.0, g_pf_vae.secs);
        r += buf;
    }
    {
        nph_te::Stats ts = nph_te::stats();
        const char* on = getenv("NPH_TE_CACHE");
        const bool te_on = !(on && (strcmp(on, "0") == 0 || strcmp(on, "false") == 0));
        char buf[512];
        snprintf(buf, sizeof buf, "Caché TE: %s · RAM aciertos %llu · disco aciertos %llu (carga Σ%.0f ms) · fallos %llu · guardados %llu · rechazados %llu%s%s\n",
                 te_on ? "activa" : "desactivada", (unsigned long long)ts.ram_hits, (unsigned long long)ts.disk_hits,
                 ts.disk_load_us / 1000.0, (unsigned long long)ts.misses, (unsigned long long)ts.disk_stores,
                 (unsigned long long)ts.disk_rejects, ts.last.empty() ? "" : " · último: ", ts.last.c_str());
        r += buf;
    }
    (void)gen_s;
    return r;
}

static void progress_cb(int step, int steps, float time, void*) {
    if (steps == g_sampler_steps.load() && step >= 1 && step <= steps) take_snap("step", step);
    std::lock_guard<std::mutex> lk(g_cb_mutex);
    if (!g_cb || !g_on_progress) return;
    JNIEnv* env = get_env(); if (!env) return;
    env->CallVoidMethod(g_cb, g_on_progress, (jint)step, (jint)steps, (jfloat)time);
    if (env->ExceptionCheck()) env->ExceptionClear();
}

struct JStr {
    JNIEnv* env; jstring js; const char* c;
    JStr(JNIEnv* e, jstring s) : env(e), js(s), c(s ? e->GetStringUTFChars(s, nullptr) : "") {}
    ~JStr() { if (js) env->ReleaseStringUTFChars(js, c); }
};

struct NphHandle {
    sd_ctx_t* ctx = nullptr;
    std::string dit, te, vae, backend, params_backend;
    int threads = 4;
    int flags = 0;
    int gens = 0;  // generations run on this ctx (0 = DiT weights not loaded yet)
    // LoRA identity baked into this ctx (changing forces recreate)
    std::string lora_key;
};

static void bind_cb(JNIEnv* env, jobject cb) {
    std::lock_guard<std::mutex> lk(g_cb_mutex);
    if (g_cb) env->DeleteGlobalRef(g_cb);
    g_cb = cb ? env->NewGlobalRef(cb) : nullptr;
    if (cb) {
        jclass cls = env->GetObjectClass(cb);
        g_on_log = env->GetMethodID(cls, "onLog", "(ILjava/lang/String;)V");
        g_on_progress = env->GetMethodID(cls, "onProgress", "(IIF)V");
        env->DeleteLocalRef(cls);
    }
}

static void clear_cb(JNIEnv* env) {
    sd_set_progress_callback(nullptr, nullptr);
    std::lock_guard<std::mutex> lk(g_cb_mutex);
    if (g_cb) { env->DeleteGlobalRef(g_cb); g_cb = nullptr; }
}

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
    g_vm = vm;
    nph_set_hex_hooks(&g_hex_hooks);
    nph_set_mark_cb(mark_cb);
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_zithexagon_engine_NphNative_info(JNIEnv* env, jclass) {
    sd_set_log_callback(log_cb, nullptr);
    std::string s = "NPH (sd.cpp Hexagon)\nversion: ";
    s += sd_version(); s += " commit: "; s += sd_commit();
    s += "\nsystem: "; s += sd_get_system_info();
    s += "\ndevices:\n";
    size_t n = sd_list_devices(nullptr, 0);
    std::vector<char> buf(n + 1, 0);
    sd_list_devices(buf.data(), buf.size());
    s += buf.data();
    return new_jstring_safe(env, s.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_zithexagon_engine_NphNative_memAvailableMb(JNIEnv*, jclass) {
    struct sysinfo si{};
    if (sysinfo(&si) != 0) return -1;
    // MemAvailable-ish: freeram + bufferram (conservative; /proc/meminfo would be better)
    long long avail = (long long)(si.freeram + si.bufferram) * si.mem_unit;
    // Prefer MemAvailable from /proc
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            long v = 0;
            if (sscanf(line, "MemAvailable: %ld kB", &v) == 1) { avail = (long long)v * 1024; break; }
        }
        fclose(f);
    }
    return (jint)(avail / (1024 * 1024));
}

extern "C" JNIEXPORT void JNICALL
Java_com_zithexagon_engine_NphNative_cancel(JNIEnv*, jclass) {
    sd_ctx_t* c = g_cancel_target.load();
    if (c) sd_cancel_generation(c, SD_CANCEL_ALL);
}

// flags: bit0 fa, bit1 vae_conv_direct, bit2 mmap, bit3 eager_load
extern "C" JNIEXPORT jlong JNICALL
Java_com_zithexagon_engine_NphNative_create(JNIEnv* env, jclass,
        jstring jdiff, jstring jllm, jstring jvae,
        jstring jbackend, jstring jparams_backend,
        jint threads, jint flags, jobject cb, jdoubleArray jtimings) {
    JStr diff(env, jdiff), llm(env, jllm), vae(env, jvae);
    JStr backend(env, jbackend), params_backend(env, jparams_backend);
    bind_cb(env, cb);
    sd_set_log_callback(log_cb, nullptr);

    sd_ctx_params_t cp;
    sd_ctx_params_init(&cp);
    cp.diffusion_model_path = diff.c;
    cp.llm_path = llm.c;
    cp.vae_path = vae.c;
    cp.n_threads = threads > 0 ? threads : 4;
    cp.flash_attn = (flags & 1) != 0;
    cp.vae_conv_direct = (flags & 2) != 0;
    cp.enable_mmap = (flags & 4) != 0;
    cp.eager_load = (flags & 8) != 0;
    cp.backend = backend.c;
    cp.params_backend = params_backend.c;
    cp.lora_apply_mode = LORA_APPLY_AUTO; // at_runtime for quantized/FP8

    char* cps = sd_ctx_params_to_str(&cp);
    if (cps) { log_cb(SD_LOG_INFO, cps, nullptr); free(cps); }

    double t0 = now_s();
    sd_ctx_t* ctx = new_sd_ctx(&cp);
    double create_s = now_s() - t0;
    if (jtimings && env->GetArrayLength(jtimings) > 0) {
        double t = create_s;
        env->SetDoubleArrayRegion(jtimings, 0, 1, &t);
    }
    clear_cb(env);
    if (!ctx) {
        log_cb(SD_LOG_ERROR, "nph: new_sd_ctx failed\n", nullptr);
        return 0;
    }
    auto* h = new NphHandle();
    h->ctx = ctx;
    h->dit = diff.c; h->te = llm.c; h->vae = vae.c;
    h->backend = backend.c; h->params_backend = params_backend.c;
    h->threads = cp.n_threads; h->flags = flags;
    g_cancel_target.store(ctx);
    return reinterpret_cast<jlong>(h);
}

extern "C" JNIEXPORT void JNICALL
Java_com_zithexagon_engine_NphNative_free(JNIEnv*, jclass, jlong handle) {
    if (!handle) return;
    auto* h = reinterpret_cast<NphHandle*>(handle);
    if (h->ctx) {
        if (g_cancel_target.load() == h->ctx) g_cancel_target.store(nullptr);
        free_sd_ctx(h->ctx);
        h->ctx = nullptr;
    }
    delete h;
}


// Same sd_img_gen_params for generate() and teProbe() (the TE-cache key must match exactly).
struct NphGenArgs {
    std::vector<sd_lora_t> loras;
    std::vector<std::string> lora_paths_owned;
    sd_img_gen_params_t gp;
};

static void nph_build_gen_args(JNIEnv* env, NphGenArgs& ga, const char* prompt, jint steps, jint width, jint height,
                               jlong seed, jfloat cfg, jobjectArray jloraPaths, jfloatArray jloraStrengths) {
    auto& loras            = ga.loras;
    auto& lora_paths_owned = ga.lora_paths_owned;
    if (jloraPaths && jloraStrengths) {
        jsize n = env->GetArrayLength(jloraPaths);
        jsize ns = env->GetArrayLength(jloraStrengths);
        jsize m = std::min(n, ns);
        jfloat* str = env->GetFloatArrayElements(jloraStrengths, nullptr);
        lora_paths_owned.reserve(m);
        for (jsize i = 0; i < m; i++) {
            auto js = (jstring)env->GetObjectArrayElement(jloraPaths, i);
            if (!js) continue;
            const char* p = env->GetStringUTFChars(js, nullptr);
            lora_paths_owned.emplace_back(p ? p : "");
            env->ReleaseStringUTFChars(js, p);
            env->DeleteLocalRef(js);
            if (lora_paths_owned.back().empty() || str[i] == 0.f) continue;
            sd_lora_t L{};
            L.path = lora_paths_owned.back().c_str();
            L.multiplier = str[i];
            L.is_high_noise = false;
            loras.push_back(L);
        }
        env->ReleaseFloatArrayElements(jloraStrengths, str, JNI_ABORT);
    }
    sd_img_gen_params_t& gp = ga.gp;
    sd_img_gen_params_init(&gp);
    gp.prompt = prompt;
    gp.negative_prompt = "";
    gp.width = width;
    gp.height = height;
    gp.seed = seed;
    gp.batch_count = 1;
    gp.sample_params.sample_steps = steps;
    gp.sample_params.sample_method = EULER_SAMPLE_METHOD;
    gp.sample_params.guidance.txt_cfg = cfg > 0 ? cfg : 1.0f;
    if (!loras.empty()) {
        gp.loras = loras.data();
        gp.lora_count = (uint32_t)loras.size();
    }
}

// 0.30: 2 = TE-cache RAM hit, 1 = disk hit, 0 = miss (TE ~4 GB will load), -1 = n/a / no ctx.
extern "C" JNIEXPORT jint JNICALL
Java_com_zithexagon_engine_NphNative_teProbe(JNIEnv* env, jclass, jlong handle,
        jstring jprompt, jint steps, jint width, jint height, jfloat cfg,
        jobjectArray jloraPaths, jfloatArray jloraStrengths) {
    if (!handle) return -1;
    auto* h = reinterpret_cast<NphHandle*>(handle);
    if (!h->ctx) return -1;
    JStr prompt(env, jprompt);
    NphGenArgs ga;
    nph_build_gen_args(env, ga, prompt.c, steps, width, height, 0, cfg, jloraPaths, jloraStrengths);
    return (jint)nph_te_cache_probe(h->ctx, &ga.gp);
}

// timings out: [0] generate_s, [1] cpu_user, [2] cpu_sys  (create time reported separately)
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_zithexagon_engine_NphNative_generate(JNIEnv* env, jclass, jlong handle,
        jstring jprompt, jint steps, jint width, jint height, jlong seed, jfloat cfg,
        jobjectArray jloraPaths, jfloatArray jloraStrengths,
        jobject cb, jdoubleArray jtimings) {
    if (!handle) return nullptr;
    auto* h = reinterpret_cast<NphHandle*>(handle);
    if (!h->ctx) return nullptr;
    JStr prompt(env, jprompt);
    bind_cb(env, cb);
    sd_set_log_callback(log_cb, nullptr);
    sd_set_progress_callback(progress_cb, nullptr);
    nph_npg_notes_clear();
    g_cancel_target.store(h->ctx);

    NphGenArgs ga;
    nph_build_gen_args(env, ga, prompt.c, steps, width, height, seed, cfg, jloraPaths, jloraStrengths);
    sd_img_gen_params_t& gp = ga.gp;

    // phase timing + prefetch
    {
        std::lock_guard<std::mutex> lk(g_snap_mtx);
        g_snaps.clear();
    }
    g_sampler_steps = steps;
    g_recording = true;
    g_pf_dit.join(); g_pf_dit.what.clear();
    g_pf_vae.join(); g_pf_vae.what.clear();
    if (h->gens == 0 && nph_env_flag("NPH_PREFETCH_DIT", true)) {
        // weights will load lazily during step 1: warm the page cache while the TE runs
        const char* cdir = getenv("NPH_CACHE_DIR");
        std::string cache = (cdir && *cdir && nph_env_flag("NPH_REPACK_CACHE", true)) ? NphRepackCache::find_any(cdir) : "";
        g_pf_dit.start(cache.empty() ? h->dit : cache, cache.empty() ? "DiT (durante TE)" : "caché repack (durante TE)");
    }
    g_pf_vae_path = (h->gens == 0 && nph_env_flag("NPH_PREFETCH_VAE", false)) ? h->vae : "";

    rusage ru0{}, ru1{};
    getrusage(RUSAGE_SELF, &ru0);
    double t0 = now_s();
    take_snap("gen_begin", -1);
    sd_image_t* images = nullptr;
    int n_images = 0;
    bool ok = generate_image(h->ctx, &gp, &images, &n_images);
    double gen_s = now_s() - t0;
    take_snap("gen_end", -1);
    g_recording = false;
    g_pf_dit.join();
    g_pf_vae.join();
    h->gens++;
    {
        std::string rep = build_phase_report(gen_s);
        log_cb(SD_LOG_INFO, ("nph-phases\n" + rep).c_str(), nullptr);
        std::lock_guard<std::mutex> lk(g_report_mtx);
        g_last_report = rep;
    }
    getrusage(RUSAGE_SELF, &ru1);
    double user = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) + (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) * 1e-6;
    double sys  = (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) + (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) * 1e-6;

    jbyteArray result = nullptr;
    if (ok && images && n_images > 0 && images[0].data) {
        sd_image_t& im = images[0];
        size_t n = (size_t)im.width * im.height * im.channel;
        if (im.channel == 3 && (int)im.width == width && (int)im.height == height) {
            result = env->NewByteArray((jsize)n);
            if (result) env->SetByteArrayRegion(result, 0, (jsize)n, (const jbyte*)im.data);
        } else {
            log_cb(SD_LOG_ERROR, "nph: unexpected image shape\n", nullptr);
        }
    } else {
        log_cb(SD_LOG_ERROR, "nph: generate_image failed\n", nullptr);
    }
    if (images) free_sd_images(images, n_images);

    if (jtimings) {
        jsize len = env->GetArrayLength(jtimings);
        double t[3] = {gen_s, user, sys};
        env->SetDoubleArrayRegion(jtimings, 0, std::min<jsize>(3, len), t);
    }
    clear_cb(env);
    return result;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_zithexagon_engine_NphNative_setEnv(JNIEnv* env, jclass, jstring jkey, jstring jval) {
    JStr key(env, jkey), val(env, jval);
    if (!key.c || !key.c[0]) return JNI_FALSE;
    int rc = setenv(key.c, val.c ? val.c : "", 1);
    if (rc != 0) {
        char buf[256];
        snprintf(buf, sizeof buf, "nph: setenv(%s) failed errno=%d\n", key.c, errno);
        log_cb(SD_LOG_WARN, buf, nullptr);
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

/** Sequential read to pull file into OS page cache. Returns bytes read (0 on error). */
extern "C" JNIEXPORT jlong JNICALL
Java_com_zithexagon_engine_NphNative_warmupFile(JNIEnv* env, jclass, jstring jpath) {
    JStr path(env, jpath);
    if (!path.c || !path.c[0]) return 0;
    FILE* f = fopen(path.c, "rb");
    if (!f) return 0;
    const size_t BUFSZ = 8u << 20;
    std::vector<char> buf(BUFSZ);
    jlong total = 0;
    while (true) {
        size_t n = fread(buf.data(), 1, BUFSZ, f);
        total += (jlong)n;
        if (n < BUFSZ) break;
    }
    fclose(f);
    return total;
}

// ---------------- NPH 0.28 extras ----------------
extern "C" JNIEXPORT jstring JNICALL
Java_com_zithexagon_engine_NphNative_lastPhaseReport(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lk(g_report_mtx);
    return new_jstring_safe(env, g_last_report.c_str());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_zithexagon_engine_NphNative_trimPool(JNIEnv*, jclass) {
    return (jlong)ggml_backend_hexagon_nph_trim_pool();
}

static int clear_plain_files(const std::string& dir) {
    int n  = 0;
    DIR* d = opendir(dir.c_str());
    if (!d) return 0;
    std::vector<std::string> victims;
    while (auto* e = readdir(d)) {
        std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        struct stat st{};
        std::string p = dir + "/" + name;
        if (stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode)) victims.push_back(p);
    }
    closedir(d);
    for (auto& v : victims) n += unlink(v.c_str()) == 0;
    return n;
}

// 0.29: the NPH cache dir also holds te/ (TE output cache) and oclk/ (OpenCL kernel binaries, NPG).
extern "C" JNIEXPORT jlong JNICALL
Java_com_zithexagon_engine_NphNative_cacheSizeBytes(JNIEnv* env, jclass, jstring jdir) {
    JStr dir(env, jdir);
    std::string d = dir.c;
    return (jlong)(NphRepackCache::dir_size_bytes(d) + nph_te::disk_size(d + "/te") +
                   NphRepackCache::dir_size_bytes(d + "/oclk"));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_zithexagon_engine_NphNative_cacheClear(JNIEnv* env, jclass, jstring jdir) {
    JStr dir(env, jdir);
    std::string d = dir.c;
    int n = NphRepackCache::clear_dir(d);
    n += nph_te::disk_clear(d + "/te");
    n += clear_plain_files(d + "/oclk");
    return (jint)n;
}

// ---------------- NPH 0.29 NPG (experimental: VAE on the Adreno GPU via upstream ggml-opencl) ----------------
static std::mutex g_npg_mtx;
static std::string g_npg_sentinel;   // crash-guard file written by the app; OpenCL error lines are appended
static std::string g_npg_prepare;    // text of the last npgPrepare()
static std::string g_npg_last_err;   // last captured OpenCL / NPG error line
static bool g_npg_registered = false;
static ggml_backend_t g_npg_keep = nullptr;  // keeps the OpenCL backend context (compiled kernels) alive

static void npg_append_sentinel_locked(const char* text) {
    if (g_npg_sentinel.empty()) return;
    int fd = open(g_npg_sentinel.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) return;
    size_t n = strnlen(text, 600);
    (void)!write(fd, text, n);
    if (n == 0 || text[n - 1] != '\n') (void)!write(fd, "\n", 1);
    fsync(fd);
    close(fd);
}

static void npg_capture(const char* text) {
    const bool rel = strstr(text, "ggml_opencl") || strstr(text, "nph-npg") || strstr(text, "OpenCL");
    if (!rel) return;
    const bool err = strstr(text, "error") || strstr(text, "Error") || strstr(text, "fail") || strstr(text, "rejected") ||
                     strstr(text, "CL_") || strstr(text, "Unsupported");
    if (!err) return;
    std::lock_guard<std::mutex> lk(g_npg_mtx);
    g_npg_last_err.assign(text, strnlen(text, 400));
    while (!g_npg_last_err.empty() && (g_npg_last_err.back() == '\n' || g_npg_last_err.back() == '\r')) g_npg_last_err.pop_back();
    npg_append_sentinel_locked(text);
}

static ggml_abort_callback_t g_prev_abort = nullptr;
static void npg_abort_cb(const char* msg) {
    {
        std::lock_guard<std::mutex> lk(g_npg_mtx);
        std::string m = std::string("abort: ") + (msg ? msg : "?");
        npg_append_sentinel_locked(m.c_str());
    }
    __android_log_print(ANDROID_LOG_ERROR, TAG, "ggml abort: %s", msg ? msg : "?");
    if (g_prev_abort) g_prev_abort(msg);
}

static void npg_atexit() {
    std::lock_guard<std::mutex> lk(g_npg_mtx);
    npg_append_sentinel_locked("exit() llamado durante NPG (p. ej. fallo al compilar kernels OpenCL)");
}

static void npg_ggml_log(enum ggml_log_level level, const char* text, void*) {
    if (!text) return;
    log_cb(level >= GGML_LOG_LEVEL_ERROR ? SD_LOG_ERROR : level == GGML_LOG_LEVEL_WARN ? SD_LOG_WARN : SD_LOG_INFO, text, nullptr);
}

// Sets (or clears, with "") the crash-guard file used while NPG work is in flight.
extern "C" JNIEXPORT void JNICALL
Java_com_zithexagon_engine_NphNative_npgSentinel(JNIEnv* env, jclass, jstring jpath) {
    JStr p(env, jpath);
    static bool hooks = false;
    std::lock_guard<std::mutex> lk(g_npg_mtx);
    g_npg_sentinel = p.c ? p.c : "";
    if (!hooks && !g_npg_sentinel.empty()) {
        hooks = true;
        g_prev_abort = ggml_set_abort_callback(npg_abort_cb);
        atexit(npg_atexit);
    }
}

#ifdef NPH_HAVE_OPENCL
#include <CL/cl.h>
// 0.30: identity of the OpenCL driver for the self-test cache (device name + driver + CL version).
static std::string npg_driver_key() {
    cl_uint np = 0;
    if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) return "sin-plataforma";
    std::vector<cl_platform_id> ps(np);
    clGetPlatformIDs(np, ps.data(), nullptr);
    for (auto p : ps) {
        cl_device_id d = nullptr;
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, &nd) != CL_SUCCESS || nd == 0) continue;
        char name[256] = {0}, drv[256] = {0}, ver[256] = {0};
        clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof name - 1, name, nullptr);
        clGetDeviceInfo(d, CL_DRIVER_VERSION, sizeof drv - 1, drv, nullptr);
        clGetDeviceInfo(d, CL_DEVICE_VERSION, sizeof ver - 1, ver, nullptr);
        std::string k = std::string(name) + " | driver " + drv + " | " + ver;
        for (auto& c : k) if (c == '\n' || c == '\r') c = ' ';
        return k;
    }
    return "sin-GPU";
}
#endif

static bool npg_read_file(const std::string& path, std::string& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char b[4096];
    size_t n;
    out.clear();
    while ((n = fread(b, 1, sizeof b, f)) > 0 && out.size() < 65536) out.append(b, n);
    fclose(f);
    return true;
}

static void npg_write_file(const std::string& path, const std::string& data) {
    const std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    const bool ok = fwrite(data.data(), 1, data.size(), f) == data.size();
    fclose(f);
    if (ok) rename(tmp.c_str(), path.c_str()); else unlink(tmp.c_str());
}

// Loads libOpenCL lazily (no DT_NEEDED), registers upstream ggml-opencl, checks the Adreno device and runs a
// tiny synthetic self-test. Returns "OK\n<details>" or "FAIL\n<reason>". Never touches model weights.
extern "C" JNIEXPORT jstring JNICALL
Java_com_zithexagon_engine_NphNative_npgPrepare(JNIEnv* env, jclass, jstring jcache, jboolean self_test) {
    JStr cache(env, jcache);
    std::string out;
    bool ok = false;
#ifdef NPH_HAVE_OPENCL
    sd_set_log_callback(log_cb, nullptr);
    ggml_log_set(npg_ggml_log, nullptr);
    const double t0 = now_s();
    std::string kdir = std::string(cache.c ? cache.c : "") + "/oclk";
    if (cache.c && *cache.c) {
        mkdir(kdir.c_str(), 0700);
        setenv("GGML_OPENCL_KERNEL_CACHE_DIR", kdir.c_str(), 1);
    }
    // Only used by ggml-opencl when a normal allocation is refused (big 1024² VAE activations).
    setenv("GGML_OPENCL_ADRENO_USE_LARGE_BUFFER", "1", 0);
    const char* info = nullptr;
    char buf[640];
    if (!nph_ocl_shim_load(&info)) {
        out = std::string("FAIL\nlibOpenCL: ") + (info ? info : "no se pudo cargar");
    } else {
        out = std::string("libOpenCL: ") + (info ? info : "?") + "\n";
        if (!g_npg_registered) {
            ggml_backend_register(ggml_backend_opencl_reg());
            g_npg_registered = true;
        }
        ggml_backend_dev_t dev = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (d && ggml_backend_dev_name(d) && strcmp(ggml_backend_dev_name(d), "GPUOpenCL") == 0) { dev = d; break; }
        }
        const double t_probe = now_s() - t0;
        // 0.30 Adreno mul_mat precision: upstream's default Adreno "xmem" F16xF32 GEMM converts the F32
        // activations to half and accumulates in half (gemm_xmem_f16_f32_os8.cl). NPH_NPG_MM_F32 (default
        // on) disables it before the backend is created, so F16xF32 matmuls use the local-memory GEMM
        // (F16 weights widened to float, F32 activations, F32 accumulation).
        const bool mm_f32 = nph_env_flag("NPH_NPG_MM_F32", true);
        const bool mm_npu = nph_env_flag("NPH_NPG_MM_NPU", false);
        const bool xmem_fixed = g_npg_keep != nullptr;  // env is read once at backend init
        if (mm_f32 && !xmem_fixed) setenv("GGML_OPENCL_ADRENO_XMEM_GEMM", "0", 1);
        const char* xm = getenv("GGML_OPENCL_ADRENO_XMEM_GEMM");
        const bool xmem_off = xm && atoi(xm) == 0;
        const std::string drv = npg_driver_key();
        if (!dev) {
            out = "FAIL\n" + out + "ggml-opencl no encontró un dispositivo compatible (requiere Adreno/Intel con OpenCL C ≥ 2.0)";
        } else {
            size_t fr = 0, tot = 0;
            ggml_backend_dev_memory(dev, &fr, &tot);
            snprintf(buf, sizeof buf, "dispositivo: GPUOpenCL · %s · memoria %.0f/%.0f MB · sondeo %.2f s\n",
                     ggml_backend_dev_description(dev), fr / 1048576.0, tot / 1048576.0, t_probe);
            out += buf;
            const double t1 = now_s();
            ggml_backend_t be = g_npg_keep ? g_npg_keep : ggml_backend_dev_init(dev, nullptr);
            const double t_init = now_s() - t1;
            if (!be) {
                out = "FAIL\n" + out + "no se pudo inicializar el backend OpenCL";
            } else {
                g_npg_keep = be;
                snprintf(buf, sizeof buf, "init backend + kernels: %.2f s (caché de kernels: %s)\n", t_init, kdir.c_str());
                out += buf;
                out += "driver: " + drv + "\n";
                out += std::string("mul_mat GPU: ") + (xmem_off ? "ruta FP32 (GEMM xmem FP16 desactivado)" : "GEMM xmem FP16 de upstream activo") +
                       (mm_f32 && xmem_fixed && !xmem_off ? " (backend ya creado en este proceso; aplica tras reiniciar la app)" : "") +
                       " · mul_mat/atención→NPU: " + (mm_npu ? "sí" : "no") + "\n";
                ok = true;
                if (self_test) {
                    // 0.30: cache keyed by driver + app + engine + mm options; re-run only when one changes.
                    const char* appv = getenv("NPH_APP_VERSION");
                    const std::string key = std::string("npg-selftest v2|") + drv + "|app=" + (appv ? appv : "?") +
                                            "|engine=nph-0.30.0|xmem_off=" + (xmem_off ? "1" : "0") + "|mm_npu=" + (mm_npu ? "1" : "0");
                    const std::string cpath = std::string(cache.c ? cache.c : "") + "/npg_selftest.txt";
                    std::string cached;
                    bool st_ok = false, from_cache = false;
                    std::string st_text;
                    if (cache.c && *cache.c && npg_read_file(cpath, cached)) {
                        const size_t nl = cached.find('\n');
                        if (nl != std::string::npos && cached.compare(0, nl, key) == 0) {
                            const size_t nl2 = cached.find('\n', nl + 1);
                            const std::string verdict = cached.substr(nl + 1, nl2 == std::string::npos ? std::string::npos : nl2 - nl - 1);
                            if (verdict == "OK" || verdict == "FAIL") {
                                st_ok = verdict == "OK";
                                st_text = nl2 == std::string::npos ? "" : cached.substr(nl2 + 1);
                                from_cache = true;
                            }
                        }
                    }
                    if (!from_cache) {
                        NpgSelftestResult st = npg_selftest(be);
                        // mul_mat / attention misses are acceptable when those ops run on the NPU anyway.
                        const bool mm_routed = mm_npu && st.bad_other == 0 && st.bad_mm > 0;
                        st_ok = (st.ok || mm_routed) && st.supported > 0;
                        snprintf(buf, sizeof buf, "autoprueba sintética (sin modelo): %s · %d ops en GPU, %d irían a CPU · %.0f ms\n",
                                 st_ok ? "OK" : "FALLÓ", st.supported, st.unsupported, st.ms);
                        st_text = buf + st.text;
                        if (mm_routed) st_text += "  mul_mat/atención fuera de tolerancia en GPU → se ejecutan en la NPU\n";
                        if (cache.c && *cache.c) npg_write_file(cpath, key + "\n" + (st_ok ? "OK" : "FAIL") + "\n" + st_text);
                    } else {
                        st_text = "autoprueba: resultado en caché (mismo driver/app/motor; no se repite)\n" + st_text;
                    }
                    out += st_text;
                    if (!st_ok) {
                        ok  = false;
                        out = "FAIL\n" + out + "autoprueba fuera de tolerancia → VAE en NPU";
                    }
                }
                if (ok) out = "OK\n" + out;
            }
        }
    }
#else
    out = "FAIL\nesta compilación no incluye OpenCL";
#endif
    (void)ok;
    {
        std::lock_guard<std::mutex> lk(g_npg_mtx);
        g_npg_prepare = out;
    }
    return new_jstring_safe(env, out.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_zithexagon_engine_NphNative_npgLastError(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lk(g_npg_mtx);
    return new_jstring_safe(env, g_npg_last_err.c_str());
}

// Notes recorded during the last generate (node placement per device, VAE output check).
extern "C" JNIEXPORT jstring JNICALL
Java_com_zithexagon_engine_NphNative_npgNotes(JNIEnv* env, jclass) {
    std::vector<char> b(8192, 0);
    nph_npg_notes(b.data(), b.size());
    return new_jstring_safe(env, b.data());
}
