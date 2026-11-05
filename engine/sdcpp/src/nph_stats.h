// NPH 0.28 instrumentation + hooks shared by sd.cpp and the libnph JNI (ZIT-HEXAGON, MIT).
#pragma once
#include <cstddef>
#include <cstdint>

struct ggml_tensor;

enum nph_sd_counter {
    NPH_SD_RUNNER_START_NS = 0,  // runner_start (param residency / lazy weight load)
    NPH_SD_GRAPH_BUILD_NS,       // building the ggml graph
    NPH_SD_MEASURE_NS,           // graph-cut plan + compute buffer measure
    NPH_SD_WEIGHTS_NS,           // segment weight preparation (lazy load of params)
    NPH_SD_WS_ALLOC_NS,          // workspace prepare + allocate (compute buffers)
    NPH_SD_INPUT_NS,             // host->backend input copies
    NPH_SD_EXEC_NS,              // backend graph compute (submit + NPU + wait)
    NPH_SD_OUTPUT_NS,            // output readback
    NPH_SD_LOAD_NS,              // ModelLoader::load_tensors wall
    NPH_SD_LOAD_BYTES,
    NPH_SD_READ_NS,              // file read time (sum over loader threads)
    NPH_SD_SETT_NS,              // ggml_backend_tensor_set time (sum over loader threads)
    NPH_SD_CACHE_HIT_BYTES,      // repack cache: bytes read already tiled
    NPH_SD_CACHE_HIT_NS,
    NPH_SD_CACHE_WRITE_BYTES,
    NPH_SD_CACHE_WRITE_NS,
    NPH_SD_CACHE_REJECT,         // checksum/size mismatches (fell back to normal load)
    NPH_SD_COUNT
};

uint64_t nph_now_ns();
void nph_sd_add(int idx, uint64_t v);
void nph_sd_counters(uint64_t* out, int n);

struct NphScopedTimer {
    int idx;
    uint64_t t0;
    explicit NphScopedTimer(int i) : idx(i), t0(nph_now_ns()) {}
    ~NphScopedTimer() { nph_sd_add(idx, nph_now_ns() - t0); }
};

// Phase marks ("te_begin", "te_end", "sample_begin", "vae_begin", ...). The JNI registers a callback
// that snapshots all counters at each mark.
typedef void (*nph_mark_cb_t)(const char* name);
void nph_set_mark_cb(nph_mark_cb_t cb);
void nph_mark(const char* name);

// Backend hooks, set by libnph (keeps sd.cpp free of a hard dependency on ggml-hexagon).
struct nph_hex_hooks {
    size_t (*f8_weight_size)(ggml_tensor* t);                // 0 = not a cacheable tiled F8 weight
    bool (*tiled_view)(ggml_tensor* t, void** data, size_t* size);
    bool (*mark_prepacked)(ggml_tensor* t);
    bool (*parallel_set_ok)(ggml_tensor* t);                 // backend set_tensor is thread-safe per tensor
    // 0.30.2 fused LoRA matmul (HTP_OP_MUL_MAT_LORA) toggle + planned-op counter (may be null).
    void (*set_lora_fuse)(int on);
    int (*get_lora_fuse)();
    uint64_t (*lora_fused_count)();
};
void nph_set_hex_hooks(const nph_hex_hooks* hooks);
const nph_hex_hooks* nph_get_hex_hooks();

bool nph_env_flag(const char* name, bool def);

// NPH 0.29 NPG: free-form notes for the in-app "NPG" report block (device placement of graph nodes
// for non-HTP runners, VAE output checks). Only written when NPH_NPG_STATS=1.
void nph_npg_note(const char* key, const char* value);
// Returns "key: value\n" lines in insertion order (latest value per key).
void nph_npg_notes(char* out, size_t cap);
void nph_npg_notes_clear();
