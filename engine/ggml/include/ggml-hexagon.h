#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_hexagon_init(void);

GGML_BACKEND_API bool ggml_backend_is_hexagon(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);

// ---- NPH 0.28 extensions (ZIT-HEXAGON) ----
enum ggml_hexagon_nph_counter {
    GGML_HEXAGON_NPH_REPACK_NS = 0,
    GGML_HEXAGON_NPH_REPACK_BYTES,
    GGML_HEXAGON_NPH_SET_NS,        // total time inside buffer set_tensor (incl. repack/memcpy)
    GGML_HEXAGON_NPH_SET_BYTES,
    GGML_HEXAGON_NPH_RPCMEM_NS,     // rpcmem_alloc2 (shared memory alloc)
    GGML_HEXAGON_NPH_RPCMEM_N,
    GGML_HEXAGON_NPH_RPCMEM_BYTES,
    GGML_HEXAGON_NPH_MMAP_NS,       // fastrpc_mmap (NPU map)
    GGML_HEXAGON_NPH_MMAP_N,
    GGML_HEXAGON_NPH_FREE_NS,       // unmap + rpcmem_free
    GGML_HEXAGON_NPH_SUBMIT_NS,     // host side opbatch build+dspqueue_write (incl. queue backpressure)
    GGML_HEXAGON_NPH_SUBMIT_N,
    GGML_HEXAGON_NPH_WAIT_NS,       // host waiting for NPU responses
    GGML_HEXAGON_NPH_POOL_HIT,
    GGML_HEXAGON_NPH_POOL_HIT_BYTES,
    GGML_HEXAGON_NPH_POOL_MISS,
    GGML_HEXAGON_NPH_PREPACKED_BYTES, // F8 weights written already tiled (repack cache hit)
    GGML_HEXAGON_NPH_CTR_COUNT
};
GGML_BACKEND_API void ggml_backend_hexagon_nph_counters(uint64_t * out, int n);
// Repack cache support. If `tensor` lives in a hexagon WEIGHTS buffer and is an F8_E4M3 repack
// tensor, returns true and the location/size of its device (tiled) layout.
GGML_BACKEND_API bool ggml_backend_hexagon_nph_tiled_view(struct ggml_tensor * tensor, void ** data, size_t * size);
// Mark a tensor whose tiled bytes were written directly (from the repack cache) as loaded.
GGML_BACKEND_API bool ggml_backend_hexagon_nph_mark_prepacked(struct ggml_tensor * tensor);
// Size of the tiled layout if `tensor` is an F8_E4M3 weight in a hexagon WEIGHTS buffer, else 0.
GGML_BACKEND_API size_t ggml_backend_hexagon_nph_f8_weight_size(struct ggml_tensor * tensor);
// True if the tensor lives in a hexagon device buffer (set_tensor is per-tensor thread-safe there).
GGML_BACKEND_API bool ggml_backend_hexagon_nph_parallel_set_ok(struct ggml_tensor * tensor);
// Release compute buffers kept mapped in the reuse pool. Returns bytes released.
GGML_BACKEND_API size_t ggml_backend_hexagon_nph_trim_pool(void);
// NPH 0.30.2: fused LoRA matmul (HTP_OP_MUL_MAT_LORA). Default from env NPH_LORA_FUSE (on unless "0").
// set: 1 on, 0 off (takes effect for graphs planned afterwards). get: current state.
GGML_BACKEND_API void ggml_backend_hexagon_nph_set_lora_fuse(int on);
GGML_BACKEND_API int  ggml_backend_hexagon_nph_get_lora_fuse(void);
// Number of HTP_OP_MUL_MAT_LORA ops planned so far (all sessions; for self-tests/diagnostics).
GGML_BACKEND_API uint64_t ggml_backend_hexagon_nph_lora_fused_count(void);

#ifdef  __cplusplus
}
#endif
