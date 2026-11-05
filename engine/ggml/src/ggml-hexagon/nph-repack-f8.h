// NPH fast FP8 E4M3 -> HMX tile repack (MIT, written from scratch for ZIT-HEXAGON).
// Produces exactly the same bytes as the reference byte loop repack_f8_e4m3_tiled():
//   tile[(col/4*4 + row/8)*32 + (row%8)*4 + lane[col%4]] = v == 0x80 ? 0 : v,  lane = {0,2,1,3}
// i.e. every 4 source bytes of one row (one 32-bit word) land in one 4-byte slot with the two
// middle bytes swapped. So we move whole words (NEON: 16 bytes at a time) and split the work by
// column tiles (32 source rows) across threads. Padding (ne0/ne1 not multiple of 32) uses the
// exact reference formula for that tile.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <thread>
#include <vector>
#include <algorithm>
#if !defined(NPH_REPACK_NO_NEON) && (defined(__ARM_NEON) || defined(__aarch64__))
#include <arm_neon.h>
#define NPH_REPACK_NEON 1
#elif defined(NPH_REPACK_NEON_EMU)
#define NPH_REPACK_NEON 1   // unit test: arm_neon intrinsics emulated on x86 by the test
#endif

namespace nph_repack {

static inline size_t f8_tile_offset(int row, int col) {
    static const uint8_t lane[4] = { 0, 2, 1, 3 };
    const int tile_row = (col / 4) * 4 + row / 8;
    const int tile_col = (row % 8) * 4 + lane[col % 4];
    return (size_t) tile_row * 32 + tile_col;
}

// zero bytes equal to 0x80 (negative zero in e4m3) and swap the two middle bytes
static inline uint32_t f8_fix_word(uint32_t w) {
    const uint32_t x = w ^ 0x80808080u;
    uint32_t t = ((x & 0x7F7F7F7Fu) + 0x7F7F7F7Fu);
    t = ~(t | x | 0x7F7F7F7Fu);           // 0x80 in each byte where x byte == 0 (exact)
    w &= ~((t >> 7) * 0xFFu);
    return (w & 0xFF0000FFu) | ((w >> 8) & 0x0000FF00u) | ((w << 8) & 0x00FF0000u);
}

// reference (old) algorithm for one tile; used for partial edge tiles and by the unit test
static inline void f8_tile_ref(uint8_t * tile, const uint8_t * m, int64_t ne0, int64_t ne1, int ct, int kt) {
    for (int row = 0; row < 32; ++row) {
        const int64_t j = (int64_t) ct * 32 + row;
        for (int col = 0; col < 32; ++col) {
            const int64_t i = (int64_t) kt * 32 + col;
            const uint8_t value = i < ne0 && j < ne1 ? m[j * ne0 + i] : 0;
            tile[f8_tile_offset(row, col)] = value == 0x80 ? 0 : value;
        }
    }
}

static inline void f8_tile_fast(uint8_t * tile, const uint8_t * m, int64_t ne0, int ct, int kt) {
    const uint8_t * base = m + (int64_t) ct * 32 * ne0 + (int64_t) kt * 32;
#ifdef NPH_REPACK_NEON
    static const uint8_t idx_tbl[16] = { 0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15 };
    const uint8x16_t idx = vld1q_u8(idx_tbl);
    const uint8x16_t k80 = vdupq_n_u8(0x80);
#endif
    for (int row = 0; row < 32; ++row) {
        const uint8_t * s = base + (int64_t) row * ne0;
        uint8_t * d = tile + (row >> 3) * 32 + (row & 7) * 4;   // + c4*128
#ifdef NPH_REPACK_NEON
        for (int h = 0; h < 2; ++h) {
            uint8x16_t v = vld1q_u8(s + h * 16);
            v = vbicq_u8(v, vceqq_u8(v, k80));
            v = vqtbl1q_u8(v, idx);
            uint32x4_t w = vreinterpretq_u32_u8(v);
            uint8_t * dd = d + h * 4 * 128;
            vst1q_lane_u32((uint32_t *) (dd + 0 * 128), w, 0);
            vst1q_lane_u32((uint32_t *) (dd + 1 * 128), w, 1);
            vst1q_lane_u32((uint32_t *) (dd + 2 * 128), w, 2);
            vst1q_lane_u32((uint32_t *) (dd + 3 * 128), w, 3);
        }
#else
        for (int c4 = 0; c4 < 8; ++c4) {
            uint32_t w;
            memcpy(&w, s + c4 * 4, 4);
            w = f8_fix_word(w);
            memcpy(d + c4 * 128, &w, 4);
        }
#endif
    }
}

static inline void f8_matrix_range(uint8_t * mdst, const uint8_t * msrc, int64_t ne0, int64_t ne1,
                                   int n_k_tiles, int ct0, int ct1, size_t tile_size) {
    for (int ct = ct0; ct < ct1; ++ct) {
        const bool full_rows = (int64_t) (ct + 1) * 32 <= ne1;
        for (int kt = 0; kt < n_k_tiles; ++kt) {
            uint8_t * tile = mdst + ((size_t) ct * n_k_tiles + kt) * tile_size;
            if (full_rows && (int64_t) (kt + 1) * 32 <= ne0) {
                f8_tile_fast(tile, msrc, ne0, ct, kt);
            } else {
                f8_tile_ref(tile, msrc, ne0, ne1, ct, kt);
            }
        }
    }
}

// dst: tiled tensor data (n_col_tiles*n_k_tiles*1024 per 2D slice); src: row-major e4m3 bytes.
static inline void repack_f8_e4m3_fast(uint8_t * dst, const uint8_t * src, int64_t ne0, int64_t ne1,
                                       int64_t ne2, int64_t ne3, int n_threads, size_t tile_size = 1024) {
    const int n_k_tiles   = (int) ((ne0 + 31) / 32);
    const int n_col_tiles = (int) ((ne1 + 31) / 32);
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;
    const int64_t n_mat = ne2 * ne3;
    const int64_t total_ct = n_mat * n_col_tiles;
    if (n_threads < 1) n_threads = 1;
    // below ~1 MiB per thread spawning costs more than it saves
    const int64_t bytes = ne0 * ne1 * n_mat;
    n_threads = (int) std::min<int64_t>(n_threads, std::max<int64_t>(1, bytes >> 20));
    n_threads = (int) std::min<int64_t>(n_threads, total_ct);
    auto work = [&](int64_t g0, int64_t g1) {
        for (int64_t g = g0; g < g1;) {
            const int64_t mat = g / n_col_tiles;
            const int ct0 = (int) (g % n_col_tiles);
            const int ct1 = (int) std::min<int64_t>(n_col_tiles, ct0 + (g1 - g));
            f8_matrix_range(dst + mat * matrix_size, src + mat * ne0 * ne1, ne0, ne1, n_k_tiles, ct0, ct1, tile_size);
            g += ct1 - ct0;
        }
    };
    if (n_threads <= 1) { work(0, total_ct); return; }
    std::vector<std::thread> th;
    th.reserve(n_threads - 1);
    const int64_t per = (total_ct + n_threads - 1) / n_threads;
    for (int t = 1; t < n_threads; ++t) {
        const int64_t g0 = std::min<int64_t>(total_ct, t * per), g1 = std::min<int64_t>(total_ct, g0 + per);
        if (g0 < g1) th.emplace_back(work, g0, g1);
    }
    work(0, std::min<int64_t>(total_ct, per));
    for (auto & t : th) t.join();
}

}  // namespace nph_repack
