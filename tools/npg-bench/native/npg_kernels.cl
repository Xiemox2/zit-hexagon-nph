// NPG bench OpenCL kernels. Copyright (c) 2026 ZIT-HEXAGON authors. MIT License.
// Own minimal kernels for the Phase-2 measurement only (not tuned production code).
// Precision: FP8 E4M3 / FP16 weights are expanded exactly to float; all accumulation in FP32.
#ifdef HAS_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 4
#endif

// E4M3 byte -> half bits (exact): sign to bit15, exp+mantissa shifted by 7.
// The resulting half equals value * 2^-8, so callers fold 256 into alpha.
inline float4 f8x4_to_f4(uchar4 b) {
    ushort4 u = convert_ushort4(b);
    ushort4 h = ((u & (ushort4)0x80) << (ushort4)8) | ((u & (ushort4)0x7F) << (ushort4)7);
#ifdef HAS_FP16
    return convert_float4(as_half4(h));
#else
    return vload_half4(0, (const __private half *) &h);
#endif
}

#if defined(WT_F8)
  #define WTYPE uchar
  #define LOADW4(p) f8x4_to_f4(vload4(0, (p)))
#elif defined(WT_F16)
  #define WTYPE half
  #define LOADW4(p) vload_half4(0, (p))
#else
  #define WTYPE float
  #define LOADW4(p) vload4(0, (p))
#endif

// Y[m, n] = alpha * sum_k X[m, k] * W[n, k]  (+ Y[m, n] if accumulate)
// X: M x K row-major float; W: N x K row-major (same layout as ggml weights); Y: M x N float.
// Requires M % TM == 0, N % TN == 0, K % 4 == 0.
__kernel void gemm(__global const float * restrict X, __global const WTYPE * restrict W,
                   __global float * restrict Y, int M, int N, int K, float alpha, int accumulate, int m_base) {
    const int n0 = get_global_id(0) * TN;
    const int m0 = m_base + get_global_id(1) * TM;
    if (n0 >= N || m0 >= M) return;
    float acc[TM][TN];
    for (int i = 0; i < TM; ++i) for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    const __global float * xr = X + (size_t) m0 * K;
    const __global WTYPE * wr = W + (size_t) n0 * K;
    for (int k = 0; k < K; k += 4) {
        float4 xv[TM];
        for (int i = 0; i < TM; ++i) xv[i] = vload4(0, xr + (size_t) i * K + k);
        for (int j = 0; j < TN; ++j) {
            float4 wv = LOADW4(wr + (size_t) j * K + k);
            for (int i = 0; i < TM; ++i) acc[i][j] += dot(xv[i], wv);
        }
    }
    for (int i = 0; i < TM; ++i) {
        __global float * yr = Y + (size_t) (m0 + i) * N + n0;
        for (int j = 0; j < TN; ++j) {
            float v = acc[i][j] * alpha;
            yr[j] = accumulate ? yr[j] + v : v;
        }
    }
}

// Local-memory tiled variant: work-group 16x16, each WI computes TM x TN outputs,
// tile 64(m) x 64(n) for TM=TN=4, K step 16.
#define BK 16
__kernel void gemm_tiled(__global const float * restrict X, __global const WTYPE * restrict W,
                         __global float * restrict Y, int M, int N, int K, float alpha, int accumulate, int m_base) {
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int BN = 16 * TN, BM = 16 * TM;
    const int nb = get_group_id(0) * BN, mb = m_base + get_group_id(1) * BM;
    __local float xs[16 * TM][BK + 1];
    __local float ws[16 * TN][BK + 1];
    float acc[TM][TN];
    for (int i = 0; i < TM; ++i) for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    const int lid = ly * 16 + lx;
    for (int k0 = 0; k0 < K; k0 += BK) {
        // cooperative loads: BM*BK floats of X, BN*BK of W; 256 WIs, 4 elems per load
        for (int e = lid; e < BM * (BK / 4); e += 256) {
            int r = e / (BK / 4), c = (e % (BK / 4)) * 4;
            float4 v = (mb + r < M) ? vload4(0, X + (size_t) (mb + r) * K + k0 + c) : (float4) 0.0f;
            xs[r][c] = v.x; xs[r][c + 1] = v.y; xs[r][c + 2] = v.z; xs[r][c + 3] = v.w;
        }
        for (int e = lid; e < BN * (BK / 4); e += 256) {
            int r = e / (BK / 4), c = (e % (BK / 4)) * 4;
            float4 v = (nb + r < N) ? LOADW4(W + (size_t) (nb + r) * K + k0 + c) : (float4) 0.0f;
            ws[r][c] = v.x; ws[r][c + 1] = v.y; ws[r][c + 2] = v.z; ws[r][c + 3] = v.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < BK; ++kk) {
            float xa[TM], wb[TN];
            for (int i = 0; i < TM; ++i) xa[i] = xs[ly + 16 * i][kk];
            for (int j = 0; j < TN; ++j) wb[j] = ws[lx + 16 * j][kk];
            for (int i = 0; i < TM; ++i) for (int j = 0; j < TN; ++j) acc[i][j] += xa[i] * wb[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < TM; ++i) {
        int m = mb + ly + 16 * i;
        if (m >= M) continue;
        for (int j = 0; j < TN; ++j) {
            int n = nb + lx + 16 * j;
            if (n >= N) continue;
            float v = acc[i][j] * alpha;
            size_t o = (size_t) m * N + n;
            Y[o] = accumulate ? Y[o] + v : v;
        }
    }
}

// Flash attention (non-optimised reference-style kernel, FP32 accumulation, online softmax).
// Q: [H][S][128] float, K,V: [H][S][128] half, O: [H][S][128] float.
// Work-group 64 = 16 queries x 4 slices of 32 dims. Requires S % 16 == 0.
#define FA_D 128
#define FA_TK 16
__kernel void flash_attn(__global const float * restrict Q, __global const half * restrict Kh,
                         __global const half * restrict Vh, __global float * restrict O, int S, float scale) {
    const int lid = get_local_id(0);
    const int ql = lid >> 2, part = lid & 3;
    const int h = get_global_id(1);
    const int q = get_group_id(0) * 16 + ql;
    __local float ks[FA_TK][FA_D];
    __local float vs[FA_TK][FA_D];
    __local float ps[FA_TK][64];
    const size_t hb = (size_t) h * S * FA_D;
    float4 qv[8], ov[8];
    for (int i = 0; i < 8; ++i) { qv[i] = vload4(0, Q + hb + (size_t) q * FA_D + part * 32 + i * 4); ov[i] = (float4) 0.0f; }
    float m = -INFINITY, l = 0.0f;
    for (int t0 = 0; t0 < S; t0 += FA_TK) {
        for (int e = lid; e < FA_TK * FA_D / 4; e += 64) {
            int r = e / (FA_D / 4), c = (e % (FA_D / 4)) * 4;
            float4 kv = vload_half4(0, Kh + hb + (size_t) (t0 + r) * FA_D + c);
            float4 vv = vload_half4(0, Vh + hb + (size_t) (t0 + r) * FA_D + c);
            vstore4(kv, 0, &ks[r][c]);
            vstore4(vv, 0, &vs[r][c]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < FA_TK; ++j) {
            float s = 0.0f;
            for (int i = 0; i < 8; ++i) s += dot(qv[i], vload4(0, &ks[j][part * 32 + i * 4]));
            ps[j][lid] = s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float sj[FA_TK];
        float mt = m;
        for (int j = 0; j < FA_TK; ++j) {
            int b = ql * 4;
            sj[j] = (ps[j][b] + ps[j][b + 1] + ps[j][b + 2] + ps[j][b + 3]) * scale;
            mt = fmax(mt, sj[j]);
        }
        float corr = exp(m - mt);
        l *= corr;
        for (int i = 0; i < 8; ++i) ov[i] *= corr;
        for (int j = 0; j < FA_TK; ++j) {
            float p = exp(sj[j] - mt);
            l += p;
            for (int i = 0; i < 8; ++i) ov[i] += p * vload4(0, &vs[j][part * 32 + i * 4]);
        }
        m = mt;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float il = 1.0f / l;
    for (int i = 0; i < 8; ++i) vstore4(ov[i] * il, 0, O + hb + (size_t) q * FA_D + part * 32 + i * 4);
}

// y = rms_norm(x) * w * (1 + scale) + shift ; one work-group (256) per row
__kernel void rmsnorm_mod(__global const float * X, __global const float * w, __global const float * sc,
                          __global const float * sh, __global float * Y, int C, float eps) {
    __local float red[256];
    const int row = get_group_id(0), lid = get_local_id(0);
    const __global float * x = X + (size_t) row * C;
    float s = 0.0f;
    for (int c = lid; c < C; c += 256) { float v = x[c]; s += v * v; }
    red[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = 128; o > 0; o >>= 1) { if (lid < o) red[lid] += red[lid + o]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float r = rsqrt(red[0] / (float) C + eps);
    __global float * y = Y + (size_t) row * C;
    for (int c = lid; c < C; c += 256) y[c] = x[c] * r * w[c] * (1.0f + sc[c]) + sh[c];
}

// y = base + branch * gate[c]
__kernel void gated_res(__global const float * B, __global const float * R, __global const float * g,
                        __global float * Y, int C) {
    size_t i = get_global_id(0);
    Y[i] = B[i] + R[i] * g[i % C];
}

// y = silu(a) * b
__kernel void swiglu(__global const float * A, __global const float * Bv, __global float * Y) {
    size_t i = get_global_id(0);
    float a = A[i];
    Y[i] = a / (1.0f + exp(-a)) * Bv[i];
}

__kernel void copy4(__global const float4 * a, __global float4 * b) {
    size_t i = get_global_id(0);
    b[i] = a[i];
}

__kernel void add_one_u32(__global uint * a) {
    size_t i = get_global_id(0);
    a[i] = a[i] + 1u;
}

__kernel void fill_f32(__global float * a, int off, float base, float step) {
    size_t i = get_global_id(0);
    a[off + i] = base + step * (float) (i % 1024);
}

__kernel void copy_f32(__global const float * a, int off, __global float * b) {
    size_t i = get_global_id(0);
    b[i] = a[off + i];
}

__kernel void empty_k(__global float * a) {
    if (get_global_id(0) == 0x7fffffff) a[0] = 0.0f;
}
