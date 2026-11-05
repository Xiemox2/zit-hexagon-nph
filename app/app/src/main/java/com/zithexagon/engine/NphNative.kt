package com.zithexagon.engine

/** Callback from libnph.so (any thread). */
interface NphCallback {
    fun onLog(level: Int, msg: String)
    fun onProgress(step: Int, steps: Int, time: Float)
}

/**
 * JNI to the NPH engine (sd.cpp Hexagon / ggml-htp). Branded NPH = NPU + Hexagon.
 * create/generate/free allow Extremo residency (keep DiT+VAE between images).
 */
object NphNative {
    @JvmStatic external fun info(): String
    @JvmStatic external fun memAvailableMb(): Int
    @JvmStatic external fun cancel()

    /** Set process env before first Hexagon init (PROFILE/NHVX/…). Returns false if setenv failed. */
    @JvmStatic external fun setEnv(key: String, value: String): Boolean

    /** Prefault [path] into the page cache (sequential read). Safe / output-identical. */
    @JvmStatic external fun warmupFile(path: String): Long

    /** flags: bit0 flash_attn, bit1 vae_conv_direct, bit2 mmap, bit3 eager_load. timings[0]=create_s */
    @JvmStatic external fun create(
        diffusion: String, llm: String, vae: String,
        backend: String, paramsBackend: String,
        threads: Int, flags: Int, cb: NphCallback?, timings: DoubleArray,
    ): Long

    @JvmStatic external fun free(handle: Long)

    // ---- 0.28 ----
    /** Per-phase report of the last generate (TE / step 1 breakdown / steps / VAE). */
    @JvmStatic external fun lastPhaseReport(): String
    /** Release parked (reusable) NPU compute buffers. Returns bytes freed. */
    @JvmStatic external fun trimPool(): Long
    /** Size in bytes of the repacked-DiT cache files (*.nphc) in [dir]. */
    @JvmStatic external fun cacheSizeBytes(dir: String): Long
    /** Delete the repacked-DiT cache in [dir] (engine must be freed first). Returns files removed. */
    @JvmStatic external fun cacheClear(dir: String): Int

    // ---- 0.29 NPG (experimental GPU VAE via upstream ggml-opencl) ----
    /** Crash-guard file for NPG work in flight ("" clears). OpenCL errors/aborts are appended to it. */
    @JvmStatic external fun npgSentinel(path: String)
    /** Lazy-load libOpenCL, register ggml-opencl, check the Adreno device, optional synthetic self-test.
     *  Returns "OK\n…" or "FAIL\n<reason>". No model weights involved. */
    @JvmStatic external fun npgPrepare(cacheDir: String, selfTest: Boolean): String
    /** Last captured OpenCL/NPG error line ("" if none). */
    @JvmStatic external fun npgLastError(): String
    /** Notes from the last generate: node placement per device, VAE output check. */
    @JvmStatic external fun npgNotes(): String

    // ---- 0.30 ----
    /** Predict whether generate() with the same arguments skips the text encoder (TE output cache).
     *  2 = RAM hit, 1 = disk hit, 0 = miss (TE ~4 GB will be loaded), -1 = n/a. Read-only. */
    @JvmStatic external fun teProbe(
        handle: Long, prompt: String,
        steps: Int, width: Int, height: Int, cfg: Float,
        loraPaths: Array<String>, loraStrengths: FloatArray,
    ): Int

    /** timings: [0]=generate_s [1]=cpu_user [2]=cpu_sys. Returns RGB888 or null. */
    @JvmStatic external fun generate(
        handle: Long, prompt: String,
        steps: Int, width: Int, height: Int, seed: Long, cfg: Float,
        loraPaths: Array<String>, loraStrengths: FloatArray,
        cb: NphCallback?, timings: DoubleArray,
    ): ByteArray?
}
