package com.zithexagon.data

import android.content.Context
import com.zithexagon.engine.Sampler
import org.json.JSONObject

/** Small settings store (app-private SharedPreferences). */
class Prefs(context: Context) {
    private val sp = context.getSharedPreferences("zithexagon", Context.MODE_PRIVATE)

    /** Last form on the generate screen (prompt, params, LoRAs). */
    var lastParams: GenParams
        get() = sp.getString("last_params", null)?.let { runCatching { GenParams.fromJson(JSONObject(it)) }.getOrNull() }
            ?: GenParams(prompt = "", sampler = Sampler.EULER)
        set(v) = sp.edit().putString("last_params", v.toJson().toString()).apply()

    var selectedModel: String?
        get() = sp.getString("model", null)
        set(v) = sp.edit().putString("model", v).apply()

    var randomSeed: Boolean
        get() = sp.getBoolean("random_seed", true)
        set(v) = sp.edit().putBoolean("random_seed", v).apply()

    var upscaleEnabled: Boolean
        get() = sp.getBoolean("upscale_on", false)
        set(v) = sp.edit().putBoolean("upscale_on", v).apply()
    var upscaleFactor: Int
        get() = sp.getInt("upscale_factor", 2)
        set(v) = sp.edit().putInt("upscale_factor", v).apply()
    var upscalerModel: String
        get() = sp.getString("upscaler", null)?.takeIf { id -> UPSCALERS.any { it.id == id } } ?: UPSCALERS.first().id
        set(v) = sp.edit().putString("upscaler", v).apply()

    /** NPU execution mode (0.25.1): Extremo (all DiT blocks resident) or Seguro (2 groups of 3). */
    var execMode: com.zithexagon.engine.ExecMode
        get() = com.zithexagon.engine.ExecMode.entries.firstOrNull { it.name == sp.getString("exec_mode", null) } ?: com.zithexagon.engine.ExecMode.EXTREMO
        set(v) = sp.edit().putString("exec_mode", v.name).apply()

    /** 0.25.2 diagnostics: dump block-0 inputs/output of step 1 to Download/ZITHEXAGON/debug (+ A/B block 0 from test/). */
    var debugDump: Boolean
        get() = sp.getBoolean("debug_dump_b0", false)
        set(v) = sp.edit().putBoolean("debug_dump_b0", v).apply()

    var copyToPhoneGallery: Boolean
        get() = sp.getBoolean("copy_to_phone", false)
        set(v) = sp.edit().putBoolean("copy_to_phone", v).apply()

    var accent: String
        get() = sp.getString("accent", "vino") ?: "vino"
        set(v) = sp.edit().putString("accent", v).apply()

    var showNegative: Boolean
        get() = sp.getBoolean("show_negative", false)
        set(v) = sp.edit().putBoolean("show_negative", v).apply()

    // ---- 0.26 NPH speed / profiling (Avanzado) ----
    /** GGML_HEXAGON_PROFILE=1 → per-op HMX/HVX/CPU/wait in report. */
    var nphProfile: Boolean
        get() = sp.getBoolean("nph_profile", false)
        set(v) = sp.edit().putBoolean("nph_profile", v).apply()

    /** Prefault page cache of DiT/TE/VAE before create (default ON; output-identical). */
    var nphPageCacheWarmup: Boolean
        get() = sp.getBoolean("nph_page_cache", true)
        set(v) = sp.edit().putBoolean("nph_page_cache", v).apply()

    /**
     * H1 / 0.27.1: keep DiT gallocr/HTP compute buffers across Extremo gens (same res/settings).
     * Default ON — same reuse already used between steps within a gen (bit-identical).
     */
    var nphGraphCache: Boolean
        get() = sp.getBoolean("nph_graph_cache", true)
        set(v) = sp.edit().putBoolean("nph_graph_cache", v).apply()

    /** Opt-in: try overlapping DiT page-warm with TE path prep (unmeasured; low risk). */
    var nphOverlapDitTe: Boolean
        get() = sp.getBoolean("nph_overlap_dit_te", false)
        set(v) = sp.edit().putBoolean("nph_overlap_dit_te", v).apply()

    /** Opt-in: prefer more CPU graph segments when idle (unmeasured; may change scheduling). */
    var nphCpuOffload: Boolean
        get() = sp.getBoolean("nph_cpu_offload", false)
        set(v) = sp.edit().putBoolean("nph_cpu_offload", v).apply()

    // ---- 0.28 load-path improvements (default ON; output bit-identical) ----
    var nphFastRepack: Boolean
        get() = sp.getBoolean("nph_fast_repack", true)
        set(v) = sp.edit().putBoolean("nph_fast_repack", v).apply()
    var nphParallelSet: Boolean
        get() = sp.getBoolean("nph_parallel_set", true)
        set(v) = sp.edit().putBoolean("nph_parallel_set", v).apply()
    var nphFastRead: Boolean
        get() = sp.getBoolean("nph_fast_read", true)
        set(v) = sp.edit().putBoolean("nph_fast_read", v).apply()
    var nphRepackCache: Boolean
        get() = sp.getBoolean("nph_repack_cache", true)
        set(v) = sp.edit().putBoolean("nph_repack_cache", v).apply()
    var nphBufPool: Boolean
        get() = sp.getBoolean("nph_buf_pool", true)
        set(v) = sp.edit().putBoolean("nph_buf_pool", v).apply()
    var nphPrefetchDit: Boolean
        get() = sp.getBoolean("nph_prefetch_dit", true)
        set(v) = sp.edit().putBoolean("nph_prefetch_dit", v).apply()

    // ---- 0.29: TE output cache is a normal feature (default ON; identical output) ----
    /** RAM cache (≤4 prompts) of the text-encoder output. New key in 0.29 so it starts ON for everyone. */
    var nphTeCache: Boolean
        get() = sp.getBoolean("nph_te_cache_029", true)
        set(v) = sp.edit().putBoolean("nph_te_cache_029", v).apply()
    /** Also keep it on disk (nph-cache/te, ≤128 MB) so it survives restarts / engine release. */
    var nphTeCacheDisk: Boolean
        get() = sp.getBoolean("nph_te_cache_disk_029", true)
        set(v) = sp.edit().putBoolean("nph_te_cache_disk_029", v).apply()

    // ---- 0.30: LoRA combined adapter (default ON; FP16-exact packing + FP32 scale, HMX-eligible) ----
    var nphLoraCombined: Boolean
        get() = sp.getBoolean("nph_lora_combined_030", true)
        set(v) = sp.edit().putBoolean("nph_lora_combined_030", v).apply()

    // ---- 0.30.2: LoRA B·(A·x) + scale fused into the HMX matmul epilogue (FP32; default ON, self-tested on device) ----
    var nphLoraFuse: Boolean
        get() = sp.getBoolean("nph_lora_fuse_0302", false)
        set(v) = sp.edit().putBoolean("nph_lora_fuse_0302", v).apply()

    /** 0.30.1: Mejoras (default ON) that are currently OFF, so a report shows they were changed by hand. */
    fun nonDefaultNphToggles(): List<String> = buildList {
        if (!nphFastRepack) add("repack-rápido")
        if (!nphParallelSet) add("set-paralelo")
        if (!nphFastRead) add("lectura-rápida")
        if (!nphRepackCache) add("caché-repack")
        if (!nphBufPool) add("pool-buffers")
        if (!nphPrefetchDit) add("prefetch-DiT")
        if (!nphTeCache) add("caché-TE")
        if (!nphTeCacheDisk) add("caché-TE-disco")
        if (!nphLoraCombined) add("LoRA-combinado")
        if (!nphGraphCache) add("caché-grafo")
    }

    /** 0.30.1: «Restaurar predeterminados» in Ajustes → Mejoras. */
    fun resetNphToggles() {
        nphFastRepack = true; nphParallelSet = true; nphFastRead = true; nphRepackCache = true
        nphBufPool = true; nphPrefetchDit = true; nphTeCache = true; nphTeCacheDisk = true
        nphLoraCombined = true; nphGraphCache = true; nphLoraFuse = false
    }

    // ---- 0.30 NPG sub-options (only matter while the experimental GPU VAE is on) ----
    /** GPU mul_mat with FP32 accumulation (disables upstream's FP16-accumulating Adreno xmem GEMM). */
    var nphNpgMmF32: Boolean
        get() = sp.getBoolean("nph_npg_mm_f32_030", true)
        set(v) = sp.edit().putBoolean("nph_npg_mm_f32_030", v).apply()
    /** Check every GPU VAE part (NaN/Inf/overflow/error) and redo only the failing part on the NPU. */
    var nphNpgPartFallback: Boolean
        get() = sp.getBoolean("nph_npg_part_fb_030", true)
        set(v) = sp.edit().putBoolean("nph_npg_part_fb_030", v).apply()
    /** Experimental: run the VAE matmuls / attention on the NPU, the rest on the GPU. */
    var nphNpgMmNpu: Boolean
        get() = sp.getBoolean("nph_npg_mm_npu_030", false)
        set(v) = sp.edit().putBoolean("nph_npg_mm_npu_030", v).apply()

    // ---- 0.29 experimental NPG (default OFF) ----
    /** VAE decode on the Adreno GPU (upstream ggml-opencl). Falls back to the NPU on any error. */
    var nphNpgVae: Boolean
        get() = sp.getBoolean("nph_npg_vae", false)
        set(v) = sp.edit().putBoolean("nph_npg_vae", v).apply()
    /** Synthetic (no model) GPU-vs-reference self-test before using the GPU VAE. */
    var nphNpgSelfTest: Boolean
        get() = sp.getBoolean("nph_npg_selftest", true)
        set(v) = sp.edit().putBoolean("nph_npg_selftest", v).apply()
    /** Non-empty = NPG VAE auto-disabled after a crash during NPG work (reason shown in Ajustes/reporte). */
    var nphNpgBlocked: String
        get() = sp.getString("nph_npg_blocked", "") ?: ""
        set(v) = sp.edit().putString("nph_npg_blocked", v).apply()

    // ---- 0.28 experimental (default OFF) ----
    var nphExpPrefetchVae: Boolean
        get() = sp.getBoolean("nph_exp_prefetch_vae", false)
        set(v) = sp.edit().putBoolean("nph_exp_prefetch_vae", v).apply()

    /** Empty = engine default (0 = all HVX threads). */
    var nphNhvx: String
        get() = sp.getString("nph_nhvx", "") ?: ""
        set(v) = sp.edit().putString("nph_nhvx", v).apply()
    var nphNhmx: String
        get() = sp.getString("nph_nhmx", "") ?: ""
        set(v) = sp.edit().putString("nph_nhmx", v).apply()
    var nphOpbatch: String
        get() = sp.getString("nph_opbatch", "") ?: ""
        set(v) = sp.edit().putString("nph_opbatch", v).apply()
    var nphOpqueue: String
        get() = sp.getString("nph_opqueue", "") ?: ""
        set(v) = sp.edit().putString("nph_opqueue", v).apply()
    var nphOpfusion: String
        get() = sp.getString("nph_opfusion", "") ?: ""
        set(v) = sp.edit().putString("nph_opfusion", v).apply()
    /** MiB; empty = default (~3200 on v81). */
    var nphVmem: String
        get() = sp.getString("nph_vmem", "") ?: ""
        set(v) = sp.edit().putString("nph_vmem", v).apply()
    /** MiB; empty = default 1024. */
    var nphMbuf: String
        get() = sp.getString("nph_mbuf", "") ?: ""
        set(v) = sp.edit().putString("nph_mbuf", v).apply()
}


/** Upscaler families shown as groups in the picker. */
enum class UpscalerKind { REALISM, ANIME }

/**
 * Upscaler models (placeholders until they are converted for the NPU). All four are 4× ESRGAN/RRDB
 * networks; 2× output = run at 4× and downsample (or tile at half scale), decided by the engine.
 * [nonCommercial] flags licences that forbid commercial use (fine for this personal app).
 */
data class UpscalerModel(
    val id: String, val name: String, val kind: UpscalerKind, val author: String,
    val license: String, val nonCommercial: Boolean, val url: String,
)

val UPSCALERS = listOf(
    UpscalerModel("4x-ultrasharp", "4x-UltraSharp", UpscalerKind.REALISM, "Kim2091",
        "CC BY-NC-SA 4.0", nonCommercial = true, url = "https://huggingface.co/Kim2091/UltraSharp"),
    UpscalerModel("realesrgan-x4plus", "Real-ESRGAN x4plus", UpscalerKind.REALISM, "Xintao Wang (xinntao)",
        "BSD-3-Clause", nonCommercial = false, url = "https://github.com/xinntao/Real-ESRGAN"),
    UpscalerModel("realesrgan-x4plus-anime-6b", "Real-ESRGAN x4plus Anime 6B", UpscalerKind.ANIME, "Xintao Wang (xinntao)",
        "BSD-3-Clause", nonCommercial = false, url = "https://github.com/xinntao/Real-ESRGAN"),
    UpscalerModel("4x-animesharp", "4x-AnimeSharp", UpscalerKind.ANIME, "Kim2091",
        "CC BY-NC-SA 4.0", nonCommercial = true, url = "https://huggingface.co/Kim2091/AnimeSharp"),
)
