package com.zithexagon.engine

import android.app.ActivityManager
import android.content.ComponentCallbacks2
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Build
import android.system.Os
import com.zithexagon.ZitDiag
import com.zithexagon.data.LoraLimits
import com.zithexagon.data.NphModelStore
import com.zithexagon.data.Prefs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

/**
 * NPH = Hexagon sd.cpp engine (proven in SdBench). Default and only diffusion engine in 0.26.
 * Extremo: keep DiT+VAE resident between gens (TE=disk → encode then free). Seguro: free after each.
 */
class NphEngine(private val context: Context) : ZitEngine {
    override val info = EngineInfo("ZIT-HEXAGON", "NPH · Hexagon HTP (sd.cpp)", isFake = false)
    private val _state = MutableStateFlow<EngineState>(EngineState.Idle)
    override val state: StateFlow<EngineState> = _state.asStateFlow()

    @Volatile private var handle = 0L
    @Volatile private var loadedId: String? = null
    @Volatile private var loadedLoraKey: String = ""
    @Volatile private var createSec = 0.0
    /** 0.30: generations run on the current ctx (2+ = DiT stayed resident). */
    @Volatile private var gensOnCtx = 0
    private val residencyNotes = ArrayList<String>()
    /** PROFILE flag that was applied when the current HTP session was created. */
    @Volatile private var ctxProfile = false
    /** 0.29 NPG: the current ctx decodes the VAE on the GPU (OpenCL). */
    @Volatile private var ctxNpgVae = false
    /** 0.29 NPG: reason the GPU VAE was turned off for this app session (automatic fallback). */
    @Volatile private var npgSessionOff: String? = null
    @Volatile private var npgPrepareText: String = ""
    private val npgEvents = ArrayList<String>()
    private val cancelled = AtomicBoolean(false)
    private val logBuf = StringBuilder(8192)
    private val logLock = Any()
    @Volatile private var heldResolved: NphModelStore.Resolved? = null
    private val prefs = Prefs(context)

    companion object {
        private val lib: Result<Unit> = runCatching {
            // HTP skels from assets → filesDir/htp; ADSP_LIBRARY_PATH before loadLibrary
            true // placeholder; real load in ensureLib
        }
        @Volatile private var libReady = false
        @Volatile private var libError: String? = null
        /**
         * 0.30 residency (Extremo). 0.29 required 2800 MB free after a gen, but a resident DiT leaves
         * ~1.8–1.9 GB, so it always freed → 5.9 GB reload + graph-cache MISS every image.
         * Now: keep while MemAvailable ≥ floor (1000 MB; 1200 MB with LoRAs, phone min seen 1.2 GB),
         * and before the next gen free only if the TE cache predicts a miss (TE ≈ 4 GB loads) and
         * there is not enough RAM for it next to the resident DiT.
         */
        const val KEEP_FLOOR_MB = 1000
        const val KEEP_FLOOR_LORA_MB = 1200
        const val TE_LOAD_MB = 3974      // Q8_0 Qwen3-4B TE loaded on a TE-cache miss (phone log)
        const val TE_MARGIN_MB = 600
        fun keepFloorMb(loras: Int): Int = if (loras > 0) KEEP_FLOOR_LORA_MB else KEEP_FLOOR_MB
        /** Pre-gen decision with a resident ctx: true = free first (TE miss and not enough RAM for the TE). */
        fun mustFreeBeforeGen(teProbe: Int, availMb: Int): Boolean =
            teProbe <= 0 && availMb in 0 until (TE_LOAD_MB + TE_MARGIN_MB)  // unknown (-1) treated as miss
        const val FLAGS_DEFAULT = 0b0011 // fa + vae_conv_direct; mmap off (proven phone config)
        const val BACKEND = "diffusion=HTP0,te=HTP0,vae=HTP0"
        /** 0.29 NPG: VAE on the Adreno GPU (upstream ggml-opencl). default=HTP0 keeps any other module on the NPU. */
        const val BACKEND_NPG_VAE = "default=HTP0,diffusion=HTP0,te=HTP0,vae=GPUOpenCL"
        /** 0.30.1: NPG GPU VAE removed from the app (native default build has no ggml-opencl). */
        const val NPG_BUILD = false
        @Volatile private var npgStartupChecked = false
        const val PARAMS_BACKEND = "te=disk"

        /** 0.28 repacked-DiT cache dir (internal storage; survives app updates, removed on uninstall). */
        fun cacheDir(ctx: Context): File = File(ctx.filesDir, "nph-cache").apply { mkdirs() }

        /** Clear via the engine (closes its open cache files) if libnph is loaded; null → caller deletes files. */
        fun clearCacheIfLoaded(dir: File): Int? =
            if (libReady) runCatching { NphNative.cacheClear(dir.absolutePath) }.getOrNull() else null
    }

    private fun ensureLib() {
        if (libReady) return
        synchronized(this) {
            if (libReady) return
            try {
                val dir = File(context.filesDir, "htp").apply { mkdirs() }
                val assets = context.assets.list("htp") ?: emptyArray()
                for (name in assets) {
                    val out = File(dir, name)
                    context.assets.open("htp/$name").use { inp ->
                        val bytes = inp.readBytes()
                        if (!out.exists() || out.length() != bytes.size.toLong()) {
                            out.outputStream().use { it.write(bytes) }
                        }
                    }
                    out.setReadable(true, false)
                }
                val adsp = listOf(
                    dir.absolutePath, context.applicationInfo.nativeLibraryDir,
                    "/vendor/dsp/cdsp", "/vendor/lib/rfsa/adsp", "/system/lib/rfsa/adsp", "/vendor/dsp", "/dsp",
                ).joinToString(";")
                runCatching { Os.setenv("ADSP_LIBRARY_PATH", adsp, true) }
                runCatching { Os.setenv("LD_LIBRARY_PATH", context.applicationInfo.nativeLibraryDir + ":/vendor/lib64", true) }
                System.loadLibrary("nph")
                libReady = true
                ZitDiag.bc("NPH lib loaded; ADSP=$adsp")
            } catch (t: Throwable) {
                libError = t.toString()
                ZitDiag.bc("NPH loadLibrary FAILED: $t")
                throw t
            }
        }
    }

    private fun cb(): NphCallback = object : NphCallback {
        override fun onLog(level: Int, msg: String) {
            synchronized(logLock) {
                if (logBuf.length > 60_000) logBuf.delete(0, logBuf.length - 40_000)
                logBuf.append(msg)
                if (!msg.endsWith('\n')) logBuf.append('\n')
            }
            if (level >= 2) ZitDiag.bc("nph: ${msg.trim()}")
        }
        override fun onProgress(step: Int, steps: Int, time: Float) { /* forwarded per-generate */ }
    }

    private fun memAvailMb(): Int = runCatching { NphNative.memAvailableMb() }.getOrDefault(-1)

    private fun freeHandle(reason: String) {
        val h = handle
        if (h == 0L) {
            NphModelStore.closeResolved(heldResolved); heldResolved = null
            return
        }
        ZitDiag.bc("NPH free ($reason) h=$h avail=${memAvailMb()}MB")
        runCatching { NphNative.free(h) }
        runCatching { NphNative.trimPool() }.onSuccess { if (it > 0) ZitDiag.bc("NPH pool liberado ${it shr 20} MB") }
        handle = 0L
        loadedId = null
        loadedLoraKey = ""
        createSec = 0.0
        gensOnCtx = 0
        ctxProfile = false
        ctxNpgVae = false
        NphModelStore.closeResolved(heldResolved); heldResolved = null
    }

    /** Called from Application.onTrimMemory / low-memory. */
    fun onTrimMemory(level: Int) {
        if (level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL ||
            level >= ComponentCallbacks2.TRIM_MEMORY_COMPLETE ||
            level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW) {
            if (_state.value !is EngineState.Busy) {
                freeHandle("onTrimMemory/$level")
                if (_state.value is EngineState.Ready) _state.value = EngineState.Idle
            }
        }
    }

    override suspend fun load(request: LoadRequest) = withContext(Dispatchers.IO) {
        ensureLib()
        libError?.let { error("libnph no disponible: $it") }
        // Prefer request.nph; else file paths(); else SAF resolve() (hold PFDs until freeHandle).
        val paths = request.nph ?: NphModelStore.paths(context) ?: run {
            val r = NphModelStore.resolve(context)
                ?: error("Faltan modelos NPH (DiT FP8 + TE Qwen3 + VAE). Impórtalos en Modelos.")
            NphModelStore.closeResolved(heldResolved)
            heldResolved = r
            r.paths
        }
        val loraKey = request.loras.filter { it.strength != 0f && File(it.path).isFile }
            .sortedBy { it.path }.joinToString("|") { "${it.path}@${it.strength}" }
        if (handle != 0L && loadedId == request.modelId && loadedLoraKey == loraKey) {
            _state.value = EngineState.Ready(request.modelId, request.loras)
            return@withContext
        }
        // LoRA change or new model → recreate ctx (LoRA applied at DiT load / first gen)
        if (handle != 0L) freeHandle("reload")
        _state.value = EngineState.Loading(request.modelId, 0.05f, "Preparando NPH…")
        ZitDiag.bc("NPH load dit=${paths.dit} te=${paths.te} vae=${paths.vae} loras=${request.loras.size}")
        // Actual ctx create is deferred to first generate (so Extremo can skip recreate).
        // We still mark Ready so UI can proceed; generate() creates if needed.
        loadedId = request.modelId
        loadedLoraKey = loraKey
        _state.value = EngineState.Ready(request.modelId, request.loras)
    }

    override suspend fun unload() = withContext(Dispatchers.IO) {
        ZitDiag.bc("NPH unload")
        freeHandle("unload")
        _state.value = EngineState.Idle
    }

    override fun cancel() {
        cancelled.set(true)
        runCatching { NphNative.cancel() }
    }

    private fun rgbToBitmap(rgb: ByteArray, w: Int, h: Int): Bitmap {
        val px = IntArray(w * h)
        var i = 0
        for (p in 0 until w * h) {
            val r = rgb[i++].toInt() and 0xff
            val g = rgb[i++].toInt() and 0xff
            val b = rgb[i++].toInt() and 0xff
            px[p] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
    }

    /** Apply Avanzado / profiling env BEFORE Hexagon session init (lib must be loaded). */
    private fun applyHexagonEnv() {
        ensureLib()
        fun set(k: String, v: String?) {
            if (v.isNullOrBlank()) return
            runCatching { NphNative.setEnv(k, v.trim()) }
                .onSuccess { ZitDiag.bc("NPH env $k=$v") }
                .onFailure { ZitDiag.bc("NPH env $k failed: $it") }
        }
        // Always write 0/1 so a prior PROFILE=1 cannot stick after the user turns it off.
        set("GGML_HEXAGON_PROFILE", if (prefs.nphProfile) "1" else "0")
        set("GGML_HEXAGON_NHVX", prefs.nphNhvx)
        set("GGML_HEXAGON_NHMX", prefs.nphNhmx)
        set("GGML_HEXAGON_OPBATCH", prefs.nphOpbatch)
        set("GGML_HEXAGON_OPQUEUE", prefs.nphOpqueue)
        set("GGML_HEXAGON_OPFUSION", prefs.nphOpfusion)
        set("GGML_HEXAGON_VMEM", prefs.nphVmem)
        set("GGML_HEXAGON_MBUF", prefs.nphMbuf)
        applyNphLoadEnv()
        // cpuOffload / overlapDiT∥TE removed from UI (phone: slower / no gain). Prefs ignored.
    }

    /** 0.28 load-path switches + experiments (read by libnph at create / load / generate). */
    private fun applyNphLoadEnv() {
        fun b(k: String, v: Boolean) { runCatching { NphNative.setEnv(k, if (v) "1" else "0") } }
        b("NPH_FAST_REPACK", prefs.nphFastRepack)
        b("NPH_PARALLEL_SET", prefs.nphParallelSet)
        b("NPH_FAST_READ", prefs.nphFastRead)
        b("NPH_REPACK_CACHE", prefs.nphRepackCache)
        b("NPH_BUF_POOL", prefs.nphBufPool)
        b("NPH_PREFETCH_DIT", prefs.nphPrefetchDit)
        b("NPH_TE_CACHE", prefs.nphTeCache)
        b("NPH_TE_CACHE_DISK", prefs.nphTeCacheDisk)
        b("NPH_PREFETCH_VAE", prefs.nphExpPrefetchVae)
        // NPG report/check hooks only while the GPU VAE is in use (default path untouched).
        b("NPH_NPG_STATS", ctxNpgVae)
        b("NPH_NPG_VAE_CHECK", ctxNpgVae)
        b("NPH_LORA_COMBINED", prefs.nphLoraCombined)
        b("NPH_LORA_FUSE", prefs.nphLoraFuse)
        b("NPH_NPG_MM_F32", prefs.nphNpgMmF32)
        b("NPH_NPG_MM_NPU", NPG_BUILD && prefs.nphNpgVae && prefs.nphNpgMmNpu)  // native: only when the VAE is on a non-HTP device
        b("NPH_NPG_PART_FALLBACK", NPG_BUILD && prefs.nphNpgVae && prefs.nphNpgPartFallback)
        // Upstream ggml-opencl reads this once at OpenCL backend init (FP32-accumulating GEMM when 0).
        if (NPG_BUILD && prefs.nphNpgVae && prefs.nphNpgMmF32) runCatching { NphNative.setEnv("GGML_OPENCL_ADRENO_XMEM_GEMM", "0") }
        runCatching {
            @Suppress("DEPRECATION")
            val pi = context.packageManager.getPackageInfo(context.packageName, 0)
            val vc = if (Build.VERSION.SDK_INT >= 28) pi.longVersionCode else pi.versionCode.toLong()
            NphNative.setEnv("NPH_APP_VERSION", "$vc/${pi.versionName}")
        }
        runCatching { NphNative.setEnv("NPH_CACHE_DIR", cacheDir(context).absolutePath) }
    }

    private fun nphTogglesLine(): String =
        "Mejoras: repack-rápido=${prefs.nphFastRepack} set-paralelo=${prefs.nphParallelSet} lectura-rápida=${prefs.nphFastRead} " +
            "caché-repack=${prefs.nphRepackCache} pool-buffers=${prefs.nphBufPool} prefetch-DiT=${prefs.nphPrefetchDit} " +
            "caché-TE=${prefs.nphTeCache} (disco=${prefs.nphTeCacheDisk && prefs.nphTeCache}) LoRA-combinado=${prefs.nphLoraCombined} LoRA-fusionada-HMX=${prefs.nphLoraFuse} " +
            "residencia-adaptativa=true caché-LoRA-disco=${prefs.nphLoraCombined}\n" +
            "Experimental: prefetch-VAE=${prefs.nphExpPrefetchVae}\n" +
            prefs.nonDefaultNphToggles().let { if (it.isEmpty()) "" else "Cambiados a mano (no predeterminado): ${it.joinToString(", ")}\n" }

    // ---------------- 0.29 NPG ----------------
    private fun npgSentinelFile(): File = File(cacheDir(context), "npg_inflight.txt")

    private fun npgEvent(msg: String) {
        synchronized(npgEvents) {
            if (npgEvents.size > 20) npgEvents.removeAt(0)
            npgEvents += msg
        }
        ZitDiag.bc("NPG: $msg")
    }

    /** A leftover in-flight file means the process died during NPG work → auto-disable NPG VAE. */
    private fun npgCheckStartup() {
        if (npgStartupChecked) return
        npgStartupChecked = true
        val f = npgSentinelFile()
        if (!f.exists()) return
        if (!NPG_BUILD) {
            // Stale file from a 0.29/0.30 NPG session: just drop it. This guard only ever touches the NPG
            // GPU-VAE block flag (never the Mejoras / other toggles).
            runCatching { f.delete() }
            return
        }
        val txt = runCatching { f.readText() }.getOrDefault("").trim()
        runCatching { f.delete() }
        val why = txt.lines().filter { it.isNotBlank() }.takeLast(4).joinToString(" | ").take(600)
        prefs.nphNpgBlocked = if (why.isBlank()) "cierre inesperado" else why
        npgEvent("arranque: la sesión anterior se cerró durante trabajo NPG → VAE GPU desactivado ($why)")
    }

    private fun npgInflightBegin(stage: String) {
        val f = npgSentinelFile()
        runCatching { f.writeText("etapa=$stage ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.ROOT).format(java.util.Date())}\n") }
        runCatching { NphNative.npgSentinel(f.absolutePath) }
    }

    private fun npgInflightEnd() {
        runCatching { NphNative.npgSentinel("") }
        runCatching { npgSentinelFile().delete() }
    }

    // 0.30.1: NPG (GPU VAE) removed from the UI and the default native build (code kept for git history /
    // a -DNPH_OPENCL=ON build). Never used, whatever an old pref says.
    private fun npgWanted(): Boolean = NPG_BUILD && prefs.nphNpgVae && prefs.nphNpgBlocked.isBlank() && npgSessionOff == null

    /** Report block "NPG" (only when the toggle is on or something happened). */
    private fun npgReport(phaseReport: String?): String {
        val events = synchronized(npgEvents) { npgEvents.toList() }
        if (!NPG_BUILD) return ""
        if (!prefs.nphNpgVae && events.isEmpty() && prefs.nphNpgBlocked.isBlank()) return ""
        return buildString {
            append("— NPG (experimental) —\n")
            append("Etapas: TE=NPU (HTP0) · DiT=NPU (HTP0) · VAE=")
            append(if (ctxNpgVae) "GPU Adreno (GPUOpenCL, ggml-opencl upstream)" else "NPU (HTP0)")
            append('\n')
            if (prefs.nphNpgVae && !ctxNpgVae) {
                val why = prefs.nphNpgBlocked.ifBlank { npgSessionOff ?: "" }
                if (why.isNotBlank()) append("VAE GPU desactivado: $why\n")
            }
            if (npgPrepareText.isNotBlank()) {
                append("Preparación OpenCL:\n")
                npgPrepareText.lines().filter { it.isNotBlank() }.forEach { append("  ").append(it.trim()).append('\n') }
            }
            if (ctxNpgVae) {
                phaseReport?.lines()?.firstOrNull { it.startsWith("VAE:") }?.let { append("Tiempo VAE en GPU: ").append(it.removePrefix("VAE:").trim()).append('\n') }
                val notes = runCatching { NphNative.npgNotes() }.getOrDefault("")
                notes.lines().filter { it.isNotBlank() }.forEach { append("  ").append(it.trim()).append('\n') }
                append("Precisión VAE GPU: pesos FP16, activaciones y acumulación FP32 (conv2d ggml-opencl")
                append(if (prefs.nphNpgMmF32) "; mul_mat sin GEMM xmem FP16" else "; AVISO: GEMM xmem de Adreno acumula en FP16")
                append("); ops no soportadas → CPU FP32\n")
                append("Revisión por partes: ").append(if (prefs.nphNpgPartFallback) "activa (ver «partes VAE»)" else "apagada (solo revisión de imagen completa)").append('\n')
            }
            append("Zero-copy (dmabuf): no usado en 0.30 — pesos del VAE en buffers OpenCL propios; latente (~1 MB) e imagen pasan por el host (ver «entrada/salida» en la línea VAE)\n")
            append("Solape VAE(N) ∥ DiT(N+1): no incluido (la app no genera N seguidas) · TE NPU ∥ GPU: no incluido\n")
            val err = runCatching { NphNative.npgLastError() }.getOrDefault("")
            if (err.isNotBlank()) append("Último error OpenCL: ${err.take(300)}\n")
            append("Fallbacks: ").append(if (events.isEmpty()) "ninguno" else "").append('\n')
            events.forEach { append("  • ").append(it).append('\n') }
        }
    }

    /** Prefault models into page cache. Safe / bit-identical. Optional parallel DiT∥TE warm. */
    private fun warmupModels(paths: NphModelStore.Paths) {
        if (!prefs.nphPageCacheWarmup && !prefs.nphOverlapDitTe) return
        val t0 = System.nanoTime()
        fun warm(label: String, path: String) {
            val n = runCatching { NphNative.warmupFile(path) }.getOrDefault(0L)
            ZitDiag.bc("NPH warmup $label ${n / (1 shl 20)} MB")
        }
        if (prefs.nphOverlapDitTe) {
            // Approximate "overlap DiT load with TE": warm DiT+VAE and TE concurrently (page cache only).
            val a = Thread({ warm("DiT", paths.dit); warm("VAE", paths.vae) }, "nph-warm-dit")
            val b = Thread({ warm("TE", paths.te) }, "nph-warm-te")
            a.start(); b.start(); a.join(); b.join()
        } else if (prefs.nphPageCacheWarmup) {
            warm("DiT", paths.dit); warm("TE", paths.te); warm("VAE", paths.vae)
        }
        ZitDiag.bc("NPH warmup done in ${"%.2f".format(Locale.ROOT, (System.nanoTime() - t0) / 1e9)}s")
    }

    private fun ensureCtx(paths: NphModelStore.Paths, modelId: String): Long {
        // Profile toggle (or Liberar) must recreate HTP session so opt_profile / DSP profiler apply.
        if (handle != 0L && ctxProfile != prefs.nphProfile) {
            freeHandle("profile-toggle")
        }
        ensureLib()
        npgCheckStartup()
        val wantNpg = npgWanted()
        if (handle != 0L && ctxNpgVae != wantNpg) freeHandle("npg-toggle")
        if (handle != 0L) return handle
        _state.value = EngineState.Loading(modelId, 0.1f, "Cargando DiT/TE/VAE en HTP…")
        applyHexagonEnv()
        warmupModels(paths)
        val timings = DoubleArray(1)
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 6)
        synchronized(logLock) { logBuf.setLength(0) }
        var useNpg = false
        if (wantNpg) {
            _state.value = EngineState.Loading(modelId, 0.1f, "NPG: preparando GPU (OpenCL)…")
            npgInflightBegin("preparación OpenCL")
            val r = runCatching { NphNative.npgPrepare(cacheDir(context).absolutePath, prefs.nphNpgSelfTest) }
                .getOrElse { "FAIL\n${it.message}" }
            npgInflightEnd()
            npgPrepareText = r.substringAfter('\n', "")
            if (r.startsWith("OK")) {
                useNpg = true
            } else {
                val why = npgPrepareText.lines().lastOrNull { it.isNotBlank() }?.trim() ?: "error"
                npgSessionOff = why
                npgEvent("preparación OpenCL falló → VAE en NPU ($why)")
            }
        }
        var h = 0L
        if (useNpg) {
            ctxNpgVae = true
            applyNphLoadEnv()
            npgInflightBegin("crear contexto (VAE en GPU)")
            h = NphNative.create(
                paths.dit, paths.te, paths.vae,
                BACKEND_NPG_VAE, PARAMS_BACKEND,
                threads, FLAGS_DEFAULT, cb(), timings,
            )
            npgInflightEnd()
            if (h == 0L) {
                val why = runCatching { NphNative.npgLastError() }.getOrDefault("").ifBlank { "create devolvió 0" }
                npgSessionOff = why
                npgEvent("crear contexto con VAE GPU falló → VAE en NPU ($why)")
                ctxNpgVae = false
                applyNphLoadEnv()
            }
        }
        if (h == 0L) {
            ctxNpgVae = false
            h = NphNative.create(
                paths.dit, paths.te, paths.vae,
                BACKEND, PARAMS_BACKEND,
                threads, FLAGS_DEFAULT, cb(), timings,
            )
        }
        createSec = timings[0]
        if (h == 0L) error("NPH: no se pudo crear el contexto (¿HTP/skels/modelos?)")
        handle = h
        ctxProfile = prefs.nphProfile
        ZitDiag.bc("NPH ctx ok in ${"%.1f".format(Locale.ROOT, createSec)}s profile=$ctxProfile avail=${memAvailMb()}MB")
        return h
    }

    override fun generate(request: GenerationRequest): Flow<GenerationEvent> = callbackFlow {
        cancelled.set(false)
        val ready = _state.value as? EngineState.Ready
        if (ready == null && loadedId == null) {
            trySend(GenerationEvent.Failed("No hay modelo NPH cargado")); close(); return@callbackFlow
        }
        val modelId = ready?.modelId ?: loadedId ?: "nph"
        val resolved = NphModelStore.resolve(context)
            ?: run { trySend(GenerationEvent.Failed("Faltan rutas DiT/TE/VAE")); close(); return@callbackFlow }
        heldResolved = resolved
        val paths = resolved.paths
        val keepResident = request.mode == ExecMode.EXTREMO
        _state.value = EngineState.Busy(modelId, JobKind.GENERATE)
        val ramBefore = memAvailMb()
        var ramMin = ramBefore
        val stepTimes = ArrayList<Float>()
        var lastStepAt = 0L
        try {
            // LoRA identity: if changed since last ctx, recreate
            val loraKey = request.loras.filter { it.strength != 0f && File(it.path).isFile }
                .sortedBy { it.path }.joinToString("|") { "${it.path}@${it.strength}" }
            if (handle != 0L && loadedLoraKey != loraKey) {
                freeHandle("lora-change")
                loadedLoraKey = loraKey
            }
            synchronized(residencyNotes) { residencyNotes.clear() }
            val loras0 = request.loras.filter { it.strength != 0f && File(it.path).isFile }.take(LoraLimits.MAX_ACTIVE)
            if (handle != 0L && keepResident) {
                // 0.30: resident DiT + TE-cache miss → the ~4 GB TE must fit next to it, else free first.
                val probe = runCatching {
                    NphNative.teProbe(
                        handle, request.prompt, request.steps, request.width, request.height, request.cfg,
                        loras0.map { it.path }.toTypedArray(),
                        FloatArray(loras0.size) { loras0[it].strength.coerceIn(LoraLimits.MIN_STRENGTH, LoraLimits.MAX_STRENGTH) },
                    )
                }.getOrDefault(-1)
                val availPre = memAvailMb()
                val probeTxt = when (probe) { 2 -> "HIT RAM"; 1 -> "HIT disco"; 0 -> "MISS"; else -> "n/d" }
                val note = if (mustFreeBeforeGen(probe, availPre)) {
                    freeHandle("TE-miss con DiT residente (avail ${availPre} MB < ${TE_LOAD_MB + TE_MARGIN_MB} MB)")
                    "antes de gen: caché TE=$probeTxt y avail ${availPre} MB < TE ${TE_LOAD_MB}+${TE_MARGIN_MB} MB → DiT liberado (recarga)"
                } else {
                    "antes de gen: caché TE=$probeTxt · avail ${availPre} MB → DiT residente reutilizado (sin recarga)"
                }
                synchronized(residencyNotes) { residencyNotes += note }
                ZitDiag.bc("NPH residencia: $note")
            }
            val h = try {
                ensureCtx(paths, modelId)
            } catch (t: Throwable) {
                trySend(GenerationEvent.Failed(t.message ?: "error al cargar NPH"))
                close(); return@callbackFlow
            }
            _state.value = EngineState.Busy(modelId, JobKind.GENERATE)
            trySend(GenerationEvent.StageChanged(Stage.TEXT_ENCODER))

            val loras = request.loras.filter { it.strength != 0f && File(it.path).isFile }.take(LoraLimits.MAX_ACTIVE)
            val loraPaths = loras.map { it.path }.toTypedArray()
            val loraStr = FloatArray(loras.size) { loras[it].strength.coerceIn(LoraLimits.MIN_STRENGTH, LoraLimits.MAX_STRENGTH) }

            // Live HTP profile + host-overhead accumulator (survives log truncation).
            val prof = HtpProfileAccum()
            val samplerSteps = request.steps
            // H1: Extremo + graph-cache → keep DiT gallocr/HTP buffers across gens.
            val keepCompute = keepResident && prefs.nphGraphCache
            runCatching { NphNative.setEnv("NPH_KEEP_COMPUTE", if (keepCompute) "1" else "0") }
                .onSuccess { ZitDiag.bc("NPH env NPH_KEEP_COMPUTE=${if (keepCompute) 1 else 0} (Extremo=${keepResident} graphCache=${prefs.nphGraphCache})") }
            applyNphLoadEnv()  // runtime toggles (prefetch / TE cache) without engine release
            val progressCb = object : NphCallback {
                override fun onLog(level: Int, msg: String) {
                    cb().onLog(level, msg)
                    if (prefs.nphProfile || msg.contains("nph-host-overhead") || msg.contains("nph-lora-")) prof.ingest(msg)
                }
                override fun onProgress(step: Int, steps: Int, time: Float) {
                    // Only real sampler steps. Load uses pretty_bytes_progress (steps=tens of tensors,
                    // time~0.01s); tiling uses other totals — those must NOT enter "Por paso".
                    val isSampler = steps == samplerSteps && step in 1..samplerSteps
                    if (isSampler) {
                        stepTimes += time
                        if (prefs.nphProfile) prof.onSamplerStep(step, time)
                        trySend(GenerationEvent.StageChanged(Stage.DENOISE))
                        trySend(GenerationEvent.Step(step, steps, null))
                    }
                    ramMin = minOf(ramMin, memAvailMb())
                }
            }
            val timings = DoubleArray(3)
            ZitDiag.bc("NPH generate ${request.width}x${request.height} steps=${request.steps} mode=${request.mode} loras=${loras.size}")
            lastStepAt = System.currentTimeMillis()
            fun runGen(hh: Long): ByteArray? {
                if (ctxNpgVae) npgInflightBegin("generar (VAE en GPU)")
                try {
                    return NphNative.generate(
                        hh, request.prompt, request.steps, request.width, request.height, request.seed, request.cfg,
                        loraPaths, loraStr, progressCb, timings,
                    )
                } finally {
                    if (ctxNpgVae) npgInflightEnd()
                }
            }
            var rgb = runGen(h)
            if (rgb == null && !cancelled.get() && ctxNpgVae) {
                // NPG fallback: GPU VAE failed (OpenCL error / non-finite output) → redo with the NPU VAE.
                // The TE cache makes the retry skip the text encoder.
                val notes = runCatching { NphNative.npgNotes() }.getOrDefault("")
                val why = runCatching { NphNative.npgLastError() }.getOrDefault("").ifBlank {
                    notes.lines().firstOrNull { it.startsWith("VAE rechazado") } ?: "generate falló"
                }
                npgSessionOff = why
                npgEvent("VAE GPU falló durante la generación → se repite con VAE en NPU ($why)")
                freeHandle("npg-fallback")
                val h2 = ensureCtx(paths, modelId)
                _state.value = EngineState.Busy(modelId, JobKind.GENERATE)
                stepTimes.clear()
                rgb = runGen(h2)
            }
            val genS = timings[0]
            val ramAfter = memAvailMb()
            if (rgb != null) gensOnCtx += 1
            val floor = keepFloorMb(loras.size)
            val willKeep = keepResident && rgb != null && !cancelled.get() && ramAfter >= floor
            synchronized(residencyNotes) {
                residencyNotes += when {
                    !keepResident -> "después: modo Seguro → se libera"
                    willKeep -> "después: avail ${ramAfter} MB ≥ umbral ${floor} MB → DiT+VAE se mantienen residentes"
                    cancelled.get() -> "después: cancelado → se libera"
                    rgb == null -> "después: generación fallida → se libera"
                    else -> "después: avail ${ramAfter} MB < umbral ${floor} MB → se libera"
                }
            }
            if (cancelled.get() && rgb == null) {
                trySend(GenerationEvent.Cancelled)
            } else if (rgb == null) {
                val tail = synchronized(logLock) { logBuf.toString() }
                prof.ingestLogTail(tail)
                trySend(GenerationEvent.Failed("NPH: generación fallida", buildPerf(request, createSec, genS, stepTimes, ramBefore, ramMin, ramAfter, false, null, prof) + "\n--- log ---\n${tail.takeLast(2000)}"))
            } else {
                trySend(GenerationEvent.StageChanged(Stage.VAE_DECODE))
                val bmp = rgbToBitmap(rgb, request.width, request.height)
                val teMs = estimateTeMs(stepTimes, genS, createSec)
                val denoiseMs = (stepTimes.sum() * 1000).toLong().coerceAtLeast(0)
                val decodeMs = ((genS * 1000) - teMs - denoiseMs).toLong().coerceAtLeast(0)
                run {
                    val tail = synchronized(logLock) { logBuf.toString() }
                    prof.ingestLogTail(tail)
                }
                val perf = buildPerf(request, createSec, genS, stepTimes, ramBefore, ramMin, ramAfter, willKeep, loras, prof)
                trySend(GenerationEvent.Completed(bmp, Timings(teMs, denoiseMs, decodeMs, 0, perf)))
            }

            // Residency policy (decided above so the report label matches what happens)
            val avail = ramAfter
            if (!willKeep) {
                // 0.30.2: the reason must match the actual cause (failure/cancel vs low RAM vs Seguro).
                val reason = when {
                    !keepResident -> "Seguro"
                    cancelled.get() -> "Extremo→cancelado"
                    rgb == null -> "Extremo→generación fallida"
                    else -> "Extremo→poco RAM ($avail MB < $floor MB)"
                }
                freeHandle(reason)
                _state.value = EngineState.Idle
            } else {
                _state.value = EngineState.Ready(modelId, loras)
                ZitDiag.bc("NPH Extremo: DiT+VAE residentes, avail=${avail}MB")
            }
        } catch (t: Throwable) {
            if (t is kotlinx.coroutines.CancellationException) throw t
            freeHandle("error")
            trySend(GenerationEvent.Failed(t.message ?: t.javaClass.simpleName))
            _state.value = EngineState.Error(t.message ?: "error")
        } finally {
            close()
        }
        awaitClose { }
    }.flowOn(Dispatchers.Default)

    private fun estimateTeMs(steps: List<Float>, genS: Double, createS: Double): Long {
        // TE is inside generate_image before steps; approximate from gen - sum(steps) - ~2s VAE
        val stepSum = steps.sum().toDouble()
        val rest = (genS - stepSum).coerceAtLeast(0.0)
        val te = (rest - 2.0).coerceAtLeast(0.5) // rough VAE ~1.9s on phone
        return (te * 1000).toLong()
    }

    private fun buildPerf(
        req: GenerationRequest, createS: Double, genS: Double, steps: List<Float>,
        ramBefore: Int, ramMin: Int, ramAfter: Int, kept: Boolean, loras: List<LoraBinding>?,
        prof: HtpProfileAccum = HtpProfileAccum(),
    ): String = buildString {
        append("Motor: NPH (sd.cpp Hexagon HTP)\n")
        append("Backend: ${if (ctxNpgVae) BACKEND_NPG_VAE else BACKEND} · params: $PARAMS_BACKEND · fa · vae_conv_direct · mmap off\n")
        append("Modo: ${req.mode.label}" + (if (kept && req.mode == ExecMode.EXTREMO) " (DiT+VAE quedan residentes)" else if (req.mode == ExecMode.EXTREMO) " (se libera tras esta imagen)" else "") + "\n")
        append("Residencia: gen #$gensOnCtx en este contexto" + (if (gensOnCtx >= 2) " (DiT ya residente: sin recarga de 5.9 GB)" else " (contexto nuevo: DiT cargado)") + "\n")
        synchronized(residencyNotes) { residencyNotes.forEach { append("  · ").append(it).append('\n') } }
        append("Resolución: ${req.width}×${req.height} · ${req.steps} pasos · ${req.sampler.label} · CFG ${req.cfg}\n")
        if (!loras.isNullOrEmpty()) append("LoRAs: ${loras.joinToString { "${File(it.path).name}×${it.strength}" }}\n")
        append("Carga contexto: ${"%.1f".format(Locale.ROOT, createS)} s\n")
        append("Generación: ${"%.1f".format(Locale.ROOT, genS)} s\n")
        if (steps.isNotEmpty()) {
            val avg = steps.average()
            append("Por paso (solo sampler ${steps.size}/${req.steps}): ${steps.map { "%.2f".format(Locale.ROOT, it) }.joinToString(", ")} s  (media ${"%.2f".format(Locale.ROOT, avg)} s/paso)\n")
        } else append("Por paso: (sin pasos de sampler registrados)\n")
        append("RAM (MemAvailable): antes ${ramBefore} · mín ${ramMin} · después ${ramAfter} MB\n")
        append("Dispositivo: ${Build.MODEL} · Android ${Build.VERSION.RELEASE} · ${Build.SOC_MODEL}\n")
        append("Precisión: FP8 e4m3 pesos · FP32 residual · TE Q8_0 (sin bajar)\n")
        append("Ajustes: page-cache=${prefs.nphPageCacheWarmup} profile=${prefs.nphProfile} graph-cache=${prefs.nphGraphCache}\n")
        val knobs = listOf(
            "NHVX" to prefs.nphNhvx, "NHMX" to prefs.nphNhmx, "OPBATCH" to prefs.nphOpbatch,
            "OPQUEUE" to prefs.nphOpqueue, "OPFUSION" to prefs.nphOpfusion,
            "VMEM" to prefs.nphVmem, "MBUF" to prefs.nphMbuf,
        ).filter { it.second.isNotBlank() }
        if (knobs.isNotEmpty()) append("Avanzado: ${knobs.joinToString { "${it.first}=${it.second}" }}\n")
        else append("Avanzado: defaults (NHVX=all, NHMX=1, OPBATCH=1280, OPQUEUE=32, OPFUSION=1)\n")
        append(nphTogglesLine())
        val phase = runCatching { NphNative.lastPhaseReport() }.getOrNull()?.takeIf { it.isNotBlank() }
        phase?.let { append(it) }
        append(npgReport(phase))
        append(prof.loraReport())
        if (prefs.nphProfile) append(prof.report(steps, gensOnCtx))
        else append(prof.hostReport(gensOnCtx))
        append("Veredicto: NPH OK")
        if (steps.isNotEmpty()) append(" · ${"%.1f".format(Locale.ROOT, steps.average())} s/paso")
        append('\n')
    }

    override fun upscale(request: UpscaleRequest): Flow<GenerationEvent> = callbackFlow {
        trySend(GenerationEvent.Failed("Upscale con NPH: próximamente (Real-ESRGAN no va en este motor)"))
        close()
        awaitClose { }
    }.flowOn(Dispatchers.Default)
}
