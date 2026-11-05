package com.zithexagon.engine

import java.util.Locale
import android.content.Context
import android.graphics.Bitmap
import android.system.Os
import com.zithexagon.ZitDiag
import com.zithexagon.data.LoraLimits
import com.zithexagon.data.ModelManifest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

/** Callback from native code (any thread). stage: 0 load, 1 TE, 2 denoise, 3 VAE, 4 upscale. */
interface NativeProgress { fun onEvent(stage: Int, step: Int, total: Int, detail: String, progress: Float) }

/**
 * Real engine: in-process JNI library (libzitengine.so). Text encoder Qwen3-4B Q8 on CPU (llama.cpp),
 * DiT as N HTP context binaries + VAE (+ upscaler) on the Hexagon NPU through QNN 2.50, all bound by
 * IO tensor name. See /native in the source tree.
 */
class NativeZitEngine(private val context: Context) : ZitEngine {
    override val info = EngineInfo("ZIT-HEXAGON", "QNN 2.50 · HTP v81 + CPU (llama.cpp)", isFake = false)
    private val _state = MutableStateFlow<EngineState>(EngineState.Idle)
    override val state: StateFlow<EngineState> = _state.asStateFlow()
    private var handle = 0L
    private var manifest: ModelManifest? = null
    private var modelDir: File? = null

    override suspend fun load(request: LoadRequest) = withContext(Dispatchers.IO) {
        val cur = _state.value
        if (cur is EngineState.Ready && cur.modelId == request.modelId && handle != 0L) return@withContext
        lib.exceptionOrNull()?.let { ZitDiag.bc("loadLibrary FAILED: $it"); error("libzitengine no disponible: ${it.message}") }
        ZitDiag.bc("engine load ${request.modelId} dir=${request.modelDir}")
        if (handle != 0L) { nativeFree(handle); handle = 0L }
        val dir = File(request.modelDir ?: error("paquete sin carpeta"))
        val mf = ModelManifest.parse(File(dir, ModelManifest.FILE_NAME).readText())
        fun f(key: String) = mf.files[key]?.let { File(dir, it).absolutePath } ?: ""
        val chunks = mf.files.keys.filter { it.startsWith("dit_") }.sortedBy { it.removePrefix("dit_").toInt() }.map { f(it) }
        require(chunks.isNotEmpty()) { "zit.json sin bloques dit_N" }
        val eng = JSONObject(mf.engineJson)
        val libDir = context.applicationInfo.nativeLibraryDir
        // HTP skel lookup path for FastRPC (DSP side) -> the extracted jniLibs folder.
        runCatching { Os.setenv("ADSP_LIBRARY_PATH", "$libDir;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp", true) }
        _state.value = EngineState.Loading(request.modelId, 0f, "Iniciando NPU")
        val err = arrayOfNulls<String>(1)
        val h = onBigStack { nativeLoad(
            arrayOf("$libDir/libQnnHtp.so", "$libDir/libQnnSystem.so", f("text_encoder"), f("t_embedder"), f("vae"),
                f("upscaler_realesrgan-x4plus"),
                // learned per-device NPU residency cap (how many DiT blocks fit on the NPU at once)
                File(ZitDiag.dir(context).apply { mkdirs() }, "npu_residency_${request.modelId.replace(Regex("[^A-Za-z0-9._-]"), "_")}.txt").absolutePath),
            chunks.toTypedArray(),
            intArrayOf(eng.optInt("caption_len", 512), eng.optInt("lat_h", 128), eng.optInt("lat_w", 128),
                Runtime.getRuntime().availableProcessors().coerceIn(2, 6)),
            doubleArrayOf(eng.optDouble("shift", 3.0), eng.optDouble("vae_scale", 0.3611), eng.optDouble("vae_shift", 0.1159)),
            object : NativeProgress {
                override fun onEvent(stage: Int, step: Int, total: Int, detail: String, progress: Float) {
                    _state.value = EngineState.Loading(request.modelId, progress, detail)
                }
            }, err) }
        ZitDiag.bc("engine load -> " + if (h == 0L) "FAILED ${err[0]}" else "ok")
        if (h == 0L) { _state.value = EngineState.Error(err[0] ?: "error"); error(err[0] ?: "no se pudo cargar el motor") }
        handle = h; manifest = mf; modelDir = dir
        _state.value = EngineState.Ready(request.modelId, emptyList())
    }

    override suspend fun unload() = withContext(Dispatchers.IO) {
        ZitDiag.bc("engine unload")
        if (handle != 0L) { nativeFree(handle); handle = 0L }
        _state.value = EngineState.Idle
    }

    override fun cancel() { if (handle != 0L) nativeCancel(handle) }

    private fun toBitmap(px: IntArray, w: Int, h: Int) = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)

    override fun generate(request: GenerationRequest): Flow<GenerationEvent> = callbackFlow {
        val ready = _state.value as? EngineState.Ready
        val mf = manifest
        if (ready == null || handle == 0L || mf == null) { trySend(GenerationEvent.Failed("No hay ningún modelo cargado")); close(); return@callbackFlow }
        val eng = JSONObject(mf.engineJson)
        val ow = eng.optInt("lat_w", 128) * 8; val oh = eng.optInt("lat_h", 128) * 8
        // 0.25: fixed 1024×1024 (the package's compiled size); any other requested size is ignored, never an error
        if (request.width != ow || request.height != oh) ZitDiag.bc("size ${request.width}x${request.height} -> fixed ${ow}x$oh")
        // 0.25: LoRA swap (rank 96, merged on the host into the graphs' LoRA inputs); 0-strength / missing files are skipped
        val loras = if (!mf.loraSupported) emptyList() else
            request.loras.filter { it.strength != 0f && File(it.path).isFile }.take(LoraLimits.MAX_ACTIVE)
        if (request.loras.isNotEmpty() && !mf.loraSupported) ZitDiag.bc("loras ignored: package without LoRA support")
        _state.value = EngineState.Busy(ready.modelId, JobKind.GENERATE)
        try {
            val times = DoubleArray(4); val err = arrayOfNulls<String>(1); val perf = arrayOfNulls<String>(1)
            var lastStage = -1
            val cb = object : NativeProgress {
                override fun onEvent(stage: Int, step: Int, total: Int, detail: String, progress: Float) {
                    if (stage != lastStage) {
                        lastStage = stage
                        trySend(GenerationEvent.StageChanged(when (stage) { 1 -> Stage.TEXT_ENCODER; 2 -> Stage.DENOISE; 3 -> Stage.VAE_DECODE; 4 -> Stage.UPSCALE; else -> Stage.LOADING }))
                    }
                    if (stage == 2 && total > 0) trySend(GenerationEvent.Step(step, total, null))
                }
            }
            ZitDiag.bc("JOB start generate loras=${loras.size} steps=${request.steps} cfg=${request.cfg} sampler=${request.sampler.id}/${request.scheduler.id} neg=${request.negativePrompt.isNotBlank()} up=${request.upscale?.scale ?: 0}")
            val dbgDir = if (!request.debugDump) "" else
                (if (com.zithexagon.data.Storage.hasAllFilesAccess()) File(com.zithexagon.data.Storage.sharedRoot, "debug")
                 else File(context.getExternalFilesDir(null) ?: context.filesDir, "debug")).absolutePath
            val c0Alt = if (!request.debugDump) "" else
                listOf(File(com.zithexagon.data.Storage.sharedRoot, "test/$C0_AB_FILE"),
                       File(android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS), C0_AB_FILE))
                    .firstOrNull { runCatching { it.isFile && it.length() > 0 }.getOrDefault(false) }?.absolutePath ?: ""
            if (request.debugDump) ZitDiag.bc("debug dump -> $dbgDir, c0 A/B '${c0Alt}'")
            val px = onBigStack { nativeGenerate(handle, request.prompt, request.negativePrompt, request.sampler.id, request.scheduler.id,
                request.steps, request.cfg.toDouble(), request.seed, cb, times, perf, err,
                loras.map { it.path }.toTypedArray(), FloatArray(loras.size) { loras[it].strength.coerceIn(LoraLimits.MIN_STRENGTH, LoraLimits.MAX_STRENGTH) }, request.mode.id,
                dbgDir, c0Alt) }
            ZitDiag.bc("generate -> " + (if (px != null) "ok" else "null ${err[0]}"))
            when {
                px == null && err[0] == "cancelled" -> trySend(GenerationEvent.Cancelled)
                px == null -> trySend(GenerationEvent.Failed(err[0] ?: "error", perf[0]))
                else -> {
                    var bmp = toBitmap(px, ow, oh)
                    var upMs = 0L
                    val up = request.upscale
                    if (up != null) {
                        trySend(GenerationEvent.StageChanged(Stage.UPSCALE))
                        val t = System.currentTimeMillis()
                        bmp = runUpscale(bmp, up, cb) ?: bmp.also { trySend(GenerationEvent.Failed(lastErr ?: "upscale")) }
                        upMs = System.currentTimeMillis() - t
                    }
                    val pt = buildString {
                        append(perf[0] ?: "")
                        if (up != null) append("Upscale x${up.scale} (Real-ESRGAN): ${"%.1f".format(Locale.ROOT, upMs / 1000.0)} s · ${bmp.width}x${bmp.height}\n")
                    }.ifBlank { null }
                    trySend(GenerationEvent.Completed(bmp, Timings(times[0].toLong(), times[1].toLong(), times[2].toLong(), upMs, pt)))
                }
            }
        } finally {
            ZitDiag.bc("JOB end generate")
            if (handle != 0L) _state.value = EngineState.Ready(ready.modelId, loras)
            close()
        }
        awaitClose { }
    }.flowOn(Dispatchers.Default)

    @Volatile private var lastErr: String? = null

    private fun runUpscale(src: Bitmap, opt: UpscaleOptions, cb: NativeProgress): Bitmap? {
        if (opt.modelId != "realesrgan-x4plus") { lastErr = "Upscaler ${opt.modelId}: próximamente (este paquete incluye Real-ESRGAN x4plus)"; return null }
        val w = src.width; val h = src.height
        val px = IntArray(w * h); src.getPixels(px, 0, w, 0, 0, w, h)
        val dims = IntArray(2); val err = arrayOfNulls<String>(1)
        ZitDiag.bc("upscale ${w}x$h x${opt.scale}")
        val out = onBigStack { nativeUpscale(handle, px, w, h, opt.scale, cb, dims, err) } ?: run { lastErr = err[0]; ZitDiag.bc("upscale FAILED ${err[0]}"); return null }
        return toBitmap(out, dims[0], dims[1])
    }

    override fun upscale(request: UpscaleRequest): Flow<GenerationEvent> = callbackFlow {
        val ready = _state.value as? EngineState.Ready
        if (ready == null || handle == 0L) { trySend(GenerationEvent.Failed("Carga primero un modelo NPU")); close(); return@callbackFlow }
        _state.value = EngineState.Busy(ready.modelId, JobKind.UPSCALE)
        ZitDiag.bc("JOB start upscale")
        try {
            val t = System.currentTimeMillis()
            val cb = object : NativeProgress {
                override fun onEvent(stage: Int, step: Int, total: Int, detail: String, progress: Float) {
                    if (total > 0) trySend(GenerationEvent.Step(step, total, null))
                }
            }
            val bmp = runUpscale(request.source, request.options, cb)
            if (bmp == null) trySend(GenerationEvent.Failed(lastErr ?: "upscale"))
            else {
                val ms = System.currentTimeMillis() - t
                val pt = "Upscale x${request.options.scale} (Real-ESRGAN): ${"%.1f".format(Locale.ROOT, ms / 1000.0)} s · " +
                    "${request.source.width}x${request.source.height} → ${bmp.width}x${bmp.height}\n"
                trySend(GenerationEvent.Completed(bmp, Timings(upscaleMs = ms, perf = pt)))
            }
        } finally {
            ZitDiag.bc("JOB end upscale")
            _state.value = EngineState.Ready(ready.modelId, emptyList()); close()
        }
        awaitClose { }
    }.flowOn(Dispatchers.Default)

    private external fun nativeLoad(paths: Array<String>, chunks: Array<String>, ints: IntArray, dbl: DoubleArray,
                                    cb: NativeProgress, err: Array<String?>): Long
    private external fun nativeGenerate(h: Long, prompt: String, negative: String, sampler: String, scheduler: String,
                                        steps: Int, cfg: Double, seed: Long, cb: NativeProgress, times: DoubleArray, perf: Array<String?>,
                                        err: Array<String?>, loraPaths: Array<String>, loraStrengths: FloatArray, mode: Int,
                                        debugDir: String, c0Override: String): IntArray?
    private external fun nativeUpscale(h: Long, argb: IntArray, w: Int, hgt: Int, scale: Int, cb: NativeProgress,
                                       outDims: IntArray, err: Array<String?>): IntArray?
    private external fun nativeCancel(h: Long)
    private external fun nativeFree(h: Long)

    companion object {
        /** 0.25.2 A/B: block 0 built WITHOUT extended_udma (same DLC/precision), used only with the debug toggle. */
        const val C0_AB_FILE = "zit-test-c0-noudma.bin"
        /** Loaded on first real load() so the demo/fake path (and JVM tests) never need the .so. */
        private val lib: Result<Unit> by lazy {
            runCatching {
                System.loadLibrary("zitengine")
                // Native breadcrumbs + fatal-signal handler write to the same diag file as the Kotlin ones.
                ZitDiag.stagePath?.let { p -> runCatching { nativeDiagInit(p) } }
                Unit
            }
        }

        /** Native work runs on its own thread with a 64 MB stack (llama.cpp / QNN can use deep stacks). */
        private fun <T> onBigStack(block: () -> T): T {
            var result: Result<T>? = null
            val t = Thread(null, { result = runCatching(block) }, "zit-native", 64L shl 20)
            t.start(); t.join()
            return result!!.getOrThrow()
        }

        /** Diagnostics ("Probar bloque de prueba"): load each context file, run it [runs] times with dummy inputs and once
         *  profiled; the first file is the reference. The pipeline engine must be released first. Returns the report. */
        fun bench(context: Context, files: List<File>, runs: Int, onProgress: (String) -> Unit): String {
            lib.exceptionOrNull()?.let { error("libzitengine no disponible: ${it.message}") }
            val libDir = context.applicationInfo.nativeLibraryDir
            runCatching { Os.setenv("ADSP_LIBRARY_PATH", "$libDir;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp", true) }
            val err = arrayOfNulls<String>(1)
            val cb = object : NativeProgress {
                override fun onEvent(stage: Int, step: Int, total: Int, detail: String, progress: Float) = onProgress(detail)
            }
            val paths = arrayOf("$libDir/libQnnHtp.so", "$libDir/libQnnSystem.so") + files.map { it.absolutePath }
            return onBigStack { nativeBench(paths, runs, cb, err) } ?: error(err[0] ?: "la prueba falló")
        }

        @JvmStatic private external fun nativeBench(paths: Array<String>, runs: Int, cb: NativeProgress?, err: Array<String?>): String?
        @JvmStatic private external fun nativeDiagInit(path: String)
        @JvmStatic external fun nativeBreadcrumb(msg: String)
    }
}
