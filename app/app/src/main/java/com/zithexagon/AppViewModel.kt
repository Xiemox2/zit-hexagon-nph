package com.zithexagon

import com.zithexagon.service.EngineHost

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.annotation.StringRes
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.zithexagon.data.GalleryItem
import com.zithexagon.data.GalleryRepository
import com.zithexagon.data.GenParams
import com.zithexagon.data.LoraEntry
import com.zithexagon.data.LoraRepository
import com.zithexagon.data.LoraUse
import com.zithexagon.data.ModelRepository
import com.zithexagon.data.Prefs
import com.zithexagon.data.Resolution
import com.zithexagon.data.PromptKeywords
import com.zithexagon.data.UPSCALERS
import com.zithexagon.data.ZitModel
import com.zithexagon.engine.EngineState
import com.zithexagon.engine.FakeEngine
import com.zithexagon.data.ModelSource
import com.zithexagon.data.NphModelStore
import com.zithexagon.data.Role
import com.zithexagon.engine.GenerationEvent
import com.zithexagon.engine.GenerationRequest
import com.zithexagon.engine.LoadRequest
import com.zithexagon.engine.LoraBinding
import com.zithexagon.engine.Stage
import com.zithexagon.engine.UpscaleOptions
import com.zithexagon.engine.UpscaleRequest
import com.zithexagon.engine.ZitEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.random.Random

/** UI state of a running job (generation or upscale). */
data class JobUi(
    val kind: Kind,
    val stage: Stage,
    val step: Int = 0,
    val total: Int = 0,
    val preview: Bitmap? = null,
    val startedAt: Long = System.currentTimeMillis(),
    val loadDetail: String? = null,
    val loadProgress: Float = 0f,
) {
    enum class Kind { GENERATE, UPSCALE }
    val fraction: Float get() = when (stage) {
        Stage.LOADING -> loadProgress * 0.1f
        Stage.TEXT_ENCODER -> 0.1f
        Stage.DENOISE -> 0.12f + 0.78f * (if (total > 0) step / total.toFloat() else 0f)
        Stage.VAE_DECODE -> 0.93f
        Stage.UPSCALE -> if (kind == Kind.UPSCALE && total > 0) step / total.toFloat() else 0.97f
    }
}

/** A one-shot message for the snackbar: a string resource and its format args. */
data class UiMessage(@StringRes val res: Int, val args: List<Any> = emptyList(), val share: String? = null)

class AppViewModel(
    app: Application,
    val engine: ZitEngine,
) : AndroidViewModel(app) {
    /** Used by the default ViewModelProvider factory (needs exactly (Application)); 0.23.1 lacked it -> crash on launch. */
    constructor(app: Application) : this(app, com.zithexagon.service.EngineHost.engine(app))

    val prefs = Prefs(app)
    val modelRepo = ModelRepository(app)
    val loraRepo = LoraRepository(app)
    val galleryRepo = GalleryRepository(app)

    var params by mutableStateOf(prefs.lastParams); private set
    @set:JvmName("assign_randomSeed") var randomSeed by mutableStateOf(prefs.randomSeed); private set
    var upscaleOn by mutableStateOf(prefs.upscaleEnabled); private set
    @set:JvmName("assign_upscaleFactor") var upscaleFactor by mutableStateOf(prefs.upscaleFactor); private set
    var upscalerModel by mutableStateOf(prefs.upscalerModel); private set
    @set:JvmName("assign_copyToPhone") var copyToPhone by mutableStateOf(prefs.copyToPhoneGallery); private set
    @set:JvmName("assign_accent") var accent by mutableStateOf(prefs.accent); private set
    @set:JvmName("assign_showNegative") var showNegative by mutableStateOf(prefs.showNegative); private set

    var models by mutableStateOf<List<ZitModel>>(listOf(modelRepo.demo)); private set
    var selectedModelId by mutableStateOf(prefs.selectedModel ?: modelRepo.demo.id); private set
    val selectedModel: ZitModel get() = models.firstOrNull { it.id == selectedModelId } ?: modelRepo.demo

    var loras by mutableStateOf<List<LoraEntry>>(emptyList()); private set
    var gallery by mutableStateOf<List<GalleryItem>>(emptyList()); private set
    var lastResult by mutableStateOf<GalleryItem?>(null); private set

    // the job lives in the process-wide EngineHost (owned by InferenceService), so a re-created activity/ViewModel
    // sees a generation that kept running in the background, and the notification shows the same state
    private var jobState by mutableStateOf(EngineHost.job.value)
    var job: JobUi?
        get() = jobState
        private set(v) { jobState = v; EngineHost.job.value = v }
    /** Gallery item to open (from the "Imagen lista" notification). */
    var openItemRequest by mutableStateOf<String?>(null)
    var engineState by mutableStateOf<EngineState>(EngineState.Idle); private set
    var busyImport by mutableStateOf<String?>(null); private set

    private val _messages = MutableSharedFlow<UiMessage>(extraBufferCapacity = 8)
    val messages = _messages.asSharedFlow()
    private fun say(@StringRes res: Int, vararg args: Any) { _messages.tryEmit(UiMessage(res, args.toList())) }

    private var running: Job?
        get() = EngineHost.running
        set(v) { EngineHost.running = v }

    init {
        viewModelScope.launch { engine.state.collect { engineState = it } }
        viewModelScope.launch { EngineHost.job.collect { if (it !== jobState) jobState = it } }
        refreshAll()
    }

    fun refreshAll() {
        viewModelScope.launch {
            val (m, l, g) = runCatching { withContext(Dispatchers.IO) { Triple(modelRepo.list(), loraRepo.all(), galleryRepo.list()) } }
                .getOrElse { e -> CrashReporter.record(getApplication(), e, "no fatal, lectura de almacenamiento"); return@launch }
            models = m; loras = l; gallery = g
            if (lastResult == null) lastResult = g.firstOrNull()
            // Prefer NPH model when available and current selection is missing/demo
            if (m.any { it.id.startsWith("nph-") } && (selectedModelId == modelRepo.demo.id || m.none { it.id == selectedModelId })) {
                val nph = m.first { it.id.startsWith("nph-") }
                selectedModelId = nph.id; prefs.selectedModel = nph.id
            }
        }
    }

    // ---------- form ----------
    fun updateParams(transform: (GenParams) -> GenParams) { params = transform(params); prefs.lastParams = params }
    fun toggleKeyword(kw: String) = updateParams { it.copy(prompt = PromptKeywords.toggle(it.prompt, kw)) }
    fun setRandomSeed(v: Boolean) { randomSeed = v; prefs.randomSeed = v }
    fun newSeed() = updateParams { it.copy(seed = Random.nextLong(0, 4_294_967_295L)) }
    fun setUpscale(on: Boolean) { upscaleOn = on; prefs.upscaleEnabled = on }
    fun setUpscaleFactor(f: Int) { upscaleFactor = f; prefs.upscaleFactor = f }
    fun setUpscaler(id: String) { upscalerModel = id; prefs.upscalerModel = id }
    @set:JvmName("assign_execMode") var execMode by mutableStateOf(prefs.execMode); private set
    fun setExecMode(m: com.zithexagon.engine.ExecMode) { execMode = m; prefs.execMode = m }
    @set:JvmName("assign_debugDump") var debugDump by mutableStateOf(prefs.debugDump); private set
    fun setDebugDump(v: Boolean) { debugDump = v; prefs.debugDump = v }

    // 0.26 NPH Avanzado
    @set:JvmName("assign_nphProfile") var nphProfile by mutableStateOf(prefs.nphProfile); private set
    fun setNphProfile(v: Boolean) {
        if (nphProfile == v) return
        nphProfile = v; prefs.nphProfile = v
        // PROFILE is read into sticky opt_profile / DSP profiler at HTP session create.
        // Extremo may keep DiT resident — force full recreation so Liberar is not required by hand.
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { EngineHost.release(getApplication()).join() }
            ZitDiag.bc("NPH profile=$v → motor liberado (recrear sesión HTP)")
        }
    }
    @set:JvmName("assign_nphPageCache") var nphPageCacheWarmup by mutableStateOf(prefs.nphPageCacheWarmup); private set
    fun setNphPageCacheWarmup(v: Boolean) { nphPageCacheWarmup = v; prefs.nphPageCacheWarmup = v }
    @set:JvmName("assign_nphGraphCache") var nphGraphCache by mutableStateOf(prefs.nphGraphCache); private set
    fun setNphGraphCache(v: Boolean) { nphGraphCache = v; prefs.nphGraphCache = v }
    @set:JvmName("assign_nphOverlap") var nphOverlapDitTe by mutableStateOf(prefs.nphOverlapDitTe); private set
    fun setNphOverlapDitTe(v: Boolean) { nphOverlapDitTe = v; prefs.nphOverlapDitTe = v }
    // ---- 0.28 ----
    private fun releaseEngineAsync(why: String) = viewModelScope.launch(Dispatchers.IO) {
        runCatching { EngineHost.release(getApplication()).join() }
        ZitDiag.bc("NPH $why → motor liberado")
    }
    @set:JvmName("assign_nphFastRepack") var nphFastRepack by mutableStateOf(prefs.nphFastRepack); private set
    fun setNphFastRepack(v: Boolean) { nphFastRepack = v; prefs.nphFastRepack = v; releaseEngineAsync("fast-repack=$v") }
    @set:JvmName("assign_nphParallelSet") var nphParallelSet by mutableStateOf(prefs.nphParallelSet); private set
    fun setNphParallelSet(v: Boolean) { nphParallelSet = v; prefs.nphParallelSet = v; releaseEngineAsync("parallel-set=$v") }
    @set:JvmName("assign_nphFastRead") var nphFastRead by mutableStateOf(prefs.nphFastRead); private set
    fun setNphFastRead(v: Boolean) { nphFastRead = v; prefs.nphFastRead = v; releaseEngineAsync("fast-read=$v") }
    @set:JvmName("assign_nphRepackCache") var nphRepackCache by mutableStateOf(prefs.nphRepackCache); private set
    fun setNphRepackCache(v: Boolean) { nphRepackCache = v; prefs.nphRepackCache = v; releaseEngineAsync("repack-cache=$v") }
    @set:JvmName("assign_nphBufPool") var nphBufPool by mutableStateOf(prefs.nphBufPool); private set
    fun setNphBufPool(v: Boolean) { nphBufPool = v; prefs.nphBufPool = v; releaseEngineAsync("buf-pool=$v") }
    @set:JvmName("assign_nphPrefetchDit") var nphPrefetchDit by mutableStateOf(prefs.nphPrefetchDit); private set
    fun setNphPrefetchDit(v: Boolean) { nphPrefetchDit = v; prefs.nphPrefetchDit = v }
    @set:JvmName("assign_nphTeCache") var nphTeCache by mutableStateOf(prefs.nphTeCache); private set
    fun setNphTeCache(v: Boolean) { nphTeCache = v; prefs.nphTeCache = v }
    @set:JvmName("assign_nphTeCacheDisk") var nphTeCacheDisk by mutableStateOf(prefs.nphTeCacheDisk); private set
    fun setNphTeCacheDisk(v: Boolean) { nphTeCacheDisk = v; prefs.nphTeCacheDisk = v }
    @set:JvmName("assign_nphNpgVae") var nphNpgVae by mutableStateOf(prefs.nphNpgVae); private set
    @set:JvmName("assign_nphNpgBlocked") var nphNpgBlocked by mutableStateOf(prefs.nphNpgBlocked); private set
    fun setNphNpgVae(v: Boolean) {
        nphNpgVae = v; prefs.nphNpgVae = v
        // Re-enabling by hand clears a previous crash-guard block.
        if (v) { prefs.nphNpgBlocked = ""; nphNpgBlocked = "" }
    }
    fun resetNphToggles() {
        prefs.resetNphToggles()
        nphFastRepack = true; nphParallelSet = true; nphFastRead = true; nphRepackCache = true
        nphBufPool = true; nphPrefetchDit = true; nphTeCache = true; nphTeCacheDisk = true
        nphLoraCombined = true; nphGraphCache = true; nphLoraFuse = false
        releaseEngineAsync("mejoras restauradas")
    }
    fun refreshNphNpgBlocked() { nphNpgBlocked = prefs.nphNpgBlocked; nphNpgVae = prefs.nphNpgVae }
    @set:JvmName("assign_nphNpgSelfTest") var nphNpgSelfTest by mutableStateOf(prefs.nphNpgSelfTest); private set
    fun setNphNpgSelfTest(v: Boolean) { nphNpgSelfTest = v; prefs.nphNpgSelfTest = v }
    @set:JvmName("assign_nphLoraCombined") var nphLoraCombined by mutableStateOf(prefs.nphLoraCombined); private set
    fun setNphLoraCombined(v: Boolean) { nphLoraCombined = v; prefs.nphLoraCombined = v; releaseEngineAsync("lora-combined=$v") }
    @set:JvmName("assign_nphLoraFuse") var nphLoraFuse by mutableStateOf(prefs.nphLoraFuse); private set
    // Read by native before every DiT use (no engine release needed).
    fun setNphLoraFuse(v: Boolean) { nphLoraFuse = v; prefs.nphLoraFuse = v }
    @set:JvmName("assign_nphNpgMmF32") var nphNpgMmF32 by mutableStateOf(prefs.nphNpgMmF32); private set
    fun setNphNpgMmF32(v: Boolean) { nphNpgMmF32 = v; prefs.nphNpgMmF32 = v; releaseEngineAsync("npg-mm-f32=$v") }
    @set:JvmName("assign_nphNpgPartFallback") var nphNpgPartFallback by mutableStateOf(prefs.nphNpgPartFallback); private set
    fun setNphNpgPartFallback(v: Boolean) { nphNpgPartFallback = v; prefs.nphNpgPartFallback = v; releaseEngineAsync("npg-part-fb=$v") }
    @set:JvmName("assign_nphNpgMmNpu") var nphNpgMmNpu by mutableStateOf(prefs.nphNpgMmNpu); private set
    fun setNphNpgMmNpu(v: Boolean) { nphNpgMmNpu = v; prefs.nphNpgMmNpu = v; releaseEngineAsync("npg-mm-npu=$v") }
    @set:JvmName("assign_nphExpPrefetchVae") var nphExpPrefetchVae by mutableStateOf(prefs.nphExpPrefetchVae); private set
    fun setNphExpPrefetchVae(v: Boolean) { nphExpPrefetchVae = v; prefs.nphExpPrefetchVae = v }

    var nphCacheBytes by mutableStateOf(-1L); private set
    var nphCacheFreeBytes by mutableStateOf(-1L); private set
    var nphCacheBusy by mutableStateOf(false); private set
    fun refreshNphCacheSize() = viewModelScope.launch(Dispatchers.IO) {
        val dir = com.zithexagon.engine.NphEngine.cacheDir(getApplication())
        nphCacheBytes = (dir.listFiles()?.filter { it.name.endsWith(".nphc") }?.sumOf { it.length() }) ?: 0L
        nphCacheFreeBytes = runCatching { dir.usableSpace }.getOrDefault(-1L)
    }
    fun clearNphCache() = viewModelScope.launch(Dispatchers.IO) {
        nphCacheBusy = true
        try {
            // engine must not hold the cache file open → free the handle first
            runCatching { EngineHost.release(getApplication()).join() }
            val dir = com.zithexagon.engine.NphEngine.cacheDir(getApplication())
            val n = com.zithexagon.engine.NphEngine.clearCacheIfLoaded(dir)
                ?: (dir.listFiles()?.filter { it.name.endsWith(".nphc") }?.count { it.delete() } ?: 0)
            ZitDiag.bc("NPH caché repack borrada ($n archivos)")
        } finally {
            nphCacheBusy = false
            refreshNphCacheSize()
        }
    }
    @set:JvmName("assign_nphCpu") var nphCpuOffload by mutableStateOf(prefs.nphCpuOffload); private set
    fun setNphCpuOffload(v: Boolean) { nphCpuOffload = v; prefs.nphCpuOffload = v }
    @set:JvmName("assign_nphNhvx") var nphNhvx by mutableStateOf(prefs.nphNhvx); private set
    fun setNphNhvx(v: String) { nphNhvx = v; prefs.nphNhvx = v }
    @set:JvmName("assign_nphNhmx") var nphNhmx by mutableStateOf(prefs.nphNhmx); private set
    fun setNphNhmx(v: String) { nphNhmx = v; prefs.nphNhmx = v }
    @set:JvmName("assign_nphOpbatch") var nphOpbatch by mutableStateOf(prefs.nphOpbatch); private set
    fun setNphOpbatch(v: String) { nphOpbatch = v; prefs.nphOpbatch = v }
    @set:JvmName("assign_nphOpqueue") var nphOpqueue by mutableStateOf(prefs.nphOpqueue); private set
    fun setNphOpqueue(v: String) { nphOpqueue = v; prefs.nphOpqueue = v }
    @set:JvmName("assign_nphOpfusion") var nphOpfusion by mutableStateOf(prefs.nphOpfusion); private set
    fun setNphOpfusion(v: String) { nphOpfusion = v; prefs.nphOpfusion = v }
    @set:JvmName("assign_nphVmem") var nphVmem by mutableStateOf(prefs.nphVmem); private set
    fun setNphVmem(v: String) { nphVmem = v; prefs.nphVmem = v }
    @set:JvmName("assign_nphMbuf") var nphMbuf by mutableStateOf(prefs.nphMbuf); private set
    fun setNphMbuf(v: String) { nphMbuf = v; prefs.nphMbuf = v }
    fun setCopyToPhone(v: Boolean) { copyToPhone = v; prefs.copyToPhoneGallery = v }
    fun setAccent(id: String) { accent = id; prefs.accent = id }
    fun setShowNegative(v: Boolean) { showNegative = v; prefs.showNegative = v }

    fun selectModel(m: ZitModel) {
        selectedModelId = m.id; prefs.selectedModel = m.id
        val mf = m.manifest
        updateParams { it.copy(steps = mf.defaultSteps, cfg = mf.defaultCfg, width = mf.defaultWidth, height = mf.defaultHeight,
            sampler = com.zithexagon.engine.Sampler.byId(mf.defaultSampler)) }
    }

    // ---------- LoRAs on the form ----------
    fun isLoraActive(id: String) = params.loras.any { it.id == id }
    fun toggleLora(e: LoraEntry) = updateParams { p ->
        if (p.loras.any { it.id == e.id }) p.copy(loras = p.loras.filterNot { it.id == e.id })
        else {
            if (p.loras.size >= selectedModel.manifest.maxLoras) { say(R.string.msg_lora_limit, selectedModel.manifest.maxLoras); p }
            else p.copy(loras = p.loras + LoraUse(e.id, e.name, e.strength))
        }
    }
    fun setActiveLoraStrength(id: String, s: Float) = updateParams { p ->
        p.copy(loras = p.loras.map { if (it.id == id) it.copy(strength = com.zithexagon.data.LoraLimits.snap(s)) else it })
    }

    /** 0.25: one fixed output size per package (defaults.width/height = the compiled size, 1024×1024 for ReaZIT). */
    val fixedResolution: Resolution get() = selectedModel.manifest.let { Resolution(it.defaultWidth, it.defaultHeight) }

    // ---------- generation ----------
    val isBusy get() = job != null

    fun generate() {
        if (running?.isActive == true) return
        if (params.prompt.isBlank()) { say(R.string.msg_prompt_empty); return }
        val seed = if (randomSeed || params.seed < 0) Random.nextLong(0, 4_294_967_295L) else params.seed
        val model = selectedModel
        val fr = fixedResolution
        val p = params.copy(seed = seed, modelId = model.id, modelName = model.name, width = fr.width, height = fr.height,
            upscale = if (upscaleOn) upscaleFactor else 0,
            loras = params.loras.mapNotNull { u -> loraRepo.get(u.id)?.let { u.copy(name = it.name) } })
        updateParams { it.copy(seed = seed) }
        val bindings = if (!model.manifest.loraSupported) emptyList() else p.loras.mapNotNull { u -> loraRepo.get(u.id)?.let { LoraBinding(u.id, it.path, u.strength) } }
        ZitDiag.bc("UI generate model=${model.id} ${p.width}x${p.height} steps=${p.steps} loras=${bindings.size}")
        EngineHost.ensureService(getApplication())
        EngineHost.lastOutcome.value = null
        // runs in the engine host scope: keeps going with the app in the background or the screen off
        running = EngineHost.scope.launch {
            val started = System.currentTimeMillis()
            job = JobUi(JobUi.Kind.GENERATE, Stage.LOADING, startedAt = started)
            try {
                val watcher = launch {
                    engine.state.collect { st ->
                        if (st is EngineState.Loading) job = job?.copy(stage = Stage.LOADING, loadDetail = st.detail, loadProgress = st.progress)
                    }
                }
                // NPH: use resolve() (SAF uri + file). paths() alone returns null for SAF imports,
                // which previously made LoadRequest.nph=null and EngineRouter silently picked FakeEngine.
                val isNph = model.id.startsWith("nph-") ||
                    model.manifest.engineJson.contains("\"engine\":\"nph\"") ||
                    model.manifest.engineJson.contains("\"kind\":\"nph\"")
                val nphPaths = if (isNph) {
                    val resolved = NphModelStore.resolve(getApplication())
                        ?: error(
                            "No se pueden abrir los modelos NPH (DiT/TE/VAE). " +
                                "Si los importaste con el selector de archivos, vuelve a importarlos; " +
                                "si están en almacenamiento, concede el permiso. " +
                                "Ajustes → Diagnóstico NPH tiene más detalle."
                        )
                    // generate() re-resolves and holds PFDs; close this preflight handle
                    val paths = resolved.paths
                    NphModelStore.closeResolved(resolved)
                    paths
                } else null
                if (isNph && nphPaths == null) {
                    error("Modelos NPH incompletos. Importa DiT FP8 + TE Qwen3 + VAE en Modelos.")
                }
                engine.load(
                    LoadRequest(
                        model.id,
                        if (isNph) null else model.dir?.absolutePath,
                        bindings,
                        nphPaths,
                    )
                )
                watcher.cancel()
                val req = GenerationRequest(p.prompt, p.negativePrompt, p.steps, p.cfg, seed, p.width, p.height, p.sampler, bindings,
                    if (upscaleOn) UpscaleOptions(upscaleFactor, upscalerModel) else null, p.scheduler, execMode, debugDump)
                engine.generate(req).flowOn(Dispatchers.Default).collect { ev -> onEvent(ev, p, started) }
            } catch (t: Throwable) {
                if (t is kotlinx.coroutines.CancellationException) throw t
                ZitDiag.bc("UI generate error: $t")
                EngineHost.lastOutcome.value = EngineHost.Outcome.Failed(t.message ?: t.javaClass.simpleName)
                say(R.string.msg_error, t.message ?: t.javaClass.simpleName)
            } finally { job = null }
        }
    }

    private suspend fun onEvent(ev: GenerationEvent, p: GenParams, started: Long, upscaledFrom: GalleryItem? = null) {
        when (ev) {
            is GenerationEvent.StageChanged -> job = job?.copy(stage = ev.stage, step = 0, total = 0)
            is GenerationEvent.Step -> job = job?.copy(step = ev.step, total = ev.total, preview = ev.preview ?: job?.preview)
            is GenerationEvent.Completed -> {
                val total = System.currentTimeMillis() - started
                val perf = ev.timings.perf?.let {
                    it + "Total (desde que tocaste el botón): ${"%.1f".format(java.util.Locale.ROOT, total / 1000.0)} s\n"
                }
                perf?.let { ZitDiag.perf(it) }
                val item = withContext(Dispatchers.IO) {
                    galleryRepo.save(ev.image, p, total, upscaledFrom?.id, if (upscaledFrom != null) p.upscale else 0, copyToPhone, perf)
                }
                gallery = listOf(item) + gallery
                lastResult = item
                EngineHost.lastOutcome.value = EngineHost.Outcome.Done(item.id)
                if (upscaledFrom != null) say(R.string.msg_upscaled, item.width, item.height)
            }
            is GenerationEvent.Failed -> {
                val total = System.currentTimeMillis() - started
                val detail = buildString {
                    append("Error: ").append(ev.message).append("\n\n")
                    ev.perf?.takeIf { it.isNotBlank() }?.let { append(it) }
                    append("Total hasta el fallo: ${"%.1f".format(java.util.Locale.ROOT, total / 1000.0)} s\n")
                    val app = getApplication<android.app.Application>()
                    val ver = runCatching { app.packageManager.getPackageInfo(app.packageName, 0).versionName }.getOrNull() ?: "?"
                    append("App $ver · ${android.os.Build.MODEL} · Android ${android.os.Build.VERSION.RELEASE}\n")
                    val tail = withContext(Dispatchers.IO) { ZitDiag.stageTail(60) }
                    if (tail.isNotBlank()) append("\n--- log del motor (últimas líneas) ---\n").append(tail).append("\n")
                }
                withContext(Dispatchers.IO) { ZitDiag.perf("FALLÓ\n$detail") }
                EngineHost.lastOutcome.value = EngineHost.Outcome.Failed(ev.message)
                _messages.tryEmit(UiMessage(R.string.msg_error, listOf(ev.message), share = detail))
            }
            GenerationEvent.Cancelled -> { EngineHost.lastOutcome.value = EngineHost.Outcome.Cancelled(); say(R.string.msg_cancelled) }
        }
    }

    fun cancel() { EngineHost.cancel() }

    /** "Liberar memoria": cancel a running job first, then unload NPH context; stop the service. */
    fun releaseModel() {
        viewModelScope.launch {
            EngineHost.release(getApplication()).join()
            // Explicit cancel: some OEMs leave the FGS notification after stopService alone.
            com.zithexagon.service.InferenceService.clearNotification(getApplication())
            say(R.string.msg_released)
        }
    }
    val canRelease: Boolean get() = job != null || engineState.let { it is EngineState.Ready || it is EngineState.Loading || it is EngineState.Busy || it is EngineState.Error }

    fun upscale(item: GalleryItem, factor: Int) {
        if (running?.isActive == true) { say(R.string.msg_busy); return }
        EngineHost.ensureService(getApplication())
        EngineHost.lastOutcome.value = null
        running = EngineHost.scope.launch {
            val started = System.currentTimeMillis()
            job = JobUi(JobUi.Kind.UPSCALE, Stage.UPSCALE, startedAt = started)
            try {
                val src = withContext(Dispatchers.IO) { BitmapFactory.decodeFile(item.file.path) } ?: error("imagen no legible")
                engine.upscale(UpscaleRequest(src, UpscaleOptions(factor, upscalerModel))).flowOn(Dispatchers.Default)
                    .collect { onEvent(it, item.params.copy(upscale = factor), started, upscaledFrom = item) }
            } catch (t: Throwable) {
                if (t is kotlinx.coroutines.CancellationException) throw t
                say(R.string.msg_error, t.message ?: "")
            } finally { job = null }
        }
    }

    // ---------- diagnostics: test block ----------
    var benchProgress by mutableStateOf<String?>(null); private set
    var benchReport by mutableStateOf<String?>(null)

    /** Diagnóstico NPH: info del motor + MemAvailable (ya no hay bloques QNN). */
    fun runTestBlocks() {
        if (running?.isActive == true || benchProgress != null) { say(R.string.msg_busy); return }
        benchProgress = "Consultando NPH…"
        viewModelScope.launch {
            val text = withContext(Dispatchers.IO) {
                runCatching {
                    buildString {
                        append("ZIT-HEXAGON ${com.zithexagon.BuildConfigInfo.versionName(getApplication())} — diagnóstico NPH\n")
                        append("${android.os.Build.MODEL} · Android ${android.os.Build.VERSION.RELEASE}\n\n")
                        val info = runCatching {
                            val app = getApplication<Application>()
                            val dir = java.io.File(app.filesDir, "htp").apply { mkdirs() }
                            app.assets.list("htp")?.forEach { name ->
                                val out = java.io.File(dir, name)
                                if (!out.exists()) app.assets.open("htp/$name").use { inp -> out.outputStream().use { it.write(inp.readBytes()) } }
                            }
                            android.system.Os.setenv("ADSP_LIBRARY_PATH", dir.absolutePath, true)
                            System.loadLibrary("nph")
                            com.zithexagon.engine.NphNative.info()
                        }.getOrElse { "No se pudo cargar libnph: $it" }
                        append(info).append('\n')
                        val mb = runCatching { com.zithexagon.engine.NphNative.memAvailableMb() }.getOrDefault(-1)
                        append("\nMemAvailable: $mb MB\n")
                        append("allReady: ${NphModelStore.allReady(getApplication())}\n")
                        for (r in Role.entries) append(NphModelStore.statusLine(getApplication(), r)).append('\n')
                        val filePaths = NphModelStore.paths(getApplication())
                        append("paths() file-only: ${if (filePaths != null) "OK" else "null (¿SAF/uri?)"}\n")
                        val resolved = NphModelStore.resolve(getApplication())
                        if (resolved != null) {
                            val ld = resolved.paths.labels[Role.DIT] ?: java.io.File(resolved.paths.dit).name
                            val lt = resolved.paths.labels[Role.TE] ?: java.io.File(resolved.paths.te).name
                            val lv = resolved.paths.labels[Role.VAE] ?: java.io.File(resolved.paths.vae).name
                            append("resolve(): OK\n  DiT: $ld\n  TE: $lt\n  VAE: $lv\n")
                            NphModelStore.closeResolved(resolved)
                        } else {
                            append("resolve(): FALLÓ — no se pueden abrir DiT+TE+VAE\n")
                        }
                        val app = getApplication<Application>()
                        val htpDir = java.io.File(app.filesDir, "htp")
                        val skels = htpDir.list()?.filter { it.endsWith(".so") }?.sorted().orEmpty()
                        append("\nSkeleton HTP (${htpDir.absolutePath}):\n")
                        if (skels.isEmpty()) append("  (ninguno extraído aún — se copian al cargar libnph)\n")
                        else skels.forEach { append("  $it\n") }
                        val nativeDir = app.applicationInfo.nativeLibraryDir
                        append("nativeLibraryDir: $nativeDir\n")
                        val libnph = java.io.File(nativeDir, "libnph.so")
                        append("libnph.so: ${if (libnph.isFile) "OK ${libnph.length()} B" else "AUSENTE"}\n")
                        append("\nNota: NPH ejecuta el DiT completo en HTP (no hay bloques QNN).\n")
                    }
                }.getOrElse { "Error: ${it.message}" }
            }
            com.zithexagon.ZitDiag.perf(text)
            benchReport = text; benchProgress = null
        }
    }

    // ---------- gallery ----------
    fun deleteItem(item: GalleryItem) {
        viewModelScope.launch(Dispatchers.IO) { galleryRepo.delete(item) }
        gallery = gallery.filterNot { it.id == item.id }
        if (lastResult?.id == item.id) lastResult = gallery.firstOrNull()
        say(R.string.msg_deleted)
    }

    fun reuseParams(item: GalleryItem) {
        val p = item.params
        updateParams { p.copy(loras = p.loras.filter { loraRepo.get(it.id) != null }) }
        setRandomSeed(false)
        setUpscale(p.upscale > 0); if (p.upscale > 0) setUpscaleFactor(p.upscale)
        models.firstOrNull { it.id == p.modelId }?.let { selectedModelId = it.id; prefs.selectedModel = it.id }
        say(R.string.msg_params_reused)
    }

    fun saveToPhone(item: GalleryItem) {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { galleryRepo.copyToMediaStore(item.file) }
                .onSuccess { say(R.string.msg_saved_phone) }.onFailure { say(R.string.msg_error, it.message ?: "") }
        }
    }

    // ---------- LoRA library ----------
    private fun loraOp(@StringRes busy: Int, block: suspend () -> Unit) {
        viewModelScope.launch {
            busyImport = getApplication<Application>().getString(busy)
            try { withContext(Dispatchers.IO) { block() } }
            catch (t: Throwable) { say(R.string.msg_error, t.message ?: t.javaClass.simpleName) }
            finally { busyImport = null; loras = loraRepo.all() }
        }
    }

    fun importLoras(uris: List<Uri>) = loraOp(R.string.busy_importing) {
        var n = 0; uris.forEach { runCatching { loraRepo.importUri(it) }.onSuccess { n++ }.onFailure { e -> say(R.string.msg_error, e.message ?: "") } }
        say(R.string.msg_loras_imported, n)
    }
    fun importLoraFolder(tree: Uri) = loraOp(R.string.busy_importing) { say(R.string.msg_loras_imported, loraRepo.importTree(tree)) }
    fun scanLoraFolders() = loraOp(R.string.busy_scanning) { say(R.string.msg_loras_found, loraRepo.scanFolders()) }
    fun setLoraStrength(id: String, s: Float) { loraRepo.setStrength(id, com.zithexagon.data.LoraLimits.snap(s)); loras = loraRepo.all() }
    fun setLoraTriggers(id: String, words: List<String>) { loraRepo.setTriggerWords(id, words); loras = loraRepo.all() }
    fun renameLora(id: String, name: String) {
        loraRepo.rename(id, name); loras = loraRepo.all()
        updateParams { p -> p.copy(loras = p.loras.map { if (it.id == id) it.copy(name = loraRepo.get(id)?.name ?: it.name) else it }) }
    }
    fun removeLora(id: String) { loraRepo.remove(id); loras = loraRepo.all(); updateParams { p -> p.copy(loras = p.loras.filterNot { it.id == id }) } }
    fun setLoraPreviewFromUri(id: String, uri: Uri) = loraOp(R.string.busy_saving) { loraRepo.setPreviewFromUri(id, uri); say(R.string.msg_preview_set) }
    fun setLoraPreviewFromGallery(id: String, item: GalleryItem) = loraOp(R.string.busy_saving) { loraRepo.setPreviewFromFile(id, item.file); say(R.string.msg_preview_set) }
    fun clearLoraPreview(id: String) { loraRepo.clearPreview(id); loras = loraRepo.all() }

    // ---------- models ----------
    private fun modelOp(block: suspend () -> ZitModel) {
        viewModelScope.launch {
            busyImport = getApplication<Application>().getString(R.string.busy_importing)
            try {
                val m = withContext(Dispatchers.IO) { block() }
                say(R.string.msg_model_imported, m.name)
            } catch (t: Throwable) { say(R.string.msg_error, t.message ?: t.javaClass.simpleName) }
            finally { busyImport = null; models = withContext(Dispatchers.IO) { modelRepo.list() } }
        }
    }

    fun importNphUri(uri: Uri, preferCopy: Boolean = false, expectRole: Role? = null) = modelOp {
        val app = getApplication<Application>()
        // preferCopy=true → force copy (allowCopy only); else try in-place first via importUri
        val sel = if (preferCopy) {
            val name = NphModelStore.uriName(app, uri)
            val size = NphModelStore.uriSize(app, uri)
            val probe = NphModelStore.probeUri(app, uri, size)
            if (!probe.ok) error(probe.error ?: "archivo no reconocido")
            if (expectRole != null && probe.role != expectRole)
                error("esperaba ${expectRole.label}, detecté ${probe.role?.label ?: "?"}")
            val copied = NphModelStore.copyIn(
                app, uri, name, probe,
                onProgress = { done, total ->
                    busyImport = "Copiando: ${done / 1_000_000} / ${total / 1_000_000} MB"
                },
            )
            NphModelStore.save(app, copied)
            copied
        } else {
            NphModelStore.importUri(
                app, uri,
                expectedRole = expectRole,
                allowCopy = true,
                onProgress = { done, total ->
                    busyImport = "Copiando: ${done / 1_000_000} / ${total / 1_000_000} MB"
                },
            )
        }
        val m = NphModelStore.asZitModel(app)
        if (m != null) {
            withContext(Dispatchers.Main) { selectModel(m) }
            m
        } else {
            // Triple incomplete — keep demo selected; toast still shows imported file via sel.name
            ZitModel(
                manifest = modelRepo.demo.manifest.copy(name = "Importado: ${sel.name}"),
                dir = null,
                source = ModelSource.DEMO,
                sizeBytes = sel.size,
                missing = emptyList(),
            )
        }
    }

    fun scanNphDownloads() = modelOp {
        val app = getApplication<Application>()
        NphModelStore.autoselectMissing(app)
        refreshModels()
        NphModelStore.asZitModel(app) ?: error("No se encontró DiT/TE/VAE en Descargas")
    }

    fun nphStatusLines(): List<String> {
        val app = getApplication<Application>()
        return Role.entries.map { NphModelStore.statusLine(app, it) }
    }

    fun importModelZip(uri: Uri) = modelOp { modelRepo.importZip(uri) }
    fun importModelFolder(uri: Uri) = modelOp { modelRepo.importTree(uri) }
    fun deleteModel(m: ZitModel) {
        viewModelScope.launch {
            if (engineState.let { it is EngineState.Ready && it.modelId == m.id }) EngineHost.release(getApplication()).join()
            withContext(Dispatchers.IO) { modelRepo.delete(m) }
            models = withContext(Dispatchers.IO) { modelRepo.list() }
            if (selectedModelId == m.id) selectModel(modelRepo.demo)
            say(R.string.msg_deleted)
        }
    }
    fun refreshModels() { viewModelScope.launch { models = withContext(Dispatchers.IO) { modelRepo.list() } } }

    val upscalerName get() = UPSCALERS.firstOrNull { it.id == upscalerModel }?.name ?: upscalerModel
}
