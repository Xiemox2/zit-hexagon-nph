package com.zithexagon.engine

import android.graphics.Bitmap
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow

/**
 * The inference backend behind the UI. 0.26: [NphEngine] (sd.cpp Hexagon, branded NPH) is the
 * default diffusion engine; [FakeEngine] remains for the demo model.
 *
 * Flow of a session:
 *  1. [load] a model package (+ the swap-type LoRAs, baked in at load time) -> [state] goes
 *     Loading(progress) -> Ready.
 *  2. [generate] emits [GenerationEvent]s: Stage changes, one Step per denoising step (with an
 *     optional low-res preview), then exactly one terminal Completed / Failed / Cancelled.
 *  3. [upscale] works the same way for an existing picture.
 *  4. [cancel] stops the running job at the next step boundary.
 */
interface ZitEngine {
    val info: EngineInfo
    val state: StateFlow<EngineState>

    /** Loads (or re-swaps) the model. Calling it with the same model and LoRAs is a no-op. */
    suspend fun load(request: LoadRequest)

    suspend fun unload()

    /** Cold flow: collecting it runs one generation. Requires state == Ready. */
    fun generate(request: GenerationRequest): Flow<GenerationEvent>

    /** Cold flow: upscales [UpscaleRequest.source] with the upscaler model. */
    fun upscale(request: UpscaleRequest): Flow<GenerationEvent>

    fun cancel()
}

data class EngineInfo(
    val name: String,
    /** e.g. "Simulado (sin NPU)" or "QNN · HTP v81". */
    val backend: String,
    val isFake: Boolean,
)

sealed interface EngineState {
    data object Idle : EngineState
    data class Loading(val modelId: String, val progress: Float, val detail: String) : EngineState
    data class Ready(val modelId: String, val loras: List<LoraBinding>) : EngineState
    data class Busy(val modelId: String, val kind: JobKind) : EngineState
    data class Error(val message: String) : EngineState
}

enum class JobKind { GENERATE, UPSCALE }

/** One swap-type LoRA as the engine sees it: the file and the strength baked into the weights. */
data class LoraBinding(val id: String, val path: String, val strength: Float)  // strength in [-3, 3] (slider LoRAs go negative)

data class LoadRequest(
    val modelId: String,
    /** Folder of the model package (contains zit.json); null for the built-in demo model. */
    val modelDir: String?,
    val loras: List<LoraBinding> = emptyList(),
    /** 0.26 NPH: paths to DiT FP8 + TE GGUF + VAE; when set, EngineRouter uses NphEngine. */
    val nph: com.zithexagon.data.NphModelStore.Paths? = null,
)

/** 0.26 NPH: EXTREMO keeps DiT+VAE resident between gens (TE=disk); SEGURO frees everything after each image. */
enum class ExecMode(val id: Int, val label: String) { EXTREMO(0, "Extremo"), SEGURO(1, "Seguro") }

data class GenerationRequest(
    val prompt: String,
    val negativePrompt: String,
    val steps: Int,
    val cfg: Float,
    val seed: Long,
    val width: Int,
    val height: Int,
    val sampler: Sampler,
    val loras: List<LoraBinding>,
    /** Optional upscale run right after the generation (same flow, extra UPSCALE stage). */
    val upscale: UpscaleOptions? = null,
    val scheduler: Scheduler = Scheduler.SIMPLE,
    val mode: ExecMode = ExecMode.EXTREMO,
    /** 0.25.2 diagnostics: dump block 0 (step 1) to Download/ZITHEXAGON/debug; A/B block 0 from test/zit-test-c0-noudma.bin if present. */
    val debugDump: Boolean = false,
)

data class UpscaleOptions(val scale: Int, val modelId: String)

data class UpscaleRequest(val source: Bitmap, val options: UpscaleOptions)

enum class Stage { LOADING, TEXT_ENCODER, DENOISE, VAE_DECODE, UPSCALE }

/**
 * The 5 samplers most used with Z-Image Turbo in ComfyUI (ids = ComfyUI KSampler names):
 * euler (most popular, default), euler_ancestral and dpmpp_sde (top picks in community sampler
 * comparisons for ZIT), dpmpp_2m (classic deterministic multistep), res_multistep (ComfyUI's
 * official Z-Image Turbo template).
 */
enum class Sampler(val id: String, val label: String) {
    EULER("euler", "Euler"),
    EULER_A("euler_ancestral", "Euler Ancestral"),
    DPMPP_SDE("dpmpp_sde", "DPM++ SDE"),
    DPMPP_2M("dpmpp_2m", "DPM++ 2M"),
    RES_MULTISTEP("res_multistep", "Res Multistep");

    companion object {
        private val legacy = mapOf("euler_a" to EULER_A, "dpmpp_2m_sde" to DPMPP_SDE)
        fun byId(id: String?) = entries.firstOrNull { it.id == id } ?: legacy[id] ?: EULER
    }
}

/** Sigma schedule (ComfyUI scheduler names). simple = official ZIT default; beta / ddim_uniform pair well with SDE/ancestral. */
enum class Scheduler(val id: String, val label: String) {
    SIMPLE("simple", "Simple"),
    BETA("beta", "Beta"),
    DDIM_UNIFORM("ddim_uniform", "DDIM uniform");

    companion object {
        fun byId(id: String?) = entries.firstOrNull { it.id == id } ?: SIMPLE
    }
}

sealed interface GenerationEvent {
    data class StageChanged(val stage: Stage) : GenerationEvent
    data class Step(val step: Int, val total: Int, val preview: Bitmap?) : GenerationEvent
    data class Completed(val image: Bitmap, val timings: Timings) : GenerationEvent
    data class Failed(val message: String, val perf: String? = null) : GenerationEvent
    data object Cancelled : GenerationEvent
}

data class Timings(
    val textEncoderMs: Long = 0,
    val denoiseMs: Long = 0,
    val decodeMs: Long = 0,
    val upscaleMs: Long = 0,
    /** Human-readable per-stage breakdown (Spanish), null when the engine did not provide one. */
    val perf: String? = null,
) {
    val totalMs get() = textEncoderMs + denoiseMs + decodeMs + upscaleMs
}
