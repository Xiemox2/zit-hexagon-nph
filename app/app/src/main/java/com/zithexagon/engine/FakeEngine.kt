package com.zithexagon.engine

import android.graphics.Bitmap
import android.graphics.BlurMaskFilter
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.RadialGradient
import android.graphics.Shader
import android.graphics.Typeface
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

/**
 * Stand-in for the QNN engine: draws a deterministic abstract picture from the seed and the
 * prompt, with simulated timings and per-step previews, so every screen works end to end.
 * [speed] scales all delays (0 = instant, used by tests).
 */
class FakeEngine(private val speed: Float = 1f) : ZitEngine {
    override val info = EngineInfo("FakeEngine", "Simulado (sin NPU)", isFake = true)

    private val _state = MutableStateFlow<EngineState>(EngineState.Idle)
    override val state: StateFlow<EngineState> = _state.asStateFlow()

    @Volatile private var cancelled = false

    private suspend fun pause(ms: Long) { if (speed > 0f) delay((ms * speed).toLong()) }

    override suspend fun load(request: LoadRequest) {
        val cur = _state.value
        if (cur is EngineState.Ready && cur.modelId == request.modelId && cur.loras == request.loras) return
        val parts = listOf("Codificador de texto", "Transformer (DiT)", "VAE") +
            request.loras.map { "LoRA ${it.id}" }
        parts.forEachIndexed { i, p ->
            _state.value = EngineState.Loading(request.modelId, i / parts.size.toFloat(), p)
            pause(250)
        }
        _state.value = EngineState.Ready(request.modelId, request.loras)
    }

    override suspend fun unload() { _state.value = EngineState.Idle }

    override fun cancel() { cancelled = true }

    override fun generate(request: GenerationRequest): Flow<GenerationEvent> = flow {
        val ready = _state.value as? EngineState.Ready
        if (ready == null) { emit(GenerationEvent.Failed("No hay ningún modelo cargado")); return@flow }
        cancelled = false
        _state.value = EngineState.Busy(ready.modelId, JobKind.GENERATE)
        try {
            val t0 = System.currentTimeMillis()
            emit(GenerationEvent.StageChanged(Stage.TEXT_ENCODER)); pause(350)
            val t1 = System.currentTimeMillis()
            emit(GenerationEvent.StageChanged(Stage.DENOISE))
            val steps = max(1, request.steps)
            val pw = 128; val ph = max(16, 128 * request.height / request.width)
            for (s in 1..steps) {
                if (cancelled) { emit(GenerationEvent.Cancelled); return@flow }
                pause(320)
                emit(GenerationEvent.Step(s, steps, render(request, pw, ph, noise = 1f - s / steps.toFloat())))
            }
            val t2 = System.currentTimeMillis()
            emit(GenerationEvent.StageChanged(Stage.VAE_DECODE)); pause(400)
            var img = render(request, request.width, request.height, noise = 0f)
            val t3 = System.currentTimeMillis()
            var upMs = 0L
            request.upscale?.let { up ->
                emit(GenerationEvent.StageChanged(Stage.UPSCALE)); pause(600L * up.scale / 2)
                img = Bitmap.createScaledBitmap(img, img.width * up.scale, img.height * up.scale, true)
                upMs = System.currentTimeMillis() - t3
            }
            emit(GenerationEvent.Completed(img, Timings(t1 - t0, t2 - t1, t3 - t2, upMs)))
        } finally {
            _state.value = EngineState.Ready(ready.modelId, ready.loras)
        }
    }

    override fun upscale(request: UpscaleRequest): Flow<GenerationEvent> = flow {
        val prev = _state.value
        cancelled = false
        _state.value = EngineState.Busy((prev as? EngineState.Ready)?.modelId ?: "", JobKind.UPSCALE)
        try {
            val t0 = System.currentTimeMillis()
            emit(GenerationEvent.StageChanged(Stage.UPSCALE))
            val tiles = 4 * request.options.scale
            for (i in 1..tiles) {
                if (cancelled) { emit(GenerationEvent.Cancelled); return@flow }
                pause(120); emit(GenerationEvent.Step(i, tiles, null))
            }
            val s = request.options.scale
            val out = Bitmap.createScaledBitmap(request.source, request.source.width * s, request.source.height * s, true)
            emit(GenerationEvent.Completed(out, Timings(upscaleMs = System.currentTimeMillis() - t0)))
        } finally {
            _state.value = if (prev is EngineState.Busy || prev is EngineState.Loading) EngineState.Idle else prev
        }
    }

    companion object {
        /** Deterministic placeholder art: wine/black palette mixed with hues from the prompt. */
        fun render(r: GenerationRequest, w: Int, h: Int, noise: Float): Bitmap {
            val rnd = Random(r.seed xor r.prompt.hashCode().toLong())
            val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val c = Canvas(bmp)
            val hue0 = (r.prompt.hashCode().toLong() and 0xffff).toFloat() / 0xffff * 360f
            fun hsv(hh: Float, s: Float, v: Float) = Color.HSVToColor(floatArrayOf(((hh % 360) + 360) % 360, s, v))
            val p = Paint(Paint.ANTI_ALIAS_FLAG)
            p.shader = LinearGradient(0f, 0f, w.toFloat(), h.toFloat(),
                hsv(hue0, 0.65f, 0.18f), hsv(hue0 + 40, 0.75f, 0.42f), Shader.TileMode.CLAMP)
            c.drawRect(0f, 0f, w.toFloat(), h.toFloat(), p)
            val m = min(w, h).toFloat()
            p.maskFilter = if (m > 64) BlurMaskFilter(m / 30f, BlurMaskFilter.Blur.NORMAL) else null
            repeat(9) { i ->
                val cx = rnd.nextFloat() * w; val cy = rnd.nextFloat() * h
                val rad = m * (0.12f + rnd.nextFloat() * 0.38f)
                val col = if (i % 3 == 0) Color.rgb(142, 42, 107) else hsv(hue0 + rnd.nextInt(-70, 70), 0.55f + rnd.nextFloat() * 0.4f, 0.5f + rnd.nextFloat() * 0.5f)
                p.shader = RadialGradient(cx, cy, rad, col, Color.TRANSPARENT, Shader.TileMode.CLAMP)
                c.drawCircle(cx, cy, rad, p)
            }
            p.maskFilter = null; p.shader = null
            // Horizon-like band so pictures read as "scenes".
            p.shader = LinearGradient(0f, h * 0.62f, 0f, h.toFloat(), Color.TRANSPARENT, Color.argb(200, 0, 0, 0), Shader.TileMode.CLAMP)
            c.drawRect(0f, h * 0.62f, w.toFloat(), h.toFloat(), p); p.shader = null
            if (noise > 0f) {
                val n = Random(r.seed)
                val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
                for (i in px.indices) {
                    val g = n.nextInt(256)
                    val a = noise
                    val cr = (Color.red(px[i]) * (1 - a) + g * a).toInt()
                    val cg = (Color.green(px[i]) * (1 - a) + g * a).toInt()
                    val cb = (Color.blue(px[i]) * (1 - a) + g * a).toInt()
                    px[i] = Color.rgb(cr, cg, cb)
                }
                bmp.setPixels(px, 0, w, 0, 0, w, h)
            } else if (w >= 256) {
                p.color = Color.argb(150, 255, 255, 255); p.typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
                p.textSize = m / 26f
                val label = "ZIT-HEXAGON · demo · seed ${r.seed}"
                c.drawText(label, m / 26f, m / 26f * 2f, p)  // top-left, clear of UI captions
            }
            return bmp
        }
    }
}
