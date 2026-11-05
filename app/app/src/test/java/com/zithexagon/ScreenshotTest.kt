package com.zithexagon

import android.app.Application
import android.os.Looper
import org.robolectric.Shadows
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onRoot
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performSemanticsAction
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.onAllNodesWithText
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.github.takahirom.roborazzi.captureRoboImage
import com.github.takahirom.roborazzi.captureScreenRoboImage
import com.zithexagon.data.GalleryItem
import com.zithexagon.data.GalleryRepository
import com.zithexagon.data.GenParams
import com.zithexagon.data.LoraRepository
import com.zithexagon.data.LoraUse
import com.zithexagon.data.Prefs
import com.zithexagon.data.Storage
import com.zithexagon.engine.*
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import org.json.JSONObject
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import java.io.File

/**
 * Renders every main screen with demo data (Robolectric native graphics + Roborazzi).
 * Run: ./gradlew :app:recordRoborazziDebug  -> out/screenshots (PNG)
 * Device: Nubia Z80 Ultra-like portrait (412×912 dp, xxhdpi).
 */
@RunWith(AndroidJUnit4::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [35], qualifiers = "w412dp-h912dp-xxhdpi")
class ScreenshotTest {
    @get:Rule val compose = createComposeRule()
    private lateinit var app: Application
    private val out = "../out/screenshots"
    private var items: List<GalleryItem> = emptyList()

    private val prompts = listOf(
        "cinematic portrait of a woman in a wine-red velvet dress, soft rim light, 85mm",
        "neon city street at night after rain, reflections, ultra detailed",
        "a cozy cabin in a snowy forest, warm windows, dusk",
        "macro photo of a purple orchid with dew drops",
        "ancient temple in the mountains, mist, golden hour",
        "cyberpunk samurai, dramatic lighting, film grain",
        "still life of grapes and wine glass, chiaroscuro",
        "astronaut floating above a violet nebula",
        "minimalist architecture, white concrete, deep shadows",
    )

    /** Minimal safetensors: 8-byte header length + JSON header with __metadata__. */
    private fun fakeLora(name: String, triggers: String): File {
        val header = JSONObject().put("__metadata__", JSONObject().put("ss_trigger_words", triggers)
            .put("modelspec.architecture", "z-image-turbo/lora")).toString().toByteArray()
        val f = File(Storage.loras(app), name)
        f.outputStream().use { o ->
            val n = header.size.toLong()
            o.write(ByteArray(8) { i -> ((n shr (8 * i)) and 0xff).toByte() }); o.write(header)
            o.write(ByteArray(1 shl 16))
        }
        return f
    }

    @Before
    fun seed() {
        app = ApplicationProvider.getApplicationContext()
        val gallery = GalleryRepository(app)
        items = prompts.mapIndexed { i, pr ->
            val (w, h) = listOf(768 to 768, 640 to 896, 896 to 640)[i % 3]
            val p = GenParams(prompt = pr, steps = 8, cfg = 1f, seed = 1000L + i * 7919, width = w, height = h,
                sampler = Sampler.EULER, modelId = "demo-reazit", modelName = "ReaZIT v1.0 (demo)",
                loras = if (i % 2 == 0) listOf(LoraUse("velvet_style_zit.safetensors", "Velvet Style", 0.8f)) else emptyList())
            val req = GenerationRequest(pr, "", 8, 1f, p.seed, w, h, Sampler.EULER, emptyList())
            gallery.save(FakeEngine.render(req, w, h, 0f), p, 9_400L + i * 310)
        }.reversed()
        val loras = LoraRepository(app)
        val l1 = loras.register(fakeLora("velvet_style_zit.safetensors", "velvet style, wine red, soft rim light"), "app")
        loras.register(fakeLora("neon_noir_zit.safetensors", "neon noir, rain reflections, cinematic"), "app")
        val l3 = loras.register(fakeLora("rea_character_v2.safetensors", "rea_girl, long dark hair, freckles"), "app")
        loras.register(fakeLora("film_grain_zit.safetensors", "film grain, kodak portra"), "app")
        val l5 = loras.register(fakeLora("age_slider_zit.safetensors", "age slider"), "app")
        loras.rename(l5.id, "Age Slider"); loras.setStrength(l5.id, -1.5f)
        loras.rename(l1.id, "Velvet Style"); loras.rename(l3.id, "Rea (personaje) v2")
        loras.setStrength(l3.id, 0.75f)
        loras.setPreviewFromFile(l1.id, items[0].file)
        loras.setPreviewFromFile(l3.id, items[3].file)
        loras.setPreviewFromFile("neon_noir_zit.safetensors", items[1].file)
        Prefs(app).lastParams = GenParams(
            prompt = "rea_girl, long dark hair, cinematic portrait in a wine-red velvet dress, soft rim light, 85mm",
            negativePrompt = "blurry, lowres", steps = 8, cfg = 1f, seed = 424242, width = 1024, height = 1024,
            loras = listOf(LoraUse(l3.id, "Rea (personaje) v2", 0.75f), LoraUse(l1.id, "Velvet Style", 0.6f),
                LoraUse("neon_noir_zit.safetensors", "neon_noir_zit", 1f), LoraUse(l5.id, "Age Slider", -1.5f)),
        )
        Prefs(app).randomSeed = false
    }

    private fun vm(engine: ZitEngine = FakeEngine(speed = 0f)): AppViewModel {
        val vm = AppViewModel(app, engine)
        pump { vm.gallery.isNotEmpty() && vm.loras.isNotEmpty() }
        return vm
    }

    /** Runs the main looper until [cond] holds (coroutines hop IO -> Main). */
    private fun pump(timeoutMs: Long = 20_000, cond: () -> Boolean) {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (!cond()) {
            check(System.currentTimeMillis() < deadline) { "timeout waiting for condition" }
            Shadows.shadowOf(Looper.getMainLooper()).idle(); Thread.sleep(20)
        }
    }

    private fun shot(name: String) {
        compose.waitForIdle()
        compose.onRoot().captureRoboImage("$out/$name.png")
    }

    @Test fun s01_generate() { val v = vm(); compose.setContent { ZitApp(v) }; shot("01_generar") }

    @Test @Config(qualifiers = "w412dp-h3000dp-xxhdpi")
    fun s02_generate_full() {
        val v = vm(); v.setUpscale(true); v.setShowNegative(true)
        compose.setContent { ZitApp(v) }; shot("02_generar_completo")
    }

    @Test fun s03_generating() {
        val v = vm(SuspendedEngine())
        compose.setContent { ZitApp(v) }
        v.generate()
        pump { v.job?.total == 8 }
        shot("03_generando")
    }

    @Test fun s04_lora_picker() {
        val v = vm(); compose.setContent { ZitApp(v) }
        compose.onNodeWithText("Añadir").performClick()
        compose.mainClock.advanceTimeBy(1000)
        shot("04_selector_loras")
    }

    @Test fun s12_many_loras_warning() {
        val v = vm(); compose.setContent { ZitApp(v) }
        compose.onNodeWithText("Más de 3 LoRAs", substring = true).performScrollTo()
        shot("12_generar_aviso_loras")
    }

    @Test fun s13_upscaler_picker() {
        val v = vm(); v.setUpscale(true); compose.setContent { ZitApp(v) }
        compose.onNodeWithText("4x-AnimeSharp").performScrollTo()
        shot("13_escalador")
    }

    @Test fun s14_strength_dialog() {
        val v = vm(); compose.setContent { ZitApp(v, Tab.LORAS, Overlay.LoraEdit("age_slider_zit.safetensors")) }
        val pill = androidx.compose.ui.test.SemanticsMatcher("strength pill") {
            it.config.getOrElseNullable(androidx.compose.ui.semantics.SemanticsProperties.Text) { null }?.map { t -> t.text } == listOf("-1.50")
        } and androidx.compose.ui.test.hasClickAction()
        compose.waitForIdle()
        compose.mainClock.autoAdvance = false   // Robolectric never reports idle once this dialog's window is up
        compose.onNode(pill).performClick()
        compose.mainClock.advanceTimeBy(500)
        captureScreenRoboImage("$out/14_fuerza_exacta.png")
    }

    @Test fun s05_gallery() { val v = vm(); compose.setContent { ZitApp(v, Tab.GALLERY) }; shot("05_galeria") }

    @Test fun s06_detail() {
        val v = vm(); compose.setContent { ZitApp(v, Tab.GALLERY, Overlay.GalleryDetail(items[0].id)) }; shot("06_detalle")
    }

    @Test fun s07_loras() { val v = vm(); compose.setContent { ZitApp(v, Tab.LORAS) }; shot("07_loras") }

    @Test fun s08_lora_edit() {
        val v = vm(); compose.setContent { ZitApp(v, Tab.LORAS, Overlay.LoraEdit("rea_character_v2.safetensors")) }; shot("08_editar_lora")
    }

    @Test fun s09_models() { val v = vm(); compose.setContent { ZitApp(v, Tab.MODELS) }; shot("09_modelos") }

    @Test @Config(qualifiers = "w412dp-h1300dp-xxhdpi")
    fun s10_settings() { val v = vm(); compose.setContent { ZitApp(v, Tab.SETTINGS) }; shot("10_ajustes") }

    @Test @Config(qualifiers = "w412dp-h1200dp-xxhdpi")
    fun s11_about() { val v = vm(); compose.setContent { ZitApp(v, Tab.SETTINGS, Overlay.About) }; shot("11_licencias") }

    /** End-to-end check of the FakeEngine flow through the ViewModel: a new picture lands in the gallery. */
    @Test fun e2e_generate_saves_to_gallery() {
        val v = vm(); compose.setContent { ZitApp(v) }
        val before = v.gallery.size
        v.generate()
        pump { v.gallery.size == before + 1 && v.job == null }
        val it = v.gallery.first()
        check(it.width == 1024 && it.params.seed == 424242L && it.params.loras.size == 4) { "unexpected $it" }
    }

    /** Engine that stops mid-denoise so the progress UI can be captured. */
    private class SuspendedEngine : ZitEngine by FakeEngine(0f) {
        private val inner = FakeEngine(0f)
        override val state get() = inner.state
        override suspend fun load(request: LoadRequest) = inner.load(request)
        override fun generate(request: GenerationRequest): Flow<GenerationEvent> = flow {
            emit(GenerationEvent.StageChanged(Stage.DENOISE))
            emit(GenerationEvent.Step(5, 8, FakeEngine.render(request, 256, 256, noise = 0.35f)))
            awaitCancellation()
        }
    }
}
