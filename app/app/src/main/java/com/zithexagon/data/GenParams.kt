package com.zithexagon.data

import com.zithexagon.engine.Sampler
import com.zithexagon.engine.Scheduler
import org.json.JSONArray
import org.json.JSONObject

data class LoraUse(val id: String, val name: String, val strength: Float)

/** LoRA limits: up to [MAX_ACTIVE] at once, but more than [RECOMMENDED] shows a quality/speed warning. */
object LoraLimits {
    const val RECOMMENDED = 3
    const val MAX_ACTIVE = 7
    const val MIN_STRENGTH = -3f
    const val MAX_STRENGTH = 3f
    const val STRENGTH_STEP = 0.05f
    const val DEFAULT_STRENGTH = 1f
    fun snap(v: Float) = (Math.round(v / STRENGTH_STEP) * STRENGTH_STEP).coerceIn(MIN_STRENGTH, MAX_STRENGTH)
}

/** Everything needed to reproduce a picture; saved next to each output as JSON. */
data class GenParams(
    val prompt: String = "",
    val negativePrompt: String = "",
    val steps: Int = 8,
    val cfg: Float = 1.0f,
    val seed: Long = -1L,
    val width: Int = 1024,
    val height: Int = 1024,
    val sampler: Sampler = Sampler.EULER,
    val scheduler: Scheduler = Scheduler.SIMPLE,
    val loras: List<LoraUse> = emptyList(),
    val modelId: String = "",
    val modelName: String = "",
    /** 0 = no upscale, else 2 or 4. */
    val upscale: Int = 0,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("prompt", prompt); put("negative_prompt", negativePrompt)
        put("steps", steps); put("cfg", cfg.toDouble()); put("seed", seed)
        put("width", width); put("height", height); put("sampler", sampler.id); put("scheduler", scheduler.id)
        put("model_id", modelId); put("model_name", modelName); put("upscale", upscale)
        put("loras", JSONArray().apply {
            loras.forEach { put(JSONObject().put("id", it.id).put("name", it.name).put("strength", it.strength.toDouble())) }
        })
    }

    companion object {
        fun fromJson(o: JSONObject) = GenParams(
            prompt = o.optString("prompt"),
            negativePrompt = o.optString("negative_prompt"),
            steps = o.optInt("steps", 8),
            cfg = o.optDouble("cfg", 1.0).toFloat(),
            seed = o.optLong("seed", -1),
            width = o.optInt("width", 1024),
            height = o.optInt("height", 1024),
            sampler = Sampler.byId(o.optString("sampler")),
            scheduler = Scheduler.byId(o.optString("scheduler")),
            modelId = o.optString("model_id"),
            modelName = o.optString("model_name"),
            upscale = o.optInt("upscale", 0),
            loras = o.optJSONArray("loras")?.let { a ->
                (0 until a.length()).map { i ->
                    val l = a.getJSONObject(i)
                    LoraUse(l.optString("id"), l.optString("name"), l.optDouble("strength", 1.0).toFloat())
                }
            } ?: emptyList(),
        )
    }
}

/** Resolution presets that suit Z-Image Turbo (multiples of 64, ~1 MP and a fast 0.5 MP set). */
data class Resolution(val width: Int, val height: Int) {
    val label get() = "$width×$height"
    val ratio: String get() {
        fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
        val g = gcd(width, height); return "${width / g}:${height / g}"
    }
    companion object {
        val PRESETS = listOf(
            Resolution(1024, 1024), Resolution(896, 1152), Resolution(1152, 896),
            Resolution(832, 1216), Resolution(1216, 832), Resolution(768, 1344), Resolution(1344, 768),
            Resolution(768, 768), Resolution(512, 512),
        )
    }
}
