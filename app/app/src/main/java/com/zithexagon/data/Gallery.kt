package com.zithexagon.data

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import android.provider.MediaStore
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class GalleryItem(
    val id: String,
    val file: File,
    val params: GenParams,
    val createdAt: Long,
    val width: Int,
    val height: Int,
    val totalMs: Long,
    /** Set when the picture is an upscale of another one. */
    val upscaledFrom: String? = null,
    val upscaleFactor: Int = 0,
    /** Per-image performance breakdown (Spanish text), null for older images. */
    val perf: String? = null,
)

/** Generations: outputs/<id>.png + outputs/<id>.json (params, timings). */
class GalleryRepository(private val context: Context) {
    private val dir get() = Storage.outputs(context)

    fun list(): List<GalleryItem> =
        dir.listFiles { f -> f.name.endsWith(".json") }?.mapNotNull { read(it) }?.sortedByDescending { it.createdAt } ?: emptyList()

    private fun read(json: File): GalleryItem? = runCatching {
        val o = JSONObject(json.readText())
        val png = File(dir, json.name.removeSuffix(".json") + ".png")
        if (!png.isFile) return null
        GalleryItem(
            id = json.name.removeSuffix(".json"), file = png,
            params = GenParams.fromJson(o.getJSONObject("params")),
            createdAt = o.optLong("created_at", png.lastModified()),
            width = o.optInt("width"), height = o.optInt("height"), totalMs = o.optLong("total_ms"),
            upscaledFrom = o.optString("upscaled_from").ifBlank { null }, upscaleFactor = o.optInt("upscale_factor"),
            perf = o.optString("perf").ifBlank { null },
        )
    }.getOrNull()

    fun save(bitmap: Bitmap, params: GenParams, totalMs: Long, upscaledFrom: String? = null, factor: Int = 0,
             copyToPhone: Boolean = false, perf: String? = null): GalleryItem {
        val now = System.currentTimeMillis()
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date(now))
        val id = "zit_${stamp}_${params.seed}" + (if (factor > 0) "_x$factor" else "")
        val png = File(dir, "$id.png")
        png.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        File(dir, "$id.json").writeText(JSONObject().apply {
            put("params", params.toJson()); put("created_at", now)
            put("width", bitmap.width); put("height", bitmap.height); put("total_ms", totalMs)
            upscaledFrom?.let { put("upscaled_from", it) }; if (factor > 0) put("upscale_factor", factor)
            perf?.let { put("perf", it) }
        }.toString(2))
        if (copyToPhone) runCatching { copyToMediaStore(png) }
        return GalleryItem(id, png, params, now, bitmap.width, bitmap.height, totalMs, upscaledFrom, factor, perf)
    }

    fun delete(item: GalleryItem) {
        item.file.delete(); File(dir, item.id + ".json").delete()
    }

    /** Copies a picture to Pictures/ZIT-HEXAGON (visible in the phone's gallery). No permission needed. */
    fun copyToMediaStore(png: File) {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, png.name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/ZIT-HEXAGON")
        }
        val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
        context.contentResolver.openOutputStream(uri)?.use { out -> png.inputStream().use { it.copyTo(out) } }
    }
}
