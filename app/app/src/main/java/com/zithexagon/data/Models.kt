package com.zithexagon.data

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.util.zip.ZipInputStream

/**
 * zit.json — manifest of a converted ZIT-HEXAGON model package. formatVersion 2 (engine 0.23):
 * files = text_encoder (GGUF, CPU llama.cpp), t_embedder (raw fp32), dit_0..dit_N (QNN HTP context
 * binaries, run in order), vae (HTP ctx), optional upscaler_<id> (HTP ctx, 4x tile model);
 * "engine" = { caption_len, shift, vae_scale, vae_shift, lat_h, lat_w }; lora.supported=false
 * means LoRA merge is not available yet. Only "format", "id" and "name" are required.
 *
 * {
 *   "format": "zit-hexagon", "formatVersion": 1,
 *   "id": "reazit-v1.0", "name": "ReaZIT v1.0", "version": "1.0",
 *   "base": "Z-Image Turbo", "author": "…", "description": "…", "license": "…",
 *   "target": { "soc": "SM8850", "htp": "v81", "qnn": "2.x" },
 *   "defaults": { "steps": 8, "cfg": 1.0, "sampler": "euler", "width": 1024, "height": 1024 },
 *   "resolutions": ["1024x1024", "832x1216"],
 *   "lora": { "type": "swap", "maxActive": 7, "recommended": 3 },
 *   "files": { "text_encoder": "text_encoder.bin", "transformer": "transformer.bin", "vae": "vae_decoder.bin" },
 *   "preview": "preview.jpg"
 * }
 */
data class ModelManifest(
    val id: String,
    val name: String,
    val version: String,
    val base: String,
    val author: String,
    val description: String,
    val license: String,
    val soc: String,
    val htp: String,
    val defaultSteps: Int,
    val defaultCfg: Float,
    val defaultSampler: String,
    val defaultWidth: Int,
    val defaultHeight: Int,
    val resolutions: List<Resolution>,
    val loraType: String,
    val maxLoras: Int,
    val recommendedLoras: Int,
    val files: Map<String, String>,
    val preview: String?,
    /** lora.supported: false = the engine cannot merge LoRAs yet (UI shows "próximamente"). */
    val loraSupported: Boolean = true,
    val defaultScheduler: String = "simple",
    /** Raw "engine" block (caption_len, shift, vae_scale, vae_shift, ...), read by NativeZitEngine. */
    val engineJson: String = "{}",
) {
    companion object {
        const val FILE_NAME = "zit.json"
        const val FORMAT = "zit-hexagon"

        fun parse(text: String): ModelManifest {
            val o = JSONObject(text)
            require(o.optString("format") in setOf(FORMAT, "zit-npu")) { "no es un paquete ZIT-HEXAGON (\"format\" ≠ \"$FORMAT\")" }
            val id = o.optString("id").ifBlank { error("falta \"id\"") }
            val d = o.optJSONObject("defaults") ?: JSONObject()
            val t = o.optJSONObject("target") ?: JSONObject()
            val l = o.optJSONObject("lora") ?: JSONObject()
            val f = o.optJSONObject("files") ?: JSONObject()
            val res = o.optJSONArray("resolutions")?.let { a ->
                (0 until a.length()).mapNotNull { i ->
                    a.optString(i).split('x', '×').takeIf { it.size == 2 }?.let { p ->
                        val w = p[0].trim().toIntOrNull(); val h = p[1].trim().toIntOrNull()
                        if (w != null && h != null) Resolution(w, h) else null
                    }
                }
            } ?: emptyList()
            return ModelManifest(
                id = id, name = o.optString("name", id), version = o.optString("version"),
                base = o.optString("base", "Z-Image Turbo"), author = o.optString("author"),
                description = o.optString("description"), license = o.optString("license"),
                soc = t.optString("soc"), htp = t.optString("htp"),
                defaultSteps = d.optInt("steps", 8), defaultCfg = d.optDouble("cfg", 1.0).toFloat(),
                defaultSampler = d.optString("sampler", "euler"),
                defaultWidth = d.optInt("width", 1024), defaultHeight = d.optInt("height", 1024),
                resolutions = res, loraType = l.optString("type", "swap"), maxLoras = l.optInt("maxActive", LoraLimits.MAX_ACTIVE).coerceIn(1, LoraLimits.MAX_ACTIVE),
                recommendedLoras = l.optInt("recommended", LoraLimits.RECOMMENDED),
                files = f.keys().asSequence().associateWith { f.getString(it) },
                preview = o.optString("preview").ifBlank { null },
                loraSupported = l.optBoolean("supported", true),
                defaultScheduler = d.optString("scheduler", "simple"),
                engineJson = (o.optJSONObject("engine") ?: JSONObject()).toString(),
            )
        }

        /** Template the user can export and fill in for the converter. */
        fun template(): String = JSONObject().apply {
            put("format", FORMAT); put("formatVersion", 1)
            put("id", "reazit-v1.0"); put("name", "ReaZIT v1.0"); put("version", "1.0")
            put("base", "Z-Image Turbo"); put("author", ""); put("description", ""); put("license", "")
            put("target", JSONObject().put("soc", "SM8850").put("htp", "v81").put("qnn", ""))
            put("defaults", JSONObject().put("steps", 8).put("cfg", 1.0).put("sampler", "euler").put("scheduler", "simple").put("width", 1024).put("height", 1024))
            put("resolutions", JSONArray(listOf("1024x1024")))   // 0.25: one fixed compiled size
            put("lora", JSONObject().put("type", "swap").put("maxActive", LoraLimits.MAX_ACTIVE).put("recommended", LoraLimits.RECOMMENDED))
            put("files", JSONObject().put("text_encoder", "text_encoder.bin").put("transformer", "transformer.bin").put("vae", "vae_decoder.bin"))
            put("preview", "preview.jpg")
        }.toString(2)
    }
}

enum class ModelSource { DEMO, APP, SHARED }

data class ZitModel(
    val manifest: ModelManifest,
    /** null for the demo model. */
    val dir: File?,
    val source: ModelSource,
    val sizeBytes: Long,
    /** Files named in the manifest that are not in the folder. */
    val missing: List<String>,
) {
    val id get() = manifest.id
    val name get() = manifest.name
    val isDemo get() = source == ModelSource.DEMO
    val previewFile: File? get() = manifest.preview?.let { p -> dir?.let { File(it, p) } }?.takeIf { it.isFile }
}

class ModelRepository(private val context: Context) {

    val demo = ZitModel(
        ModelManifest(
            id = "demo-reazit", name = "ReaZIT v1.0 (demo)", version = "1.0", base = "Z-Image Turbo",
            author = "—", description = "Modelo de demostración (motor simulado). Para generar de verdad, importa DiT FP8 + TE + VAE en Modelos (NPH).",
            license = "", soc = "SM8850", htp = "v81", defaultSteps = 8, defaultCfg = 1.0f, defaultSampler = "euler",
            defaultWidth = 1024, defaultHeight = 1024, resolutions = Resolution.PRESETS, loraType = "swap", maxLoras = LoraLimits.MAX_ACTIVE, recommendedLoras = LoraLimits.RECOMMENDED,
            files = emptyMap(), preview = null,
        ),
        dir = null, source = ModelSource.DEMO, sizeBytes = 0, missing = emptyList(),
    )

    fun list(): List<ZitModel> {
        val out = mutableListOf(demo)
        // 0.26: NPH triplet (DiT+TE+VAE) from content-based selections
        NphModelStore.asZitModel(context)?.let { out += it }
        val seen = HashSet<String>().also { it.add("nph-reazit-fp8") }
        fun scan(root: File, source: ModelSource) {
            root.listFiles()?.filter { it.isDirectory }?.sortedBy { it.name.lowercase() }?.forEach { d ->
                read(d, source)?.let { if (seen.add(it.id)) out += it }
            }
        }
        scan(Storage.models(context), ModelSource.APP)
        if (Storage.hasAllFilesAccess()) scan(Storage.sharedModels(), ModelSource.SHARED)
        return out
    }

    fun read(dir: File, source: ModelSource): ZitModel? {
        val mf = File(dir, ModelManifest.FILE_NAME)
        if (!mf.isFile) return null
        return runCatching {
            val m = ModelManifest.parse(mf.readText())
            ZitModel(m, dir, source, dir.dirSize(), m.files.values.filterNot { File(dir, it).exists() })
        }.getOrNull()
    }

    /** Imports a .zip with zit.json at its root (or inside a single top folder). */
    fun importZip(uri: Uri, onProgress: (Long) -> Unit = {}): ZitModel {
        val tmp = File(Storage.models(context), ".import-${System.currentTimeMillis()}")
        try {
            tmp.mkdirs()
            context.contentResolver.openInputStream(uri)!!.use { unzip(it, tmp, onProgress) }
            return finishImport(tmp)
        } finally { if (tmp.exists()) tmp.deleteRecursively() }
    }

    /** Imports (copies) a model folder picked with the system folder picker. */
    fun importTree(treeUri: Uri, onProgress: (Long) -> Unit = {}): ZitModel {
        val root = DocumentFile.fromTreeUri(context, treeUri) ?: error("carpeta no legible")
        val tmp = File(Storage.models(context), ".import-${System.currentTimeMillis()}")
        try {
            tmp.mkdirs()
            var copied = 0L
            fun copy(doc: DocumentFile, into: File) {
                for (c in doc.listFiles()) {
                    val name = c.name ?: continue
                    if (c.isDirectory) copy(c, File(into, name).apply { mkdirs() })
                    else context.contentResolver.openInputStream(c.uri)!!.use { input ->
                        File(into, name).outputStream().use { out ->
                            val buf = ByteArray(1 shl 16)
                            while (true) { val n = input.read(buf); if (n < 0) break; out.write(buf, 0, n); copied += n; onProgress(copied) }
                        }
                    }
                }
            }
            copy(root, tmp)
            return finishImport(tmp)
        } finally { if (tmp.exists()) tmp.deleteRecursively() }
    }

    private fun finishImport(tmp: File): ZitModel {
        val base = if (File(tmp, ModelManifest.FILE_NAME).isFile) tmp
            else tmp.listFiles()?.singleOrNull { it.isDirectory && File(it, ModelManifest.FILE_NAME).isFile }
                ?: error("no se encontró ${ModelManifest.FILE_NAME}")
        val manifest = ModelManifest.parse(File(base, ModelManifest.FILE_NAME).readText())
        val safeId = manifest.id.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val dest = File(Storage.models(context), safeId)
        require(!dest.exists()) { "ya existe un modelo con id \"${manifest.id}\"" }
        check(base.renameTo(dest)) { "no se pudo mover el paquete" }
        return read(dest, ModelSource.APP)!!
    }

    fun delete(model: ZitModel): Boolean =
        model.source == ModelSource.APP && model.dir?.deleteRecursively() == true

    companion object {
        fun unzip(input: InputStream, dest: File, onProgress: (Long) -> Unit) {
            val canon = dest.canonicalPath + File.separator
            var total = 0L
            ZipInputStream(input.buffered()).use { zip ->
                while (true) {
                    val e = zip.nextEntry ?: break
                    val f = File(dest, e.name)
                    require(f.canonicalPath.startsWith(canon)) { "entrada zip inválida: ${e.name}" }
                    if (e.isDirectory) { f.mkdirs(); continue }
                    f.parentFile?.mkdirs()
                    f.outputStream().use { out ->
                        val buf = ByteArray(1 shl 16)
                        while (true) { val n = zip.read(buf); if (n < 0) break; out.write(buf, 0, n); total += n; onProgress(total) }
                    }
                }
            }
        }
    }
}
