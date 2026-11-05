package com.zithexagon.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.OpenableColumns
import androidx.documentfile.provider.DocumentFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile

/** One swap-type LoRA in the library. Metadata lives in lora_library.json (app files dir). */
data class LoraEntry(
    /** Stable key: the file name (same LoRA found in two folders = one entry). */
    val id: String,
    val name: String,
    val path: String,
    val sizeBytes: Long,
    /** "app" = copied into the app; "folder" = read in place from a scanned folder. */
    val source: String,
    val triggerWords: List<String>,
    val strength: Float,
    val previewPath: String?,
    val addedAt: Long,
    val notes: String = "",
) {
    val file get() = File(path)
    val exists get() = file.isFile
    val preview: File? get() = previewPath?.let(::File)?.takeIf { it.isFile }

    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id); put("name", name); put("path", path); put("size", sizeBytes); put("source", source)
        put("trigger_words", JSONArray(triggerWords)); put("strength", strength.toDouble())
        previewPath?.let { put("preview", it) }; put("added_at", addedAt); put("notes", notes)
    }

    companion object {
        fun fromJson(o: JSONObject) = LoraEntry(
            id = o.getString("id"), name = o.optString("name", o.getString("id")), path = o.optString("path"),
            sizeBytes = o.optLong("size"), source = o.optString("source", "app"),
            triggerWords = o.optJSONArray("trigger_words")?.let { a -> (0 until a.length()).map { a.getString(it) } } ?: emptyList(),
            strength = o.optDouble("strength", 1.0).toFloat(),
            previewPath = o.optString("preview").ifBlank { null },
            addedAt = o.optLong("added_at"), notes = o.optString("notes"),
        )
    }
}

class LoraRepository(private val context: Context) {
    private val file = File(context.filesDir, "lora_library.json")
    private val entries = LinkedHashMap<String, LoraEntry>()

    init { load() }

    fun all(): List<LoraEntry> = synchronized(entries) { entries.values.sortedBy { it.name.lowercase() } }
    fun get(id: String): LoraEntry? = synchronized(entries) { entries[id] }

    fun update(e: LoraEntry) { synchronized(entries) { entries[e.id] = e }; save() }

    fun setStrength(id: String, s: Float) = get(id)?.let { update(it.copy(strength = s)) }
    fun setTriggerWords(id: String, words: List<String>) =
        get(id)?.let { update(it.copy(triggerWords = words.map { w -> w.trim() }.filter { w -> w.isNotEmpty() }.distinct())) }
    fun rename(id: String, name: String) = get(id)?.let { update(it.copy(name = name.trim().ifEmpty { it.name })) }

    /** Removes from the library; deletes the file only when it was copied into the app. */
    fun remove(id: String) {
        val e = synchronized(entries) { entries.remove(id) } ?: return
        if (e.source == "app") e.file.delete()
        e.preview?.takeIf { it.parentFile == Storage.previews(context) }?.delete()
        save()
    }

    /** null = usable Z-Image LoRA; otherwise the reason (Spanish, shown to the user). */
    fun checkZImage(f: File): String? {
        val n = SafetensorsMeta.zimageTargets(f)
        return when {
            n > 0 -> null
            n < 0 -> "\"${f.name}\" no es un .safetensors válido"
            SafetensorsMeta.isLycoris(f) -> "\"${f.name}\" es LoKr/LoHa: solo se admiten LoRA normales de Z-Image"
            else -> "\"${f.name}\" no parece un LoRA de Z-Image (ninguna capa del DiT reconocida; ¿es de SDXL/Flux?)"
        }
    }

    /** Import through the system picker (SAF): the file is copied into the app's LoRA folder. */
    fun importUri(uri: Uri): LoraEntry {
        val name = displayName(uri) ?: "lora_${System.currentTimeMillis()}.safetensors"
        require(name.endsWith(".safetensors", true)) { "\"$name\" no es un .safetensors" }
        val dest = File(Storage.loras(context), name)
        context.contentResolver.openInputStream(uri)!!.use { i -> dest.outputStream().use { o -> i.copyTo(o, 1 shl 16) } }
        checkZImage(dest)?.let { msg -> dest.delete(); throw IllegalArgumentException(msg) }
        return register(dest, "app")
    }

    /** Import every .safetensors (and its sidecar preview / .json) of a folder picked with SAF. */
    fun importTree(treeUri: Uri): Int {
        val root = DocumentFile.fromTreeUri(context, treeUri) ?: return 0
        val kids = root.listFiles()
        var n = 0
        for (doc in kids) {
            val nm = doc.name ?: continue
            if (!doc.isFile || !nm.endsWith(".safetensors", true)) continue
            val dest = File(Storage.loras(context), nm)
            if (!dest.isFile || dest.length() != doc.length()) {
                context.contentResolver.openInputStream(doc.uri)!!.use { i -> dest.outputStream().use { o -> i.copyTo(o, 1 shl 16) } }
            }
            if (checkZImage(dest) != null) { dest.delete(); continue }   // not a Z-Image LoRA: skipped
            val base = nm.removeSuffix(".safetensors")
            kids.filter { k -> k.name?.let { it != nm && it.startsWith("$base.") } == true }.forEach { side ->
                runCatching {
                    context.contentResolver.openInputStream(side.uri)!!.use { i ->
                        File(Storage.loras(context), side.name!!).outputStream().use { o -> i.copyTo(o) }
                    }
                }
            }
            register(dest, "app"); n++
        }
        return n
    }

    /** Hunyuan-NPU style scan: app LoRA folder + Download/ZITHEXAGON/loras (in place, needs all-files access). */
    fun scanFolders(): Int {
        val dirs = buildList {
            add(Storage.loras(context) to "app")
            if (Storage.hasAllFilesAccess()) add(Storage.sharedLoras() to "folder")
        }
        var added = 0
        for ((dir, src) in dirs) {
            dir.listFiles { f -> f.isFile && f.name.endsWith(".safetensors", true) }?.forEach { f ->
                if (get(f.name) == null) { register(f, src); added++ }
            }
        }
        // Drop entries whose in-place file disappeared from a scanned folder.
        all().filter { it.source == "folder" && !it.exists }.forEach { remove(it.id) }
        return added
    }

    fun register(f: File, source: String): LoraEntry {
        val meta = SafetensorsMeta.read(f)
        val old = get(f.name)
        val e = LoraEntry(
            id = f.name, name = old?.name ?: meta.title ?: f.name.removeSuffix(".safetensors"),
            path = f.absolutePath, sizeBytes = f.length(), source = source,
            triggerWords = old?.triggerWords?.takeIf { it.isNotEmpty() } ?: meta.keywords.take(12),
            strength = old?.strength ?: 1f,
            previewPath = old?.previewPath ?: meta.preview?.let { copyPreview(f.name, BitmapFactory.decodeFile(it.path)) },
            addedAt = old?.addedAt ?: System.currentTimeMillis(), notes = old?.notes ?: "",
        )
        update(e); return e
    }

    /** Preview from the photo picker (content Uri). */
    fun setPreviewFromUri(id: String, uri: Uri) {
        val bmp = context.contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it) } ?: error("imagen no válida")
        get(id)?.let { update(it.copy(previewPath = copyPreview(id, bmp))) }
    }

    /** Preview from one of the app's own generations. */
    fun setPreviewFromFile(id: String, image: File) {
        val bmp = BitmapFactory.decodeFile(image.path) ?: error("imagen no válida")
        get(id)?.let { update(it.copy(previewPath = copyPreview(id, bmp))) }
    }

    fun clearPreview(id: String) = get(id)?.let { e -> e.preview?.delete(); update(e.copy(previewPath = null)) }

    private fun copyPreview(id: String, src: Bitmap?): String? {
        src ?: return null
        val max = 512
        val s = minOf(1f, max / maxOf(src.width, src.height).toFloat())
        val scaled = if (s < 1f) Bitmap.createScaledBitmap(src, (src.width * s).toInt(), (src.height * s).toInt(), true) else src
        val out = File(Storage.previews(context), id.replace(Regex("[^A-Za-z0-9._-]"), "_") + "_${System.currentTimeMillis()}.jpg")
        Storage.previews(context).listFiles { f -> f.name.startsWith(id.replace(Regex("[^A-Za-z0-9._-]"), "_") + "_") }?.forEach { it.delete() }
        out.outputStream().use { scaled.compress(Bitmap.CompressFormat.JPEG, 90, it) }
        return out.absolutePath
    }

    private fun displayName(uri: Uri): String? = runCatching {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }
    }.getOrNull()

    fun toJson(): JSONObject = JSONObject().put("version", 1).put("loras", JSONArray().apply { all().forEach { put(it.toJson()) } })

    private fun load() {
        if (!file.isFile) return
        runCatching {
            val a = JSONObject(file.readText()).getJSONArray("loras")
            for (i in 0 until a.length()) LoraEntry.fromJson(a.getJSONObject(i)).let { entries[it.id] = it }
        }
    }

    private fun save() { runCatching { file.writeText(toJson().toString(2)) } }
}

/** Trigger words / title / sidecar preview from a .safetensors header (adapted from Hunyuan NPU's LoraMetaReader). */
object SafetensorsMeta {
    data class Meta(val title: String?, val keywords: List<String>, val preview: File?)

    private fun header(file: File): JSONObject = RandomAccessFile(file, "r").use { raf ->
        val lenBytes = ByteArray(8); raf.readFully(lenBytes)
        var n = 0L
        for (i in 7 downTo 0) n = (n shl 8) or (lenBytes[i].toLong() and 0xff)
        require(n in 2..100_000_000) { "cabecera safetensors inválida" }
        val buf = ByteArray(n.toInt()); raf.readFully(buf)
        JSONObject(String(buf, Charsets.UTF_8))
    }

    // same key forms the native merger accepts (zit_lora.cpp parseKey): PEFT/diffusers lora_A/B, kohya lora_down/up
    private val ZIMAGE_KEY = Regex("(?:^|[._])(?:layers|noise_refiner|context_refiner)[._]\\d+[._](?:attention|feed_forward|adaLN_modulation)[._].*\\.(?:lora_A|lora_down)(?:\\.default)?\\.weight$")

    /** 0.25: number of Z-Image DiT layers this file can modify (0 = not a Z-Image LoRA, e.g. SDXL/Flux/LoKr). -1 = unreadable. */
    fun zimageTargets(file: File): Int = runCatching { header(file).keys().asSequence().count { ZIMAGE_KEY.containsMatchIn(it) } }.getOrDefault(-1)

    fun isLycoris(file: File): Boolean = runCatching { header(file).keys().asSequence().any { "lokr" in it || "hada" in it } }.getOrDefault(false)

    private fun split(s: String) = s.split(',', ';', '\n').map { it.trim().trim('"') }.filter { it.isNotEmpty() }

    fun read(file: File): Meta {
        val kws = LinkedHashSet<String>()
        var title: String? = null
        runCatching {
            val meta = header(file).optJSONObject("__metadata__")
            if (meta != null) {
                for (key in listOf("modelspec.trigger_phrase", "ss_trigger_words", "trigger_words", "trained_words",
                        "trainedWords", "activation_text", "ss_activation_text")) {
                    val v = meta.optString(key, "")
                    if (v.isNotBlank()) {
                        kws += if (v.trim().startsWith("[")) runCatching {
                            val a = JSONArray(v); (0 until a.length()).map { a.getString(it) }
                        }.getOrDefault(split(v)) else split(v)
                    }
                }
                val tf = meta.optString("ss_tag_frequency", "")
                if (tf.isNotBlank() && kws.isEmpty()) runCatching {
                    val freq = HashMap<String, Int>()
                    val o = JSONObject(tf)
                    for (ds in o.keys()) { val d = o.getJSONObject(ds); for (t in d.keys()) freq[t.trim()] = (freq[t.trim()] ?: 0) + d.optInt(t, 1) }
                    kws += freq.entries.sortedByDescending { it.value }.take(12).map { it.key }.filter { it.isNotEmpty() }
                }
                title = meta.optString("modelspec.title").ifBlank { null }
            }
        }
        val base = file.path.removeSuffix(".safetensors")
        for (side in listOf("$base.civitai.info", "$base.json", "$base.info")) {
            val f = File(side)
            if (f.isFile && f.length() < 5_000_000) runCatching {
                val o = JSONObject(f.readText())
                val arr = o.optJSONArray("trainedWords") ?: o.optJSONArray("trained_words") ?: o.optJSONArray("activation_text")
                if (arr != null) kws += (0 until arr.length()).flatMap { split(arr.getString(it)) }
            }
        }
        val preview = listOf(".preview.png", ".preview.jpg", ".png", ".jpg", ".jpeg", ".webp").map { File(base + it) }.firstOrNull { it.isFile }
        return Meta(title, kws.toList(), preview)
    }
}
