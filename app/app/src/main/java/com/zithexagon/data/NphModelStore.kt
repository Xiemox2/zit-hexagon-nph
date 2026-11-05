package com.zithexagon.data

import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.os.Environment
import android.os.ParcelFileDescriptor
import android.os.StatFs
import android.provider.OpenableColumns
import android.system.Os
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream

/**
 * Persisted DiT / TE / VAE selections for the NPH engine (content-based, same as SdBench).
 * Prefers "usar sin copiar" via real path or /proc/self/fd; copies into app files as fallback.
 */
object NphModelStore {
    data class Paths(val dit: String, val te: String, val vae: String, val labels: Map<Role, String> = emptyMap())
    data class Sel(
        val role: Role,
        val kind: String, // "file" | "uri"
        val ref: String,
        val name: String,
        val size: Long,
        val detail: String,
        val where: String,
    )

    private fun sp(ctx: Context): SharedPreferences =
        ctx.getSharedPreferences("nph_models", Context.MODE_PRIVATE)

    fun appModelDir(ctx: Context): File =
        File(ctx.getExternalFilesDir(null) ?: ctx.filesDir, "models").apply { mkdirs() }

    fun scanDirs(ctx: Context): List<File> = buildList {
        val dl = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        add(File(dl, "ZITHEXAGON/sdcpp"))
        add(File(dl, "ZITHEXAGON"))
        add(dl)
        add(appModelDir(ctx))
        // SdBench private models (same key / shared user)
        add(File("/storage/emulated/0/Android/data/com.zithexagon.sdbench/files/models"))
    }

    fun save(ctx: Context, s: Sel) {
        sp(ctx).edit()
            .putString("sel_${s.role.key}_kind", s.kind)
            .putString("sel_${s.role.key}_ref", s.ref)
            .putString("sel_${s.role.key}_name", s.name)
            .putLong("sel_${s.role.key}_size", s.size)
            .putString("sel_${s.role.key}_detail", s.detail)
            .putString("sel_${s.role.key}_where", s.where)
            .remove("sel_${s.role.key}_problem")
            .apply()
    }

    fun load(ctx: Context, r: Role): Sel? = loadSel(ctx, r)

    fun loadSel(ctx: Context, r: Role): Sel? {
        val p = sp(ctx)
        val kind = p.getString("sel_${r.key}_kind", null) ?: return null
        val ref = p.getString("sel_${r.key}_ref", null) ?: return null
        return Sel(
            r, kind, ref,
            p.getString("sel_${r.key}_name", "") ?: "",
            p.getLong("sel_${r.key}_size", 0),
            p.getString("sel_${r.key}_detail", "") ?: "",
            p.getString("sel_${r.key}_where", "") ?: "",
        )
    }

    fun problem(ctx: Context, r: Role): String? = sp(ctx).getString("sel_${r.key}_problem", null)

    fun statusLine(ctx: Context, r: Role): String {
        val s = loadSel(ctx, r) ?: return "${r.label}: — no seleccionado"
        val prob = problem(ctx, r)
        return if (prob != null) "${r.label}: ✘ ${s.name} — $prob"
        else "${r.label}: ✔ ${s.name} (${ModelProbe.human(s.size)}) · ${s.where}"
    }

    fun allReady(ctx: Context): Boolean =
        Role.entries.all { loadSel(ctx, it) != null && problem(ctx, it) == null }

    data class Resolved(val paths: Paths, val pfds: List<ParcelFileDescriptor>)

    fun resolve(ctx: Context): Resolved? {
        val labels = HashMap<Role, String>()
        val pathMap = HashMap<Role, String>()
        val pfds = ArrayList<ParcelFileDescriptor>()
        for (r in Role.entries) {
            val s = loadSel(ctx, r) ?: return null
            if (problem(ctx, r) != null) return null
            labels[r] = s.name
            if (s.kind == "file") {
                val f = File(s.ref)
                if (!f.canRead()) {
                    sp(ctx).edit().putString("sel_${r.key}_problem", "no se puede leer").apply()
                    pfds.forEach { runCatching { it.close() } }
                    return null
                }
                pathMap[r] = f.absolutePath
            } else {
                try {
                    val pfd = ctx.contentResolver.openFileDescriptor(Uri.parse(s.ref), "r")
                        ?: throw java.io.IOException("null pfd")
                    pfds.add(pfd)
                    pathMap[r] = "/proc/self/fd/${pfd.fd}"
                } catch (_: Throwable) {
                    sp(ctx).edit().putString("sel_${r.key}_problem", "sin acceso SAF; reimporta").apply()
                    pfds.forEach { runCatching { it.close() } }
                    return null
                }
            }
        }
        return Resolved(Paths(pathMap[Role.DIT]!!, pathMap[Role.TE]!!, pathMap[Role.VAE]!!, labels), pfds)
    }

    fun closeResolved(r: Resolved?) = r?.pfds?.forEach { runCatching { it.close() } }

    /** File-backed paths only (no open PFDs). Prefer [resolve] when URIs may be selected. */
    fun paths(ctx: Context): Paths? {
        val m = HashMap<Role, String>()
        val labels = HashMap<Role, String>()
        for (r in Role.entries) {
            val s = loadSel(ctx, r) ?: return null
            if (problem(ctx, r) != null) return null
            if (s.kind != "file") return null
            val f = File(s.ref)
            if (!f.canRead()) return null
            m[r] = f.absolutePath
            labels[r] = s.name
        }
        return Paths(m[Role.DIT]!!, m[Role.TE]!!, m[Role.VAE]!!, labels)
    }

    fun asZitModel(ctx: Context): ZitModel? = asModel(ctx)

    fun uriName(ctx: Context, uri: Uri): String = try {
        ctx.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        } ?: uri.lastPathSegment ?: "archivo"
    } catch (_: Throwable) { uri.lastPathSegment ?: "archivo" }

    fun uriSize(ctx: Context, uri: Uri): Long = try {
        ctx.contentResolver.openFileDescriptor(uri, "r")?.use { it.statSize } ?: -1
    } catch (_: Throwable) { -1 }

    fun probeUri(ctx: Context, uri: Uri, size: Long): Probe =
        ModelProbe.probe(size) {
            ctx.contentResolver.openInputStream(uri) ?: throw java.io.IOException("no se pudo abrir")
        }

    private fun whereOf(ctx: Context, f: File): String {
        val a = f.absolutePath
        return when {
            a.startsWith(appModelDir(ctx).absolutePath) -> "copia en la app"
            else -> a.removePrefix(Environment.getExternalStorageDirectory().absolutePath + "/")
        }
    }

    fun selFromFile(ctx: Context, f: File, p: Probe): Sel =
        Sel(p.role!!, "file", f.absolutePath, f.name, p.size, p.describe(), whereOf(ctx, f))

    /** Overload without ctx for scan helpers. */
    fun selFromFile(f: File, p: Probe): Sel =
        Sel(p.role!!, "file", f.absolutePath, f.name, p.size, p.describe(), f.parentFile?.absolutePath ?: "")

    fun tryInPlace(ctx: Context, uri: Uri, name: String, p: Probe): Sel? {
        try {
            ctx.contentResolver.openFileDescriptor(uri, "r")?.use { pfd ->
                val proc = "/proc/self/fd/${pfd.fd}"
                val target = try { Os.readlink(proc) } catch (_: Throwable) { "" }
                if (target.startsWith("/")) {
                    val f = File(target)
                    if (f.canRead() && f.length() == p.size) {
                        val q = ModelProbe.probe(f.length()) { FileInputStream(f) }
                        if (q.ok && q.role == p.role) return selFromFile(ctx, f, q)
                    }
                }
                val q = ModelProbe.probe(p.size) { FileInputStream(proc) }
                if (q.ok && q.role == p.role)
                    return Sel(p.role!!, "uri", uri.toString(), name, p.size, p.describe(), "sin copiar (SAF)")
            }
        } catch (_: Throwable) {}
        return null
    }

    fun copyIn(
        ctx: Context, uri: Uri, name: String, p: Probe,
        onProgress: (Long, Long) -> Unit = { _, _ -> },
        isCancelled: () -> Boolean = { false },
    ): Sel {
        val dir = appModelDir(ctx)
        val safe = name.replace(Regex("[/\\\\:*?\"<>|]"), "_").ifEmpty { "${p.role!!.key}.bin" }
        val dst = File(dir, safe)
        if (dst.exists() && dst.length() == p.size) {
            val q = ModelProbe.probe(dst.length()) { FileInputStream(dst) }
            if (q.ok && q.role == p.role) return selFromFile(ctx, dst, q)
        }
        val free = StatFs(dir.absolutePath).availableBytes
        val need = p.size + 200_000_000L
        if (free < need) throw java.io.IOException(
            "no hay espacio: hacen falta ${ModelProbe.human(need)} y quedan ${ModelProbe.human(free)}"
        )
        val tmp = File(dir, "$safe.copying")
        try {
            (ctx.contentResolver.openInputStream(uri) ?: throw java.io.IOException("no se pudo abrir")).use { ins ->
                FileOutputStream(tmp).use { out ->
                    val buf = ByteArray(8 shl 20)
                    var done = 0L
                    while (true) {
                        if (isCancelled()) throw java.io.IOException("copia cancelada")
                        val k = ins.read(buf); if (k < 0) break
                        out.write(buf, 0, k); done += k
                        onProgress(done, p.size)
                    }
                    out.fd.sync()
                }
            }
            if (tmp.length() != p.size) throw java.io.IOException("copia incompleta")
            if (dst.exists()) dst.delete()
            if (!tmp.renameTo(dst)) throw java.io.IOException("no se pudo renombrar")
        } catch (t: Throwable) { tmp.delete(); throw t }
        val q = ModelProbe.probe(dst.length()) { FileInputStream(dst) }
        if (!q.ok || q.role != p.role) { dst.delete(); throw java.io.IOException("copia inválida: ${q.error}") }
        return selFromFile(ctx, dst, q)
    }

    /**
     * Import one URI for [expectedRole] (or auto-detect if null). Tries in-place first when [allowCopy].
     * Returns the saved Sel.
     */
    fun importUri(
        ctx: Context,
        uri: Uri,
        expectedRole: Role?,
        allowCopy: Boolean = true,
        onProgress: (Long, Long) -> Unit = { _, _ -> },
        isCancelled: () -> Boolean = { false },
    ): Sel {
        val name = uriName(ctx, uri)
        val size = uriSize(ctx, uri)
        val probe = probeUri(ctx, uri, size)
        if (!probe.ok) throw java.io.IOException(probe.error ?: "archivo no reconocido")
        if (expectedRole != null && probe.role != expectedRole)
            throw java.io.IOException("esperaba ${expectedRole.label}, detecté ${probe.role?.label ?: "?"}: ${probe.describe()}")
        val role = probe.role!!
        val inplace = tryInPlace(ctx, uri, name, probe)
        val sel = when {
            inplace != null -> inplace
            allowCopy -> copyIn(ctx, uri, name, probe, onProgress, isCancelled)
            else -> throw java.io.IOException("no se pudo usar sin copiar; activa copia o concede acceso")
        }
        save(ctx, sel)
        return sel
    }

    /** Fill missing roles from Download/ZITHEXAGON/sdcpp etc. */
    fun autoselectMissing(ctx: Context) {
        val found = ArrayList<Pair<File, Probe>>()
        for (d in scanDirs(ctx)) {
            val files = try { d.listFiles() } catch (_: Throwable) { null } ?: continue
            for (f in files) {
                if (!f.isFile) continue
                val n = f.name.lowercase()
                if (n.endsWith(".crdownload") || n.endsWith(".part") || n.endsWith(".tmp")) continue
                val ext = n.endsWith(".safetensors") || n.endsWith(".gguf") || n.endsWith(".sft")
                if (f.length() < 50_000_000L && !(ext && f.length() > 1_000_000L)) continue
                val p = runCatching { ModelProbe.probe(f.length()) { FileInputStream(f) } }.getOrNull() ?: continue
                if (p.ok) found += f to p
            }
        }
        for (r in Role.entries) {
            if (loadSel(ctx, r) != null && problem(ctx, r) == null) continue
            val best = found.filter { it.second.role == r }
                .sortedWith(compareBy(
                    { if (it.first.parentFile?.name == "sdcpp") 0 else 1 },
                    { -it.first.lastModified() },
                ))
                .firstOrNull() ?: continue
            save(ctx, selFromFile(ctx, best.first, best.second))
        }
    }

    fun asModel(ctx: Context): ZitModel? {
        if (!allReady(ctx)) return null
        val dit = loadSel(ctx, Role.DIT)!!
        val te = loadSel(ctx, Role.TE)!!
        val vae = loadSel(ctx, Role.VAE)!!
        val mf = ModelManifest(
            id = "nph-reazit-fp8",
            name = "ReaZIT FP8 (NPH)",
            version = "1.0",
            base = "Z-Image Turbo",
            author = "",
            description = "DiT FP8 + Qwen3-4B Q8_0 + VAE vía motor NPH (sd.cpp Hexagon).",
            license = "Apache-2.0",
            soc = "SM8850",
            htp = "v81",
            defaultSteps = 8,
            defaultCfg = 1.0f,
            defaultSampler = "euler",
            defaultWidth = 1024,
            defaultHeight = 1024,
            resolutions = listOf(Resolution(1024, 1024)),
            loraType = "runtime",
            maxLoras = LoraLimits.MAX_ACTIVE,
            recommendedLoras = LoraLimits.RECOMMENDED,
            files = mapOf("dit" to dit.name, "te" to te.name, "vae" to vae.name),
            preview = null,
            loraSupported = true,
            defaultScheduler = "simple",
            engineJson = """{"engine":"nph"}""",
        )
        return ZitModel(mf, dir = null, source = ModelSource.SHARED, sizeBytes = dit.size + te.size + vae.size, missing = emptyList())
    }
}
