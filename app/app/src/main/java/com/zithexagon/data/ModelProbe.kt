package com.zithexagon.data

import java.io.BufferedInputStream
import java.io.EOFException
import java.io.InputStream
import java.util.Locale

/** Role of a model file, detected from its content (never from the file name). */
enum class Role(val key: String, val label: String) {
    DIT("dit", "DiT (Z-Image)"), TE("te", "Codificador de texto (Qwen3)"), VAE("vae", "VAE (ae)");
}

/**
 * Result of reading a model header. [role] null = not a usable model; [error] explains why in Spanish.
 * [expectedSize] is the size the file must have according to its own header (-1 if unknown):
 * a smaller file is an unfinished download.
 */
data class Probe(
    val format: String,          // "safetensors" | "gguf" | "?"
    val role: Role?,
    val detail: String,          // e.g. "FP8 e4m3 (662 tensores)"
    val size: Long,
    val expectedSize: Long,
    val error: String?,
) {
    val ok get() = role != null && error == null
    fun describe(): String = buildString {
        append(format); if (detail.isNotEmpty()) append(", ").append(detail)
        if (role != null) append(" → ").append(role.label)
        if (error != null) append(" — ").append(error)
    }
}

object ModelProbe {
    private const val MAX_ST_HEADER = 100L shl 20   // 100 MB JSON header limit (sane safetensors)

    fun gb(b: Long) = String.format(Locale.US, "%.2f GB", b / 1e9)
    fun human(b: Long) = if (b >= 1_000_000_000L) gb(b) else String.format(Locale.US, "%.1f MB", b / 1e6)

    /** [open] must return a fresh stream positioned at offset 0. [size] = real file length. */
    fun probe(size: Long, open: () -> InputStream): Probe {
        if (size < 16) return Probe("?", null, "", size, -1, "archivo vacío o demasiado pequeño (${size} B)")
        return try {
            open().use { raw ->
                val s = BufferedInputStream(raw, 1 shl 20)
                val magic = ByteArray(8); readFully(s, magic)
                if (magic[0] == 'G'.code.toByte() && magic[1] == 'G'.code.toByte() && magic[2] == 'U'.code.toByte() && magic[3] == 'F'.code.toByte())
                    probeGguf(s, magic, size)
                else probeSafetensors(s, magic, size)
            }
        } catch (e: EOFException) {
            Probe("?", null, "", size, -1, "archivo cortado (descarga incompleta)")
        } catch (e: SecurityException) {
            Probe("?", null, "", size, -1, "sin permiso para leerlo ($e)")
        } catch (e: Throwable) {
            Probe("?", null, "", size, -1, "no se pudo leer: ${e.message ?: e}")
        }
    }

    // ---------------- safetensors ----------------
    private fun probeSafetensors(s: InputStream, first8: ByteArray, size: Long): Probe {
        var n = 0L
        for (i in 7 downTo 0) n = (n shl 8) or (first8[i].toLong() and 0xFF)
        if (n <= 1 || n > MAX_ST_HEADER || n + 8 > size)
            return Probe("?", null, "", size, -1, "no es un modelo .safetensors ni .gguf")
        val hb = ByteArray(n.toInt()); readFully(s, hb)
        val json = String(hb, Charsets.UTF_8).trim()
        if (!json.startsWith("{")) return Probe("?", null, "", size, -1, "no es un modelo .safetensors ni .gguf")
        // Minimal scan of the flat header: "name":{"dtype":"X","shape":[..],"data_offsets":[a,b]}
        val names = ArrayList<String>()
        val dtypes = HashMap<String, Int>()
        var maxEnd = 0L
        val re = Regex("\"((?:[^\"\\\\]|\\\\.)*)\"\\s*:\\s*\\{([^{}]*)\\}")
        val reD = Regex("\"dtype\"\\s*:\\s*\"([^\"]+)\"")
        val reO = Regex("\"data_offsets\"\\s*:\\s*\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\]")
        for (m in re.findAll(json)) {
            val name = m.groupValues[1]
            if (name == "__metadata__") continue
            val body = m.groupValues[2]
            val o = reO.find(body) ?: continue
            names.add(name)
            reD.find(body)?.let { dtypes.merge(it.groupValues[1], 1, Int::plus) }
            maxEnd = maxOf(maxEnd, o.groupValues[2].toLong())
        }
        if (names.isEmpty()) return Probe("safetensors", null, "", size, -1, "safetensors sin tensores")
        val expected = 8 + n + maxEnd
        val role = classify(names)
        val dt = dtypes.entries.sortedByDescending { it.value }.joinToString("/") { it.key }
        val fp8 = dtypes.keys.any { it.startsWith("F8") }
        var detail = "${names.size} tensores, $dt"
        if (role == Role.DIT) detail = (if (fp8) "FP8 e4m3, " else "SIN FP8 (más lento/más RAM), ") + detail
        val err = when {
            size < expected -> "INCOMPLETO: ${human(size)} de ${human(expected)} (la descarga no terminó)"
            role == null -> "no parece DiT Z-Image, Qwen3 ni VAE"
            else -> null
        }
        return Probe("safetensors", role, detail, size, expected, err)
    }

    private fun classify(names: List<String>): Role? {
        fun has(vararg keys: String) = names.any { n -> keys.any { n.contains(it) } }
        return when {
            has("cap_embedder", "context_refiner", "noise_refiner", "x_embedder.", "t_embedder.mlp") && has("layers.") -> Role.DIT
            has("embed_tokens", "self_attn.q_proj", "token_embd", "attn_q.weight") -> Role.TE
            has("decoder.conv_in", "first_stage_model.", "decoder.up.", "decoder.mid.") -> Role.VAE
            else -> null
        }
    }

    // ---------------- gguf ----------------
    private class LE(val s: InputStream) {
        val b = ByteArray(8)
        var pos = 0L
        fun u8(): Int { readFully(s, b, 1); pos += 1; return b[0].toInt() and 0xFF }
        fun u32(): Long { readFully(s, b, 4); pos += 4; var v = 0L; for (i in 3 downTo 0) v = (v shl 8) or (b[i].toLong() and 0xFF); return v }
        fun u64(): Long { readFully(s, b, 8); pos += 8; var v = 0L; for (i in 7 downTo 0) v = (v shl 8) or (b[i].toLong() and 0xFF); return v }
        fun skip(n: Long) { var r = n; while (r > 0) { val k = s.skip(r); if (k <= 0) { if (s.read() < 0) throw EOFException(); r -= 1 } else r -= k }; pos += n }
        fun str(): String { val n = u64(); if (n < 0 || n > (64 shl 20)) throw IllegalStateException("cadena gguf inválida"); val a = ByteArray(n.toInt()); readFully(s, a); pos += n; return String(a, Charsets.UTF_8) }
    }

    private fun scalarSize(t: Long): Int = when (t.toInt()) { 0, 1, 7 -> 1; 2, 3 -> 2; 4, 5, 6 -> 4; 10, 11, 12 -> 8; else -> -1 }

    /** ggml type → (block bytes, block elements); null = unknown. */
    private fun typeSize(t: Int): Pair<Int, Int>? = when (t) {
        0 -> 4 to 1; 1 -> 2 to 1; 2 -> 18 to 32; 3 -> 20 to 32; 6 -> 22 to 32; 7 -> 24 to 32; 8 -> 34 to 32; 9 -> 36 to 32
        10 -> 84 to 256; 11 -> 110 to 256; 12 -> 144 to 256; 13 -> 176 to 256; 14 -> 210 to 256; 15 -> 292 to 256
        24 -> 1 to 1; 25 -> 2 to 1; 26 -> 4 to 1; 27 -> 8 to 1; 28 -> 8 to 1; 30 -> 2 to 1
        else -> null
    }

    private fun probeGguf(s: InputStream, first8: ByteArray, size: Long): Probe {
        val r = LE(s); r.pos = 8
        val ver = (first8[4].toLong() and 0xFF) or ((first8[5].toLong() and 0xFF) shl 8)
        if (ver < 2) return Probe("gguf", null, "", size, -1, "versión GGUF $ver no soportada")
        val nT = r.u64(); val nKv = r.u64()
        if (nT <= 0 || nT > 1_000_000 || nKv < 0 || nKv > 1_000_000) return Probe("gguf", null, "", size, -1, "cabecera GGUF dañada")
        var arch = ""; var align = 32L; var ftype = -1L
        for (i in 0 until nKv) {
            val key = r.str(); val t = r.u32()
            when {
                t == 8L -> { val v = r.str(); if (key == "general.architecture") arch = v }
                t == 9L -> {
                    val et = r.u32(); val cnt = r.u64()
                    if (et == 8L) for (j in 0 until cnt) { val n = r.u64(); r.skip(n) }
                    else { val es = scalarSize(et); if (es < 0) throw IllegalStateException("tipo gguf $et"); r.skip(cnt * es) }
                }
                else -> {
                    val es = scalarSize(t); if (es < 0) throw IllegalStateException("tipo gguf $t")
                    if (key == "general.alignment" && es == 4) align = r.u32()
                    else if (key == "general.file_type" && es == 4) ftype = r.u32()
                    else r.skip(es.toLong())
                }
            }
        }
        val names = ArrayList<String>()
        var maxEnd = 0L; var known = true
        val types = HashMap<Int, Int>()
        for (i in 0 until nT) {
            val name = r.str(); val nd = r.u32().toInt()
            var ne = 1L; repeat(nd) { ne *= r.u64() }
            val t = r.u32().toInt(); val off = r.u64()
            names.add(name); types.merge(t, 1, Int::plus)
            val ts = typeSize(t)
            if (ts == null) known = false else maxEnd = maxOf(maxEnd, off + ne / ts.second * ts.first)
        }
        val dataStart = (r.pos + align - 1) / align * align
        val expected = if (known) dataStart + maxEnd else -1L
        var role = classify(names)
        if (role == null && arch.isNotEmpty() && names.any { it.startsWith("blk.") }) role = Role.TE
        val q = when (ftype.toInt()) { 7 -> "Q8_0"; 1 -> "F16"; 0 -> "F32"; 2 -> "Q4_0"; 15 -> "Q4_K_M"; 17 -> "Q5_K_M"; 18 -> "Q6_K"; 32 -> "BF16"; else -> if (ftype >= 0) "ftype $ftype" else "" }
        val detail = listOf(arch, q, "${names.size} tensores").filter { it.isNotEmpty() }.joinToString(", ")
        val err = when {
            expected > 0 && size < expected -> "INCOMPLETO: ${human(size)} de ${human(expected)} (la descarga no terminó)"
            role == null -> "GGUF que no es Qwen3/DiT/VAE"
            role == Role.TE && arch.isNotEmpty() && !arch.startsWith("qwen3") -> "es '$arch'; Z-Image necesita Qwen3-4B"
            else -> null
        }
        return Probe("gguf", role, detail, size, expected, err)
    }

    private fun readFully(s: InputStream, a: ByteArray, n: Int = a.size) {
        var o = 0
        while (o < n) { val k = s.read(a, o, n - o); if (k < 0) throw EOFException(); o += k }
    }
}
