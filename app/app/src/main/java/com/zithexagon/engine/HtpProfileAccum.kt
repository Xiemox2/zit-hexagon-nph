package com.zithexagon.engine

import java.util.Locale

/**
 * Parses live `profile-batch` INFO lines from ggml-hexagon (DSP qurt usecs by class + host FastRPC wait).
 * Also parses `nph-host-overhead` (Phase B / H1: build+alloc + cache HIT/MISS).
 * Snapshots per sampler step so the report table reconciles with wall-clock.
 * Host-side unit-tested against the exact format string in libnph.
 */
class HtpProfileAccum {
    data class Bucket(
        var hmxUs: Long = 0, var hvxUs: Long = 0, var dmaUs: Long = 0,
        var flushUs: Long = 0, var fenceUs: Long = 0, var otherUs: Long = 0,
        var dspUs: Long = 0, var hostWaitUs: Long = 0,
        var hmxN: Int = 0, var hvxN: Int = 0, var dmaN: Int = 0,
        var flushN: Int = 0, var fenceN: Int = 0, var otherN: Int = 0, var batches: Int = 0,
        /** 0.30.1: combined-LoRA ops (nphL_*), already included in the classes above. */
        var loraMmUs: Long = 0, var loraEwUs: Long = 0, var loraMmN: Int = 0, var loraEwN: Int = 0,
        /** 0.30.2: fused base+LoRA matmuls (MUL_MAT_LORA; included in HMX) and folded no-op slots. */
        var loraFusedUs: Long = 0, var loraFusedN: Int = 0, var nopN: Int = 0,
    )
    data class HostOverhead(
        val buildMs: Double, val allocMs: Double, val cache: String,
        val reserves: Int, val wsMb: Double, val keep: Int,
        /** 0.30: runner desc (z_image = DiT, qwen3/llm = TE, *vae* = VAE, lora = LoRA merge); null = pre-0.30 line. */
        val runner: String? = null,
    ) {
        val isDit: Boolean get() = runner == null || !(
            runner == "lora" || runner.contains("vae", true) || runner.contains("qwen", true) ||
                runner.contains("llm", true) || runner.contains("clip", true) || runner.contains("t5", true) ||
                runner.contains("mistral", true) || runner.contains("gemma", true) || runner.contains("tae", true)
            )
    }
    private val lock = Any()
    private var cur = Bucket()
    private val perStep = ArrayList<Pair<Float, Bucket>>()
    private var linesReceived = 0
    private var firstRaw: String? = null
    private var motorActive = false
    private val diag = ArrayList<String>()
    private val hostOverheads = ArrayList<HostOverhead>()
    private val loraLines = ArrayList<String>()

    companion object {
        /** Exact field layout from ggml_hexagon_dump_batch_prof (INFO). */
        val RE = Regex(
            """profile-batch\s+n=(\d+)\s+dsp_us=(\d+)\s+host_wait_us=(\d+)\s+""" +
                """hmx_us=(\d+)\((\d+)\)\s+hvx_us=(\d+)\((\d+)\)\s+""" +
                """dma_us=(\d+)\((\d+)\)\s+flush_us=(\d+)\((\d+)\)\s+""" +
                """fence_us=(\d+)\((\d+)\)\s+other_us=(\d+)\((\d+)\)""" +
                """(?:\s+lora_mm_us=(\d+)\((\d+)\)\s+lora_ew_us=(\d+)\((\d+)\))?""" +
                """(?:\s+lora_fused_us=(\d+)\((\d+)\)\s+nop=(\d+))?"""
        )
        val HOST_RE = Regex(
            """nph-host-overhead\s+build_ms=([0-9.]+)\s+alloc_ms=([0-9.]+)\s+did_reserve=(\d+)\s+""" +
                """cache=(\w+)\s+reserves=(\d+)\s+ws_mb=([0-9.]+)\s+keep=(\d+)(?:\s+runner=(\S+))?"""
        )
        private val DIAG_RE = Regex("""(?i)hex|htp|profile|nph-host""")
    }

    fun ingest(msg: String) {
        val t = msg.trimEnd()
        if (DIAG_RE.containsMatchIn(t)) {
            synchronized(lock) {
                if (diag.size < 64) diag += t.take(400)
                if (t.contains("opt_profile=") || t.contains("DSP profiler enabled") ||
                    t.contains("Profiling mode") || t.contains("PROFILE env refresh")) {
                    val m = Regex("""opt_profile=(\d+)""").find(t)
                    motorActive = when {
                        m != null -> m.groupValues[1].toInt() > 0
                        t.contains("DSP profiler enabled") -> true
                        t.contains("PROFILE env refresh") -> {
                            val mm = Regex("""->\s*(\d+)""").find(t)
                            (mm?.groupValues?.get(1)?.toIntOrNull() ?: 0) > 0
                        }
                        else -> true
                    }
                }
            }
        }
        if (msg.contains("nph-host-overhead")) {
            val hm = HOST_RE.find(msg)
            if (hm != null) {
                val g = hm.groupValues
                synchronized(lock) {
                    hostOverheads += HostOverhead(
                        g[1].toDouble(), g[2].toDouble(), g[4],
                        g[5].toInt(), g[6].toDouble(), g[7].toInt(),
                        g.getOrNull(8)?.takeIf { it.isNotEmpty() },
                    )
                }
            }
        }
        if (msg.contains("nph-lora-")) {
            val i = t.indexOf("nph-lora-")
            synchronized(lock) { if (loraLines.size < 16) loraLines += t.substring(i).take(400) }
        }
        if (!msg.contains("profile-batch")) return
        val m = RE.find(msg) ?: return
        val g = m.groupValues
        synchronized(lock) {
            linesReceived += 1
            if (firstRaw == null) firstRaw = t.take(300)
            motorActive = true
            cur.batches += 1
            cur.dspUs += g[2].toLong()
            cur.hostWaitUs += g[3].toLong()
            cur.hmxUs += g[4].toLong(); cur.hmxN += g[5].toInt()
            cur.hvxUs += g[6].toLong(); cur.hvxN += g[7].toInt()
            cur.dmaUs += g[8].toLong(); cur.dmaN += g[9].toInt()
            cur.flushUs += g[10].toLong(); cur.flushN += g[11].toInt()
            cur.fenceUs += g[12].toLong(); cur.fenceN += g[13].toInt()
            cur.otherUs += g[14].toLong(); cur.otherN += g[15].toInt()
            if (g.size > 19 && g[16].isNotEmpty()) {
                cur.loraMmUs += g[16].toLong(); cur.loraMmN += g[17].toInt()
                cur.loraEwUs += g[18].toLong(); cur.loraEwN += g[19].toInt()
            }
            if (g.size > 22 && g[20].isNotEmpty()) {
                cur.loraFusedUs += g[20].toLong(); cur.loraFusedN += g[21].toInt(); cur.nopN += g[22].toInt()
            }
        }
    }

    /** Also scan a retained log buffer for diag lines (when profile-batch never arrived). */
    fun ingestLogTail(log: String) {
        val hadLora = synchronized(lock) { loraLines.isNotEmpty() }
        for (line in log.lineSequence()) {
            if (DIAG_RE.containsMatchIn(line)) {
                synchronized(lock) {
                    if (diag.size < 64) diag += line.trimEnd().take(400)
                    if (line.contains("opt_profile=")) {
                        val m = Regex("""opt_profile=(\d+)""").find(line)
                        if (m != null) motorActive = m.groupValues[1].toInt() > 0
                    }
                }
            }
            if (line.contains("profile-batch") || line.contains("nph-host-overhead")) ingest(line)
            if (!hadLora && line.contains("nph-lora-")) ingest(line)
        }
    }

    fun onSamplerStep(step: Int, wallS: Float) {
        synchronized(lock) {
            perStep += wallS to cur.copy()
            cur = Bucket()
        }
    }

    fun selfCheckLine(): String = synchronized(lock) {
        val ejemplo = firstRaw?.replace('\n', ' ') ?: "(ninguna)"
        "perfil: activo en motor=${if (motorActive) "sí" else "no"} · líneas recibidas=$linesReceived · formato ejemplo=$ejemplo"
    }

    /** 0.30: LoRA block from the engine's nph-lora-file / nph-lora-combined lines (empty when no LoRA). */
    fun loraReport(): String = synchronized(lock) {
        if (loraLines.isEmpty()) return@synchronized ""
        buildString {
            append("LoRA (0.30: rango concatenado → 1 par de matmul HMX por capa; FP16 exacto + escala FP32):\n")
            loraLines.forEach { append("  ").append(it).append('\n') }
        }
    }

    /**
     * Host overhead of the DiT computes only (0.30: lines carry runner=…; TE/VAE/LoRA lines are excluded).
     * [genOnCtx] = 1-based generation index on the current engine context (2+ = context was kept resident).
     */
    fun hostReport(genOnCtx: Int = 1): String = synchronized(lock) {
        val dit = hostOverheads.filter { it.isDit }
        if (dit.isEmpty()) return@synchronized "Host paso-1: (sin líneas nph-host-overhead del DiT)\n"
        buildString {
            append("Host overhead (build/alloc por compute DiT; cache=HIT reutiliza gallocr):\n")
            val first = dit.first()
            val label = if (genOnCtx >= 2) "gen #$genOnCtx en contexto residente · compute-1" else "paso-1 host"
            append(
                "  $label: build ${"%.0f".format(Locale.ROOT, first.buildMs)} ms · " +
                    "alloc ${"%.0f".format(Locale.ROOT, first.allocMs)} ms · cache=${first.cache} · " +
                    "ws=${"%.0f".format(Locale.ROOT, first.wsMb)} MB · keep=${first.keep}" +
                    (if (genOnCtx >= 2) " (esperado HIT con Extremo+graph-cache)" else "") + "\n"
            )
            if (dit.size > 1) {
                val s2 = dit[1]
                append(
                    "  compute-2: build ${"%.0f".format(Locale.ROOT, s2.buildMs)} ms · " +
                        "alloc ${"%.0f".format(Locale.ROOT, s2.allocMs)} ms · cache=${s2.cache}\n"
                )
            }
            val hits = dit.count { it.cache == "HIT" }
            append("  computes DiT: ${dit.size} (HIT $hits · MISS ${dit.size - hits})")
            val others = hostOverheads.filter { !it.isDit }.groupBy { it.runner ?: "?" }
            if (others.isNotEmpty()) {
                append(" · otros: ")
                append(others.entries.joinToString { (r, l) -> "$r×${l.size} alloc ${"%.0f".format(Locale.ROOT, l.sumOf { it.allocMs })} ms" })
            }
            append('\n')
        }
    }

    fun report(samplerWall: List<Float>, genOnCtx: Int = 1): String = synchronized(lock) {
        val check = selfCheckLine()
        if (linesReceived == 0) {
            val diagLines = diag.filter { DIAG_RE.containsMatchIn(it) }.take(20)
            return@synchronized buildString {
                append(check).append('\n')
                append(hostReport(genOnCtx))
                append("Perfil HTP: activado, 0 profile-batch (¿GGML_HEXAGON_PROFILE antes del init? reabrir motor tras activar)\n")
                if (diagLines.isNotEmpty()) {
                    append("Diagnóstico (primeras ${diagLines.size} líneas hex/htp/profile del log):\n")
                    diagLines.forEach { append("  | ").append(it).append('\n') }
                }
            }
        }
        fun ms(u: Long) = "%.1f".format(Locale.ROOT, u / 1000.0)
        fun pct(part: Long, wallUs: Long) =
            if (wallUs <= 0) "—" else "%.0f%%".format(Locale.ROOT, 100.0 * part / wallUs)
        buildString {
            append(check).append('\n')
            append(hostReport(genOnCtx))
            append("Perfil HTP (device qurt + host FastRPC wait; una línea profile-batch por op-batch):\n")
            val steps = if (perStep.isNotEmpty()) perStep else listOf(0f to cur)
            steps.forEachIndexed { i, (wallS, b) ->
                val wallUs = (wallS * 1_000_000).toLong().coerceAtLeast(1)
                val accounted = b.hmxUs + b.hvxUs + b.dmaUs + b.flushUs + b.fenceUs + b.otherUs
                append("  paso ${i + 1}: wall ${"%.2f".format(Locale.ROOT, wallS)} s · batches ${b.batches}\n")
                append("    HMX ${ms(b.hmxUs)} ms (${b.hmxN} ops, ${pct(b.hmxUs, wallUs)}) · HVX ${ms(b.hvxUs)} ms (${b.hvxN}, ${pct(b.hvxUs, wallUs)})\n")
                append("    DMA ${ms(b.dmaUs)} ms (${b.dmaN}) · flush ${ms(b.flushUs)} ms (${b.flushN}) · fence ${ms(b.fenceUs)} ms (${b.fenceN}) · otros ${ms(b.otherUs)} ms (${b.otherN})\n")
                if (b.loraMmN + b.loraEwN + b.loraFusedN > 0) {
                    append("    LoRA (incluido arriba): matmul ${ms(b.loraMmUs)} ms (${b.loraMmN}) · elementwise ${ms(b.loraEwUs)} ms (${b.loraEwN}) · ${pct(b.loraMmUs + b.loraEwUs, wallUs)} del paso\n")
                    if (b.loraFusedN > 0) {
                        append("    LoRA fusionada en HMX (base+LoRA, incluida en HMX): ${ms(b.loraFusedUs)} ms (${b.loraFusedN} ops) · no-ops ${b.nopN}\n")
                    }
                }
                append("    DSP-batch ${ms(b.dspUs)} ms · host-wait/RPC ${ms(b.hostWaitUs)} ms · suma-ops ${ms(accounted)} ms\n")
                append("    (DSP+host-wait ${ms(b.dspUs + b.hostWaitUs)} ms vs wall ${ms(wallUs)} ms)\n")
            }
            val all = steps.map { it.second } + listOf(cur)
            val tot = Bucket()
            for (b in all) {
                tot.hmxUs += b.hmxUs; tot.hvxUs += b.hvxUs; tot.dmaUs += b.dmaUs
                tot.flushUs += b.flushUs; tot.fenceUs += b.fenceUs; tot.otherUs += b.otherUs
                tot.dspUs += b.dspUs; tot.hostWaitUs += b.hostWaitUs
                tot.hmxN += b.hmxN; tot.hvxN += b.hvxN; tot.dmaN += b.dmaN
                tot.flushN += b.flushN; tot.fenceN += b.fenceN; tot.otherN += b.otherN
                tot.batches += b.batches
                tot.loraMmUs += b.loraMmUs; tot.loraEwUs += b.loraEwUs; tot.loraMmN += b.loraMmN; tot.loraEwN += b.loraEwN
                tot.loraFusedUs += b.loraFusedUs; tot.loraFusedN += b.loraFusedN; tot.nopN += b.nopN
            }
            val wallTotUs = (samplerWall.sum() * 1_000_000).toLong().coerceAtLeast(1)
            append("  TOTAL sampler: HMX ${ms(tot.hmxUs)} · HVX ${ms(tot.hvxUs)} · DMA ${ms(tot.dmaUs)} · flush ${ms(tot.flushUs)} · fence ${ms(tot.fenceUs)} · otros ${ms(tot.otherUs)}\n")
            if (tot.loraMmN + tot.loraEwN + tot.loraFusedN > 0) {
                append("  TOTAL LoRA: matmul ${ms(tot.loraMmUs)} ms (${tot.loraMmN}) · elementwise ${ms(tot.loraEwUs)} ms (${tot.loraEwN})")
                if (tot.loraFusedN > 0) append(" · fusionada base+LoRA ${ms(tot.loraFusedUs)} ms (${tot.loraFusedN})")
                append("\n")
            }
            append("  TOTAL DSP ${ms(tot.dspUs)} · host-wait ${ms(tot.hostWaitUs)} · wall pasos ${ms(wallTotUs)}\n")
        }
    }
}
