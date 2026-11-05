package com.zithexagon

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.util.Base64
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Diagnostics without adb:
 *  - filesDir/diag/stage.log: breadcrumbs (Kotlin + native, one flushed line each). Rotated to stage_prev.log at process start.
 *  - On start, reads ApplicationExitInfo for the previous process; if it died abnormally (native crash, LMK, signal,
 *    ANR...) or while a job was running, builds a report (reason, memory, decoded tombstone, last breadcrumbs) and hands
 *    it to CrashReporter so the app offers "Compartir reporte de error" (also in Ajustes).
 */
object ZitDiag {
    @Volatile var stagePath: String? = null; private set
    private val lock = Any()
    private const val PREFS = "zitdiag"
    private const val KEY_LAST_TS = "last_exit_ts"
    private const val MAX_REPORT_CHARS = 180_000

    fun dir(ctx: Context) = File(ctx.filesDir, "diag")

    /** Call first thing in Application.onCreate. Never throws. */
    fun start(ctx: Context) {
        runCatching {
            val d = dir(ctx).apply { mkdirs() }
            val cur = File(d, "stage.log"); val prev = File(d, "stage_prev.log")
            if (cur.isFile && cur.length() > 0) { prev.delete(); cur.renameTo(prev) }
            stagePath = cur.absolutePath
            bc("APP start v" + runCatching { BuildConfigInfo.versionName(ctx) }.getOrDefault("?") + " pid=" + android.os.Process.myPid())
        }
        runCatching { checkPreviousExit(ctx) }.onFailure { bc("exit-info check failed: $it") }
    }

    /** Kotlin breadcrumb (same file as native ones). Never throws. */
    fun bc(msg: String) {
        val p = stagePath ?: return
        runCatching {
            synchronized(lock) {
                FileOutputStream(p, true).use { fo ->
                    val rt = Runtime.getRuntime()
                    val heapMb = (rt.totalMemory() - rt.freeMemory()) shr 20
                    fo.write("[${ts()}] K heap=${heapMb}MB $msg\n".toByteArray()); fo.fd.sync()
                }
            }
        }
    }

    fun perfFile(ctx: Context) = File(dir(ctx), "perf.log")

    /** Appends one image's performance breakdown to diag/perf.log (kept under ~256 KB). Never throws. */
    fun perf(text: String) {
        val p = stagePath ?: return
        runCatching {
            synchronized(lock) {
                val f = File(File(p).parentFile, "perf.log")
                if (f.length() > 256 * 1024) f.writeText(f.readText().takeLast(128 * 1024))
                val stamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.ROOT).format(Date())
                f.appendText("=== $stamp ===\n$text\n")
            }
        }
    }

    /** Last [n] lines of the current stage log (native breadcrumbs, QNN errors). Never throws. */
    fun stageTail(n: Int = 60): String = runCatching {
        val p = stagePath ?: return ""
        synchronized(lock) { File(p).readLines().takeLast(n).joinToString("\n") }
    }.getOrDefault("")

    private fun ts() = SimpleDateFormat("HH:mm:ss.SSS", Locale.ROOT).format(Date())

    private fun reasonName(r: Int) = when (r) {
        ApplicationExitInfo.REASON_UNKNOWN -> "UNKNOWN"
        ApplicationExitInfo.REASON_EXIT_SELF -> "EXIT_SELF"
        ApplicationExitInfo.REASON_SIGNALED -> "SIGNALED"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "LOW_MEMORY (el sistema mató la app por falta de RAM)"
        ApplicationExitInfo.REASON_CRASH -> "CRASH (Java)"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "CRASH_NATIVE"
        ApplicationExitInfo.REASON_ANR -> "ANR"
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "INITIALIZATION_FAILURE"
        ApplicationExitInfo.REASON_PERMISSION_CHANGE -> "PERMISSION_CHANGE"
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "EXCESSIVE_RESOURCE_USAGE"
        ApplicationExitInfo.REASON_USER_REQUESTED -> "USER_REQUESTED"
        ApplicationExitInfo.REASON_USER_STOPPED -> "USER_STOPPED"
        ApplicationExitInfo.REASON_DEPENDENCY_DIED -> "DEPENDENCY_DIED"
        ApplicationExitInfo.REASON_OTHER -> "OTHER"
        ApplicationExitInfo.REASON_FREEZER -> "FREEZER"
        ApplicationExitInfo.REASON_PACKAGE_STATE_CHANGE -> "PACKAGE_STATE_CHANGE"
        ApplicationExitInfo.REASON_PACKAGE_UPDATED -> "PACKAGE_UPDATED"
        else -> "reason=$r"
    }

    private val ABNORMAL = setOf(
        ApplicationExitInfo.REASON_UNKNOWN, ApplicationExitInfo.REASON_SIGNALED, ApplicationExitInfo.REASON_LOW_MEMORY,
        ApplicationExitInfo.REASON_CRASH, ApplicationExitInfo.REASON_CRASH_NATIVE, ApplicationExitInfo.REASON_ANR,
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE, ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE,
        ApplicationExitInfo.REASON_DEPENDENCY_DIED,
    )

    private fun checkPreviousExit(ctx: Context) {
        val am = ctx.getSystemService(ActivityManager::class.java) ?: return
        val list = am.getHistoricalProcessExitReasons(ctx.packageName, 0, 5)
        val prefs = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val lastSeen = prefs.getLong(KEY_LAST_TS, 0L)
        val newest = list.maxOfOrNull { it.timestamp } ?: return
        prefs.edit().putLong(KEY_LAST_TS, maxOf(lastSeen, newest)).apply()
        val prevLog = File(dir(ctx), "stage_prev.log").takeIf { it.isFile }?.let { runCatching { it.readLines() }.getOrNull() } ?: emptyList()
        val jobRunning = prevLog.lastOrNull { it.contains("JOB start") || it.contains("JOB end") }?.contains("JOB start") == true
        val fresh = list.filter { it.timestamp > lastSeen }.sortedByDescending { it.timestamp }
        val main = fresh.firstOrNull { it.reason in ABNORMAL } ?: fresh.firstOrNull() ?: return
        val abnormal = main.reason in ABNORMAL || jobRunning
        bc("prev exit: ${reasonName(main.reason)} status=${main.status} jobRunning=$jobRunning")
        if (!abnormal) return
        if (main.reason == ApplicationExitInfo.REASON_CRASH && CrashReporter.hasUnseen(ctx)) return // Java crash already recorded
        CrashReporter.recordText(ctx, buildReport(ctx, main, fresh, prevLog), "cierre inesperado: ${reasonName(main.reason)}")
    }

    private fun mb(kb: Long) = "${kb / 1024} MB"

    private fun buildReport(ctx: Context, e: ApplicationExitInfo, all: List<ApplicationExitInfo>, prevLog: List<String>): String = buildString {
        val df = SimpleDateFormat("yyyy-MM-dd HH:mm:ss Z", Locale.ROOT)
        appendLine("== Cierre anterior (ApplicationExitInfo) ==")
        appendLine("motivo: ${reasonName(e.reason)}  status/señal: ${e.status}  importancia: ${e.importance}")
        appendLine("descripción: ${e.description ?: "-"}")
        appendLine("hora: ${df.format(Date(e.timestamp))}  pid: ${e.pid}  proceso: ${e.processName}")
        appendLine("memoria al morir: PSS ${mb(e.pss)} · RSS ${mb(e.rss)}")
        if (all.size > 1) {
            appendLine("otros cierres recientes:")
            all.drop(1).forEach { appendLine("  - ${df.format(Date(it.timestamp))} ${reasonName(it.reason)} st=${it.status} pss=${mb(it.pss)} rss=${mb(it.rss)} ${it.description ?: ""}") }
        }
        val mi = ActivityManager.MemoryInfo(); ctx.getSystemService(ActivityManager::class.java)?.getMemoryInfo(mi)
        appendLine("RAM ahora: total ${mi.totalMem shr 20} MB · libre ${mi.availMem shr 20} MB · umbral LMK ${mi.threshold shr 20} MB")
        appendLine()
        appendLine("== Últimos pasos registrados (stage_prev.log, ${prevLog.size} líneas) ==")
        prevLog.takeLast(220).forEach { appendLine(it) }
        appendLine()
        val trace = runCatching { e.traceInputStream?.use { it.readBytes() } }.getOrNull()
        if (trace != null && trace.isNotEmpty()) {
            if (e.reason == ApplicationExitInfo.REASON_CRASH_NATIVE) {
                appendLine("== Tombstone (decodificado) ==")
                append(runCatching { Tombstone.decode(trace) }.getOrElse { "no se pudo decodificar: $it\n" })
                if (length < MAX_REPORT_CHARS - 40_000) {
                    appendLine(); appendLine("== Tombstone base64 (recortado a 24 KB) ==")
                    appendLine(Base64.encodeToString(trace.copyOf(minOf(trace.size, 24 * 1024)), Base64.NO_WRAP))
                }
            } else {
                appendLine("== Traza del sistema ==")
                val txt = String(trace, Charsets.UTF_8)
                appendLine(if (txt.length > 60_000) txt.take(60_000) + "\n[...recortado]" else txt)
            }
        } else appendLine("(el sistema no guardó traza/tombstone)")
    }.let { if (it.length > MAX_REPORT_CHARS) it.take(MAX_REPORT_CHARS) + "\n[...recortado]" else it }
}

/** Minimal protobuf reader for Android's tombstone.proto (only the fields useful to us). */
internal object Tombstone {
    private class R(val b: ByteArray, var p: Int, val end: Int) {
        fun more() = p < end
        fun varint(): Long { var r = 0L; var s = 0; while (true) { val x = b[p++].toInt() and 0xff; r = r or ((x and 0x7f).toLong() shl s); if (x < 0x80) return r; s += 7; require(s < 70) } }
        fun skip(wt: Int) { when (wt) { 0 -> varint(); 1 -> p += 8; 2 -> { val n = varint().toInt(); p += n }; 5 -> p += 4; else -> error("wt $wt") } }
        fun sub(): R { val n = varint().toInt(); require(n >= 0 && p + n <= end); val r = R(b, p, p + n); p += n; return r }
        fun str(): String { val s = sub(); return String(b, s.p, s.end - s.p, Charsets.UTF_8) }
        inline fun each(f: (Int, Int) -> Unit) { while (more()) { val k = varint().toInt(); f(k ushr 3, k and 7) } }
    }

    private data class Frame(var relPc: Long = 0, var pc: Long = 0, var fn: String = "", var off: Long = 0, var file: String = "", var buildId: String = "")
    private data class Thr(var id: Int = 0, var name: String = "", val frames: MutableList<Frame> = mutableListOf())

    private fun frame(r: R) = Frame().also { f -> r.each { fld, wt -> when {
        fld == 1 && wt == 0 -> f.relPc = r.varint(); fld == 2 && wt == 0 -> f.pc = r.varint()
        fld == 4 && wt == 2 -> f.fn = r.str(); fld == 5 && wt == 0 -> f.off = r.varint()
        fld == 6 && wt == 2 -> f.file = r.str(); fld == 8 && wt == 2 -> f.buildId = r.str()
        else -> r.skip(wt) } } }

    private fun thread(r: R) = Thr().also { t -> r.each { fld, wt -> when {
        fld == 1 && wt == 0 -> t.id = r.varint().toInt(); fld == 2 && wt == 2 -> t.name = r.str()
        fld == 4 && wt == 2 -> t.frames += frame(r.sub()); else -> r.skip(wt) } } }

    fun decode(b: ByteArray): String = buildString {
        val r = R(b, 0, b.size)
        var tid = 0; val threads = mutableMapOf<Int, Thr>(); val causes = mutableListOf<String>()
        val logs = mutableListOf<String>()
        r.each { fld, wt -> when {
            fld == 2 && wt == 2 -> appendLine("build: ${r.str()}")
            fld == 4 && wt == 2 -> appendLine("timestamp: ${r.str()}")
            fld == 5 && wt == 0 -> appendLine("pid: ${r.varint()}")
            fld == 6 && wt == 0 -> tid = r.varint().toInt()
            fld == 10 && wt == 2 -> { val s = r.sub(); var num = 0L; var name = ""; var code = 0L; var cname = ""; var addr = 0L
                s.each { f2, w2 -> when { f2 == 1 && w2 == 0 -> num = s.varint(); f2 == 2 && w2 == 2 -> name = s.str()
                    f2 == 3 && w2 == 0 -> code = s.varint(); f2 == 4 && w2 == 2 -> cname = s.str(); f2 == 9 && w2 == 0 -> addr = s.varint(); else -> s.skip(w2) } }
                appendLine("señal: $num ($name) code $code ($cname) fault addr 0x${java.lang.Long.toHexString(addr)}") }
            fld == 14 && wt == 2 -> appendLine("abort: ${r.str()}")
            fld == 15 && wt == 2 -> { val s = r.sub(); s.each { f2, w2 -> if (f2 == 1 && w2 == 2) causes += s.str() else s.skip(w2) } }
            fld == 16 && wt == 2 -> { val s = r.sub(); var key = 0; var t: Thr? = null
                s.each { f2, w2 -> when { f2 == 1 && w2 == 0 -> key = s.varint().toInt(); f2 == 2 && w2 == 2 -> t = thread(s.sub()); else -> s.skip(w2) } }
                t?.let { threads[key] = it } }
            fld == 18 && wt == 2 -> { val s = r.sub(); var bname = ""
                s.each { f2, w2 -> when { f2 == 1 && w2 == 2 -> bname = s.str()
                    f2 == 2 && w2 == 2 -> { val m = s.sub(); var tsx = ""; var pr = 0L; var tag = ""; var msg = ""; var mtid = 0L
                        m.each { f3, w3 -> when { f3 == 1 && w3 == 2 -> tsx = m.str(); f3 == 3 && w3 == 0 -> mtid = m.varint()
                            f3 == 4 && w3 == 0 -> pr = m.varint(); f3 == 5 && w3 == 2 -> tag = m.str(); f3 == 6 && w3 == 2 -> msg = m.str(); else -> m.skip(w3) } }
                        logs += "$tsx $mtid ${"VDIWEF".getOrElse((pr - 2).toInt().coerceIn(0, 5)) { '?' }} $tag: ${msg.trimEnd()}" }
                    else -> s.skip(w2) } } }
            else -> r.skip(wt)
        } }
        causes.forEach { appendLine("causa: $it") }
        fun dump(t: Thr, max: Int) {
            appendLine("hilo ${t.id} \"${t.name}\":")
            t.frames.take(max).forEachIndexed { i, f ->
                val sym = if (f.fn.isNotEmpty()) " (${f.fn}+${f.off})" else ""
                appendLine("  #%02d pc %016x %s%s".format(i, f.relPc, f.file, sym))
            }
        }
        threads[tid]?.let { appendLine("-- hilo que falló --"); dump(it, 64) } ?: appendLine("(hilo $tid sin backtrace)")
        val others = threads.values.filter { it.id != tid && it.frames.any { f -> f.file.contains("zitengine") || f.file.contains("Qnn") } }
        others.take(6).forEach { appendLine("-- otro hilo con código ZIT/QNN --"); dump(it, 24) }
        if (logs.isNotEmpty()) { appendLine("-- logcat de la app (últimas ${minOf(logs.size, 150)}) --"); logs.takeLast(150).forEach { appendLine(it) } }
    }
}
