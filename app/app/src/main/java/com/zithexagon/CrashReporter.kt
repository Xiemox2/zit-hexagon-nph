package com.zithexagon

import android.app.Application
import android.content.Context
import android.content.Intent
import android.os.Build
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Application entry: installs the crash recorder before anything else runs. */
class ZitHexagonApp : Application() {
    override fun onCreate() {
        super.onCreate()
        CrashReporter.install(this)
        ZitDiag.start(this)
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        runCatching { com.zithexagon.service.EngineHost.onTrimMemory(level, this) }
    }

    override fun onLowMemory() {
        super.onLowMemory()
        runCatching { com.zithexagon.service.EngineHost.onTrimMemory(android.content.ComponentCallbacks2.TRIM_MEMORY_COMPLETE, this) }
    }
}

/**
 * Writes the stack trace of any uncaught exception (or a caught startup failure) to
 * filesDir/crash/last_crash.txt so it can be shared on the next launch without adb.
 */
object CrashReporter {
    private fun dir(ctx: Context) = File(ctx.filesDir, "crash")
    private fun reportFile(ctx: Context) = File(dir(ctx), "last_crash.txt")
    private fun newFlag(ctx: Context) = File(dir(ctx), "unseen")

    fun install(app: Context) {
        val ctx = app.applicationContext
        val prev = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            runCatching { ZitDiag.bc("JAVA CRASH en ${t.name}: $e") }
            runCatching { record(ctx, e, "fatal, hilo ${t.name}") }
            if (prev != null) prev.uncaughtException(t, e) else { android.os.Process.killProcess(android.os.Process.myPid()); System.exit(10) }
        }
    }

    fun record(ctx: Context, e: Throwable, kind: String) {
        val sw = StringWriter(); e.printStackTrace(PrintWriter(sw))
        val tail = runCatching { File(ZitDiag.stagePath ?: "").takeIf { it.isFile }?.readLines()?.takeLast(120) }.getOrNull()
        recordText(ctx, sw.toString() + (tail?.let { "\n== Últimos pasos (stage.log) ==\n" + it.joinToString("\n") } ?: ""), kind)
    }

    fun recordText(ctx: Context, body: String, kind: String) {
        val text = buildString {
            appendLine("ZIT-HEXAGON — reporte de error ($kind)")
            appendLine("fecha: " + SimpleDateFormat("yyyy-MM-dd HH:mm:ss Z", Locale.ROOT).format(Date()))
            appendLine("app: " + runCatching { BuildConfigInfo.versionName(ctx) }.getOrDefault("?") + " (" + ctx.packageName + ")")
            appendLine("dispositivo: ${Build.MANUFACTURER} ${Build.MODEL} · Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}) · SoC ${Build.SOC_MODEL} · ABI ${Build.SUPPORTED_ABIS.joinToString()}")
            runCatching {
                val mi = android.app.ActivityManager.MemoryInfo()
                ctx.getSystemService(android.app.ActivityManager::class.java)?.getMemoryInfo(mi)
                appendLine("RAM total: ${mi.totalMem shr 20} MB")
            }
            appendLine()
            append(body)
        }
        dir(ctx).mkdirs()
        reportFile(ctx).writeText(text)
        newFlag(ctx).writeText("1")
    }

    fun report(ctx: Context): String? = reportFile(ctx).takeIf { it.isFile }?.let { runCatching { it.readText() }.getOrNull() }
    fun hasUnseen(ctx: Context): Boolean = newFlag(ctx).isFile && reportFile(ctx).isFile
    fun markSeen(ctx: Context) { newFlag(ctx).delete() }
    fun clear(ctx: Context) { newFlag(ctx).delete(); reportFile(ctx).delete() }

    fun share(ctx: Context, text: String) {
        // Full report as an attached .txt (FileProvider) + the first part as text (Binder limit ~1 MB).
        val uri = runCatching {
            val f = File(ctx.cacheDir, "zit-hexagon-reporte.txt"); f.writeText(text)
            androidx.core.content.FileProvider.getUriForFile(ctx, ctx.packageName + ".files", f)
        }.getOrNull()
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, "ZIT-HEXAGON — reporte de error")
            putExtra(Intent.EXTRA_TEXT, if (text.length > 60_000) text.take(60_000) + "\n[...recortado; el reporte completo va adjunto]" else text)
            if (uri != null) {
                putExtra(Intent.EXTRA_STREAM, uri)
                clipData = android.content.ClipData.newRawUri("reporte", uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
        }
        runCatching { ctx.startActivity(Intent.createChooser(send, ctx.getString(R.string.crash_share)).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) }
    }
}
