// NPG bench (MIT) — Fase 2: mide NPU (HMX/HVX) + GPU (Adreno OpenCL) para el motor híbrido NPG de ZIT-HEXAGON.
package com.zithexagon.npgbench

import android.app.Activity
import android.app.ActivityManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Typeface
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.system.Os
import android.view.View
import android.view.WindowManager
import android.widget.*
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.concurrent.thread

class MainActivity : Activity() {
    companion object {
        const val VERSION = "0.1.2"
        var libLoaded = false
        var libError: String? = null
        var envReport = ""
        // survives activity recreation
        val live = StringBuilder()
        var finalReport = ""
        @Volatile var running = false
    }

    private lateinit var status: TextView
    private lateinit var progress: TextView
    private lateinit var reportView: TextView
    private lateinit var scroll: ScrollView
    private lateinit var runFull: Button
    private lateinit var runQuick: Button
    private lateinit var stopBtn: Button
    private lateinit var runT68: Button
    private lateinit var runT8: Button
    private lateinit var skipField: EditText
    private val ui = Handler(Looper.getMainLooper())
    private var wake: PowerManager.WakeLock? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setupNative()
        buildUi()
    }

    private fun setupNative() {
        if (libLoaded || libError != null) return
        val sb = StringBuilder()
        try {
            val dir = File(filesDir, "htp").apply { mkdirs() }
            for (name in assets.list("htp") ?: emptyArray()) {
                val out = File(dir, name)
                val bytes = assets.open("htp/$name").use { it.readBytes() }
                if (!out.exists() || out.length() != bytes.size.toLong()) FileOutputStream(out).use { it.write(bytes) }
                out.setReadable(true, false)
                sb.append("skel $name ${bytes.size} B; ")
            }
            val adsp = listOf(dir.absolutePath, applicationInfo.nativeLibraryDir,
                "/vendor/dsp/cdsp", "/vendor/lib/rfsa/adsp", "/system/lib/rfsa/adsp", "/vendor/dsp", "/dsp")
                .joinToString(";")
            Os.setenv("ADSP_LIBRARY_PATH", adsp, true)
            Os.setenv("LD_LIBRARY_PATH", applicationInfo.nativeLibraryDir + ":/vendor/lib64", true)
            System.loadLibrary("npgbench")
            libLoaded = true
        } catch (t: Throwable) {
            libError = t.toString(); sb.append("ERROR: $t")
        }
        envReport = sb.toString()
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
    private fun label(t: String, sz: Float = 14f) = TextView(this).apply { text = t; textSize = sz; setPadding(0, dp(4), 0, 0) }
    private fun row(vararg vs: View) = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        vs.forEach { addView(it, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)) }
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(10), dp(10), dp(10), dp(6)) }
        root.addView(label("NPG Bench $VERSION — Fase 2 (NPU + GPU)", 17f).apply { setTypeface(null, Typeface.BOLD) })
        root.addView(label("Pesos sintéticos FP8 con las formas de ZIT 1024² (no necesita modelos). " +
            "Completo ≈ 12–20 min, rápido ≈ 4–6 min. Deja el teléfono en reposo, sin funda, sin otras apps.", 12f))
        status = label(if (libLoaded) "Motor cargado. ${envReport}" else "Motor FALLÓ: $libError", 12f)
        root.addView(status)
        runFull = Button(this).apply { text = "Ejecutar todo"; setOnClickListener { start(false) } }
        runQuick = Button(this).apply { text = "Rápido"; setOnClickListener { start(true) } }
        stopBtn = Button(this).apply { text = "Detener"; setOnClickListener { if (libLoaded) NpgNative.cancel(); progress.text = "deteniendo…" } }
        runT68 = Button(this).apply { text = "Solo T6–T8"; setOnClickListener { start(false, onlyT6T8 = true) } }
        runT8 = Button(this).apply { text = "Solo T8"; setOnClickListener { start(false, onlyT8 = true) } }
        root.addView(row(runFull, runQuick, stopBtn))
        root.addView(row(runT68, runT8))
        root.addView(row(
            Button(this).apply { text = "Copiar"; setOnClickListener { copyReport() } },
            Button(this).apply { text = "Compartir"; setOnClickListener { shareReport() } },
            Button(this).apply { text = "Ver último"; setOnClickListener { showLast() } },
        ))
        skipField = EditText(this).apply {
            hint = "Omitir (solo si algo se cierra): ej. T6,T8 o HTP/GPU"; textSize = 12f; setSingleLine()
            setText(getSharedPreferences("npg", MODE_PRIVATE).getString("skip", ""))
        }
        root.addView(skipField)
        progress = label("", 13f).apply { setTypeface(null, Typeface.BOLD) }
        root.addView(progress)
        scroll = ScrollView(this)
        reportView = TextView(this).apply {
            typeface = Typeface.MONOSPACE; textSize = 10f; setTextIsSelectable(true)
            text = if (finalReport.isNotEmpty()) finalReport else live.toString()
        }
        scroll.addView(reportView)
        root.addView(scroll, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        setContentView(root)
        setButtons()
    }

    private fun setButtons() {
        runFull.isEnabled = libLoaded && !running
        runQuick.isEnabled = libLoaded && !running
        runT68.isEnabled = libLoaded && !running
        runT8.isEnabled = libLoaded && !running
        stopBtn.isEnabled = running
    }

    private fun deviceHeader(quick: Boolean): String {
        val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val bat = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val lvl = bat?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val plugged = (bat?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val btemp = (bat?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0
        val ts = SimpleDateFormat("yyyy-MM-dd HH:mm:ss Z", Locale.US).format(Date())
        return "=== NPG Bench $VERSION (${if (quick) "rápido" else "completo"}) — $ts\n" +
            "equipo: ${Build.MANUFACTURER} ${Build.MODEL} | SoC ${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL} | Android ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})\n" +
            "RAM total ${mi.totalMem shr 20} MB, libre ${mi.availMem shr 20} MB | batería $lvl% ${if (plugged) "cargando" else "sin cargador"} ${btemp}°C\n"
    }

    private fun thermalLine(pm: PowerManager, t0: Long): String {
        val st = pm.currentThermalStatus
        val hr = try { pm.getThermalHeadroom(10) } catch (_: Throwable) { Float.NaN }
        val bat = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val btemp = (bat?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0
        return String.format(Locale.US, "t=%4ds estado_termico=%d headroom10s=%.2f bat=%.1fC", (SystemClock.elapsedRealtime() - t0) / 1000, st, hr, btemp)
    }

    private fun appendLive(s: String) {
        synchronized(live) { live.append(s).append('\n') }
        ui.post {
            reportView.append(s + "\n")
            scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
        }
    }

    private fun start(quick: Boolean, onlyT6T8: Boolean = false, onlyT8: Boolean = false) {
        if (running || !libLoaded) return
        val typed = skipField.text.toString().trim().uppercase(Locale.US)
        getSharedPreferences("npg", MODE_PRIVATE).edit().putString("skip", typed).apply()
        // "Solo T6–T8": skip T1..T5 (≈6.5 min on the Z80 Ultra); T7 then builds its own HTP/GPU qkv.
        // "Solo T8": skip T1..T7 (the skip key is 2 chars, so T6b is skipped too).
        val skip = when {
            onlyT8 -> listOf("T1,T2,T3,T4,T5,T6,T7", typed).filter { it.isNotEmpty() }.joinToString(",")
            onlyT6T8 -> listOf("T1,T2,T3,T4,T5", typed).filter { it.isNotEmpty() }.joinToString(",")
            else -> typed
        }
        if (skip.isEmpty()) Os.unsetenv("NPG_SKIP") else Os.setenv("NPG_SKIP", skip, true)
        running = true; finalReport = ""
        synchronized(live) { live.setLength(0) }
        reportView.text = ""
        setButtons()
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wake = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "npgbench:run").apply { acquire(60 * 60 * 1000L) }
        val header = deviceHeader(quick) + (if (onlyT8) "modo: solo T8\n" else if (onlyT6T8) "modo: solo T6–T8\n" else "")
        appendLive(header)
        val thermal = StringBuilder()
        val t0 = SystemClock.elapsedRealtime()
        val sampler = thread(name = "npg-thermal") {
            try {
                while (running) {
                    thermal.append(thermalLine(pm, t0)).append('\n')
                    Thread.sleep(5000)
                }
            } catch (_: InterruptedException) {}
        }
        thread(name = "npg-run") {
            val out = File(getExternalFilesDir(null) ?: filesDir, "npg-report-native.txt")
            var rep: String
            try {
                rep = NpgNative.run(if (quick) 1 else 0, 60, 30, out.absolutePath, object : NpgCallback {
                    override fun onLine(line: String, progress: Boolean) {
                        if (progress) ui.post { this@MainActivity.progress.text = line.trim() }
                        else appendLive(line)
                    }
                })
            } catch (t: Throwable) {
                rep = synchronized(live) { live.toString() } + "\nERROR JNI: $t\n"
            }
            running = false
            sampler.interrupt()
            val full = header + rep + "\n==== Térmico (PowerManager, cada 5 s; estado 0=ninguno..6=apagado) ====\n" + thermal
            finalReport = full
            val ts = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
            val saved = File(getExternalFilesDir(null) ?: filesDir, "npg-report-$ts.txt")
            try { saved.writeText(full) } catch (_: Throwable) {}
            wake?.let { if (it.isHeld) it.release() }
            ui.post {
                reportView.text = full
                progress.text = "Terminado. Pulsa «Copiar» o «Compartir» y envía el reporte. (guardado en ${saved.path})"
                setButtons()
                scroll.post { scroll.fullScroll(View.FOCUS_UP) }
            }
        }
    }

    /** Shows the report of the last run (also when the app was closed by a crash mid-run). */
    private fun showLast() {
        if (running) return
        val dir = getExternalFilesDir(null) ?: filesDir
        val last = dir.listFiles { f -> f.name.startsWith("npg-report-2") }?.maxByOrNull { it.lastModified() }
        val nat = File(dir, "npg-report-native.txt")
        val pick = listOfNotNull(last, nat.takeIf { it.exists() }).maxByOrNull { it.lastModified() }
        if (pick == null) { Toast.makeText(this, "No hay reportes guardados", Toast.LENGTH_SHORT).show(); return }
        val txt = pick.readText()
        finalReport = if (pick == nat) "=== (reporte incompleto: la app se cerró a media prueba; esta es la última línea escrita) ===\n" + txt else txt
        reportView.text = finalReport
        progress.text = "Mostrando ${pick.name}"
    }

    private fun currentReport(): String = if (finalReport.isNotEmpty()) finalReport else synchronized(live) { live.toString() }

    private fun copyReport() {
        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText("NPG bench", currentReport()))
        Toast.makeText(this, "Reporte copiado", Toast.LENGTH_SHORT).show()
    }

    private fun shareReport() {
        val i = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, "NPG bench $VERSION")
            putExtra(Intent.EXTRA_TEXT, currentReport())
        }
        startActivity(Intent.createChooser(i, "Compartir reporte NPG"))
    }

    override fun onDestroy() {
        super.onDestroy()
        if (!running) wake?.let { if (it.isHeld) it.release() }
    }
}
