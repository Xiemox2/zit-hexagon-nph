package com.zithexagon

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModelProvider
import com.zithexagon.ui.ZitTheme

class MainActivity : ComponentActivity() {
    private var vm: AppViewModel? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
        )
        super.onCreate(savedInstanceState)
        // Never let a startup failure kill the app: record it and show it with a share button instead.
        val created = runCatching { ViewModelProvider(this)[AppViewModel::class.java] }
        created.exceptionOrNull()?.let { CrashReporter.record(applicationContext, it, "inicio") }
        vm = created.getOrNull()
        handleIntent(intent)
        askNotificationPermission()
        setContent {
            val v = vm
            if (v != null) {
                ZitApp(v)
                CrashDialog()
            } else {
                ZitTheme { StartupErrorScreen(CrashReporter.report(applicationContext) ?: created.exceptionOrNull().toString()) }
            }
        }
    }

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(i: android.content.Intent?) {
        val id = i?.getStringExtra(EXTRA_OPEN_ITEM) ?: return
        i.removeExtra(EXTRA_OPEN_ITEM)
        vm?.let { it.refreshAll(); it.openItemRequest = id }
    }

    /** Android 13+: the "inferencia" notification (progress, Cancelar/Liberar, Imagen lista) needs this. Asked once. */
    private val notifPerm = registerForActivityResult(androidx.activity.result.contract.ActivityResultContracts.RequestPermission()) { ok ->
        ZitDiag.bc("POST_NOTIFICATIONS granted=$ok")
    }
    private fun askNotificationPermission() {
        if (androidx.core.content.ContextCompat.checkSelfPermission(this, android.Manifest.permission.POST_NOTIFICATIONS) ==
            android.content.pm.PackageManager.PERMISSION_GRANTED) return
        val sp = getSharedPreferences("zit_ui", MODE_PRIVATE)
        if (sp.getBoolean("asked_notif", false)) return
        sp.edit().putBoolean("asked_notif", true).apply()
        runCatching { notifPerm.launch(android.Manifest.permission.POST_NOTIFICATIONS) }
    }

    companion object { const val EXTRA_OPEN_ITEM = "open_item" }

    override fun onResume() {
        super.onResume()
        vm?.refreshAll()   // files may have been copied over USB meanwhile
    }
}

/** Shown on the launch after a crash: offers to share the saved report. */
@Composable
fun CrashDialog() {
    val ctx = LocalContext.current
    var show by remember { mutableStateOf(CrashReporter.hasUnseen(ctx)) }
    if (!show) return
    val text = remember { CrashReporter.report(ctx).orEmpty() }
    AlertDialog(
        onDismissRequest = { CrashReporter.markSeen(ctx); show = false },
        title = { Text(stringResource(R.string.crash_title)) },
        text = { Text(stringResource(R.string.crash_body)) },
        confirmButton = { TextButton(onClick = { CrashReporter.markSeen(ctx); show = false; CrashReporter.share(ctx, text) }) { Text(stringResource(R.string.crash_share)) } },
        dismissButton = { TextButton(onClick = { CrashReporter.markSeen(ctx); show = false }) { Text(stringResource(R.string.crash_dismiss)) } },
    )
}

@Composable
private fun StartupErrorScreen(report: String) {
    val ctx = LocalContext.current
    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Column(Modifier.fillMaxSize().systemBarsPadding().padding(20.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(stringResource(R.string.crash_startup_title), style = MaterialTheme.typography.titleLarge)
            Text(stringResource(R.string.crash_startup_body), color = MaterialTheme.colorScheme.onSurfaceVariant)
            Button(onClick = { CrashReporter.share(ctx, report) }) { Text(stringResource(R.string.crash_share)) }
            Text(report, fontFamily = FontFamily.Monospace, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}
