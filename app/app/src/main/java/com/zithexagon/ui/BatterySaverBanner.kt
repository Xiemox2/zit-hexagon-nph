package com.zithexagon.ui

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.PowerManager
import android.provider.Settings
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.BatterySaver
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.zithexagon.R
import com.zithexagon.ZitDiag

/** 0.25: Android's battery saver throttles the CPU/NPU clocks and can stretch a 1024² job a lot -> visible warning
 *  while it is on, with a shortcut to the system setting. Live: follows ACTION_POWER_SAVE_MODE_CHANGED. */
@Composable
fun BatterySaverBanner(modifier: Modifier = Modifier) {
    val ctx = LocalContext.current
    val pm = remember { ctx.getSystemService(Context.POWER_SERVICE) as PowerManager }
    var on by remember { mutableStateOf(pm.isPowerSaveMode) }
    DisposableEffect(Unit) {
        val r = object : BroadcastReceiver() {
            override fun onReceive(c: Context?, i: Intent?) { on = pm.isPowerSaveMode; ZitDiag.bc("battery saver ${if (on) "ON" else "off"}") }
        }
        // protected system broadcast: NOT_EXPORTED still receives it (required flag on targetSdk 34+)
        ContextCompat.registerReceiver(ctx, r, IntentFilter(PowerManager.ACTION_POWER_SAVE_MODE_CHANGED), ContextCompat.RECEIVER_NOT_EXPORTED)
        on = pm.isPowerSaveMode
        onDispose { runCatching { ctx.unregisterReceiver(r) } }
    }
    val a = LocalAccent.current
    AnimatedVisibility(on, modifier) {
        Surface(shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.35f),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.error.copy(alpha = 0.6f)), modifier = Modifier.fillMaxWidth()) {
            Row(Modifier.padding(start = 14.dp, end = 8.dp, top = 10.dp, bottom = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.BatterySaver, null, tint = MaterialTheme.colorScheme.error)
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.battery_saver_title), style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                    Text(stringResource(R.string.battery_saver_body), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                TextButton(onClick = {
                    val i = Intent(Settings.ACTION_BATTERY_SAVER_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    runCatching { ctx.startActivity(i) }.onFailure { runCatching { ctx.startActivity(Intent(Settings.ACTION_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) } }
                }, colors = ButtonDefaults.textButtonColors(contentColor = a.light)) { Text(stringResource(R.string.battery_saver_action)) }
            }
        }
    }
}
