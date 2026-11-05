package com.zithexagon.ui

import android.content.Context
import android.content.Intent
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.zithexagon.R

/** Per-image time breakdown ("Detalles de tiempo") with a share button. */
@Composable
fun PerfDetailsDialog(perf: String, title: String? = null, subject: String = "ZIT-HEXAGON — tiempos de generación", onDismiss: () -> Unit) {
    val ctx = LocalContext.current
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title ?: stringResource(R.string.perf_details)) },
        text = {
            SelectionContainer(Modifier.heightIn(max = 460.dp).verticalScroll(rememberScrollState())) {
                Text(perf.trimEnd(), fontFamily = FontFamily.Monospace, fontSize = 11.sp, lineHeight = 15.sp)
            }
        },
        confirmButton = { TextButton(onClick = { sharePerfText(ctx, perf, subject) }) { Text(stringResource(R.string.perf_share)) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text(stringResource(R.string.perf_close)) } },
    )
}

fun sharePerfText(ctx: Context, text: String, subject: String = "ZIT-HEXAGON — tiempos de generación") {
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "text/plain"
        putExtra(Intent.EXTRA_SUBJECT, subject)
        putExtra(Intent.EXTRA_TEXT, if (text.length > 60_000) text.takeLast(60_000) else text)
    }
    runCatching { ctx.startActivity(Intent.createChooser(send, ctx.getString(R.string.perf_share)).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) }
}
