package com.zithexagon.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Remove
import androidx.compose.material.icons.filled.RestartAlt
import androidx.compose.material.icons.outlined.WarningAmber
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.zithexagon.R
import com.zithexagon.data.LoraLimits
import java.util.Locale

val Amber = Color(0xFFFFB547)
private val AmberContainer = Color(0xFF2A1E08)

fun formatStrength(v: Float): String = String.format(Locale.US, "%+.2f", v).replace("+0.00", "0.00")

/**
 * LoRA strength: slider −3…3 snapping to 0.05, tappable value pill to type an exact value,
 * and a reset-to-1.0 button. Negative values are for "slider" LoRAs.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StrengthControl(value: Float, onChange: (Float) -> Unit, modifier: Modifier = Modifier, label: String? = null) {
    val a = LocalAccent.current
    var editing by remember { mutableStateOf(false) }
    Column(modifier) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(label ?: stringResource(R.string.lora_strength), style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(1f))
            Box(Modifier.clip(RoundedCornerShape(8.dp)).background(MaterialTheme.colorScheme.surfaceContainerHigh)
                .border(1.dp, a.deep, RoundedCornerShape(8.dp))
                .clickable { editing = true }.padding(horizontal = 10.dp, vertical = 3.dp)) {
                Text(formatStrength(value), fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
                    color = if (value < 0) Amber else a.light)
            }
            IconButton(onClick = { onChange(LoraLimits.DEFAULT_STRENGTH) }, enabled = value != LoraLimits.DEFAULT_STRENGTH,
                modifier = Modifier.size(36.dp)) {
                Icon(Icons.Filled.RestartAlt, stringResource(R.string.lora_strength_reset), Modifier.size(20.dp),
                    tint = if (value != LoraLimits.DEFAULT_STRENGTH) a.light else MaterialTheme.colorScheme.outline)
            }
        }
        val trackBg = MaterialTheme.colorScheme.surfaceContainerHighest
        val tick = MaterialTheme.colorScheme.outline
        val fill = if (value < 0) Amber else a.main
        Slider(value = value, onValueChange = { onChange(LoraLimits.snap(it)) },
            valueRange = LoraLimits.MIN_STRENGTH..LoraLimits.MAX_STRENGTH, modifier = Modifier.height(30.dp),
            colors = SliderDefaults.colors(thumbColor = if (value < 0) Amber else a.light),
            track = {
                // Bipolar track: filled from 0 towards the value (negative = amber, for slider LoRAs).
                Canvas(Modifier.fillMaxWidth().height(6.dp)) {
                    val r = size.height / 2
                    fun x(v: Float) = (v - LoraLimits.MIN_STRENGTH) / (LoraLimits.MAX_STRENGTH - LoraLimits.MIN_STRENGTH) * size.width
                    drawRoundRect(trackBg, cornerRadius = CornerRadius(r, r))
                    val x0 = x(0f); val xv = x(value)
                    drawRoundRect(fill, topLeft = Offset(minOf(x0, xv), 0f), size = Size(kotlin.math.abs(xv - x0), size.height), cornerRadius = CornerRadius(r, r))
                    drawLine(tick, Offset(x0, -3.dp.toPx()), Offset(x0, size.height + 3.dp.toPx()), strokeWidth = 1.5.dp.toPx())
                }
            })
        Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp)) {
            listOf("−3", "0", "+3").forEachIndexed { i, t ->
                if (i > 0) Spacer(Modifier.weight(1f))
                Text(t, fontSize = 10.sp, color = MaterialTheme.colorScheme.outline)
            }
        }
    }
    if (editing) StrengthDialog(value, onDismiss = { editing = false }, onConfirm = { onChange(it); editing = false })
}

@Composable
fun StrengthDialog(initial: Float, onDismiss: () -> Unit, onConfirm: (Float) -> Unit) {
    val a = LocalAccent.current
    var text by remember { mutableStateOf(String.format(Locale.US, "%.2f", initial)) }
    val parsed = text.replace(',', '.').replace('−', '-').toFloatOrNull()
    val valid = parsed != null && parsed in LoraLimits.MIN_STRENGTH..LoraLimits.MAX_STRENGTH
    fun bump(d: Float) { val v = LoraLimits.snap((parsed ?: initial) + d); text = String.format(Locale.US, "%.2f", v) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.lora_strength_exact)) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    FilledTonalIconButton(onClick = { bump(-LoraLimits.STRENGTH_STEP) }) { Icon(Icons.Filled.Remove, "−0.05") }
                    OutlinedTextField(text, { text = it.take(6) }, singleLine = true, isError = !valid,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                        modifier = Modifier.weight(1f).padding(horizontal = 8.dp), shape = RoundedCornerShape(12.dp), colors = fieldColors())
                    FilledTonalIconButton(onClick = { bump(LoraLimits.STRENGTH_STEP) }) { Icon(Icons.Filled.Add, "+0.05") }
                }
                Text(stringResource(R.string.lora_strength_range), fontSize = 12.sp,
                    color = if (valid) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.error)
            }
        },
        confirmButton = { TextButton(onClick = { parsed?.let { onConfirm(LoraLimits.snap(it)) } }, enabled = valid) { Text(stringResource(R.string.action_ok), color = a.light) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text(stringResource(R.string.action_cancel)) } },
        containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
    )
}

/** Non-blocking amber hint shown when more than the recommended number of LoRAs is active. */
@Composable
fun LoraCountWarning(active: Int, recommended: Int, modifier: Modifier = Modifier) {
    if (active <= recommended) return
    Row(modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(AmberContainer)
        .border(1.dp, Amber.copy(alpha = 0.45f), RoundedCornerShape(12.dp)).padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically) {
        Icon(Icons.Outlined.WarningAmber, null, tint = Amber, modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(8.dp))
        Text(stringResource(R.string.lora_many_warning, recommended), fontSize = 12.sp, color = Amber, lineHeight = 16.sp)
    }
}
