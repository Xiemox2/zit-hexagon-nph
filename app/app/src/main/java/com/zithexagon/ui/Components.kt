package com.zithexagon.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.LruCache
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/** Section card: near-black surface, hairline wine-tinted border, small caps title. */
@Composable
fun SectionCard(
    title: String? = null,
    modifier: Modifier = Modifier,
    trailing: (@Composable RowScope.() -> Unit)? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        color = MaterialTheme.colorScheme.surfaceContainerLow,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
    ) {
        Column(Modifier.padding(horizontal = 16.dp, vertical = 14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            if (title != null || trailing != null) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (title != null) Text(title.uppercase(), style = SectionLabel, color = LocalAccent.current.light,
                        modifier = Modifier.weight(1f))
                    else Spacer(Modifier.weight(1f))
                    trailing?.invoke(this)
                }
            }
            content()
        }
    }
}

/** Label + value header over a slider (common mobile UI pattern). */
@Composable
fun SliderRow(
    label: String,
    value: Float,
    valueText: String,
    range: ClosedFloatingPointRange<Float>,
    steps: Int = 0,
    enabled: Boolean = true,
    onChange: (Float) -> Unit,
) {
    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
            ValuePill(valueText)
        }
        Slider(value = value, onValueChange = onChange, valueRange = range, steps = steps, enabled = enabled,
            modifier = Modifier.height(36.dp),
            colors = SliderDefaults.colors(inactiveTrackColor = MaterialTheme.colorScheme.surfaceContainerHighest))
    }
}

@Composable
fun ValuePill(text: String, modifier: Modifier = Modifier) {
    Box(modifier.clip(RoundedCornerShape(8.dp)).background(MaterialTheme.colorScheme.surfaceContainerHigh)
        .padding(horizontal = 10.dp, vertical = 3.dp)) {
        Text(text, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = LocalAccent.current.light)
    }
}

@Composable
fun SwitchRow(title: String, subtitle: String? = null, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            subtitle?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
        }
        Switch(checked, onChange, colors = SwitchDefaults.colors(
            uncheckedTrackColor = MaterialTheme.colorScheme.surfaceContainerHighest,
            uncheckedBorderColor = MaterialTheme.colorScheme.outline))
    }
}

/** Trigger word chip: tap to add/remove it from the prompt; filled when it is in the prompt. */
@Composable
fun TriggerChip(text: String, selected: Boolean, onClick: () -> Unit) {
    val a = LocalAccent.current
    FilterChip(
        selected = selected, onClick = onClick,
        label = { Text(text, maxLines = 1, fontSize = 13.sp) },
        shape = RoundedCornerShape(10.dp),
        colors = FilterChipDefaults.filterChipColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
            labelColor = MaterialTheme.colorScheme.onSurface,
            selectedContainerColor = a.deep, selectedLabelColor = a.light,
        ),
        border = FilterChipDefaults.filterChipBorder(true, selected,
            borderColor = MaterialTheme.colorScheme.outline, selectedBorderColor = a.main),
    )
}

/** Small LRU of decoded thumbnails (key: path + size + mtime). */
object Thumbs {
    private val cache = object : LruCache<String, Bitmap>(64 * 1024 * 1024) {
        override fun sizeOf(key: String, value: Bitmap) = value.byteCount
    }
    fun key(f: File, max: Int) = "${f.path}:${f.length()}:${f.lastModified()}:$max"
    fun peek(f: File, max: Int): Bitmap? = cache.get(key(f, max))
    fun load(f: File, max: Int): Bitmap? {
        cache.get(key(f, max))?.let { return it }
        val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(f.path, o)
        if (o.outWidth <= 0) return null
        var s = 1
        while (o.outWidth / (s * 2) >= max || o.outHeight / (s * 2) >= max) s *= 2
        val b = BitmapFactory.decodeFile(f.path, BitmapFactory.Options().apply { inSampleSize = s }) ?: return null
        cache.put(key(f, max), b); return b
    }
}

@Composable
fun rememberThumb(file: File?, maxPx: Int = 512): ImageBitmap? {
    var img by remember(file?.path, file?.lastModified(), maxPx) {
        mutableStateOf(file?.let { Thumbs.peek(it, maxPx) }?.asImageBitmap())
    }
    LaunchedEffect(file?.path, file?.lastModified(), maxPx) {
        if (img == null && file != null) img = withContext(Dispatchers.IO) { Thumbs.load(file, maxPx) }?.asImageBitmap()
    }
    return img
}

/** Image from a file, or a wine gradient placeholder with an icon. */
@Composable
fun FileImage(file: File?, modifier: Modifier = Modifier, maxPx: Int = 512, contentScale: ContentScale = ContentScale.Crop,
              placeholderIcon: Dp = 28.dp) {
    val img = rememberThumb(file, maxPx)
    Box(modifier, contentAlignment = Alignment.Center) {
        if (img != null) Image(img, null, Modifier.fillMaxSize(), contentScale = contentScale)
        else PlaceholderArt(Modifier.fillMaxSize(), placeholderIcon)
    }
}

@Composable
fun PlaceholderArt(modifier: Modifier = Modifier, icon: Dp = 28.dp) {
    val a = LocalAccent.current
    Box(modifier.background(Brush.linearGradient(listOf(a.deep, Color(0xFF0B070A)))), contentAlignment = Alignment.Center) {
        if (icon > 0.dp) Icon(Icons.Outlined.Image, null, tint = a.light.copy(alpha = 0.55f), modifier = Modifier.size(icon))
    }
}

@Composable
fun EmptyState(icon: androidx.compose.ui.graphics.vector.ImageVector, title: String, body: String, modifier: Modifier = Modifier,
               action: (@Composable () -> Unit)? = null) {
    val a = LocalAccent.current
    Column(modifier.fillMaxWidth().padding(32.dp), horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Box(Modifier.size(72.dp).clip(RoundedCornerShape(24.dp)).background(a.deep)
            .border(1.dp, a.main.copy(alpha = 0.5f), RoundedCornerShape(24.dp)), contentAlignment = Alignment.Center) {
            Icon(icon, null, tint = a.light, modifier = Modifier.size(34.dp))
        }
        Text(title, style = MaterialTheme.typography.titleMedium)
        Text(body, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        action?.invoke()
    }
}

@Composable
fun ConfirmDialog(title: String, body: String, confirm: String, onConfirm: () -> Unit, onDismiss: () -> Unit, destructive: Boolean = true) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = { Text(body) },
        confirmButton = {
            TextButton(onClick = { onConfirm(); onDismiss() }) {
                Text(confirm, color = if (destructive) MaterialTheme.colorScheme.error else LocalAccent.current.light)
            }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text(androidx.compose.ui.res.stringResource(com.zithexagon.R.string.action_cancel)) } },
        containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
    )
}

/** Top header used by every tab: big title, optional subtitle and actions. */
@Composable
fun ScreenHeader(title: String, subtitle: String? = null, actions: @Composable RowScope.() -> Unit = {}) {
    Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp, top = 12.dp, bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.headlineSmall)
            subtitle?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1) }
        }
        actions()
    }
}

fun formatMs(ms: Long): String = if (ms < 60_000) String.format(java.util.Locale.US, "%.1f s", ms / 1000.0)
    else "${ms / 60_000} min ${(ms / 1000) % 60} s"
