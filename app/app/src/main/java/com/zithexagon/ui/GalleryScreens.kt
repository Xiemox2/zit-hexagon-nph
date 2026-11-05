package com.zithexagon.ui

import android.content.Context
import android.content.Intent
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import com.zithexagon.AppViewModel
import com.zithexagon.R
import com.zithexagon.data.GalleryItem
import com.zithexagon.data.humanBytes
import java.text.DateFormat
import java.util.Date

@Composable
fun GalleryScreen(vm: AppViewModel, onOpen: (GalleryItem) -> Unit, onGoGenerate: () -> Unit) {
    val items = vm.gallery
    Column(Modifier.fillMaxSize()) {
        val size = remember(items) { items.sumOf { it.file.length() } }
        ScreenHeader(stringResource(R.string.tab_gallery),
            if (items.isEmpty()) null else stringResource(R.string.gallery_count, items.size, size.humanBytes()))
        if (items.isEmpty()) {
            EmptyState(Icons.Outlined.PhotoLibrary, stringResource(R.string.gallery_empty_title), stringResource(R.string.gallery_empty_body),
                Modifier.padding(top = 48.dp), action = { Button(onClick = onGoGenerate) { Text(stringResource(R.string.action_generate)) } })
        } else LazyVerticalGrid(
            columns = GridCells.Adaptive(112.dp),
            contentPadding = PaddingValues(start = 12.dp, end = 12.dp, bottom = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            items(items, key = { it.id }) { item ->
                Box(Modifier.aspectRatio(1f).clip(RoundedCornerShape(14.dp)).clickable { onOpen(item) }) {
                    FileImage(item.file, Modifier.fillMaxSize(), maxPx = 384)
                    if (item.upscaleFactor > 0) Badge(Modifier.align(Alignment.TopEnd).padding(6.dp)) { Text("${item.upscaleFactor}×") }
                    if (item.params.loras.isNotEmpty()) Icon(Icons.Filled.Layers, null, tint = Color.White.copy(alpha = 0.85f),
                        modifier = Modifier.align(Alignment.BottomStart).padding(6.dp).size(16.dp))
                }
            }
        }
    }
}

@Composable
private fun Badge(modifier: Modifier, content: @Composable () -> Unit) {
    val a = LocalAccent.current
    Box(modifier.clip(RoundedCornerShape(6.dp)).background(a.main).padding(horizontal = 6.dp, vertical = 1.dp)) {
        ProvideTextStyle(MaterialTheme.typography.labelSmall.copy(color = Color.White, fontWeight = FontWeight.Bold)) { content() }
    }
}

fun shareImage(ctx: Context, item: GalleryItem) {
    val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".files", item.file)
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "image/png"; putExtra(Intent.EXTRA_STREAM, uri)
        putExtra(Intent.EXTRA_TEXT, item.params.prompt)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    ctx.startActivity(Intent.createChooser(send, ctx.getString(R.string.action_share)))
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun GalleryDetailScreen(vm: AppViewModel, itemId: String, onBack: () -> Unit, onReuse: () -> Unit) {
    val item = vm.gallery.firstOrNull { it.id == itemId }
    if (item == null) { LaunchedEffect(Unit) { onBack() }; return }
    val ctx = LocalContext.current
    val a = LocalAccent.current
    var confirmDelete by remember { mutableStateOf(false) }
    var pickLora by remember { mutableStateOf(false) }
    var upscaleMenu by remember { mutableStateOf(false) }
    var fullscreen by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxSize().statusBarsPadding()) {
        TopAppBar(
            title = { Text(stringResource(R.string.detail_title)) },
            navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.action_back)) } },
            actions = {
                IconButton(onClick = { shareImage(ctx, item) }) { Icon(Icons.Outlined.Share, stringResource(R.string.action_share)) }
                IconButton(onClick = { confirmDelete = true }) { Icon(Icons.Outlined.Delete, stringResource(R.string.action_delete)) }
            },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
            windowInsets = WindowInsets(0),
        )
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 16.dp).navigationBarsPadding(),
            verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Box(Modifier.fillMaxWidth().aspectRatio((item.width / item.height.toFloat()).coerceIn(0.5f, 2f))
                .clip(RoundedCornerShape(20.dp)).clickable { fullscreen = true }) {
                FileImage(item.file, Modifier.fillMaxSize(), maxPx = 1536, contentScale = ContentScale.Fit)
                vm.job?.takeIf { it.kind == com.zithexagon.JobUi.Kind.UPSCALE }?.let { j ->
                    Box(Modifier.fillMaxSize().background(Color(0x99000000)), contentAlignment = Alignment.Center) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(progress = { j.fraction }, color = a.light)
                            Spacer(Modifier.height(8.dp)); Text(stringResource(R.string.stage_upscale))
                        }
                    }
                }
            }
            // Actions
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { vm.reuseParams(item); onReuse() }) {
                    Icon(Icons.Filled.Replay, null, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp)); Text(stringResource(R.string.detail_reuse))
                }
                Box {
                    OutlinedButton(onClick = { upscaleMenu = true }, enabled = vm.job == null, border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline)) {
                        Icon(Icons.Outlined.ZoomOutMap, null, Modifier.size(18.dp), tint = a.light); Spacer(Modifier.width(6.dp))
                        Text(stringResource(R.string.detail_upscale), color = MaterialTheme.colorScheme.onSurface)
                    }
                    DropdownMenu(upscaleMenu, { upscaleMenu = false }, containerColor = MaterialTheme.colorScheme.surfaceContainerHigh) {
                        listOf(2, 4).forEach { f ->
                            DropdownMenuItem(text = { Text(stringResource(R.string.detail_upscale_to, f, item.width * f, item.height * f)) },
                                onClick = { upscaleMenu = false; vm.upscale(item, f) })
                        }
                    }
                }
                OutlinedButton(onClick = { pickLora = true }, border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline)) {
                    Icon(Icons.Outlined.Layers, null, Modifier.size(18.dp), tint = a.light); Spacer(Modifier.width(6.dp))
                    Text(stringResource(R.string.detail_use_as_preview), color = MaterialTheme.colorScheme.onSurface)
                }
                OutlinedButton(onClick = { vm.saveToPhone(item) }, border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline)) {
                    Icon(Icons.Outlined.Download, null, Modifier.size(18.dp), tint = a.light); Spacer(Modifier.width(6.dp))
                    Text(stringResource(R.string.detail_save_phone), color = MaterialTheme.colorScheme.onSurface)
                }
            }
            InfoCard(item)
            Spacer(Modifier.height(16.dp))
        }
    }
    if (confirmDelete) ConfirmDialog(stringResource(R.string.detail_delete_title), stringResource(R.string.detail_delete_body),
        stringResource(R.string.action_delete), onConfirm = { vm.deleteItem(item); onBack() }, onDismiss = { confirmDelete = false })
    if (pickLora) PickLoraDialog(vm, onPick = { vm.setLoraPreviewFromGallery(it, item); pickLora = false }, onDismiss = { pickLora = false })
    if (fullscreen) ZoomDialog(item) { fullscreen = false }
}

@Composable
private fun InfoCard(item: GalleryItem) {
    val p = item.params
    val clip = LocalClipboardManager.current
    SectionCard(stringResource(R.string.gen_prompt_section), trailing = {
        IconButton(onClick = { clip.setText(AnnotatedString(p.prompt)) }, Modifier.size(32.dp)) {
            Icon(Icons.Outlined.ContentCopy, stringResource(R.string.action_copy), Modifier.size(18.dp))
        }
    }) {
        SelectionContainer { Text(p.prompt.ifBlank { "—" }, style = MaterialTheme.typography.bodyMedium) }
        if (p.negativePrompt.isNotBlank()) {
            Text(stringResource(R.string.gen_negative).uppercase(), style = SectionLabel, color = MaterialTheme.colorScheme.onSurfaceVariant)
            SelectionContainer { Text(p.negativePrompt, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant) }
        }
    }
    Spacer(Modifier.height(12.dp))
    SectionCard(stringResource(R.string.gen_params_section)) {
        val rows = buildList {
            add(stringResource(R.string.label_model) to p.modelName.ifBlank { p.modelId })
            add(stringResource(R.string.label_size) to "${item.width}×${item.height}" + if (item.upscaleFactor > 0) " (${item.upscaleFactor}×)" else "")
            add(stringResource(R.string.label_steps) to p.steps.toString())
            add(stringResource(R.string.label_cfg) to String.format(java.util.Locale.US, "%.1f", p.cfg))
            add(stringResource(R.string.label_seed) to p.seed.toString())
            add(stringResource(R.string.label_sampler) to "${p.sampler.label} · ${p.scheduler.label}")
            if (p.upscale > 0 && item.upscaleFactor == 0) add(stringResource(R.string.label_upscale) to "${p.upscale}×")
            add(stringResource(R.string.label_date) to DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(item.createdAt)))
        }
        rows.forEach { (k, v) ->
            Row { Text(k, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.width(110.dp), fontSize = 14.sp)
                Text(v, fontSize = 14.sp, modifier = Modifier.weight(1f)) }
        }
        run {
            var showPerf by remember { mutableStateOf(false) }
            val perf = item.perf
            Row(if (perf != null) Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).clickable { showPerf = true } else Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(R.string.label_time), color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.width(110.dp), fontSize = 14.sp)
                Text(formatMs(item.totalMs), fontSize = 14.sp)
                if (perf != null) {
                    Spacer(Modifier.width(10.dp))
                    Text(stringResource(R.string.perf_details) + " ›", fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
                }
            }
            if (showPerf && perf != null) PerfDetailsDialog(perf) { showPerf = false }
        }
        if (p.loras.isNotEmpty()) {
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Text(stringResource(R.string.tab_loras).uppercase(), style = SectionLabel, color = MaterialTheme.colorScheme.onSurfaceVariant)
            p.loras.forEach { Row { Text(it.name, Modifier.weight(1f), fontSize = 14.sp); ValuePill(formatStrength(it.strength)) } }
        }
    }
}

@Composable
fun PickLoraDialog(vm: AppViewModel, onPick: (String) -> Unit, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.detail_pick_lora)) },
        text = {
            if (vm.loras.isEmpty()) Text(stringResource(R.string.loras_empty_body))
            else LazyColumn(Modifier.heightIn(max = 420.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                items(vm.loras, key = { it.id }) { e ->
                    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).clickable { onPick(e.id) }.padding(6.dp),
                        verticalAlignment = Alignment.CenterVertically) {
                        FileImage(e.preview, Modifier.size(44.dp).clip(RoundedCornerShape(10.dp)), maxPx = 160, placeholderIcon = 18.dp)
                        Spacer(Modifier.width(12.dp))
                        Text(e.name, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text(stringResource(R.string.action_cancel)) } },
        containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
    )
}

@Composable
private fun ZoomDialog(item: GalleryItem, onDismiss: () -> Unit) {
    androidx.compose.ui.window.Dialog(onDismiss, androidx.compose.ui.window.DialogProperties(usePlatformDefaultWidth = false)) {
        var scale by remember { mutableFloatStateOf(1f) }
        var offset by remember { mutableStateOf(androidx.compose.ui.geometry.Offset.Zero) }
        val state = rememberTransformableState { z, pan, _ -> scale = (scale * z).coerceIn(1f, 8f); offset += pan }
        Box(Modifier.fillMaxSize().background(Color.Black).transformable(state)
            .pointerInput(Unit) { detectTapGestures(onDoubleTap = { scale = if (scale > 1f) 1f else 2.5f; offset = androidx.compose.ui.geometry.Offset.Zero }, onTap = { onDismiss() }) }) {
            FileImage(item.file, Modifier.fillMaxSize().graphicsLayer { scaleX = scale; scaleY = scale; translationX = offset.x; translationY = offset.y },
                maxPx = 4096, contentScale = ContentScale.Fit)
        }
    }
}
