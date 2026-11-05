package com.zithexagon.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.zithexagon.AppViewModel
import com.zithexagon.JobUi
import com.zithexagon.R
import com.zithexagon.data.GalleryItem
import com.zithexagon.data.LoraEntry
import com.zithexagon.data.PromptKeywords
import com.zithexagon.data.Resolution
import com.zithexagon.data.UPSCALERS
import com.zithexagon.engine.EngineState
import com.zithexagon.engine.Sampler
import com.zithexagon.engine.Scheduler
import com.zithexagon.data.UpscalerKind
import com.zithexagon.data.UpscalerModel
import com.zithexagon.engine.Stage
import kotlinx.coroutines.delay
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun GenerateScreen(vm: AppViewModel, onOpenResult: (GalleryItem) -> Unit, onGoModels: () -> Unit, onGoLoras: () -> Unit) {
    val p = vm.params
    var showLoraPicker by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxSize()) {
        GenerateHeader(vm, onGoModels)
        Column(
            Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            BatterySaverBanner()
            ResultCard(vm, onOpenResult)
            PromptCard(vm)
            LoraCard(vm, onAdd = { showLoraPicker = true })
            ParamsCard(vm)
            UpscaleCard(vm)
            Spacer(Modifier.height(8.dp))
        }
        GenerateBar(vm)
    }
    if (showLoraPicker) LoraPickerSheet(vm, onDismiss = { showLoraPicker = false }, onManage = { showLoraPicker = false; onGoLoras() })
}

@Composable
private fun GenerateHeader(vm: AppViewModel, onGoModels: () -> Unit) {
    val a = LocalAccent.current
    var menu by remember { mutableStateOf(false) }
    Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 12.dp, top = 12.dp, bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(stringResource(R.string.app_name), style = MaterialTheme.typography.headlineSmall)
            Box {
                Row(Modifier.clip(RoundedCornerShape(8.dp)).clickable { menu = true }.padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    EngineDot(vm.engineState)
                    Spacer(Modifier.width(6.dp))
                    Text(vm.selectedModel.name, style = MaterialTheme.typography.bodyMedium, color = a.light,
                        maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.widthIn(max = 220.dp))
                    Icon(Icons.Filled.ArrowDropDown, null, tint = a.light)
                }
                DropdownMenu(menu, { menu = false }, containerColor = MaterialTheme.colorScheme.surfaceContainerHigh) {
                    vm.models.forEach { m ->
                        DropdownMenuItem(
                            text = { Text(m.name) },
                            leadingIcon = { if (m.id == vm.selectedModelId) Icon(Icons.Filled.Check, null, tint = a.light) else Spacer(Modifier.size(24.dp)) },
                            onClick = { vm.selectModel(m); menu = false })
                    }
                    if (vm.canRelease) {
                        HorizontalDivider()
                        DropdownMenuItem(text = { Text(stringResource(R.string.action_release)) },
                            leadingIcon = { Icon(Icons.Filled.Memory, null) },
                            onClick = { menu = false; vm.releaseModel() })
                    }
                    HorizontalDivider()
                    DropdownMenuItem(text = { Text(stringResource(R.string.gen_manage_models)) },
                        leadingIcon = { Icon(Icons.Outlined.ViewInAr, null) }, onClick = { menu = false; onGoModels() })
                }
            }
        }
        AssistChip(onClick = {}, label = { Text(engineLabel(vm), fontSize = 12.sp) },
            leadingIcon = { Icon(Icons.Outlined.Memory, null, Modifier.size(16.dp)) },
            colors = AssistChipDefaults.assistChipColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
                labelColor = MaterialTheme.colorScheme.onSurfaceVariant, leadingIconContentColor = a.light),
            border = null)
    }
}

@Composable
private fun engineLabel(vm: AppViewModel): String {
    // Honor the *selected* model so we never show "Motor simulado" while ReaZIT FP8 (NPH) is selected
    // (EngineRouter may still be sitting on FakeEngine until the first successful load).
    val id = vm.selectedModel.id
    if (id.startsWith("nph-")) {
        val backend = (vm.engine as? com.zithexagon.engine.EngineRouter)?.infoFor(id)?.backend
            ?: "NPH · Hexagon HTP (sd.cpp)"
        return backend
    }
    if (id.startsWith("demo") || vm.engine.info.isFake) return stringResource(R.string.engine_fake)
    return vm.engine.info.backend
}

@Composable
private fun EngineDot(st: EngineState) {
    val c = when (st) {
        is EngineState.Ready -> Color(0xFF6FD08C)
        is EngineState.Loading, is EngineState.Busy -> Color(0xFFF2C36B)
        is EngineState.Error -> MaterialTheme.colorScheme.error
        EngineState.Idle -> MaterialTheme.colorScheme.outline
    }
    Box(Modifier.size(8.dp).clip(CircleShape).background(c))
}

@Composable
private fun ResultCard(vm: AppViewModel, onOpen: (GalleryItem) -> Unit) {
    val a = LocalAccent.current
    val job = vm.job
    val result = vm.lastResult
    val ratio = when {
        job?.preview != null -> job.preview.width / job.preview.height.toFloat()
        result != null && job == null -> result.width / result.height.toFloat().coerceAtLeast(1f)
        else -> vm.params.width / vm.params.height.toFloat()
    }.coerceIn(0.56f, 1.8f)
    Box(
        Modifier.fillMaxWidth().aspectRatio(ratio).heightIn(max = 460.dp).clip(RoundedCornerShape(24.dp))
            .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(24.dp))
            .background(MaterialTheme.colorScheme.surfaceContainerLow)
            .clickable(enabled = job == null && result != null) { result?.let(onOpen) },
        contentAlignment = Alignment.Center,
    ) {
        when {
            job?.preview != null -> Image(job.preview.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
            job == null && result != null -> FileImage(result.file, Modifier.fillMaxSize(), maxPx = 1024)
            else -> {
                PlaceholderArt(Modifier.fillMaxSize(), icon = 0.dp)
                if (job == null) Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(Icons.Outlined.AutoAwesome, null, tint = a.light, modifier = Modifier.size(40.dp))
                    Spacer(Modifier.height(8.dp))
                    Text(stringResource(R.string.gen_result_empty), color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
        if (job != null) JobOverlay(job)
        else if (result != null) {
            Row(Modifier.align(Alignment.BottomStart).fillMaxWidth()
                .background(Brush.verticalGradient(listOf(Color.Transparent, Color(0xCC000000))))
                .padding(horizontal = 14.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                var showPerf by remember(result.id) { mutableStateOf(false) }
                Text("${result.width}×${result.height} · ${stringResource(R.string.label_seed)} ${result.params.seed} · ",
                    fontSize = 12.sp, color = Color(0xFFE6DCE2), maxLines = 1)
                Text(formatMs(result.totalMs) + if (result.perf != null) " ⓘ" else "", fontSize = 12.sp, color = Color(0xFFE6DCE2), maxLines = 1,
                    textDecoration = if (result.perf != null) androidx.compose.ui.text.style.TextDecoration.Underline else null,
                    modifier = if (result.perf != null) Modifier.clickable { showPerf = true } else Modifier)
                Spacer(Modifier.weight(1f))
                if (showPerf && result.perf != null) PerfDetailsDialog(result.perf) { showPerf = false }
                Icon(Icons.Outlined.OpenInFull, null, tint = Color(0xFFE6DCE2), modifier = Modifier.size(16.dp))
            }
        }
    }
}

@Composable
fun stageLabel(stage: Stage): String = stringResource(when (stage) {
    Stage.LOADING -> R.string.stage_loading
    Stage.TEXT_ENCODER -> R.string.stage_text_encoder
    Stage.DENOISE -> R.string.stage_denoise
    Stage.VAE_DECODE -> R.string.stage_vae
    Stage.UPSCALE -> R.string.stage_upscale
})

@Composable
private fun BoxScope.JobOverlay(job: JobUi) {
    var now by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(job.startedAt) { while (true) { now = System.currentTimeMillis(); delay(200) } }
    val progress by animateFloatAsState(job.fraction, label = "p")
    Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth()
        .background(Brush.verticalGradient(listOf(Color.Transparent, Color(0xE6000000))))
        .padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(stageLabel(job.stage) + (if (job.stage == Stage.LOADING && job.loadDetail != null) " · ${job.loadDetail}" else ""),
                fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f), maxLines = 1, overflow = TextOverflow.Ellipsis)
            if (job.total > 0) ValuePill(stringResource(R.string.gen_step_of, job.step, job.total))
        }
        LinearProgressIndicator(progress = { progress }, Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
            trackColor = Color(0x33FFFFFF), drawStopIndicator = {})
        Text(stringResource(R.string.gen_elapsed, formatMs(now - job.startedAt)), fontSize = 12.sp, color = Color(0xFFCFC3CA))
    }
}

@Composable
private fun PromptCard(vm: AppViewModel) {
    val p = vm.params
    SectionCard(stringResource(R.string.gen_prompt_section), trailing = {
        if (p.prompt.isNotEmpty()) TextButton(onClick = { vm.updateParams { it.copy(prompt = "") } }, contentPadding = PaddingValues(horizontal = 8.dp)) {
            Text(stringResource(R.string.action_clear), fontSize = 13.sp)
        }
    }) {
        OutlinedTextField(
            value = p.prompt, onValueChange = { v -> vm.updateParams { it.copy(prompt = v) } },
            modifier = Modifier.fillMaxWidth().heightIn(min = 112.dp),
            placeholder = { Text(stringResource(R.string.gen_prompt_hint)) },
            shape = RoundedCornerShape(14.dp), colors = fieldColors(),
        )
        Text(stringResource(R.string.gen_negative), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        OutlinedTextField(
            value = p.negativePrompt, onValueChange = { v -> vm.updateParams { it.copy(negativePrompt = v) } },
            modifier = Modifier.fillMaxWidth().heightIn(min = 72.dp),
            placeholder = { Text(stringResource(R.string.gen_negative_hint)) },
            shape = RoundedCornerShape(14.dp), colors = fieldColors(),
        )
    }
}

@Composable
fun fieldColors() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor = LocalAccent.current.main,
    unfocusedBorderColor = MaterialTheme.colorScheme.outline,
    focusedContainerColor = MaterialTheme.colorScheme.surfaceContainer,
    unfocusedContainerColor = MaterialTheme.colorScheme.surfaceContainer,
    cursorColor = LocalAccent.current.light,
    focusedLabelColor = LocalAccent.current.light,
)

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun LoraCard(vm: AppViewModel, onAdd: () -> Unit) {
    val active = vm.params.loras
    val a = LocalAccent.current
    val mf = vm.selectedModel.manifest
    if (!mf.loraSupported) {
        SectionCard(stringResource(R.string.gen_loras_soon_title)) {
            Text(stringResource(R.string.gen_loras_soon_body), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        return
    }
    Text(
        stringResource(R.string.nph_lora_reload_note),
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(horizontal = 4.dp, vertical = 4.dp),
    )
    SectionCard(stringResource(R.string.gen_loras_section, active.size, mf.maxLoras), trailing = {
        FilledTonalButton(onClick = onAdd, contentPadding = PaddingValues(horizontal = 12.dp), modifier = Modifier.height(34.dp),
            colors = ButtonDefaults.filledTonalButtonColors(containerColor = a.deep, contentColor = a.light)) {
            Icon(Icons.Filled.Add, null, Modifier.size(18.dp)); Spacer(Modifier.width(4.dp)); Text(stringResource(R.string.action_add), fontSize = 13.sp)
        }
    }) {
        LoraCountWarning(active.size, mf.recommendedLoras)
        if (active.isEmpty()) {
            Text(stringResource(R.string.gen_loras_empty), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        active.forEachIndexed { i, use ->
            val e = vm.loras.firstOrNull { it.id == use.id }
            if (i > 0) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Column(Modifier.animateContentSize(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    FileImage(e?.preview, Modifier.size(44.dp).clip(RoundedCornerShape(10.dp)), maxPx = 160, placeholderIcon = 18.dp)
                    Spacer(Modifier.width(10.dp))
                    Column(Modifier.weight(1f)) {
                        Text(e?.name ?: use.name, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text(if (e == null) stringResource(R.string.lora_missing) else stringResource(R.string.gen_lora_swap_note),
                            fontSize = 12.sp, color = if (e == null) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    IconButton(onClick = { e?.let(vm::toggleLora) ?: vm.updateParams { p -> p.copy(loras = p.loras.filterNot { it.id == use.id }) } }) {
                        Icon(Icons.Filled.Close, stringResource(R.string.action_remove), tint = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                StrengthControl(use.strength, { vm.setActiveLoraStrength(use.id, it) })
                if (e != null && e.triggerWords.isNotEmpty()) {
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(0.dp)) {
                        e.triggerWords.forEach { kw -> TriggerChip(kw, PromptKeywords.contains(vm.params.prompt, kw)) { vm.toggleKeyword(kw) } }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
private fun LoraPickerSheet(vm: AppViewModel, onDismiss: () -> Unit, onManage: () -> Unit) {
    val a = LocalAccent.current
    ModalBottomSheet(onDismissRequest = onDismiss, containerColor = MaterialTheme.colorScheme.surfaceContainerLow) {
        Row(Modifier.padding(horizontal = 20.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(stringResource(R.string.gen_pick_loras), style = MaterialTheme.typography.titleLarge)
                Text(stringResource(R.string.gen_loras_section, vm.params.loras.size, vm.selectedModel.manifest.maxLoras),
                    fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            TextButton(onClick = onManage) { Text(stringResource(R.string.gen_manage_loras), color = a.light) }
        }
        LoraCountWarning(vm.params.loras.size, vm.selectedModel.manifest.recommendedLoras,
            Modifier.padding(horizontal = 16.dp, vertical = 8.dp))
        if (vm.loras.isEmpty()) {
            EmptyState(Icons.Outlined.Layers, stringResource(R.string.loras_empty_title), stringResource(R.string.loras_empty_body),
                action = { Button(onClick = onManage) { Text(stringResource(R.string.gen_manage_loras)) } })
        } else LazyColumn(contentPadding = PaddingValues(start = 16.dp, end = 16.dp, bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(vm.loras, key = { it.id }) { e ->
                val on = vm.isLoraActive(e.id)
                Surface(onClick = { vm.toggleLora(e) }, shape = RoundedCornerShape(16.dp),
                    color = if (on) a.deep else MaterialTheme.colorScheme.surfaceContainer,
                    border = BorderStroke(1.dp, if (on) a.main else MaterialTheme.colorScheme.outlineVariant)) {
                    Row(Modifier.padding(10.dp), verticalAlignment = Alignment.CenterVertically) {
                        FileImage(e.preview, Modifier.size(56.dp).clip(RoundedCornerShape(12.dp)), maxPx = 200, placeholderIcon = 20.dp)
                        Spacer(Modifier.width(12.dp))
                        Column(Modifier.weight(1f)) {
                            Text(e.name, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text(e.triggerWords.take(4).joinToString(" · ").ifEmpty { stringResource(R.string.lora_no_triggers) },
                                fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                        Checkbox(on, { vm.toggleLora(e) }, colors = CheckboxDefaults.colors(checkedColor = a.main))
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ParamsCard(vm: AppViewModel) {
    val p = vm.params
    val a = LocalAccent.current
    SectionCard(stringResource(R.string.gen_params_section)) {
        SliderRow(stringResource(R.string.label_steps), p.steps.toFloat(), p.steps.toString(), 1f..30f, steps = 28) { v ->
            vm.updateParams { it.copy(steps = v.roundToInt()) }
        }
        SliderRow(stringResource(R.string.label_cfg), p.cfg, String.format(java.util.Locale.US, "%.1f", p.cfg), 1f..5f, steps = 39) { v ->
            vm.updateParams { it.copy(cfg = (v * 10).roundToInt() / 10f) }
        }
        // Seed
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = if (vm.randomSeed) "" else (if (p.seed < 0) "" else p.seed.toString()),
                onValueChange = { s -> s.filter(Char::isDigit).take(12).toLongOrNull()?.let { v -> vm.updateParams { it.copy(seed = v) } }
                    ?: if (s.isEmpty()) vm.updateParams { it.copy(seed = -1) } else Unit },
                enabled = !vm.randomSeed,
                label = { Text(stringResource(R.string.label_seed)) },
                placeholder = { Text(stringResource(if (vm.randomSeed) R.string.seed_random else R.string.seed_hint)) },
                singleLine = true, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.weight(1f), shape = RoundedCornerShape(14.dp), colors = fieldColors(),
            )
            Spacer(Modifier.width(8.dp))
            FilledTonalIconButton(onClick = { vm.setRandomSeed(false); vm.newSeed() },
                colors = IconButtonDefaults.filledTonalIconButtonColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHigh, contentColor = a.light)) {
                Icon(Icons.Outlined.Casino, stringResource(R.string.seed_new))
            }
            FilterChip(selected = vm.randomSeed, onClick = { vm.setRandomSeed(!vm.randomSeed) },
                label = { Text(stringResource(R.string.seed_random)) },
                leadingIcon = { Icon(if (vm.randomSeed) Icons.Filled.Shuffle else Icons.Filled.Lock, null, Modifier.size(16.dp)) },
                colors = FilterChipDefaults.filterChipColors(selectedContainerColor = a.deep, selectedLabelColor = a.light, selectedLeadingIconColor = a.light))
        }
        // Resolution: fixed per package since 0.25 (graphs compiled for one size only) -> plain label, no picker
        val fixed = vm.fixedResolution
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(R.string.label_resolution), style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
            Text(stringResource(R.string.resolution_fixed, fixed.width, fixed.height), style = MaterialTheme.typography.bodyMedium,
                color = a.light, fontWeight = FontWeight.SemiBold)
        }
        // Sampler
        var open by remember { mutableStateOf(false) }
        ExposedDropdownMenuBox(open, { open = it }) {
            OutlinedTextField(
                value = p.sampler.label, onValueChange = {}, readOnly = true,
                label = { Text(stringResource(R.string.label_sampler)) },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(open) },
                modifier = Modifier.fillMaxWidth().menuAnchor(ExposedDropdownMenuAnchorType.PrimaryNotEditable),
                shape = RoundedCornerShape(14.dp), colors = fieldColors(),
            )
            ExposedDropdownMenu(open, { open = false }, containerColor = MaterialTheme.colorScheme.surfaceContainerHigh) {
                Sampler.entries.forEach { s ->
                    DropdownMenuItem(text = { Text(s.label) }, onClick = { vm.updateParams { it.copy(sampler = s) }; open = false },
                        trailingIcon = { if (s == p.sampler) Icon(Icons.Filled.Check, null, tint = a.light) })
                }
            }
        }
        // Scheduler (sigma schedule)
        Text(stringResource(R.string.label_scheduler), style = MaterialTheme.typography.bodyMedium)
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            Scheduler.entries.forEachIndexed { i, sc ->
                SegmentedButton(selected = p.scheduler == sc, onClick = { vm.updateParams { it.copy(scheduler = sc) } },
                    shape = SegmentedButtonDefaults.itemShape(i, Scheduler.entries.size),
                    colors = SegmentedButtonDefaults.colors(activeContainerColor = a.deep, activeContentColor = a.light,
                        inactiveContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh, activeBorderColor = a.main,
                        inactiveBorderColor = MaterialTheme.colorScheme.outline),
                    icon = {}) { Text(sc.label, fontSize = 13.sp, maxLines = 1) }
            }
        }
        Text(stringResource(R.string.label_exec_mode), style = MaterialTheme.typography.bodyMedium)
        ExecModeSelector(vm.execMode, vm::setExecMode)
    }
}

/** Extremo / Seguro NPU execution mode (0.25.1); same control in Generar (quick toggle) and Ajustes. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ExecModeSelector(value: com.zithexagon.engine.ExecMode, onChange: (com.zithexagon.engine.ExecMode) -> Unit) {
    val a = LocalAccent.current
    val modes = com.zithexagon.engine.ExecMode.entries
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            modes.forEachIndexed { i, m ->
                SegmentedButton(selected = value == m, onClick = { onChange(m) },
                    shape = SegmentedButtonDefaults.itemShape(i, modes.size),
                    colors = SegmentedButtonDefaults.colors(activeContainerColor = a.deep, activeContentColor = a.light,
                        inactiveContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh, activeBorderColor = a.main,
                        inactiveBorderColor = MaterialTheme.colorScheme.outline),
                    icon = {}) { Text(m.label, fontSize = 13.sp, maxLines = 1) }
            }
        }
        Text(stringResource(if (value == com.zithexagon.engine.ExecMode.EXTREMO) R.string.exec_mode_extremo_sub else R.string.exec_mode_seguro_sub),
            fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun UpscaleCard(vm: AppViewModel) {
    val a = LocalAccent.current
    SectionCard(stringResource(R.string.gen_upscale_section)) {
        SwitchRow(stringResource(R.string.upscale_after), stringResource(R.string.upscale_after_sub), vm.upscaleOn, vm::setUpscale)
        AnimatedVisibility(vm.upscaleOn) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                UpscaleFactorSelector(vm.upscaleFactor, vm::setUpscaleFactor)
                Text(stringResource(R.string.label_upscaler_model), style = MaterialTheme.typography.bodyMedium)
                UpscalerPicker(vm.upscalerModel, vm::setUpscaler)
                Text(stringResource(R.string.upscale_output, vm.params.width * vm.upscaleFactor, vm.params.height * vm.upscaleFactor),
                    fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun UpscaleFactorSelector(value: Int, onChange: (Int) -> Unit) {
    val a = LocalAccent.current
    SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
        listOf(2, 4).forEachIndexed { i, f ->
            SegmentedButton(selected = value == f, onClick = { onChange(f) }, shape = SegmentedButtonDefaults.itemShape(i, 2),
                colors = SegmentedButtonDefaults.colors(activeContainerColor = a.deep, activeContentColor = a.light,
                    inactiveContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh, activeBorderColor = a.main,
                    inactiveBorderColor = MaterialTheme.colorScheme.outline)) {
                Text("${f}×")
            }
        }
    }
}

@Composable
private fun GenerateBar(vm: AppViewModel) {
    val a = LocalAccent.current
    val job = vm.job
    Surface(color = MaterialTheme.colorScheme.background) {
        Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp)) {
            if (job == null) {
                Box(Modifier.fillMaxWidth().height(56.dp).clip(RoundedCornerShape(18.dp))
                    .background(Brush.horizontalGradient(listOf(a.main, a.main.copy(alpha = 0.85f), a.deep)))
                    .clickable { vm.generate() }, contentAlignment = Alignment.Center) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Filled.AutoAwesome, null, tint = Color.White)
                        Spacer(Modifier.width(10.dp))
                        Text(stringResource(R.string.action_generate), color = Color.White, fontSize = 17.sp, fontWeight = FontWeight.SemiBold)
                    }
                }
            } else {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(stageLabel(job.stage), fontWeight = FontWeight.SemiBold, maxLines = 1)
                        Text(stringResource(R.string.gen_percent, (job.fraction * 100).roundToInt()), fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    OutlinedButton(onClick = vm::cancel, shape = RoundedCornerShape(14.dp), border = BorderStroke(1.dp, a.main)) {
                        Icon(Icons.Filled.Stop, null, tint = a.light); Spacer(Modifier.width(6.dp))
                        Text(stringResource(R.string.action_cancel), color = a.light)
                    }
                }
            }
        }
    }
}

/** Upscaler models grouped by Realista / Anime, with licence tags (non-commercial ones flagged). */
@Composable
fun UpscalerPicker(selectedId: String, onSelect: (String) -> Unit) {
    val a = LocalAccent.current
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        listOf(UpscalerKind.REALISM to R.string.upscaler_realism, UpscalerKind.ANIME to R.string.upscaler_anime).forEach { (kind, label) ->
            Text(stringResource(label).uppercase(), style = SectionLabel, color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp, start = 2.dp))
            UPSCALERS.filter { it.kind == kind }.forEach { u -> UpscalerRow(u, u.id == selectedId) { onSelect(u.id) } }
        }
    }
}

@Composable
private fun UpscalerRow(u: UpscalerModel, selected: Boolean, onClick: () -> Unit) {
    val a = LocalAccent.current
    Surface(onClick = onClick, shape = RoundedCornerShape(14.dp),
        color = if (selected) a.deep else MaterialTheme.colorScheme.surfaceContainer,
        border = BorderStroke(1.dp, if (selected) a.main else MaterialTheme.colorScheme.outlineVariant)) {
        Row(Modifier.padding(horizontal = 12.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            RadioButton(selected, onClick, colors = RadioButtonDefaults.colors(selectedColor = a.light), modifier = Modifier.size(32.dp))
            Spacer(Modifier.width(6.dp))
            Column(Modifier.weight(1f)) {
                Text(u.name, style = MaterialTheme.typography.titleSmall, color = if (selected) a.light else MaterialTheme.colorScheme.onSurface)
                Text("${u.author} · ${u.license}", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (u.nonCommercial) Text(stringResource(R.string.upscaler_nc), fontSize = 10.sp, color = Amber,
                modifier = Modifier.clip(RoundedCornerShape(6.dp)).border(1.dp, Amber.copy(alpha = 0.5f), RoundedCornerShape(6.dp))
                    .padding(horizontal = 6.dp, vertical = 2.dp))
        }
    }
}
