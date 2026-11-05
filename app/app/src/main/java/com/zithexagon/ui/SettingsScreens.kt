package com.zithexagon.ui

import androidx.compose.foundation.layout.size
import androidx.compose.material3.CircularProgressIndicator
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.zithexagon.AppViewModel
import com.zithexagon.BuildConfigInfo
import com.zithexagon.R
import com.zithexagon.data.Storage
import com.zithexagon.data.humanBytes

@Composable
fun SettingsScreen(vm: AppViewModel, onAbout: () -> Unit) {
    val ctx = LocalContext.current
    val a = LocalAccent.current
    var access by remember { mutableStateOf(Storage.hasAllFilesAccess()) }
    LaunchedEffect(Unit) { access = Storage.hasAllFilesAccess() }
    Column(Modifier.fillMaxSize()) {
        ScreenHeader(stringResource(R.string.tab_settings))
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            SectionCard(stringResource(R.string.settings_storage)) {
                FolderRow(Icons.Outlined.ViewInAr, stringResource(R.string.settings_models_dir), Storage.models(ctx).path)
                FolderRow(Icons.Outlined.Layers, stringResource(R.string.settings_loras_dir), Storage.loras(ctx).path)
                FolderRow(Icons.Outlined.PhotoLibrary, stringResource(R.string.settings_outputs_dir), Storage.outputs(ctx).path)
                FolderRow(Icons.Outlined.FolderShared, stringResource(R.string.settings_shared_dir), Storage.sharedRoot.path + "/{models,loras}")
                Text(stringResource(R.string.settings_free, Storage.freeBytes(ctx).humanBytes()), fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(stringResource(R.string.settings_all_files), style = MaterialTheme.typography.bodyLarge)
                        Text(stringResource(if (access) R.string.settings_all_files_on else R.string.settings_all_files_off),
                            style = MaterialTheme.typography.bodySmall, color = if (access) a.light else MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    if (!access) OutlinedButton(onClick = {
                        runCatching { ctx.startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:" + ctx.packageName))) }
                    }, border = BorderStroke(1.dp, a.main)) { Text(stringResource(R.string.access_grant), color = a.light) }
                    else Icon(Icons.Outlined.CheckCircle, null, tint = a.light)
                }
                SwitchRow(stringResource(R.string.settings_copy_phone), stringResource(R.string.settings_copy_phone_sub), vm.copyToPhone, vm::setCopyToPhone)
            }
            SectionCard(stringResource(R.string.settings_appearance)) {
                Text(stringResource(R.string.settings_accent), style = MaterialTheme.typography.bodyLarge)
                Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                    ACCENTS.forEach { ac ->
                        Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.clip(RoundedCornerShape(12.dp))
                            .clickable { vm.setAccent(ac.id) }.padding(4.dp)) {
                            Box(Modifier.size(44.dp).clip(CircleShape).background(androidx.compose.ui.graphics.Brush.linearGradient(listOf(ac.light, ac.main, ac.deep)))
                                .border(2.dp, if (vm.accent == ac.id) Color.White else Color.Transparent, CircleShape), contentAlignment = Alignment.Center) {
                                if (vm.accent == ac.id) Icon(Icons.Filled.Check, null, tint = Color.White)
                            }
                            Spacer(Modifier.height(4.dp))
                            Text(stringResource(ac.nameRes), fontSize = 12.sp, color = if (vm.accent == ac.id) ac.light else MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
                AccentPreview()
                Text(stringResource(R.string.settings_dark_only), fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            SectionCard(stringResource(R.string.settings_engine)) {
                Text(stringResource(R.string.label_exec_mode), style = MaterialTheme.typography.bodyLarge)
                ExecModeSelector(vm.execMode, vm::setExecMode)
                Text(stringResource(R.string.exec_mode_settings_note), fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                InfoLine(stringResource(R.string.settings_engine_name), vm.engine.info.name)
                InfoLine(stringResource(R.string.settings_engine_backend), vm.engine.info.backend)
                InfoLine(stringResource(R.string.settings_target), stringResource(R.string.settings_target_value))
                if (vm.engine.info.isFake && !vm.selectedModel.id.startsWith("nph-")) Text(stringResource(R.string.settings_engine_fake_note), fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            SectionCard(stringResource(R.string.settings_nph_advanced)) {
                Text(stringResource(R.string.settings_nph_advanced_note), fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                SwitchRow(stringResource(R.string.settings_nph_profile), stringResource(R.string.settings_nph_profile_sub), vm.nphProfile, vm::setNphProfile)
                SwitchRow(stringResource(R.string.settings_nph_page_cache), stringResource(R.string.settings_nph_page_cache_sub), vm.nphPageCacheWarmup, vm::setNphPageCacheWarmup)
                SwitchRow(stringResource(R.string.settings_nph_graph_cache), stringResource(R.string.settings_nph_graph_cache_sub), vm.nphGraphCache, vm::setNphGraphCache)
                // overlap DiT∥TE y cpuOffload quitados (0.27): en teléfono empeoraron / sin ganancia.
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                NphKnob(stringResource(R.string.settings_nph_nhvx), vm.nphNhvx, vm::setNphNhvx)
                NphKnob(stringResource(R.string.settings_nph_nhmx), vm.nphNhmx, vm::setNphNhmx)
                NphKnob(stringResource(R.string.settings_nph_opbatch), vm.nphOpbatch, vm::setNphOpbatch)
                NphKnob(stringResource(R.string.settings_nph_opqueue), vm.nphOpqueue, vm::setNphOpqueue)
                NphKnob(stringResource(R.string.settings_nph_opfusion), vm.nphOpfusion, vm::setNphOpfusion)
                NphKnob(stringResource(R.string.settings_nph_vmem), vm.nphVmem, vm::setNphVmem)
                NphKnob(stringResource(R.string.settings_nph_mbuf), vm.nphMbuf, vm::setNphMbuf)
            }
            SectionCard(stringResource(R.string.settings_nph028)) {
                Text(stringResource(R.string.settings_nph028_note), fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                SwitchRow(stringResource(R.string.settings_nph_fast_repack), stringResource(R.string.settings_nph_fast_repack_sub), vm.nphFastRepack, vm::setNphFastRepack)
                SwitchRow(stringResource(R.string.settings_nph_parallel_set), stringResource(R.string.settings_nph_parallel_set_sub), vm.nphParallelSet, vm::setNphParallelSet)
                SwitchRow(stringResource(R.string.settings_nph_fast_read), stringResource(R.string.settings_nph_fast_read_sub), vm.nphFastRead, vm::setNphFastRead)
                SwitchRow(stringResource(R.string.settings_nph_buf_pool), stringResource(R.string.settings_nph_buf_pool_sub), vm.nphBufPool, vm::setNphBufPool)
                SwitchRow(stringResource(R.string.settings_nph_prefetch_dit), stringResource(R.string.settings_nph_prefetch_dit_sub), vm.nphPrefetchDit, vm::setNphPrefetchDit)
                SwitchRow(stringResource(R.string.settings_nph_repack_cache), stringResource(R.string.settings_nph_repack_cache_sub), vm.nphRepackCache, vm::setNphRepackCache)
                SwitchRow(stringResource(R.string.settings_nph_te_cache), stringResource(R.string.settings_nph_te_cache_sub), vm.nphTeCache, vm::setNphTeCache)
                SwitchRow(stringResource(R.string.settings_nph_te_cache_disk), stringResource(R.string.settings_nph_te_cache_disk_sub), vm.nphTeCacheDisk && vm.nphTeCache, vm::setNphTeCacheDisk)
                SwitchRow(stringResource(R.string.settings_nph_lora_combined), stringResource(R.string.settings_nph_lora_combined_sub), vm.nphLoraCombined, vm::setNphLoraCombined)
                LaunchedEffect(Unit) { vm.refreshNphCacheSize() }
                val cacheTxt = if (vm.nphCacheBytes < 0) "…" else vm.nphCacheBytes.humanBytes()
                val freeTxt = if (vm.nphCacheFreeBytes < 0) "?" else vm.nphCacheFreeBytes.humanBytes()
                Text(stringResource(R.string.settings_nph_cache_size, cacheTxt, freeTxt), fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (vm.nphCacheFreeBytes in 0 until (13L shl 30)) Text(stringResource(R.string.settings_nph_cache_low), fontSize = 12.sp, color = MaterialTheme.colorScheme.error)
                OutlinedButton(onClick = { vm.clearNphCache() }, enabled = !vm.nphCacheBusy && vm.nphCacheBytes != 0L) {
                    Text(stringResource(R.string.settings_nph_cache_clear))
                }
                OutlinedButton(onClick = { vm.resetNphToggles() }) { Text(stringResource(R.string.settings_nph_reset)) }
            }
            SectionCard(stringResource(R.string.settings_nph_exp)) {
                Text(stringResource(R.string.settings_nph_exp_note), fontSize = 12.sp, color = MaterialTheme.colorScheme.error)
                SwitchRow(stringResource(R.string.settings_nph_exp_prefetch_vae), stringResource(R.string.settings_nph_exp_prefetch_vae_sub), vm.nphExpPrefetchVae, vm::setNphExpPrefetchVae)
                // 0.30.1: NPG (GPU VAE) quitado de la UI y del build por defecto (sigue en el historial git).
            }
            SectionCard(stringResource(R.string.settings_crash)) {
                var report by remember { mutableStateOf(com.zithexagon.CrashReporter.report(ctx)) }
                Text(stringResource(if (report != null) R.string.settings_crash_has else R.string.settings_crash_none), fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (report != null) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = { com.zithexagon.CrashReporter.share(ctx, report!!) }) { Text(stringResource(R.string.crash_share)) }
                        OutlinedButton(onClick = { com.zithexagon.CrashReporter.clear(ctx); report = null }) { Text(stringResource(R.string.settings_crash_clear)) }
                    }
                }
            }
            SectionCard(stringResource(R.string.settings_perf)) {
                val perfLog = remember { com.zithexagon.ZitDiag.perfFile(ctx).takeIf { it.isFile && it.length() > 0 } }
                Text(stringResource(if (perfLog != null) R.string.settings_perf_has else R.string.settings_perf_none), fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (perfLog != null) Button(onClick = {
                    runCatching { perfLog.readText() }.getOrNull()?.let { sharePerfText(ctx, it, "ZIT-HEXAGON — log de rendimiento") }
                }) { Text(stringResource(R.string.settings_perf_share)) }
            }
            SectionCard(stringResource(R.string.settings_diag)) {
                Text(stringResource(R.string.settings_diag_desc), fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                val prog = vm.benchProgress
                if (prog != null) Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp); Spacer(Modifier.width(10.dp))
                    Text(prog, fontSize = 12.sp)
                }
                Button(onClick = { vm.runTestBlocks() }, enabled = prog == null) { Text(stringResource(R.string.settings_diag_bench)) }
                SwitchRow(stringResource(R.string.settings_debug_dump), stringResource(R.string.settings_debug_dump_sub), vm.debugDump, vm::setDebugDump)
                vm.benchReport?.let { rep ->
                    PerfDetailsDialog(rep, onDismiss = { vm.benchReport = null }, title = stringResource(R.string.settings_diag_bench),
                        subject = "ZIT-HEXAGON — prueba de bloques")
                }
            }
            SectionCard(stringResource(R.string.settings_about)) {
                Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).clickable(onClick = onAbout).padding(vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Outlined.Gavel, null, tint = a.light); Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Text(stringResource(R.string.about_title), style = MaterialTheme.typography.bodyLarge)
                        Text(stringResource(R.string.about_version, BuildConfigInfo.versionName(ctx)), fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Icon(Icons.AutoMirrored.Filled.KeyboardArrowRight, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            Spacer(Modifier.height(16.dp))
        }
    }
}

@Composable
private fun FolderRow(icon: ImageVector, title: String, path: String) {
    Row(verticalAlignment = Alignment.Top) {
        Icon(icon, null, tint = LocalAccent.current.light, modifier = Modifier.size(20.dp).padding(top = 2.dp))
        Spacer(Modifier.width(12.dp))
        Column {
            Text(title, style = MaterialTheme.typography.bodyMedium)
            Text(path, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun AccentPreview() {
    val a = LocalAccent.current
    Surface(shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.surfaceContainer, border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {}) { Icon(Icons.Filled.AutoAwesome, null, Modifier.size(16.dp)); Spacer(Modifier.width(6.dp)); Text(stringResource(R.string.action_generate)) }
                TriggerChip("trigger", true) {}
                TriggerChip("chip", false) {}
            }
            var v by remember { mutableFloatStateOf(0.6f) }
            Slider(v, { v = it }, Modifier.height(28.dp), colors = SliderDefaults.colors(inactiveTrackColor = MaterialTheme.colorScheme.surfaceContainerHighest))
            Text(stringResource(R.string.settings_accent_text_sample), color = a.light, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AboutScreen(onBack: () -> Unit) {
    val ctx = LocalContext.current
    val a = LocalAccent.current
    Column(Modifier.fillMaxSize().statusBarsPadding()) {
        TopAppBar(
            title = { Text(stringResource(R.string.about_title)) },
            navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.action_back)) } },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
            windowInsets = WindowInsets(0),
        )
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp).navigationBarsPadding(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Column(Modifier.fillMaxWidth().padding(vertical = 12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Box(Modifier.size(96.dp).clip(RoundedCornerShape(28.dp))
                    .background(androidx.compose.ui.graphics.Brush.radialGradient(listOf(Color(0xFF3A0E2D), Color(0xFF160610), Color.Black))),
                    contentAlignment = Alignment.Center) {
                    androidx.compose.foundation.Image(androidx.compose.ui.res.painterResource(R.drawable.ic_launcher_foreground), null, Modifier.size(144.dp))
                }
                Spacer(Modifier.height(10.dp))
                Text(stringResource(R.string.app_name), style = MaterialTheme.typography.headlineSmall)
                Text(stringResource(R.string.about_version, BuildConfigInfo.versionName(ctx)), color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(6.dp))
                Text(stringResource(R.string.about_tagline), color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 13.sp,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center)
            }
            SectionCard(stringResource(R.string.about_models)) {
                Credit("Z-Image Turbo", "Tongyi-MAI (Alibaba) · Apache-2.0")
                Credit("ReaZIT v1.0", stringResource(R.string.about_reazit))
            }
            SectionCard(stringResource(R.string.about_upscalers)) {
                com.zithexagon.data.UPSCALERS.forEach { u ->
                    Credit(u.name, "${u.author} · ${u.license}" + if (u.nonCommercial) " · " + stringResource(R.string.upscaler_nc) else "")
                }
                Text(stringResource(R.string.about_upscalers_note), fontSize = 12.sp, color = Amber)
            }
            SectionCard(stringResource(R.string.about_runtime)) {
                Credit(stringResource(R.string.about_nph), stringResource(R.string.about_nph_detail))
            }
            SectionCard(stringResource(R.string.about_libs)) {
                Credit("Kotlin, kotlinx.coroutines", "JetBrains · Apache-2.0")
                Credit("AndroidX, Jetpack Compose, Material 3", "Google · Apache-2.0")
                Credit("Material Symbols / Icons", "Google · Apache-2.0")
            }
            SectionCard(stringResource(R.string.about_thanks)) {
                Text(stringResource(R.string.about_inspiration), fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Text(stringResource(R.string.about_noncommercial), fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(16.dp))
        }
    }
}

@Composable
private fun Credit(title: String, body: String) {
    Column { Text(title, style = MaterialTheme.typography.bodyLarge); Text(body, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant) }
}


@Composable
private fun NphKnob(label: String, value: String, onChange: (String) -> Unit) {
    var text by remember(value) { mutableStateOf(value) }
    OutlinedTextField(
        value = text,
        onValueChange = { text = it; onChange(it) },
        label = { Text(label) },
        placeholder = { Text(stringResource(R.string.settings_nph_default_hint)) },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
}
