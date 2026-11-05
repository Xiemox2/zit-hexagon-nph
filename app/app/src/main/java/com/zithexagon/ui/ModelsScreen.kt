package com.zithexagon.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.zithexagon.AppViewModel
import com.zithexagon.R
import com.zithexagon.data.ModelSource
import com.zithexagon.data.NphModelStore
import com.zithexagon.data.Role
import com.zithexagon.data.ZitModel
import com.zithexagon.data.humanBytes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun ModelsScreen(vm: AppViewModel) {
    val ctx = LocalContext.current
    val a = LocalAccent.current
    var menu by remember { mutableStateOf(false) }
    var importRole by remember { mutableStateOf<Role?>(null) }
    var useWithoutCopy by remember { mutableStateOf(true) }
    var tick by remember { mutableIntStateOf(0) }

    val pickOne = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        val role = importRole; importRole = null
        if (uri != null) {
            vm.importNphUri(uri, preferCopy = !useWithoutCopy, expectRole = role)
            tick++
        }
    }
    val pickThree = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNullOrEmpty()) return@rememberLauncherForActivityResult
        for (uri in uris) {
            vm.importNphUri(uri, preferCopy = !useWithoutCopy, expectRole = null)
        }
        tick++
    }

    Column(Modifier.fillMaxSize()) {
        ScreenHeader(stringResource(R.string.tab_models), stringResource(R.string.models_subtitle, vm.models.size)) {
            if (vm.canRelease) IconButton(onClick = vm::releaseModel) { Icon(Icons.Filled.Memory, stringResource(R.string.action_release)) }
            IconButton(onClick = {
                vm.scanNphDownloads()
                tick++
            }) { Icon(Icons.Outlined.Refresh, stringResource(R.string.action_refresh)) }
            Box {
                FilledTonalButton(onClick = { menu = true }, colors = ButtonDefaults.filledTonalButtonColors(containerColor = a.deep, contentColor = a.light)) {
                    Icon(Icons.Filled.Add, null, Modifier.size(18.dp)); Spacer(Modifier.width(4.dp)); Text(stringResource(R.string.action_import))
                }
                DropdownMenu(menu, { menu = false }, containerColor = MaterialTheme.colorScheme.surfaceContainerHigh) {
                    DropdownMenuItem(text = { Text("Importar DiT (FP8)") }, leadingIcon = { Icon(Icons.Outlined.ViewInAr, null) },
                        onClick = { menu = false; importRole = Role.DIT; pickOne.launch(arrayOf("*/*")) })
                    DropdownMenuItem(text = { Text("Importar TE (Qwen3 GGUF)") }, leadingIcon = { Icon(Icons.Outlined.TextFields, null) },
                        onClick = { menu = false; importRole = Role.TE; pickOne.launch(arrayOf("*/*")) })
                    DropdownMenuItem(text = { Text("Importar VAE (ae)") }, leadingIcon = { Icon(Icons.Outlined.Image, null) },
                        onClick = { menu = false; importRole = Role.VAE; pickOne.launch(arrayOf("*/*")) })
                    HorizontalDivider()
                    DropdownMenuItem(text = { Text("Importar los 3 a la vez") }, leadingIcon = { Icon(Icons.Outlined.FolderOpen, null) },
                        onClick = { menu = false; pickThree.launch(arrayOf("*/*")) })
                    HorizontalDivider()
                    DropdownMenuItem(text = { Text(if (useWithoutCopy) "✓ Usar sin copiar (defecto)" else "Usar sin copiar") },
                        leadingIcon = { Icon(Icons.Outlined.Link, null) },
                        onClick = { useWithoutCopy = !useWithoutCopy; menu = false })
                }
            }
        }

        key(tick, vm.busyImport, vm.models.size) {
            NphRolesCard(vm)
        }

        vm.busyImport?.let {
            LinearProgressIndicator(Modifier.fillMaxWidth().padding(horizontal = 16.dp))
            Text(it, Modifier.padding(horizontal = 20.dp), fontSize = 12.sp)
        }
        LazyColumn(contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            items(vm.models, key = { it.id }) { m -> ModelCard(vm, m) }
            item { NphFormatCallout() }
        }
    }
}

@Composable
private fun NphRolesCard(vm: AppViewModel) {
    val ctx = LocalContext.current
    val a = LocalAccent.current
    Surface(Modifier.padding(horizontal = 16.dp, vertical = 8.dp), shape = RoundedCornerShape(18.dp),
        color = MaterialTheme.colorScheme.surfaceContainerLow, border = BorderStroke(1.dp, a.main.copy(alpha = 0.35f))) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Motor NPH — modelos", fontWeight = FontWeight.SemiBold)
            Text("DiT FP8 + TE Qwen3-4B Q8 + VAE ae. Se buscan en Download/ZITHEXAGON/sdcpp, Download/ZITHEXAGON, Download/ y en la carpeta de SdBench.",
                fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            for (r in Role.entries) {
                val s = NphModelStore.load(ctx, r)
                val problem = NphModelStore.problem(ctx, r)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        if (s != null && problem == null) Icons.Filled.CheckCircle else Icons.Outlined.RadioButtonUnchecked,
                        null, tint = if (s != null && problem == null) a.light else MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Column(Modifier.weight(1f)) {
                        Text(r.label, fontSize = 13.sp, fontWeight = FontWeight.Medium)
                        Text(
                            when {
                                problem != null -> problem
                                s != null -> "${s.name} · ${s.size.humanBytes()} · ${s.where}"
                                else -> "sin seleccionar"
                            },
                            fontSize = 11.sp, color = if (problem != null) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
        }
    }
}

@Composable
private fun ModelCard(vm: AppViewModel, m: ZitModel) {
    val a = LocalAccent.current
    val selected = m.id == vm.selectedModelId
    var expanded by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }
    Surface(onClick = { expanded = !expanded }, shape = RoundedCornerShape(22.dp), color = MaterialTheme.colorScheme.surfaceContainerLow,
        border = BorderStroke(if (selected) 1.5.dp else 1.dp, if (selected) a.main else MaterialTheme.colorScheme.outlineVariant)) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(64.dp).clip(RoundedCornerShape(16.dp))) {
                    if (m.previewFile != null) FileImage(m.previewFile, Modifier.fillMaxSize(), maxPx = 200)
                    else {
                        PlaceholderArt(Modifier.fillMaxSize(), icon = 0.dp)
                        Text(if (m.id.startsWith("nph")) "NPH" else "Z", Modifier.align(Alignment.Center),
                            fontSize = if (m.id.startsWith("nph")) 16.sp else 30.sp, fontWeight = FontWeight.Black, color = a.light)
                    }
                }
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    Text(m.name, style = MaterialTheme.typography.titleMedium)
                    Text("${m.manifest.base}" + (if (m.manifest.version.isNotBlank()) " · v${m.manifest.version}" else ""),
                        fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Tag(stringResource(when (m.source) { ModelSource.DEMO -> R.string.model_src_demo; ModelSource.APP -> R.string.model_src_app; ModelSource.SHARED -> R.string.model_src_shared }))
                        if (m.id.startsWith("nph")) Tag("NPH")
                        if (m.manifest.htp.isNotBlank()) Tag("HTP ${m.manifest.htp}")
                        if (!m.isDemo) Tag(m.sizeBytes.humanBytes())
                    }
                }
                if (selected) Icon(Icons.Filled.CheckCircle, stringResource(R.string.model_selected), tint = a.light)
            }
            if (m.missing.isNotEmpty()) Text(stringResource(R.string.model_missing_files, m.missing.size, m.missing.joinToString()),
                fontSize = 12.sp, color = MaterialTheme.colorScheme.error)
            AnimatedVisibility(expanded) {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    if (m.manifest.description.isNotBlank()) Text(m.manifest.description, fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    InfoLine(stringResource(R.string.model_id), m.id)
                    InfoLine(stringResource(R.string.model_defaults), stringResource(R.string.model_defaults_value, m.manifest.defaultSteps, m.manifest.defaultCfg.toString(), m.manifest.defaultSampler, m.manifest.defaultWidth, m.manifest.defaultHeight))
                    m.manifest.files.forEach { (k, v) -> InfoLine(k, v) }
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (!selected) Button(onClick = { vm.selectModel(m) }) { Text(stringResource(R.string.model_use)) }
                else OutlinedButton(onClick = {}, enabled = false) { Text(stringResource(R.string.model_in_use)) }
                Spacer(Modifier.weight(1f))
                if (m.source == ModelSource.APP && !m.id.startsWith("nph")) TextButton(onClick = { confirmDelete = true }) {
                    Text(stringResource(R.string.action_delete), color = MaterialTheme.colorScheme.error)
                }
            }
        }
    }
    if (confirmDelete) ConfirmDialog(stringResource(R.string.model_delete_title, m.name), stringResource(R.string.model_delete_body, m.sizeBytes.humanBytes()),
        stringResource(R.string.action_delete), onConfirm = { vm.deleteModel(m) }, onDismiss = { confirmDelete = false })
}

@Composable
fun Tag(text: String) {
    Text(text, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1,
        modifier = Modifier.clip(RoundedCornerShape(6.dp)).background(MaterialTheme.colorScheme.surfaceContainerHigh).padding(horizontal = 7.dp, vertical = 2.dp))
}

@Composable
private fun NphFormatCallout() {
    val a = LocalAccent.current
    Surface(shape = RoundedCornerShape(18.dp), color = a.deep.copy(alpha = 0.45f), border = BorderStroke(1.dp, a.main.copy(alpha = 0.4f))) {
        Row(Modifier.padding(14.dp)) {
            Icon(Icons.Outlined.Info, null, tint = a.light)
            Spacer(Modifier.width(10.dp))
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Modelos NPH (0.26)", fontWeight = FontWeight.SemiBold)
                Text("Importa o reutiliza los 3 archivos (DiT FP8 .safetensors, TE Qwen3 .gguf, VAE ae.safetensors). Por defecto se usan sin copiar (~10 GB). Si Android no lo permite, se copian a la carpeta de la app (comprueba espacio libre). También se detectan solos en Descargas/ZITHEXAGON y en Android/data/com.zithexagon.sdbench/files/models/.",
                    fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}
