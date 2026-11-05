package com.zithexagon.ui

import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.zithexagon.AppViewModel
import com.zithexagon.R
import com.zithexagon.data.LoraEntry
import com.zithexagon.data.Storage
import com.zithexagon.data.humanBytes
import kotlin.math.roundToInt

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun LorasScreen(vm: AppViewModel, onEdit: (LoraEntry) -> Unit) {
    val ctx = LocalContext.current
    val a = LocalAccent.current
    var menu by remember { mutableStateOf(false) }
    var query by remember { mutableStateOf("") }
    var askAccess by remember { mutableStateOf(false) }
    val pickFiles = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) vm.importLoras(uris)
    }
    val pickTree = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri -> uri?.let(vm::importLoraFolder) }
    val scan = { if (Storage.hasAllFilesAccess()) vm.scanLoraFolders() else askAccess = true }

    Column(Modifier.fillMaxSize()) {
        ScreenHeader(stringResource(R.string.tab_loras), stringResource(R.string.loras_subtitle, vm.loras.size)) {
            IconButton(onClick = scan) { Icon(Icons.Outlined.Refresh, stringResource(R.string.loras_scan)) }
            Box {
                FilledTonalButton(onClick = { menu = true }, colors = ButtonDefaults.filledTonalButtonColors(containerColor = a.deep, contentColor = a.light)) {
                    Icon(Icons.Filled.Add, null, Modifier.size(18.dp)); Spacer(Modifier.width(4.dp)); Text(stringResource(R.string.action_import))
                }
                DropdownMenu(menu, { menu = false }, containerColor = MaterialTheme.colorScheme.surfaceContainerHigh) {
                    DropdownMenuItem(text = { Text(stringResource(R.string.loras_import_files)) }, leadingIcon = { Icon(Icons.Outlined.FileOpen, null) },
                        onClick = { menu = false; pickFiles.launch(arrayOf("application/octet-stream", "*/*")) })
                    DropdownMenuItem(text = { Text(stringResource(R.string.loras_import_folder)) }, leadingIcon = { Icon(Icons.Outlined.DriveFolderUpload, null) },
                        onClick = { menu = false; pickTree.launch(null) })
                    DropdownMenuItem(text = { Text(stringResource(R.string.loras_scan_shared)) }, leadingIcon = { Icon(Icons.Outlined.FolderOpen, null) },
                        onClick = { menu = false; scan() })
                }
            }
        }
        vm.busyImport?.let { LinearProgressIndicator(Modifier.fillMaxWidth().padding(horizontal = 16.dp)); Text(it, Modifier.padding(horizontal = 20.dp), fontSize = 12.sp) }
        if (vm.loras.isEmpty()) {
            EmptyState(Icons.Outlined.Layers, stringResource(R.string.loras_empty_title), stringResource(R.string.loras_empty_body), Modifier.padding(top = 32.dp)) {
                Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { pickFiles.launch(arrayOf("application/octet-stream", "*/*")) }) { Text(stringResource(R.string.loras_import_files)) }
                    OutlinedButton(onClick = scan) { Text(stringResource(R.string.loras_scan_shared)) }
                }
            }
        } else {
            OutlinedTextField(query, { query = it }, Modifier.fillMaxWidth().padding(horizontal = 16.dp),
                placeholder = { Text(stringResource(R.string.loras_search)) }, leadingIcon = { Icon(Icons.Outlined.Search, null) },
                singleLine = true, shape = RoundedCornerShape(16.dp), colors = fieldColors())
            val list = vm.loras.filter { query.isBlank() || it.name.contains(query, true) || it.triggerWords.any { w -> w.contains(query, true) } }
            LazyColumn(contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                items(list, key = { it.id }) { e -> LoraRow(vm, e) { onEdit(e) } }
                item {
                    Text(stringResource(R.string.loras_folder_hint, Storage.SHARED_FOLDER), fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 4.dp))
                }
            }
        }
    }
    if (askAccess) ConfirmDialog(stringResource(R.string.access_title), stringResource(R.string.access_body, Storage.SHARED_FOLDER),
        stringResource(R.string.access_grant), destructive = false, onDismiss = { askAccess = false }, onConfirm = {
            runCatching { ctx.startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:" + ctx.packageName))) }
        })
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun LoraRow(vm: AppViewModel, e: LoraEntry, onClick: () -> Unit) {
    val a = LocalAccent.current
    val active = vm.isLoraActive(e.id)
    Surface(onClick = onClick, shape = RoundedCornerShape(20.dp), color = MaterialTheme.colorScheme.surfaceContainerLow,
        border = BorderStroke(1.dp, if (active) a.main else MaterialTheme.colorScheme.outlineVariant)) {
        Row(Modifier.padding(12.dp)) {
            FileImage(e.preview, Modifier.size(84.dp).clip(RoundedCornerShape(14.dp)), maxPx = 256)
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(e.name, style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                    ValuePill(formatStrength(e.strength))
                }
                Text(listOfNotNull(e.sizeBytes.humanBytes(), stringResource(if (e.source == "app") R.string.lora_src_app else R.string.lora_src_folder),
                    if (!e.exists) stringResource(R.string.lora_file_missing) else null).joinToString(" · "),
                    fontSize = 12.sp, color = if (e.exists) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.error)
                if (e.triggerWords.isNotEmpty()) FlowRow(horizontalArrangement = Arrangement.spacedBy(4.dp), maxLines = 1) {
                    e.triggerWords.take(4).forEach { w ->
                        Text(w, fontSize = 11.sp, color = a.light, maxLines = 1,
                            modifier = Modifier.clip(RoundedCornerShape(6.dp)).background(a.deep).padding(horizontal = 6.dp, vertical = 2.dp))
                    }
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.lora_use_in_generate), fontSize = 13.sp, modifier = Modifier.weight(1f))
                    Switch(active, { vm.toggleLora(e) }, Modifier.height(28.dp), colors = SwitchDefaults.colors(
                        uncheckedTrackColor = MaterialTheme.colorScheme.surfaceContainerHighest, uncheckedBorderColor = MaterialTheme.colorScheme.outline))
                }
            }
        }
    }
}


@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun LoraEditScreen(vm: AppViewModel, loraId: String, onBack: () -> Unit) {
    val e = vm.loras.firstOrNull { it.id == loraId }
    if (e == null) { LaunchedEffect(Unit) { onBack() }; return }
    val a = LocalAccent.current
    var confirmDelete by remember { mutableStateOf(false) }
    var pickFromGallery by remember { mutableStateOf(false) }
    var name by remember(e.id) { mutableStateOf(e.name) }
    var newWord by remember { mutableStateOf("") }
    var strength by remember(e.id) { mutableFloatStateOf(e.strength) }
    val photo = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { vm.setLoraPreviewFromUri(e.id, it) }
    }
    Column(Modifier.fillMaxSize().statusBarsPadding()) {
        TopAppBar(
            title = { Text(stringResource(R.string.lora_edit_title)) },
            navigationIcon = { IconButton(onClick = { if (name != e.name) vm.renameLora(e.id, name); onBack() }) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.action_back)) } },
            actions = { IconButton(onClick = { confirmDelete = true }) { Icon(Icons.Outlined.Delete, stringResource(R.string.action_delete)) } },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
            windowInsets = WindowInsets(0),
        )
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 16.dp).navigationBarsPadding().imePadding(),
            verticalArrangement = Arrangement.spacedBy(12.dp)) {
            // Preview
            Row(verticalAlignment = Alignment.CenterVertically) {
                FileImage(e.preview, Modifier.size(148.dp).clip(RoundedCornerShape(20.dp)), maxPx = 512, placeholderIcon = 40.dp)
                Spacer(Modifier.width(14.dp))
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(stringResource(R.string.lora_preview).uppercase(), style = SectionLabel, color = a.light)
                    FilledTonalButton(onClick = { photo.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                        colors = ButtonDefaults.filledTonalButtonColors(containerColor = a.deep, contentColor = a.light)) {
                        Icon(Icons.Outlined.Photo, null, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp)); Text(stringResource(R.string.lora_preview_phone), fontSize = 13.sp)
                    }
                    OutlinedButton(onClick = { pickFromGallery = true }, border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline)) {
                        Icon(Icons.Outlined.PhotoLibrary, null, Modifier.size(18.dp), tint = a.light); Spacer(Modifier.width(6.dp))
                        Text(stringResource(R.string.lora_preview_app), fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurface)
                    }
                    if (e.preview != null) TextButton(onClick = { vm.clearLoraPreview(e.id) }, contentPadding = PaddingValues(horizontal = 8.dp)) {
                        Text(stringResource(R.string.lora_preview_clear), color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 13.sp)
                    }
                }
            }
            SectionCard(stringResource(R.string.lora_general)) {
                OutlinedTextField(name, { name = it }, Modifier.fillMaxWidth(), label = { Text(stringResource(R.string.lora_name)) },
                    singleLine = true, shape = RoundedCornerShape(14.dp), colors = fieldColors(),
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done), keyboardActions = KeyboardActions(onDone = { vm.renameLora(e.id, name) }))
                StrengthControl(strength, { strength = it }, label = stringResource(R.string.lora_default_strength))
                LaunchedEffect(strength) { kotlinx.coroutines.delay(300); if (strength != e.strength) vm.setLoraStrength(e.id, strength) }
                SwitchRow(stringResource(R.string.lora_use_in_generate), null, vm.isLoraActive(e.id)) { vm.toggleLora(e) }
            }
            SectionCard(stringResource(R.string.lora_triggers)) {
                Text(stringResource(R.string.lora_triggers_hint), fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (e.triggerWords.isEmpty()) Text(stringResource(R.string.lora_no_triggers), color = MaterialTheme.colorScheme.onSurfaceVariant)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    e.triggerWords.forEach { w ->
                        InputChip(selected = false, onClick = {}, label = { Text(w) }, shape = RoundedCornerShape(10.dp),
                            trailingIcon = { Icon(Icons.Filled.Close, stringResource(R.string.action_remove), Modifier.size(16.dp).clickable {
                                vm.setLoraTriggers(e.id, e.triggerWords - w) }) },
                            colors = InputChipDefaults.inputChipColors(containerColor = a.deep, labelColor = a.light, trailingIconColor = a.light),
                            border = InputChipDefaults.inputChipBorder(true, false, borderColor = a.main.copy(alpha = 0.6f)))
                    }
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    val add = { if (newWord.isNotBlank()) { vm.setLoraTriggers(e.id, e.triggerWords + newWord.split(',')); newWord = "" } }
                    OutlinedTextField(newWord, { newWord = it }, Modifier.weight(1f), placeholder = { Text(stringResource(R.string.lora_add_trigger)) },
                        singleLine = true, shape = RoundedCornerShape(14.dp), colors = fieldColors(),
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done), keyboardActions = KeyboardActions(onDone = { add() }))
                    Spacer(Modifier.width(8.dp))
                    FilledIconButton(onClick = add, enabled = newWord.isNotBlank()) { Icon(Icons.Filled.Add, stringResource(R.string.action_add)) }
                }
            }
            SectionCard(stringResource(R.string.lora_file)) {
                InfoLine(stringResource(R.string.lora_file_name), e.id)
                InfoLine(stringResource(R.string.lora_file_size), e.sizeBytes.humanBytes())
                InfoLine(stringResource(R.string.lora_file_source), stringResource(if (e.source == "app") R.string.lora_src_app else R.string.lora_src_folder))
                InfoLine(stringResource(R.string.lora_file_path), e.path)
                InfoLine(stringResource(R.string.lora_type), stringResource(R.string.lora_type_swap))
            }
            Spacer(Modifier.height(16.dp))
        }
    }
    if (confirmDelete) ConfirmDialog(stringResource(R.string.lora_delete_title, e.name),
        stringResource(if (e.source == "app") R.string.lora_delete_body_app else R.string.lora_delete_body_folder),
        stringResource(R.string.action_delete), onConfirm = { vm.removeLora(e.id); onBack() }, onDismiss = { confirmDelete = false })
    if (pickFromGallery) GalleryPickerDialog(vm, onPick = { vm.setLoraPreviewFromGallery(e.id, it); pickFromGallery = false }) { pickFromGallery = false }
}

@Composable
fun InfoLine(k: String, v: String) {
    Row { Text(k, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.width(96.dp), fontSize = 13.sp)
        Text(v, fontSize = 13.sp, modifier = Modifier.weight(1f)) }
}

@Composable
private fun GalleryPickerDialog(vm: AppViewModel, onPick: (com.zithexagon.data.GalleryItem) -> Unit, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.lora_preview_app)) },
        text = {
            if (vm.gallery.isEmpty()) Text(stringResource(R.string.gallery_empty_body))
            else LazyVerticalGrid(GridCells.Fixed(3), Modifier.heightIn(max = 440.dp), horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)) {
                items(vm.gallery, key = { it.id }) { g ->
                    FileImage(g.file, Modifier.aspectRatio(1f).clip(RoundedCornerShape(10.dp)).clickable { onPick(g) }, maxPx = 256)
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text(stringResource(R.string.action_cancel)) } },
        containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
    )
}
