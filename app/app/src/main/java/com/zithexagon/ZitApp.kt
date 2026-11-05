package com.zithexagon

import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.ViewInAr
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.Layers
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.outlined.ViewInAr
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.zithexagon.ui.AboutScreen
import com.zithexagon.ui.GalleryDetailScreen
import com.zithexagon.ui.GalleryScreen
import com.zithexagon.ui.GenerateScreen
import com.zithexagon.ui.LocalAccent
import com.zithexagon.ui.LoraEditScreen
import com.zithexagon.ui.LorasScreen
import com.zithexagon.ui.ModelsScreen
import com.zithexagon.ui.SettingsScreen
import com.zithexagon.ui.ZitTheme
import com.zithexagon.ui.sharePerfText
import kotlinx.coroutines.launch

enum class Tab(val label: Int, val icon: ImageVector, val iconSelected: ImageVector) {
    GENERATE(R.string.tab_generate, Icons.Outlined.AutoAwesome, Icons.Filled.AutoAwesome),
    GALLERY(R.string.tab_gallery, Icons.Outlined.PhotoLibrary, Icons.Filled.PhotoLibrary),
    LORAS(R.string.tab_loras, Icons.Outlined.Layers, Icons.Filled.Layers),
    MODELS(R.string.tab_models, Icons.Outlined.ViewInAr, Icons.Filled.ViewInAr),
    SETTINGS(R.string.tab_settings, Icons.Outlined.Settings, Icons.Filled.Settings),
}

/** Full-screen pages shown above the tabs. */
sealed interface Overlay {
    data class GalleryDetail(val itemId: String) : Overlay
    data class LoraEdit(val loraId: String) : Overlay
    data object About : Overlay
}

class Nav(start: Tab, startOverlay: Overlay?) {
    var tab by mutableStateOf(start)
    var overlay by mutableStateOf(startOverlay)
}

@Composable
fun ZitApp(vm: AppViewModel, startTab: Tab = Tab.GENERATE, startOverlay: Overlay? = null) {
    ZitTheme(vm.accent) {
        var tabSaved by rememberSaveable { mutableStateOf(startTab.name) }
        val nav = remember { Nav(Tab.valueOf(tabSaved), startOverlay) }
        LaunchedEffect(nav.tab) { tabSaved = nav.tab.name }
        val snackbar = remember { SnackbarHostState() }
        val ctx = LocalContext.current
        LaunchedEffect(vm.openItemRequest) {
            vm.openItemRequest?.let { id -> nav.tab = Tab.GALLERY; nav.overlay = Overlay.GalleryDetail(id); vm.openItemRequest = null }
        }
        LaunchedEffect(Unit) {
            vm.messages.collect { m ->
                val text = ctx.getString(m.res, *m.args.toTypedArray())
                if (m.share == null) snackbar.showSnackbar(text)
                else launch {   // don't block later messages while this one waits for the user
                    val r = snackbar.showSnackbar(text, actionLabel = ctx.getString(R.string.msg_share_detail),
                        withDismissAction = true, duration = SnackbarDuration.Indefinite)
                    if (r == SnackbarResult.ActionPerformed) sharePerfText(ctx, m.share, "ZIT-HEXAGON — detalle del fallo")
                }
            }
        }

        Box(Modifier.fillMaxSize()) {
            Scaffold(
                containerColor = MaterialTheme.colorScheme.background,
                snackbarHost = { SnackbarHost(snackbar) { Snackbar(it, containerColor = MaterialTheme.colorScheme.surfaceContainerHighest,
                    contentColor = MaterialTheme.colorScheme.onSurface, actionColor = LocalAccent.current.light) } },
                bottomBar = { if (nav.overlay == null) BottomBar(nav) },
            ) { pad ->
                Box(Modifier.padding(pad).consumeWindowInsets(pad).fillMaxSize()) {
                    AnimatedContent(nav.tab, transitionSpec = { fadeIn() togetherWith fadeOut() }, label = "tab") { tab ->
                        when (tab) {
                            Tab.GENERATE -> GenerateScreen(vm, onOpenResult = { nav.overlay = Overlay.GalleryDetail(it.id) },
                                onGoModels = { nav.tab = Tab.MODELS }, onGoLoras = { nav.tab = Tab.LORAS })
                            Tab.GALLERY -> GalleryScreen(vm, onOpen = { nav.overlay = Overlay.GalleryDetail(it.id) },
                                onGoGenerate = { nav.tab = Tab.GENERATE })
                            Tab.LORAS -> LorasScreen(vm, onEdit = { nav.overlay = Overlay.LoraEdit(it.id) })
                            Tab.MODELS -> ModelsScreen(vm)
                            Tab.SETTINGS -> SettingsScreen(vm, onAbout = { nav.overlay = Overlay.About })
                        }
                    }
                }
            }
            nav.overlay?.let { ov ->
                BackHandler { nav.overlay = null }
                Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    when (ov) {
                        is Overlay.GalleryDetail -> GalleryDetailScreen(vm, ov.itemId, onBack = { nav.overlay = null },
                            onReuse = { nav.overlay = null; nav.tab = Tab.GENERATE })
                        is Overlay.LoraEdit -> LoraEditScreen(vm, ov.loraId, onBack = { nav.overlay = null })
                        Overlay.About -> AboutScreen(onBack = { nav.overlay = null })
                    }
                }
                Box(Modifier.fillMaxSize().navigationBarsPadding().imePadding().padding(bottom = 8.dp)) {
                    SnackbarHost(snackbar, Modifier.align(androidx.compose.ui.Alignment.BottomCenter))
                }
            }
        }
    }
}

@Composable
private fun BottomBar(nav: Nav) {
    val a = LocalAccent.current
    NavigationBar(containerColor = MaterialTheme.colorScheme.surfaceContainerLow, tonalElevation = 0.dp) {
        Tab.entries.forEach { t ->
            val sel = nav.tab == t
            NavigationBarItem(
                selected = sel, onClick = { nav.tab = t },
                icon = { Icon(if (sel) t.iconSelected else t.icon, null) },
                label = { Text(stringResource(t.label), maxLines = 1) },
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = a.light, selectedTextColor = a.light, indicatorColor = a.deep,
                    unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                    unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant,
                ),
            )
        }
    }
}
