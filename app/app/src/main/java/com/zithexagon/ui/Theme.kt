package com.zithexagon.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.zithexagon.R

/** Wine-purple accents (Tongyi / Z-Image style) on a deep black base. */
@Immutable
data class Accent(
    val id: String,
    val nameRes: Int,
    val main: Color,       // fills: buttons, slider tracks, selected chips
    val deep: Color,       // containers, selected backgrounds
    val light: Color,      // text accents, icons on black
)

val ACCENTS = listOf(
    Accent("vino", R.string.accent_vino, Color(0xFF8E2A6B), Color(0xFF3F1031), Color(0xFFE7A6CF)),
    Accent("ciruela", R.string.accent_ciruela, Color(0xFF7A2459), Color(0xFF350E28), Color(0xFFDDA3C8)),
    Accent("burdeos", R.string.accent_burdeos, Color(0xFF8A2447), Color(0xFF3D0F20), Color(0xFFF0A9BF)),
    Accent("orquidea", R.string.accent_orquidea, Color(0xFF9B3C8C), Color(0xFF431540), Color(0xFFEBB4E2)),
)

fun accentById(id: String) = ACCENTS.firstOrNull { it.id == id } ?: ACCENTS.first()

object ZitPalette {
    val Black = Color(0xFF000000)
    val Surface = Color(0xFF0A090A)
    val Surface1 = Color(0xFF110E10)
    val Surface2 = Color(0xFF171316)
    val Surface3 = Color(0xFF1F191D)
    val Surface4 = Color(0xFF282026)
    val Outline = Color(0xFF3A2F36)
    val OutlineVariant = Color(0xFF241D22)
    val Text = Color(0xFFF2ECEF)
    val TextDim = Color(0xFFB4A8AF)
    val Error = Color(0xFFFF8A9A)
}

val LocalAccent = staticCompositionLocalOf { ACCENTS.first() }

@Composable
fun ZitTheme(accentId: String = "vino", content: @Composable () -> Unit) {
    val a = accentById(accentId)
    val scheme = darkColorScheme(
        primary = a.main,
        onPrimary = Color.White,
        primaryContainer = a.deep,
        onPrimaryContainer = Color(0xFFFBE4F1),
        inversePrimary = a.light,
        secondary = a.light,
        onSecondary = Color(0xFF2A0820),
        secondaryContainer = a.deep,
        onSecondaryContainer = Color(0xFFFBE4F1),
        tertiary = a.light,
        onTertiary = Color(0xFF2A0820),
        tertiaryContainer = a.deep,
        onTertiaryContainer = Color(0xFFFBE4F1),
        background = ZitPalette.Black,
        onBackground = ZitPalette.Text,
        surface = ZitPalette.Surface,
        onSurface = ZitPalette.Text,
        surfaceVariant = ZitPalette.Surface3,
        onSurfaceVariant = ZitPalette.TextDim,
        surfaceTint = a.main,
        surfaceDim = ZitPalette.Black,
        surfaceBright = ZitPalette.Surface4,
        surfaceContainerLowest = ZitPalette.Black,
        surfaceContainerLow = ZitPalette.Surface1,
        surfaceContainer = ZitPalette.Surface2,
        surfaceContainerHigh = ZitPalette.Surface3,
        surfaceContainerHighest = ZitPalette.Surface4,
        outline = ZitPalette.Outline,
        outlineVariant = ZitPalette.OutlineVariant,
        error = ZitPalette.Error,
        onError = Color(0xFF3B0710),
        errorContainer = Color(0xFF5C1020),
        onErrorContainer = Color(0xFFFFDADF),
        scrim = Color.Black,
    )
    val base = Typography()
    val type = base.copy(
        headlineSmall = base.headlineSmall.copy(fontWeight = FontWeight.SemiBold, letterSpacing = (-0.2).sp),
        titleLarge = base.titleLarge.copy(fontWeight = FontWeight.SemiBold, letterSpacing = (-0.2).sp),
        titleMedium = base.titleMedium.copy(fontWeight = FontWeight.SemiBold),
        titleSmall = base.titleSmall.copy(fontWeight = FontWeight.SemiBold),
        labelLarge = base.labelLarge.copy(fontWeight = FontWeight.SemiBold),
    )
    CompositionLocalProvider(LocalAccent provides a) {
        MaterialTheme(colorScheme = scheme, typography = type, content = content)
    }
}

val SectionLabel = TextStyle(fontSize = 12.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.8.sp)
