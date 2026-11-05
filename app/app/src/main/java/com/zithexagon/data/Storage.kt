package com.zithexagon.data

import android.content.Context
import android.os.Environment
import java.io.File

/** Where things live. App folders need no permission; Download/ZITHEXAGON needs all-files access. */
object Storage {
    const val SHARED_FOLDER = "ZITHEXAGON"

    /** Folder the user can fill over USB: /sdcard/Download/ZITHEXAGON/{models,loras}. */
    val sharedRoot: File
        get() = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), SHARED_FOLDER)

    fun sharedModels() = File(sharedRoot, "models")
    fun sharedLoras() = File(sharedRoot, "loras")

    private fun ext(context: Context, name: String): File =
        File(context.getExternalFilesDir(null) ?: context.filesDir, name).apply { mkdirs() }

    fun models(context: Context) = ext(context, "models")
    fun loras(context: Context) = ext(context, "loras")
    fun outputs(context: Context) = ext(context, "outputs")
    fun previews(context: Context) = File(context.filesDir, "lora_previews").apply { mkdirs() }

    fun hasAllFilesAccess(): Boolean = runCatching { Environment.isExternalStorageManager() }.getOrDefault(false)

    fun freeBytes(context: Context): Long = (context.getExternalFilesDir(null) ?: context.filesDir).usableSpace
}

fun Long.humanBytes(): String = when {
    this >= 1L shl 30 -> String.format(java.util.Locale.US, "%.2f GB", this / (1L shl 30).toDouble())
    this >= 1L shl 20 -> String.format(java.util.Locale.US, "%.1f MB", this / (1L shl 20).toDouble())
    this >= 1L shl 10 -> String.format(java.util.Locale.US, "%.0f KB", this / (1L shl 10).toDouble())
    else -> "$this B"
}

fun File.dirSize(): Long = if (isFile) length() else (listFiles()?.sumOf { it.dirSize() } ?: 0L)
