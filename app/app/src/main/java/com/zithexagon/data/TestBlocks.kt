package com.zithexagon.data

import android.content.Context
import android.os.Environment
import java.io.File

/** Diagnostics: test context binaries ("zit-test-*.bin") the user downloads from the repo's test/ folder. */
object TestBlocks {
    const val PREFIX = "zit-test"
    const val MIN_BYTES = 64L shl 20   // smaller = incomplete download

    fun isTestFile(f: File): Boolean = f.isFile && f.name.startsWith(PREFIX) && f.name.endsWith(".bin")

    /** Folders searched (all-files access is needed for Download/). */
    fun folders(context: Context): List<File> {
        val dl = runCatching { Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) }.getOrNull()
        return listOfNotNull(dl, Storage.sharedRoot, File(Storage.sharedRoot, "test"),
            File(context.getExternalFilesDir(null) ?: context.filesDir, "test"))
    }

    fun find(dirs: List<File>): List<File> = dirs.flatMap { d -> d.listFiles()?.filter(::isTestFile).orEmpty() }
        .distinctBy { runCatching { it.canonicalPath }.getOrDefault(it.path) }.sortedBy { it.name }

    /** The installed model's first transformer block (dit_1), used as the reference. */
    fun reference(model: ZitModel): File? {
        val dir = model.dir ?: return null
        return runCatching { ModelManifest.parse(File(dir, ModelManifest.FILE_NAME).readText()).files["dit_1"] }.getOrNull()
            ?.let { File(dir, it) }?.takeIf { it.isFile }
    }
}
