package com.zithexagon

import com.zithexagon.data.TestBlocks
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/** 0.24.3: the diagnostics bench only picks up complete zit-test-*.bin files, once each, in name order. */
class TestBlocksTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun findsOnlyTestBinaries() {
        val a = tmp.newFolder("a"); val b = tmp.newFolder("b")
        File(a, "zit-test-c1-o3.bin").writeText("x")
        File(a, "zit-test-c1-o2.bin.crdownload").writeText("x")
        File(a, "dit_1.bin").writeText("x")
        File(b, "zit-test-c1-o3-tile8.bin").writeText("x")
        File(b, "zit-test-dir.bin").mkdirs()
        val found = TestBlocks.find(listOf(a, b, a, File(tmp.root, "missing"))).map { it.name }
        assertEquals(listOf("zit-test-c1-o3-tile8.bin", "zit-test-c1-o3.bin"), found)
    }
}
