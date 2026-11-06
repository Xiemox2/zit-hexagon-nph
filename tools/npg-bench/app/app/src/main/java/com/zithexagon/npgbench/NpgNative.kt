// NPG bench (MIT)
package com.zithexagon.npgbench

interface NpgCallback {
    fun onLine(line: String, progress: Boolean)
}

object NpgNative {
    @JvmStatic external fun run(quick: Int, sustainS: Int, fsBig: Int, reportPath: String, cb: NpgCallback): String
    @JvmStatic external fun cancel()
}
