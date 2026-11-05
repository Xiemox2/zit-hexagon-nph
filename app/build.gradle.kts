// Same toolchain pins as Hunyuan NPU (AGP 9.4.1 / Kotlin 2.4.20 / Gradle 9.7.1).
plugins {
    id("com.android.application") version "9.4.1" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.4.20" apply false
    id("io.github.takahirom.roborazzi") version "1.77.0" apply false
}
