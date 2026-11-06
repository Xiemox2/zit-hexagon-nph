// NPG bench (MIT) — standalone, does not touch the main ZIT-HEXAGON app.
import java.util.Properties

plugins {
    id("com.android.application")
}

val ksDir = File("/workspace/zit-qnn/sdbench/keystore")
val ksProps = Properties().apply {
    val f = File(ksDir, "sdbench-release.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

android {
    namespace = "com.zithexagon.npgbench"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.zithexagon.npgbench"
        minSdk = 31
        targetSdk = 36
        versionCode = 3
        versionName = "0.1.2"
        ndk { abiFilters += "arm64-v8a" }
    }

    signingConfigs {
        create("release") {
            if (ksProps.isNotEmpty()) {
                storeFile = File(ksDir, File(ksProps.getProperty("storeFile")).name)
                storePassword = ksProps.getProperty("storePassword")
                keyAlias = ksProps.getProperty("keyAlias")
                keyPassword = ksProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    androidResources { noCompress += "so" }
    packaging { jniLibs { useLegacyPackaging = false } }
}
