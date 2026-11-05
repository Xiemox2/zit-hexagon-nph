# Build

## App (APK) con `libnph.so` precompilado

Requisitos: JDK 17+, Android SDK 36, NDK (el CI local usó NDK 29).

```bash
cd app
# opcional: echo "sdk.dir=..." > local.properties
./gradlew :app:assembleRelease
```

`app/src/main/jniLibs/arm64-v8a/libnph.so` y `app/src/main/assets/htp/libggml-htp-v{79,81}.so` ya van en el árbol (Git LFS).

## Motor nativo (`libnph.so`) desde fuente

Hace falta el **Hexagon SDK** (Qualcomm) para regenerar skels; sin él puedes enlazar los skels de `engine/skels/`.

Esquema (adaptar rutas NDK / Hexagon):

```bash
cmake -S engine/native -B build-nph \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 \
  -DNPH_NO_SKELS=ON -DNPH_OPENCL=OFF
cmake --build build-nph --target nph -j
```

`engine/sdcpp/ggml` es un symlink a `engine/ggml`.

## Pruebas sintéticas (caja x86, sin modelo real)

Ver `tests/` y `engine/native/tests/`. No cargan ReaZIT ni LoRAs reales.
