# NDK / Hexagon shims

Tiny CMake overlays used when the NDK or Hexagon SDK `build/` tree is incomplete.

## NDK (`ndk/`)

```bash
NDK_REAL=$ANDROID_HOME/ndk/29.0.14206865
SHIM=$PWD/ndk-shim
mkdir -p "$SHIM" "$SHIM/build/cmake"
for x in CHANGELOG.md NOTICE NOTICE.toolchain README.md meta ndk-build ndk-gdb ndk-lldb \
         ndk-stack ndk-which package.xml prebuilt python-packages shader-tools simpleperf \
         source.properties sources toolchains wrap.sh; do
  ln -sfn "$NDK_REAL/$x" "$SHIM/$x"
done
cp -R ndk/build/cmake/. "$SHIM/build/cmake/"
ln -sfn "$NDK_REAL/build/cmake/android.toolchain.cmake" "$SHIM/build/cmake/android.toolchain.cmake"
# cmake … -DCMAKE_TOOLCHAIN_FILE=$SHIM/build/cmake/android.toolchain.cmake
```

## Hexagon (`hexagon/`)

```bash
HEX_REAL=/path/to/hexagon-sdk/6.7.0.0
SHIM=$PWD/hex-shim
mkdir -p "$SHIM" "$SHIM/build/cmake"
for x in hexagon_sdk.json incs ipc libs readme.txt rtos setup_sdk_env.source tools utils; do
  ln -sfn "$HEX_REAL/$x" "$SHIM/$x"
done
cp hexagon/build/cmake/hexagon_fun.cmake "$SHIM/build/cmake/"
```

The NDK and Hexagon SDK keep their own licenses; these overlays are MIT helpers for this project.
