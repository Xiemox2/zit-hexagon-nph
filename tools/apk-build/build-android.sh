#!/usr/bin/env bash
# Build libsdbench.so + Hexagon skels (v79, v81) and the signed release APK.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NDK:=${NDK_SHIM}}"            # NDK r29 incl. build/cmake
: "${HEXAGON_SDK_ROOT:=${HEXAGON_SDK_ROOT}}"
: "${HEXAGON_TOOLS_ROOT:=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/21.0.04}"
export PATH=$PATH
cmake -S native -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
  -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
  -DHEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT -DHEXAGON_TOOLS_ROOT=$HEXAGON_TOOLS_ROOT
cmake --build build-android -j"$(nproc)"
H=build-android/sdcpp/ggml/src/ggml-hexagon
mkdir -p app/app/src/main/jniLibs/arm64-v8a app/app/src/main/assets/htp
cp build-android/libsdbench.so app/app/src/main/jniLibs/arm64-v8a/
cp $H/libggml-htp-v79.so $H/libggml-htp-v81.so app/app/src/main/assets/htp/
(cd app && ./gradlew --offline -q assembleRelease)   # needs keystore/sdbench-release.properties
VER=$(sed -n 's/.*versionName = "\(.*\)".*/\1/p' app/app/build.gradle.kts)
mkdir -p out && cp app/app/build/outputs/apk/release/app-release.apk out/sdbench-$VER.apk
sha256sum out/sdbench-$VER.apk
