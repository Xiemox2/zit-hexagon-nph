#!/usr/bin/env bash
# Build libnpgbench.so + Hexagon skels (v79, v81) and the signed npg-bench APK. MIT.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NDK:=./ndkshim}"
: "${HEXAGON_SDK_ROOT:=./hexshim}"
: "${HEXAGON_TOOLS_ROOT:=${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/21.0.04}"
export PATH=$PATH
(cd native && ./gen_kernels.sh)
cmake -S native -B build-android -G Ninja \
  -DCMAKE_SYSTEM_NAME=Android -DCMAKE_SYSTEM_VERSION=28 -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
  -DCMAKE_ANDROID_NDK=$NDK -DCMAKE_ANDROID_STL_TYPE=c++_static -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
  -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
  -DHEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT -DHEXAGON_TOOLS_ROOT=$HEXAGON_TOOLS_ROOT
cmake --build build-android -j"$(nproc)" --target npgbench
# skels: same ggml-hexagon sources (unchanged since), built 2026-10-29 for the main app
H=./build-android/sdcpp/ggml/src/ggml-hexagon
mkdir -p app/app/src/main/jniLibs/arm64-v8a app/app/src/main/assets/htp
cp build-android/libnpgbench.so app/app/src/main/jniLibs/arm64-v8a/
cp $H/libggml-htp-v79.so $H/libggml-htp-v81.so app/app/src/main/assets/htp/
