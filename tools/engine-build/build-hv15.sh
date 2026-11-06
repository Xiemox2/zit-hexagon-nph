#!/usr/bin/env bash
# Build libhv15.so + Hexagon skels + signed HV15 Bench APK.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NDK:=${NDK_SHIM}}"
: "${HEXAGON_SDK_ROOT:=${HEXAGON_SDK_ROOT}}"
: "${HEXAGON_TOOLS_ROOT:=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/21.0.04}"
export PATH=$PATH

cmake -S native -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
  -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE" \
  -DHEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT -DHEXAGON_TOOLS_ROOT=$HEXAGON_TOOLS_ROOT
cmake --build build-android -j"$(nproc)" --target hv15

H=build-android/sdcpp/ggml/src/ggml-hexagon
mkdir -p hv15app/app/src/main/jniLibs/arm64-v8a hv15app/app/src/main/assets/htp
cp -f build-android/libhv15.so hv15app/app/src/main/jniLibs/arm64-v8a/
cp -f $H/libggml-htp-v79.so $H/libggml-htp-v81.so hv15app/app/src/main/assets/htp/

(cd hv15app && ./gradlew --offline -q assembleRelease)
VER=$(sed -n 's/.*versionName = "\(.*\)".*/\1/p' hv15app/app/build.gradle.kts)
mkdir -p out
APK=out/hv15bench-$VER.apk
cp -f hv15app/app/build/outputs/apk/release/app-release.apk "$APK"
# zipalign if available
if command -v zipalign >/dev/null 2>&1; then
  zipalign -f -p 4 "$APK" "$APK.aligned" && mv "$APK.aligned" "$APK"
elif [ -x /workspace/toolchain/cmdline-tools/bin/zipalign ]; then
  /workspace/toolchain/cmdline-tools/bin/zipalign -f -p 4 "$APK" "$APK.aligned" && mv "$APK.aligned" "$APK"
else
  # find build-tools zipalign
  Z=$(ls -1 ${ANDROID_HOME}/build-tools/*/zipalign 2>/dev/null | tail -1 || true)
  if [ -n "$Z" ]; then "$Z" -f -p 4 "$APK" "$APK.aligned" && mv "$APK.aligned" "$APK"; fi
fi
sha256sum "$APK"
ls -la "$APK"
