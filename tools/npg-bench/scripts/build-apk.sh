#!/usr/bin/env bash
# Full build: native lib + skels, then the signed APK into out/. MIT.
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/build-android.sh
export JAVA_HOME=${JAVA_HOME} PATH=${JAVA_HOME}/bin:$PATH
(cd app && ./gradlew -q assembleRelease)
VER=$(sed -n 's/.*versionName = "\(.*\)".*/\1/p' app/app/build.gradle.kts)
mkdir -p out && cp app/app/build/outputs/apk/release/app-release.apk out/npg-bench-$VER.apk
(cd out && sha256sum npg-bench-$VER.apk > npg-bench-$VER.apk.sha256 && cat npg-bench-$VER.apk.sha256)
