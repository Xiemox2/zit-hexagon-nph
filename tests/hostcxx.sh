#!/bin/sh
# Box-runnable test binaries: static x86_64 Android (bionic) build, runs on the Linux box.
exec /workspace/toolchain/android-sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=x86_64-linux-android30 -std=c++17 -static "$@"
