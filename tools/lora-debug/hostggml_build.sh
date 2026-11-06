#!/bin/sh
# Host (box) ggml CPU static lib for synthetic unit tests (x86_64-android static, runs on the box).
set -e
BIN=${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin
CC="$BIN/clang --target=x86_64-linux-android30"
CXX="$BIN/clang++ --target=x86_64-linux-android30"
G=./sdcpp/ggml
OUT=$(pwd)/obj; mkdir -p $OUT
DEFS="-DGGML_USE_CPU -DNDEBUG -D_GNU_SOURCE -DGGML_SCHED_MAX_COPIES=4"
INC="-I$(pwd)/gen -I$G/include -I$G/src -I$G/src/ggml-cpu"
FL="-O2 -fPIC -msse4.2 -mavx2 -mfma -mf16c"
pids=""
for f in ggml.c ggml-alloc.c ggml-quants.c ggml-cpu/ggml-cpu.c ggml-cpu/quants.c ggml-cpu/arch/x86/quants.c; do
  o=$OUT/$(echo $f | tr / _).o; $CC -std=c11 $FL $DEFS $INC -c $G/src/$f -o $o & pids="$pids $!"
done
for f in ggml.cpp ggml-backend.cpp ggml-backend-reg.cpp ggml-backend-meta.cpp ggml-threading.cpp ggml-opt.cpp gguf.cpp ggml-cpu/ggml-cpu.cpp ggml-cpu/ops.cpp ggml-cpu/vec.cpp ggml-cpu/binary-ops.cpp ggml-cpu/unary-ops.cpp ggml-cpu/traits.cpp ggml-cpu/repack.cpp ggml-cpu/hbm.cpp ggml-cpu/iqp.cpp ggml-cpu/arch/x86/repack.cpp ggml-cpu/arch/x86/cpu-feats.cpp; do
  o=$OUT/$(echo $f | tr / _).o; $CXX -std=c++17 $FL $DEFS $INC -c $G/src/$f -o $o & pids="$pids $!"
done
for p in $pids; do wait $p; done
rm -f libggml_host.a; $BIN/llvm-ar rcs libggml_host.a $OUT/*.o
echo OK
