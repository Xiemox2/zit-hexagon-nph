#!/usr/bin/env bash
# Fetch the pinned happyyzy/stable-diffusion.cpp fork (MIT) and apply sdbench patches.
set -euo pipefail
cd "$(dirname "$0")/.."
SDCPP_COMMIT=a63954dc794f6daade32c407da1a9c4bc2d73cf0   # branch work/qualcomm-hexagon-optimizations
GGML_COMMIT=f8758628270072da26ad44baf5e189d4695d9800
[ -d sdcpp ] || git clone -b work/qualcomm-hexagon-optimizations https://github.com/happyyzy/stable-diffusion.cpp sdcpp
git -C sdcpp checkout "$SDCPP_COMMIT"
git -C sdcpp submodule update --init --recursive
git -C sdcpp/ggml checkout "$GGML_COMMIT"
git -C sdcpp/ggml apply ../../patches/ggml-sdbench.patch
