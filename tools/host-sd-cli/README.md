# Host sd-cli (x86_64 CPU)

See `../lora-debug/cpu_ref_notes.md`.

```bash
cmake -S ../../engine/sdcpp -B build-host -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HEXAGON=OFF -DGGML_OPENCL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
cmake --build build-host -j$(nproc) --target sd-cli
```
