# CPU reference with sd-cli (host)

```bash
cmake -S ../../engine/sdcpp -B build-host -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HEXAGON=OFF -DGGML_OPENCL=OFF
ninja -C build-host sd-cli
```

Models are **not** in this repo (use public Apache-2.0 Z-Image FP8 + Qwen3-4B Q8_0 + ae).

```bash
export NPH_LORA_COMBINED=0
./build-host/bin/sd-cli \
  --diffusion-model /path/dit_fp8.safetensors \
  --llm /path/Qwen3-4B-Q8_0.gguf \
  --vae /path/ae.safetensors \
  --backend cpu --params-backend te=disk --diffusion-fa --threads 6 \
  --lora-apply-mode at_runtime --lora-model-dir /path/loras \
  -W 256 -H 256 --steps 4 --cfg-scale 1.0 --sampling-method euler \
  -s 42 -p 'a detailed photo <lora:Name:0.7>' -o out.png
```

Combined path: rebuild from post-0.30 sources and set `NPH_LORA_COMBINED=1`.
