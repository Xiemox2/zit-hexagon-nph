# Conversion helpers

`convert_fp8_scale_to_bf16.py` is from the MIT `stable-diffusion.cpp` tree (also under `engine/sdcpp/scripts/`).

```bash
sd-cli -M convert -m model_fp16.safetensors -o model-q4_0.gguf --type q4_0 -v
```

Do not commit weights.
