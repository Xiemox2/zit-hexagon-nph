# NPH engine

- `sdcpp/` — fork MIT de happyyzy stable-diffusion.cpp (cambios NPH: LoRA unificada, caches, residencia, …).
- `ggml/` — ggml + ggml-hexagon (incl. op experimental `MUL_MAT_LORA`, OFF por defecto).
- `native/` — `nph_jni.cpp` + CMake para `libnph.so`.
- `skels/` — `libggml-htp-v79.so` / `v81.so` precompilados.
- `prebuilt/libnph.so` — binario arm64-v8a de la congelación 0.30.2 (vc44).

See ../docs/build.md.
