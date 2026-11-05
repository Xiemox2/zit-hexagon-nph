# Comparación con Nightmare Mobile (hechos)

**Sin afiliación.** Nightmare Mobile: https://github.com/AbrahamPaulJ/nightmare-mobile (CC BY-NC 4.0).  
Este repo es MIT y **no copia** su código ni el de Local Dream. Solo se describen ideas observadas en árboles públicos y lo que NPH implementó por su cuenta sobre la base MIT.

Fuentes internas de la investigación (no publicadas en este repo): notas de lectura de `/workspace/research/engines-ld-nm.md` y el clon de NM.

## Compartido
- `happyyzy/stable-diffusion.cpp` + `ggml-hexagon` (MIT) como motor DiT HTP.

## Nightmare Mobile (público)
- Fork del servidor de Local Dream + ~70 parches propios sobre el DiT.
- Prefetch de segmento en paralelo con la ejecución HTP (CPU‖NPU real).
- Repack FP8 a hilos / palabras de 32 bits; lecturas `pread`; reuso de buffers; FA/RoPE/HVX según ramas.
- LoRA: `lora_apply_mode` AUTO → at_runtime B(A·x) **por LoRA** en HTP.
- MiniMax H3 y otros modelos de vídeo/audio en su producto.

## NPH 0.30
- App y JNI propios; precisión DiT FP8 E4M3 / FP32 / TE Q8_0 sin bajar.
- **LoRA unificada FP16** (concat de rangos) — ganancia medida 29.6→~9.5 s/paso con 2 LoRAs.
- Caché de repack, caché TE, residencia Extremo adaptativa, perfil HTP por op.
- Sin copiar los parches CC BY-NC de NM; ideas reimplementadas donde aplica (repack rápido, etc.).

## Propuesta
Si a NM le interesa la LoRA unificada FP16 u otras piezas MIT de NPH, pueden reimplementarlas en su árbol bajo su propia licencia. Esta publicación es una **invitación abierta**, no un parche ni un PR automático.
