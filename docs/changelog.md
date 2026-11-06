# Changelog (línea NPH)

Fechas en America/Merida (UTC-6). Cifras de teléfono solo cuando se indica.

## Recomendada: 0.30.1 (vc42) — 2026-11-06

- **Build recomendada** para generar con LoRAs en SM8850 / HTP v81.
- Release GitHub: **v0.30.1-stable** · sha256 `535f150c90cd3b11c80b351947b1bd4203f7ad436d27c435553e584636e737f5`.
- Limitación: 2.ª imagen con LoRA puede fallar si el DiT sigue residente → liberar el modelo entre imágenes.
- Incluye LoRA unificada FP16, paridad w2, caché LoRA, NPG quitado.

## 0.30.2–0.30.5 — no recomendadas (ruido LoRA en dispositivo)

En el Z80 Ultra, **todas** producen ruido oscuro con LoRAs. Probado y descartado: orden ADD, fuse HMX, toggles Mejoras, gain≥1, skels 0.30.1. Referencia CPU (stock y combinada) limpia. **Causa raíz no encontrada.** Se recomienda volver a **0.30.1**.

### Detalle por versión (historial)
- **0.30.2:** 2.ª imagen + op fuse HMX; ruido LoRA en teléfono; APK de release eliminado antes.
- **0.30.3:** restauró ADD(out,d), fuse off; ruido **sigue**.
- **0.30.4:** gain≥1; ruido **sigue**.
- **0.30.5:** skels HTP = 0.30.1; ruido **sigue** (confirmado 2026-11-06).

## SdBench 0.1.x → app ZIT-HEXAGON temprana
- Banco nativo sd.cpp + Hexagon HTP; APKs de prueba en HF privado (no publicadas aquí).

## 0.26–0.27
- Residencia Extremo (`NPH_KEEP_COMPUTE`), perfil HTP, overhead de host documentado.
- 0.27.1: soft `runner_end` mantiene buffers entre generaciones.

## 0.28.0-test
- Carga DiT: repack FP8 rápido (NEON+MT), set paralelo, lectura pread, caché de repack en disco, pool de buffers, prefetch DiT durante TE.

## 0.29.0-test
- Caché de salida del TE (RAM + disco; por defecto ON; en reportes `último: ram_hit` / disco).
- NPG experimental (VAE en GPU Adreno 840 / OpenCL junto a Hexagon). Autoprueba a veces fallaba → fallback NPU.
- Mejor corrida **sin LoRA** en Z80 Ultra: **7.25 s/paso**, generación 60.5 s (botón→imagen 84.2 s en esa corrida; otra corrida cercana 67.6 s botón→imagen a 7.54 s/paso).
- Con 2 LoRAs camino por-LoRA: **~29.4 s/paso**, ~241 s.

## NPG en 0.30.0 (probado → descartado en 0.30.1)
- Solo el **VAE en GPU** llegó a la app; el split de matmuls DiT HMX‖GPU **no** se publicó.
- Medido: VAE GPU **51.51 s** (cómputo 43.00 s, alloc ~15.5 s); autoprueba FP32 **OK 452 ms**; VAE NPU típico **~2.0–2.9 s**. Imagen muy ruidosa.
- Motivo del descarte: mucho más lento y peor calidad; LoRA unificada FP16 en HMX ya cubría el cuello de LoRA. Quitado de la app en **0.30.1**.

## 0.30.0-test
- **LoRA unificada FP16** (concat de rangos, 1 par A/B por capa).
- Medido: **9.27–10.28 s/paso**, 85–91 s con los mismos 2 LoRAs (360 matmuls/pasada vs 960).
- Residencia adaptativa; NPG aún presente pero no es el foco.

## 0.30.1-test
- Paridad w2 (ganancia potencia de 2), revisión por capa al cargar, caché LoRA en disco, unscale diferido w2, perfil LoRA.
- **NPG quitado** de la UI/build por defecto (ver arriba).
- 2.ª imagen con LoRAs: fallo `cannot update model sources during execution`.
- Corrida ~10.15 s/paso.

## 0.30.2 (esta publicación, versionCode 44)
- Arreglo 2.ª imagen (tracking idle + atajo si el set de LoRAs no cambia).
- Op experimental LoRA en epílogo HMX **OFF por defecto** (no validado en teléfono).
- Congelación del trabajo DiT de este proyecto.

## 0.30.3 (vc45) — hot-fix LoRA noise
- **Bug 0.30.2:** combined LoRA → pure dark noise; profile 360 mm/420 ew (good 0.30.1: 210/570). Fuse on/off both bad.
- **Root cause (confirmed):** `graph_apply` / `graph_apply_deferred` ADD operand swap to `ADD(d,out)` for HTP_OP_MUL_MAT_LORA changed cgraph order / MUL_MAT+ADD fusion vs working `ADD(out,d)`.
- **Fix:** restore `ADD(out,d)`; hard-disable fused HMX LoRA; hide UI toggle; cache magic NPHLC003; `nph-lora-sanity` after step 1.
- GitHub repo set private; 0.30.2 release APK deleted.
