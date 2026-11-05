# Changelog (línea NPH → 0.30.2)

Fechas en America/Merida (UTC-6). Cifras de teléfono solo cuando se indica.

## SdBench 0.1.x → app ZIT-HEXAGON temprana
- Banco nativo sd.cpp + Hexagon HTP; APKs de prueba en HF privado (no publicadas aquí).

## 0.26–0.27
- Residencia Extremo (`NPH_KEEP_COMPUTE`), perfil HTP, overhead de host documentado.
- 0.27.1: soft `runner_end` mantiene buffers entre generaciones.

## 0.28.0-test
- Carga DiT: repack FP8 rápido (NEON+MT), set paralelo, lectura pread, caché de repack en disco, pool de buffers, prefetch DiT durante TE.

## 0.29.0-test
- Caché de salida del TE (RAM + disco).
- NPG experimental (VAE GPU OpenCL): luego medido ~51 s y ruidoso → retirado en 0.30.1.
- Mejor corrida **sin LoRA** en Z80 Ultra: **7.25 s/paso**, generación 60.5 s (botón→imagen 84.2 s en esa corrida; otra corrida cercana 67.6 s botón→imagen a 7.54 s/paso).
- Con 2 LoRAs camino por-LoRA: **~29.4 s/paso**, ~241 s.

## 0.30.0-test
- **LoRA unificada FP16** (concat de rangos, 1 par A/B por capa).
- Medido: **9.27–10.28 s/paso**, 85–91 s con los mismos 2 LoRAs (360 matmuls/pasada vs 960).
- Residencia adaptativa; NPG aún presente pero no es el foco.

## 0.30.1-test
- Paridad w2 (ganancia potencia de 2), revisión por capa al cargar, caché LoRA en disco, unscale diferido w2, perfil LoRA.
- NPG quitado de la UI/build por defecto.
- 2.ª imagen con LoRAs: fallo `cannot update model sources during execution`.
- Corrida ~10.15 s/paso.

## 0.30.2 (esta publicación, versionCode 44)
- Arreglo 2.ª imagen (tracking idle + atajo si el set de LoRAs no cambia).
- Op experimental LoRA en epílogo HMX **OFF por defecto** (no validado en teléfono).
- Congelación del trabajo DiT de este proyecto.
