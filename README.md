# ZIT-HEXAGON NPH 0.30.2 (congelado)

**Experimental.** Motor NPH + aplicación Android para Z-Image Turbo / ReaZIT en la NPU Hexagon (HTP). Esta es la **última línea de trabajo sobre el DiT** de este proyecto: aquí se congela el DiT; no se planea seguir optimizándolo en NPH.

**English:** see [README.en.md](README.en.md).

## Invitación abierta (Nightmare Mobile y otros)

Publicamos este motor y la app **para que sus ideas se puedan reutilizar**, sobre todo la **LoRA unificada en FP16** (más abajo), en otros motores de código abierto basados en la misma base MIT (`happyyzy/stable-diffusion.cpp` + `ggml-hexagon`).

En particular, **invitamos a quienes mantienen [Nightmare Mobile](https://github.com/AbrahamPaulJ/nightmare-mobile)** (u otros forks) a estudiar y, si les sirve, adaptar la LoRA unificada FP16 y el resto de cambios NPH para acelerar sus propios caminos DiT. **No hay afiliación** con Nightmare Mobile ni con Local Dream: solo una propuesta abierta, sin compromiso ni respaldo implícito. Su código es CC BY-NC; el nuestro es MIT y **no incluye** nada de esos proyectos.

## Qué es

- **App:** `com.zithexagon`, versionCode **45**, versionName **0.30.3** (arreglo sobre 0.30.2).
- **Motor:** fork MIT de `stable-diffusion.cpp` / `ggml-hexagon` + capa JNI NPH (`libnph.so`).
- **Teléfono de referencia:** Nubia Z80 Ultra (`NX741J`), SoC **SM8850**, Hexagon **HTP v81**, Android 16. También se incluye el skel **v79**.
- **Precisión (no se bajó):** DiT pesos **FP8 E4M3**, residuales no-matmul **FP32**, TE **Q8_0**, LoRA nunca por debajo de la precisión del archivo (conversión exacta a FP16 + escala FP32, acumulación FP32).

Los **modelos no vienen en el repo**. Hay que conseguirlos en sus fuentes públicas (ver abajo).


## 0.30.3 — arreglo del ruido LoRA de 0.30.2

La release pública **0.30.2 (vc44)** generaba **ruido oscuro** con LoRAs combinadas (con o sin fusión HMX). Causa: el orden `ADD(d,out)` de 0.30.2 (para el fuser) rompía el grafo 0.30.1 (`ADD(out,d)`); el perfil pasaba de **210 mm / 570 ew** a **360 / 420**.

**0.30.3 (vc45)** restaura el ADD de 0.30.1, retira el op fusionado HMX, sube la magia de caché LoRA a `NPHLC003` y añade `nph-lora-sanity` tras el paso 1. El repo quedó **privado** mientras se valida; el APK de 0.30.2 se eliminó de Releases.

## Estado congelado: por qué 0.30.2

Se publica el árbol **0.30.2** (con el arreglo de la 2.ª imagen) y la **LoRA unificada FP16 activa por defecto**.

El op experimental **«LoRA fusionada en HMX»** (sumar `escala·(B·(A·x))` dentro del matmul HMX) va **apagado por defecto**: en los reportes del teléfono **no** aparece la autoverificación `nph-lora-fuse`, así que ese camino **no está validado en hardware**. El interruptor sigue en Ajustes por si alguien quiere probarlo; si falla, el motor vuelve solo al camino 0.30.1.

## Qué funciona (cifras medidas en el Z80 Ultra)

Solo números sacados de reportes reales del teléfono. Resolución típica **1024×1024**, **8 pasos**, Euler, CFG 1.0, modo Extremo.

| Caso | Media s/paso | Generación | Botón → imagen | Notas |
|---|---:|---:|---:|---|
| Sin LoRA (mejor s/paso) | **7.25** | 60.5 s | 84.2 s | Mejoras set-paralelo + pool-buffers ON; contexto nuevo (carga fría) |
| Sin LoRA (mejor botón→imagen entre corridas cercanas) | 7.54 | 62.6 s | **67.6** s | Misma línea 0.29 |
| 2 LoRAs, camino **por-LoRA** (0.29) | **~29.4–29.9** | ~241–242 s | ~247 s | LandidzuStyle×0.9 + OiledSkin×0.7 |
| 2 LoRAs, **LoRA unificada FP16** (0.30) | **9.27–10.28** | 85–91 s | ~91–93 s | Mismos 2 LoRAs; `matmuls_per_dit_pass=360` (antes 960) |
| 0.30.1 / reporte 2026-11-01 (unificada) | **10.02–10.15** | ~85–86 s | ~91–92 s | Elementwise LoRA aún ~2 s/paso (570 ops); sin fusión HMX activa |

Otras piezas que sí se usan en esos reportes:

- Caché de repack FP8 del DiT (hits de varios GB durante el prefetch).
- Caché de TE (RAM + disco).
- Residencia Extremo (DiT+VAE se quedan si hay RAM libre ≥ umbral adaptativo).
- Perfil HTP por op-batch (HMX / HVX / DMA / LoRA).
- En 0.30.2: 2.ª imagen con LoRAs **sin** el error `cannot update model sources during execution` (arreglado en fuente; la prueba sintética en caja pasó 13/13).

## Qué no funciona o no está verificado

- **Op LoRA fusionada en HMX:** código presente, **OFF por defecto**, **sin evidencia de selftest en el teléfono**. El simulador de la caja no modela HMX. Meta ≲0.5 s/paso de overhead LoRA: **no medida**.
- **Paso 1** sigue caro en host (reserva gallocr / rpcmem) en la 1.ª imagen de un contexto nuevo (~8–17 s según carga y LoRAs).
- **VAE en GPU (NPG / OpenCL):** ver sección [NPU + GPU (NPG)](#npu--gpu-npg-probado-y-descartado) — **51.51 s** y muy ruidoso vs ~2–3 s en NPU; **quitado** en 0.30.1.
- **TE Hunyuan:** no reconocido (fuera de alcance).
- **NSFW:** puede salir de los LoRAs o del checkpoint; la app no filtra contenido.
- **No se generaron imágenes en el servidor de build** al publicar: solo pruebas sintéticas y la APK firmada.

## NPU + GPU (NPG): probado y descartado

Se probó un motor híbrido **NPG** (NPU Hexagon + GPU Adreno 840 por OpenCL). En las builds de prueba **0.29 / 0.30.0** solo llegó a la app el **VAE en la GPU**; el DiT seguía en HTP0. Un **reparto de matmuls del DiT entre HMX y la GPU** se planificó en bancos NPG, pero **nunca entró en la app**.

**Medido en el Z80 Ultra** (reporte con `vae=GPUOpenCL`, p. ej. la corrida NPG de 0.30):

| | VAE GPU (Adreno 840 / OpenCL) | VAE NPU (`vae=HTP0`, reportes típicos) |
|---|---:|---:|
| Tiempo VAE (línea del reporte) | **51.51 s** | **~2.0–2.9 s** (mín. medido 2.04 s; un reporte 4.04 s) |
| Cómputo | **43.00 s** | ~1.4–1.7 s en esos reportes |
| Alloc host (perfil) | **~15.5 s** (`vae×2 alloc 15522 ms`) | mucho menor (reserva rpcmem del orden de décimas–1 s) |
| Autoprueba sintética FP32 (sin modelo) | **OK en 452 ms** (8 ops en GPU) | — |

La autoprueba pasó, pero en la generación real el VAE en GPU fue **~20× más lento** que en la NPU y la imagen salió **muy ruidosa** (sin ganancia de calidad).

**Por qué se descartó:** más lento y peor calidad; no mejoró ni velocidad ni imagen. Un camino GPU para LoRA en runtime tampoco hacía falta: la **LoRA unificada FP16 en HMX** ya bajó el coste principal (~29.6 → ~9.3–10.3 s/paso con 2 LoRAs). Se **quitó de la app en 0.30.1** (el código OpenCL opcional puede quedar apagado en el árbol del motor).

**Caché del TE / prompt** (aparte de NPG): está **encendida por defecto**. En reportes aparece p. ej. `Caché TE: … último: ram_hit` («HIT» en RAM al reutilizar el prompt ya codificado). El ahorro es **pequeño**, porque el TE en NPU ya es rápido (a menudo menos de 1 s con caché fría de pesos aparte; con `ram_hit` el cómputo del TE casi no se nota).

## LoRA unificada FP16 (por qué es más rápida)

Diseño NPH 0.30 (activo por defecto):

1. Cada LoRA (bf16, f32, f16, …) se **convierte de forma exacta a FP16** (descomposición hi/lo, desplazamientos potencia de 2, **escala FP32**, sin perder lo que traía el archivo).
2. Se **concatenan los rangos** por capa → **un solo par A/B** por objetivo (rango relleno a múltiplo de 32, apto HMX).
3. En cada pasada del DiT hay **1 matmul LoRA HMX por capa** en lugar de uno por LoRA.

En el reporte medido con 2 LoRAs de rango 32:

- camino por-LoRA: **960** matmuls LoRA por pasada DiT → **~29.6 s/paso**;
- unificada: **360** matmuls → **~9.3–10.3 s/paso**.

Eso es lo que pedimos que se considere reutilizar en otros motores MIT (p. ej. Nightmare Mobile): misma idea, implementación propia de cada proyecto.

## Frente a Nightmare Mobile (solo hechos; sin afiliación)

Base común: **happyyzy `stable-diffusion.cpp` + `ggml-hexagon` (MIT)**. Local Dream y Nightmare Mobile son **CC BY-NC**; **cero** de su código está aquí. Detalle en [docs/comparacion-nightmare-mobile.md](docs/comparacion-nightmare-mobile.md).

| | Nightmare Mobile (ideas vistas en su árbol público) | NPH 0.30 (este repo) |
|---|---|---|
| Base DiT | sd.cpp + ggml-hexagon | igual |
| Precisión DiT | (según su build) | FP8 E4M3 pesos, FP32 no-matmul, TE Q8_0 **sin bajar** |
| LoRA en DiT HTP | `lora_apply_mode` AUTO → at_runtime B(A·x) por LoRA | **LoRA unificada FP16** (concat de rangos) |
| Solape CPU‖NPU | prefetch de segmento en hilo mientras corre el actual (parche 050) | DiT residente; prefetch de pesos durante TE; sin su parche 050 |
| Repack FP8 | hilos / palabras 32-bit (061) | repack rápido NEON+MT propio + caché en disco |
| Otras | FA exp2, copias HVX a hilos, MiniMax H3, RoPE fusionado (según ramas) | caché TE, residencia adaptativa, perfil HTP por op, self-tests sintéticos |

## Cómo obtener modelos

No se alojan aquí. Fuentes públicas típicas (licencias propias de cada autor):

- Checkpoint Z-Image / ReaZIT: páginas públicas del autor en Hugging Face / Civitai (busca «ReaZIT» / «Z-Image Turbo»).
- LoRAs de estilo ZIT: las que uses en Civitai/HF (p. ej. las de las pruebas fueron LandidzuStyle y OiledSkin; descárgalas tú).
- Upscalers listados en la app (UltraSharp / AnimeSharp): enlaces HF en Ajustes; algunos son **CC BY-NC-SA**.

## APK de release

En [Releases](../../releases) está `zit-hexagon-0.30.2-release.apk` (firmada; **no** se publica el keystore).

- versionCode 44 / 0.30.2  
- Cert SHA-256: `b1c30ed727183cdf065750369bd3f62e3efeede539c1db5222fefb064c4f4e99`

## Estructura del repo

```
engine/     motor (sdcpp + ggml + JNI NPH + skels + libnph precompilado)
app/        proyecto Android (Gradle)
tests/      pruebas sintéticas de LoRA / 2.ª gen (caja; sin modelo real)
docs/       changelog, build, comparación
release/    checksums; la APK va en GitHub Releases
```

Build: [docs/build.md](docs/build.md). Changelog: [docs/changelog.md](docs/changelog.md).

## Licencia

MIT — ver [LICENSE](LICENSE) y [NOTICE](NOTICE).  
No incluye código de Local Dream ni Nightmare Mobile (CC BY-NC).
