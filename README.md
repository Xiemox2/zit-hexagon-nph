# ZIT-HEXAGON NPH — build recomendada: **0.30.1**

**Experimental.** Motor NPH + aplicación Android para Z-Image Turbo / ReaZIT en la NPU Hexagon (HTP). Esta es la **última línea de trabajo sobre el DiT** de este proyecto.

**English:** see [README.en.md](README.en.md).

## Build recomendada: 0.30.1 (vc42)

| | |
|---|---|
| Release | [**v0.30.1-stable**](https://github.com/Xiemox2/zit-hexagon-nph/releases/tag/v0.30.1-stable) |
| APK | `zit-hexagon-0.30.1-release.apk` |
| versionCode | **42** · versionName **0.30.1-test** |
| SHA-256 | `535f150c90cd3b11c80b351947b1bd4203f7ad436d27c435553e584636e737f5` |
| Firma | `b1c30ed7…4e99` (se instala encima) |

En el Nubia Z80 Ultra (**SM8850** / Hexagon **HTP v81**) es la **última build que genera bien con LoRAs**. Descárgala desde Releases y úsala.

### Limitación conocida (0.30.1)

La **2.ª imagen con LoRA** puede fallar si el DiT sigue residente (modo Extremo): `cannot update model sources during execution`.

**Solución:** libera el modelo entre imágenes (Liberar / salir de Extremo) y genera de nuevo.

## Builds 0.30.2–0.30.5 — no recomendadas

En el mismo teléfono, **0.30.2, 0.30.3, 0.30.4 y 0.30.5** producen **ruido oscuro** (textura negra/púrpura, sin sujeto) al usar LoRAs, aunque el grafo LoRA y las estadísticas del paso 1 se vean “ok”.

**No uses esas builds para generar con LoRA.** Quedan en Releases solo como historial.

### Qué se probó y se descartó (causa raíz: no encontrada)

En caja (CPU) y en el teléfono se intentó y **se descartó** como causa:

| Hipótesis | Resultado |
|---|---|
| Orden `ADD(d,out)` vs `ADD(out,d)` (0.30.2→0.30.3) | Restaurado; el ruido **sigue** |
| Op fusionado HMX (`MUL_MAT_LORA`) | Apagado / retirado; el ruido **sigue** |
| Toggles de Mejoras (set-paralelo, pool-buffers, prefetch-VAE, …) | Con el perfil de 0.30.1; el ruido **sigue** |
| `gain=0` cuando strength ≥ 1.0 (0.30.4) | `gain≥1` forzado; el ruido **sigue** |
| Skels HTP de 0.30.2+ (0.30.5 = skels de 0.30.1) | Skels restaurados; el ruido **sigue** (2026-11-06) |

Además: en CPU, **sd.cpp stock** (LoRA por-LoRA) y el **camino NPH combinado** con las LoRAs reales (OiledSkin×0.7 + Landidzu×1.0) generan imagen **coherente** a 256²/512². El fallo es **específico del dispositivo / HTP**, no del empaquetado LoRA en CPU. La causa raíz **no se encontró**; se vuelve a **0.30.1**.

## Código en este repo

El árbol `engine/` / `app/` del repo puede incluir cambios posteriores a 0.30.1 (línea 0.30.2+: 2.ª imagen, fuse HMX retirado, gain≥1, etc.). **Eso no invalida** la recomendación: el APK que debes instalar es el de **v0.30.1-stable**. Los APK 0.30.2–0.30.5 están marcados como no recomendados en Releases.

## Invitación abierta (Nightmare Mobile y otros)

Publicamos este motor y la app **para que sus ideas se puedan reutilizar**, sobre todo la **LoRA unificada en FP16**, en otros motores MIT (`happyyzy/stable-diffusion.cpp` + `ggml-hexagon`).

**Invitamos** a quienes mantienen [Nightmare Mobile](https://github.com/AbrahamPaulJ/nightmare-mobile) (u otros forks) a estudiar y adaptar la LoRA unificada FP16 si les sirve. **No hay afiliación** con Nightmare Mobile ni con Local Dream. Su código es CC BY-NC; el nuestro es MIT y **no incluye** nada de esos proyectos.

## Qué es

- **App recomendada:** `com.zithexagon`, versionCode **42**, versionName **0.30.1-test**.
- **Motor:** fork MIT de `stable-diffusion.cpp` / `ggml-hexagon` + JNI NPH (`libnph.so`).
- **Teléfono de referencia:** Nubia Z80 Ultra (`NX741J`), **SM8850**, Hexagon **HTP v81**, Android 16 (skel **v79** también incluido en 0.30.1).
- **Precisión (no se bajó):** DiT **FP8 E4M3**, residuales no-matmul **FP32**, TE **Q8_0**, LoRA nunca por debajo del archivo (FP16 exacto + escala FP32, acumulación FP32).

Los **modelos no vienen en el repo**. El paquete convertido [ReaZIT-NPH](https://huggingface.co/JiangFeng12/ReaZIT-NPH) incluye DiT FP8, TE Qwen3-4B Q8_0 y VAE `ae.safetensors`; consulta sus términos por componente. **Los pesos ReaZIT no son MIT ni se ofrecen aquí como Apache-2.0 sin restricciones**.

## Qué funciona (cifras medidas en el Z80 Ultra)

Solo números de reportes reales. Típico **1024×1024**, **8 pasos**, Euler, CFG 1.0, Extremo.

| Caso | Media s/paso | Generación | Botón → imagen | Notas |
|---|---:|---:|---:|---|
| Sin LoRA (mejor s/paso) | **7.25** | 60.5 s | 84.2 s | 0.29; set-paralelo + pool-buffers ON |
| Sin LoRA (mejor botón→imagen) | 7.54 | 62.6 s | **67.6** s | Misma línea 0.29 |
| 2 LoRAs, camino por-LoRA (0.29) | **~29.4–29.9** | ~241–242 s | ~247 s | Landidzu×0.9 + OiledSkin×0.7 |
| 2 LoRAs, **LoRA unificada FP16** (**0.30 / 0.30.1**) | **9.27–10.28** | 85–91 s | ~91–93 s | Mismos 2 LoRAs; 360 matmuls/pasada (antes 960) |

## Qué no funciona o no está verificado

- **0.30.2–0.30.5 con LoRA en SM8850/v81:** ruido oscuro; **no recomendadas**.
- **2.ª imagen con LoRA en 0.30.1** si el DiT sigue residente (ver arriba).
- **VAE en GPU (NPG):** descartado — ver sección NPG.
- **TE Hunyuan:** no reconocido.
- **NSFW** puede salir de LoRAs/checkpoints; la app no filtra.


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

### ReaZIT-QNN histórico (0.24.x / 0.25.x)

La conversión **QNN/HTP v81 para SM8850** está en **[JiangFeng12/ReaZIT-QNN](https://huggingface.co/JiangFeng12/ReaZIT-QNN)**: carpeta CL para la app QNN **0.25.0–0.25.2** (LoRA rango 96) y ZIP SL anterior **0.24.x** (sin LoRA). **No sirve como paquete NPH de 0.30.1**. Es un archivo de investigación con fallos conocidos (NaN/cuadrículas y límites de RAM); consulta la tarjeta y sus SHA-256 antes de descargar.

ReaZIT es de **[SassyAI](https://civitai.com/user/SassyAI)** — [modelo original](https://civitai.com/models/2391667/reazit). Distribución convertida no comercial; términos originales y del modelo base aplicables; pesos no MIT. Herramientas de conversión MIT: [ZIT-HEXAGON-QNN/tools](https://github.com/Xiemox2/ZIT-HEXAGON-QNN/tree/main/tools) y [diccionario](https://github.com/Xiemox2/ZIT-HEXAGON-QNN/tree/main/dictionary).

**Herramientas teóricas, no probadas:** [Krea 2 QNN](https://github.com/Xiemox2/Tools_Krea2-npu) ya tiene repo público (código Apache-2.0, sin pesos; no se relicencia MIT). El usuario no continuará esta línea por ahora y la deja para que otras personas la terminen. [Hunyuan QNN](https://github.com/Xiemox2/Tools_Hunyuan-qnn) (HunyuanVideo 1.5 y 13B → HTP v81, MIT, sin pesos) también está publicado como herramientas teóricas no probadas; los adaptadores del DiT y del VAE fallan a propósito hasta que alguien los implemente. No se incorpora código de Local Dream/Nightmare Mobile.

### ReaZIT-NPH público (0.30.1)

Descarga [JiangFeng12/ReaZIT-NPH](https://huggingface.co/JiangFeng12/ReaZIT-NPH): DiT ReaZIT FP8 E4M3, codificador Qwen3-4B Q8_0 y VAE estándar. La tarjeta incluye tamaños, SHA-256 y pasos para importar **DiT / TE / VAE** en **Modelos (NPH)** de la [0.30.1 estable](https://github.com/Xiemox2/zit-hexagon-nph/releases/tag/v0.30.1-stable).

**Autora original de ReaZIT: [SassyAI](https://civitai.com/user/SassyAI)** — [modelo original](https://civitai.com/models/2391667/reazit). Gracias por permitir publicar esta conversión con crédito y enlace a su perfil. **Distribución convertida solo para uso no comercial**; aplican los términos de ReaZIT y de Z-Image Turbo. Los componentes Apache-2.0 conservan sus términos originales. La licencia MIT de nuestro código no licencia los pesos.

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
