# tools/ — herramientas MIT de ZIT-HEXAGON NPH

Utilidades propias (y scripts del árbol MIT `stable-diffusion.cpp`) usadas para construir, firmar, depurar LoRA y validar. **No** incluye modelos, pesos, datasets, keystores ni tokens.

**English:** see [README.en.md](README.en.md).

| Carpeta | Qué es |
|---|---|
| `apk-build/` | Scripts para armar APK Android (NPH / NPG-bench) y notas de firma. `hf_upload.example.py` lee `HF_TOKEN` del entorno. |
| `engine-build/` | Build nativo sdbench / skels HTP / fetch del fork sdcpp. |
| `shims/` | Overlays CMake para NDK r29 y Hexagon SDK 6.7 cuando falta `build/`. Ver `shims/README.md`. |
| `npg-bench/` | Fuentes del banco NPG (NPU+GPU): `native/`, `app/`, `scripts/`. |
| `lora-debug/` | `stage1_lora_real.py`, tests C++ de paridad/combinado/fuse, notas de referencia CPU con `sd-cli`. |
| `host-sd-cli/` | Cómo compilar `sd-cli` en CPU (x86) desde `engine/sdcpp`. |
| `validation/` | Scripts y JSON de validación (sin tensores `.pt`). |
| `convert/` | Helpers de conversión FP8/GGUF (MIT). |
| `tokenizer/` | Código THPP (MIT): `THPP_FI_1.1` (solo src), `MINI_src`, guías `TOOLS_1.1`. **Sin** corpora grandes ni modelos entrenados. |

## Requisitos típicos

- Android SDK + NDK r29, CMake, Ninja, JDK 17+.
- Hexagon SDK 6.7 (skel HTP) si compilas backends Hexagon.
- Python 3 para los scripts de LoRA / HF.

## Qué no va aquí

Modelos ReaZIT/Kijai/Qwen/VAE, LoRAs, keystores, tokens HF, SDXL de pago, código de Local Dream / Nightmare Mobile (CC BY-NC).
