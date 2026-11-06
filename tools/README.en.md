# tools/ — ZIT-HEXAGON NPH MIT tooling

Build, sign, LoRA-debug and validation helpers. **No** models, weights, datasets, keystores, or tokens.

**Español:** [README.md](README.md).

| Folder | Contents |
|---|---|
| `apk-build/` | Android APK build scripts (NPH / NPG-bench) + signing notes. `hf_upload.example.py` uses env `HF_TOKEN`. |
| `engine-build/` | Native sdbench / HTP skel build + sdcpp fetch. |
| `shims/` | CMake overlays for NDK r29 and Hexagon SDK 6.7 when `build/` is missing. See `shims/README.md`. |
| `npg-bench/` | NPG bench sources: `native/`, `app/`, `scripts/`. |
| `lora-debug/` | `stage1_lora_real.py`, C++ parity/combined/fuse tests, CPU `sd-cli` notes. |
| `host-sd-cli/` | How to build host CPU `sd-cli` from `engine/sdcpp`. |
| `validation/` | Validation scripts/JSON (no `.pt` tensors). |
| `convert/` | FP8/GGUF conversion helpers (MIT). |
| `tokenizer/` | THPP code (MIT): `THPP_FI_1.1` src, `MINI_src`, `TOOLS_1.1` guides. **No** large corpora or trained models. |

## Not included

ReaZIT/Kijai/Qwen/VAE weights, LoRAs, keystores, HF tokens, paid SDXL, any Local Dream / Nightmare Mobile code (CC BY-NC).
