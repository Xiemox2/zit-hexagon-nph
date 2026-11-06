# ZIT-HEXAGON NPH — recommended build: **0.30.1**

**Experimental.** NPH engine + Android app for Z-Image Turbo / ReaZIT on the Hexagon NPU (HTP). This is the **last DiT work** in this project.

**Español:** [README.md](README.md).

## Recommended build: 0.30.1 (vc42)

| | |
|---|---|
| Release | [**v0.30.1-stable**](https://github.com/Xiemox2/zit-hexagon-nph/releases/tag/v0.30.1-stable) |
| APK | `zit-hexagon-0.30.1-release.apk` |
| versionCode | **42** · versionName **0.30.1-test** |
| SHA-256 | `535f150c90cd3b11c80b351947b1bd4203f7ad436d27c435553e584636e737f5` |
| Signing | `b1c30ed7…4e99` (sideload over previous) |

On the Nubia Z80 Ultra (**SM8850** / Hexagon **HTP v81**) this is the **last build that generates correctly with LoRAs**. Download it from Releases and use that.

### Known limitation (0.30.1)

A **2nd image with LoRA** can fail while the DiT stays resident (Extremo mode): `cannot update model sources during execution`.

**Workaround:** free the model between images (Liberar / leave Extremo), then generate again.

## Builds 0.30.2–0.30.5 — not recommended

On the same phone, **0.30.2, 0.30.3, 0.30.4 and 0.30.5** produce **pure dark noise** (black/purple texture, no subject) when LoRAs are used, even when the LoRA graph and step-1 latent stats look “ok”.

**Do not use those builds for LoRA generation.** They remain in Releases only as history.

### Tried and ruled out (root cause: not found)

| Hypothesis | Outcome |
|---|---|
| `ADD(d,out)` vs `ADD(out,d)` order (0.30.2→0.30.3) | Restored; noise **persists** |
| Fused HMX op (`MUL_MAT_LORA`) | Off / retired; noise **persists** |
| Mejoras toggles (set-paralelo, pool-buffers, prefetch-VAE, …) | Matched 0.30.1 profile; noise **persists** |
| `gain=0` when strength ≥ 1.0 (0.30.4) | Forced `gain≥1`; noise **persists** |
| HTP skels from 0.30.2+ (0.30.5 = 0.30.1 skels) | Skels restored; noise **persists** (2026-11-06) |

Also: on CPU, **stock sd.cpp** (per-LoRA) and the **NPH combined path** with the real LoRAs (OiledSkin×0.7 + Landidzu×1.0) produce a **coherent** image at 256²/512². The failure is **device / HTP-specific**, not the CPU LoRA packing. Root cause **not found**; we stay on **0.30.1**.

## Code in this repo

The `engine/` / `app/` tree may include post-0.30.1 changes (0.30.2+ line: 2nd-image fix, HMX fuse retired, gain≥1, etc.). That does **not** change the recommendation: install the APK from **v0.30.1-stable**. Releases 0.30.2–0.30.5 are marked not recommended.

## Open invitation (Nightmare Mobile and others)

We publish this engine and app **so their ideas can be reused**, especially the **unified FP16 LoRA**, by other MIT engines (`happyyzy/stable-diffusion.cpp` + `ggml-hexagon`).

We **invite** the maintainers of [Nightmare Mobile](https://github.com/AbrahamPaulJ/nightmare-mobile) (and other forks) to study and adapt the unified FP16 LoRA if useful. **No affiliation** with Nightmare Mobile or Local Dream. Their code is CC BY-NC; ours is MIT and **includes none** of theirs.

## What it is

- **Recommended app:** `com.zithexagon`, versionCode **42**, versionName **0.30.1-test**.
- **Engine:** MIT fork of `stable-diffusion.cpp` / `ggml-hexagon` + NPH JNI (`libnph.so`).
- **Reference phone:** Nubia Z80 Ultra (`NX741J`), **SM8850**, Hexagon **HTP v81**, Android 16 (**v79** skel also in 0.30.1).
- **Precision (never lowered):** DiT **FP8 E4M3**, non-matmul residual **FP32**, TE **Q8_0**, LoRA never below file precision (exact FP16 + FP32 scale, FP32 accumulate).

**Models are not in this repo.** The converted [ReaZIT-NPH](https://huggingface.co/JiangFeng12/ReaZIT-NPH) package includes the FP8 DiT, Qwen3-4B Q8_0 TE and `ae.safetensors` VAE; see its per-component terms. **ReaZIT weights are not MIT and are not offered here as unrestricted Apache-2.0 weights**.

## What works (phone-measured)

Numbers only from real Z80 Ultra reports. Typical **1024×1024**, **8 steps**, Euler, CFG 1.0, Extremo.

| Case | Mean s/step | Generation | Button → image | Notes |
|---|---:|---:|---:|---|
| No LoRA (best s/step) | **7.25** | 60.5 s | 84.2 s | 0.29; set-paralelo + pool-buffers ON |
| No LoRA (best button→image) | 7.54 | 62.6 s | **67.6** s | Same 0.29 line |
| 2 LoRAs, per-LoRA path (0.29) | **~29.4–29.9** | ~241–242 s | ~247 s | Landidzu×0.9 + OiledSkin×0.7 |
| 2 LoRAs, **unified FP16 LoRA** (**0.30 / 0.30.1**) | **9.27–10.28** | 85–91 s | ~91–93 s | Same 2 LoRAs; 360 matmuls/pass (was 960) |

## What does not work / unverified

- **0.30.2–0.30.5 with LoRA on SM8850/v81:** dark noise; **not recommended**.
- **2nd LoRA image on 0.30.1** while DiT stays resident (see above).
- **GPU VAE (NPG):** discarded — see NPG section.
- **Hunyuan TE:** not recognised.
- **NSFW** can come from LoRAs/checkpoints; the app does not filter.


## NPU + GPU (NPG): tried and discarded

We tried a hybrid **NPG** engine (Hexagon NPU + Adreno 840 GPU via OpenCL). In the **0.29 / 0.30.0** test builds only the **VAE on the GPU** reached the app; the DiT stayed on HTP0. A **DiT matmul split between HMX and the GPU** was planned in NPG benches but **never shipped in the app**.

**Measured on the Z80 Ultra** (report with `vae=GPUOpenCL`, e.g. the 0.30 NPG run):

| | GPU VAE (Adreno 840 / OpenCL) | NPU VAE (`vae=HTP0`, typical reports) |
|---|---:|---:|
| VAE wall (report line) | **51.51 s** | **~2.0–2.9 s** (min measured 2.04 s; one report 4.04 s) |
| Compute | **43.00 s** | ~1.4–1.7 s in those reports |
| Host alloc (profile) | **~15.5 s** (`vae×2 alloc 15522 ms`) | much smaller (rpcmem reserve tenths–~1 s) |
| Synthetic FP32 self-test (no model) | **OK in 452 ms** (8 ops on GPU) | — |

The self-test passed, but in a real generation the GPU VAE was **~20× slower** than the NPU and the image came out **very noisy** (no quality win).

**Why it was dropped:** slower and worse quality; no speed or quality improvement. A GPU path for runtime LoRA was also unnecessary: **unified FP16 LoRA on HMX** already cut the main cost (~29.6 → ~9.3–10.3 s/step with 2 LoRAs). **Removed from the app in 0.30.1** (optional OpenCL code may remain off in the engine tree).

**TE / prompt cache** (separate from NPG): **on by default**. Reports show e.g. `Caché TE: … último: ram_hit` (RAM HIT when reusing an already-encoded prompt). The saving is **small**, because NPU text encoding is already fast.

## Unified FP16 LoRA (why it is faster)

NPH 0.30 design (default ON):

1. Every LoRA (bf16, f32, f16, …) is **exactly converted to FP16** (hi/lo split, power-of-two shifts, **FP32 scale**, no precision below the file).
2. Ranks are **concatenated per layer** → **one A/B pair** per target (rank padded to a multiple of 32 for HMX).
3. Each DiT pass does **one HMX LoRA matmul per layer** instead of one per LoRA.

Measured with two rank-32 LoRAs: **960 → 360** LoRA matmuls/pass, **~29.6 → ~9.3–10.3 s/step**.

That is the idea we hope other MIT-based engines (e.g. Nightmare Mobile) will consider reimplementing themselves.

## vs Nightmare Mobile (facts only; no affiliation)

Shared base: **happyyzy `stable-diffusion.cpp` + `ggml-hexagon` (MIT)**. Local Dream / Nightmare Mobile are **CC BY-NC**; **none** of their code is here. See [docs/nightmare-mobile-comparison.md](docs/nightmare-mobile-comparison.md).

| | Nightmare Mobile (from public tree) | NPH 0.30 (this repo) |
|---|---|---|
| DiT base | sd.cpp + ggml-hexagon | same |
| DiT precision | (per their build) | FP8 E4M3 weights, FP32 non-matmul, TE Q8_0 **not lowered** |
| LoRA on HTP DiT | AUTO → at_runtime B(A·x) per LoRA | **unified FP16 LoRA** (rank concat) |
| CPU‖NPU overlap | segment preload thread (patch 050) | resident DiT; weight prefetch during TE |
| FP8 repack | threaded / 32-bit words (061) | own NEON+MT fast path + disk cache |
| Other | FA exp2, threaded HVX copies, MiniMax H3, fused RoPE (per branches) | TE cache, adaptive residency, per-op HTP profile, synthetic self-tests |

## Getting models

### Historical ReaZIT-QNN package (0.24.x / 0.25.x)

The **QNN/HTP v81 conversion for SM8850** is separately available at **[JiangFeng12/ReaZIT-QNN](https://huggingface.co/JiangFeng12/ReaZIT-QNN)**: CL folder for the **0.25.0–0.25.2 QNN app** (rank-96 LoRAs) and older **0.24.x** SL ZIP (no LoRA). **Not an NPH 0.30.1 model package.** This research archive has known NaN/grid-image and RAM issues; read its card and SHA-256 manifest before downloading.

Original ReaZIT author: **[SassyAI](https://civitai.com/user/SassyAI)** — [original model](https://civitai.com/models/2391667/reazit). Non-commercial converted distribution; original/base-model terms apply; weights are not MIT. MIT conversion tools: [ZIT-HEXAGON-QNN/tools](https://github.com/Xiemox2/ZIT-HEXAGON-QNN/tree/main/tools) and [dictionary](https://github.com/Xiemox2/ZIT-HEXAGON-QNN/tree/main/dictionary).

**Theoretical, untested tools:** [Krea 2 QNN](https://github.com/Xiemox2/Tools_Krea2-npu) already has a public repository (Apache-2.0 code, no weights; not relicensed MIT). The user will not continue this line for now and leaves it for others to finish. [Hunyuan QNN](https://github.com/Xiemox2/Tools_Hunyuan-qnn) (HunyuanVideo 1.5 and 13B → HTP v81, MIT, no weights) is also published as theoretical, untested tools; the DiT and VAE adapters fail on purpose until someone implements them. No Local Dream/Nightmare Mobile code is incorporated.

### Public ReaZIT-NPH package (0.30.1)

Download [JiangFeng12/ReaZIT-NPH](https://huggingface.co/JiangFeng12/ReaZIT-NPH): ReaZIT FP8 E4M3 DiT, Qwen3-4B Q8_0 text encoder and standard VAE. The card includes file sizes, SHA-256 and instructions to import **DiT / TE / VAE** in **Models (NPH)** in the [stable 0.30.1 app](https://github.com/Xiemox2/zit-hexagon-nph/releases/tag/v0.30.1-stable).

**Original ReaZIT author: [SassyAI](https://civitai.com/user/SassyAI)** — [original model](https://civitai.com/models/2391667/reazit). Thank you for permitting this conversion release with author credit and a profile link. **This converted distribution is for non-commercial use only**; the ReaZIT and Z-Image Turbo terms also apply. Apache-2.0 components retain their upstream terms. Our code's MIT license does not license the model weights.

Not hosted here. Use each author’s public HF/Civitai pages (ReaZIT / Z-Image Turbo, your LoRAs, upscalers). Some upscalers in Settings are **CC BY-NC-SA**.

## Release APK

See [Releases](../../releases) for `zit-hexagon-0.30.2-release.apk` (signed; **keystore not published**).

- versionCode 44 / 0.30.2  
- Cert SHA-256: `b1c30ed727183cdf065750369bd3f62e3efeede539c1db5222fefb064c4f4e99`

## Layout

```
engine/   sdcpp + ggml + NPH JNI + skels + prebuilt libnph
app/      Android Gradle project
tests/    synthetic LoRA / 2nd-gen tests (box; no real model)
docs/     changelog, build, comparison
release/  checksums; APK on GitHub Releases
```

## Licence

MIT — [LICENSE](LICENSE), [NOTICE](NOTICE).  
No Local Dream / Nightmare Mobile code (CC BY-NC).
