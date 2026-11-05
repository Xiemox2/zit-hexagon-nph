# ZIT-HEXAGON NPH 0.30.2 (frozen)

**Experimental.** NPH engine + Android app for Z-Image Turbo / ReaZIT on the Hexagon NPU (HTP). This is the **last DiT work** in this project: the DiT line stops here; no further DiT optimisation is planned on NPH.

**Español:** [README.md](README.md).

## Open invitation (Nightmare Mobile and others)

We are publishing this engine and app **so their ideas can be reused**, especially the **unified FP16 LoRA** (below), by other open-source engines on the same MIT base (`happyyzy/stable-diffusion.cpp` + `ggml-hexagon`).

In particular, we **invite the maintainers of [Nightmare Mobile](https://github.com/AbrahamPaulJ/nightmare-mobile)** (and other forks) to study and, if useful, adapt the unified FP16 LoRA and the rest of the NPH changes to speed up their own DiT paths. **There is no affiliation** with Nightmare Mobile or Local Dream: this is an open proposal only, with no partnership or endorsement implied. Their code is CC BY-NC; ours is MIT and **includes none** of theirs.

## What it is

- **App:** `com.zithexagon`, versionCode **45**, versionName **0.30.3** (fix on top of 0.30.2).
- **Engine:** MIT fork of `stable-diffusion.cpp` / `ggml-hexagon` + NPH JNI (`libnph.so`).
- **Reference phone:** Nubia Z80 Ultra (`NX741J`), **SM8850**, Hexagon **HTP v81**, Android 16. A **v79** skel is also shipped.
- **Precision (never lowered):** DiT weights **FP8 E4M3**, non-matmul residual **FP32**, TE **Q8_0**, LoRA never below file precision (exact convert to FP16 + FP32 scale, FP32 accumulate).

**Models are not in this repo.** Get them from public upstreams (below).


## 0.30.3 — fix for 0.30.2 LoRA noise

Public **0.30.2 (vc44)** produced **pure dark noise** with combined LoRAs (fuse on or off). Cause: 0.30.2’s `ADD(d,out)` reorder (for the HMX fuser) broke the working 0.30.1 graph (`ADD(out,d)`); profiles went from **210 mm / 570 ew** to **360 / 420**.

**0.30.3 (vc45)** restores the 0.30.1 ADD, retires the fused HMX op, bumps LoRA disk-cache magic to `NPHLC003`, and logs `nph-lora-sanity` after step 1. Repo set **private** while validating; the 0.30.2 APK was removed from Releases.

## Why 0.30.2

We publish the **0.30.2** tree (includes the 2nd-image fix) with **unified FP16 LoRA ON by default**.

The experimental **“LoRA fused into HMX”** op (add `scale·(B·(A·x))` inside the HMX matmul) is **OFF by default**: phone reports show **no** `nph-lora-fuse` self-test lines, so that path is **not validated on hardware**. The Settings toggle remains for anyone who wants to try; on failure the engine falls back to the 0.30.1 op path.

## What works (phone-measured)

Numbers only from real Z80 Ultra reports. Typical **1024×1024**, **8 steps**, Euler, CFG 1.0, Extremo mode.

| Case | Mean s/step | Generation | Button → image | Notes |
|---|---:|---:|---:|---|
| No LoRA (best s/step) | **7.25** | 60.5 s | 84.2 s | Mejoras set-paralelo + pool-buffers ON; cold context |
| No LoRA (best button→image among nearby runs) | 7.54 | 62.6 s | **67.6** s | Same 0.29 line |
| 2 LoRAs, **per-LoRA** path (0.29) | **~29.4–29.9** | ~241–242 s | ~247 s | LandidzuStyle×0.9 + OiledSkin×0.7 |
| 2 LoRAs, **unified FP16 LoRA** (0.30) | **9.27–10.28** | 85–91 s | ~91–93 s | Same 2 LoRAs; `matmuls_per_dit_pass=360` (was 960) |
| 0.30.1 / 2026-11-01 report (unified) | **10.02–10.15** | ~85–86 s | ~91–92 s | LoRA elementwise still ~2 s/step (570 ops); no HMX fusion active |

Also exercised in those reports: FP8 DiT repack cache, TE cache (RAM+disk), Extremo residency, per-batch HTP profiling. In 0.30.2 source: 2nd image with LoRAs no longer hits `cannot update model sources during execution` (synthetic box test 13/13).

## What does not work / unverified

- **Fused HMX LoRA op:** present, **default OFF**, **no on-phone selftest evidence**. Box Hexagon sim has no HMX. Target ≲0.5 s/step LoRA overhead: **unmeasured**.
- **Step 1** host cost on a fresh context remains high (~8–17 s depending on load/LoRAs).
- **GPU VAE (NPG):** see [NPU + GPU (NPG)](#npu--gpu-npg-tried-and-discarded) — **51.51 s** and very noisy vs ~2–3 s on NPU; **removed** in 0.30.1.
- **Hunyuan TE:** not recognised (out of scope).
- **NSFW** can come from LoRAs/checkpoints; the app does not filter.
- **No real-model image generation** was done on the build box for this publish.

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
