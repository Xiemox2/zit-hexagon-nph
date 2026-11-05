# Comparison with Nightmare Mobile (facts)

**No affiliation.** Nightmare Mobile: https://github.com/AbrahamPaulJ/nightmare-mobile (CC BY-NC 4.0).  
This repo is MIT and **does not copy** their code or Local Dream’s. We only describe ideas seen in public trees and what NPH implemented independently on the MIT base.

## Shared
- `happyyzy/stable-diffusion.cpp` + `ggml-hexagon` (MIT) as the HTP DiT engine.

## Nightmare Mobile (public)
- Fork of Local Dream’s server + many private DiT patches.
- True CPU‖NPU segment preload while HTP runs.
- Threaded / 32-bit-word FP8 repack; `pread` I/O; buffer reuse; FA / RoPE / HVX work on various branches.
- LoRA: AUTO → at_runtime B(A·x) **per LoRA** on HTP.
- Broader product (video/audio models, etc.).

## NPH 0.30
- Own app + JNI; DiT precision FP8 E4M3 / FP32 / TE Q8_0 without lowering.
- **Unified FP16 LoRA** (rank concat) — measured 29.6→~9.5 s/step with 2 LoRAs.
- Repack cache, TE cache, adaptive Extremo residency, per-op HTP profiling.
- CC BY-NC patches from NM were not copied; where we took inspiration (e.g. faster FP8 repack), we reimplemented on MIT code.

## Proposal
If NM wants the unified FP16 LoRA or other MIT pieces from NPH, they can reimplement them in their tree under their licence. This release is an **open invitation**, not a patch drop or automatic PR.
