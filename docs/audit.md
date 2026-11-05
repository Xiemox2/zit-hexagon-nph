# Licence & secrets audit (2026-11-05)

## Licence
- Project: MIT (`LICENSE`).
- Upstream notices: `NOTICE` (happyyzy stable-diffusion.cpp, ggml / ggml-hexagon — MIT).
- Local Dream / Nightmare Mobile: **no source files copied**. Only prose comparison + UI string that the interface was design-inspired (explicitly “no incluye su código”).
- Upscaler catalogue entries mention third-party CC BY-NC-SA models by URL only (not bundled).

## Secrets / weights scan
- `rg` for `hf_*` tokens, PEM/SSH private keys, `gho_`/`ghp_`/`github_pat_`, `JiangFeng12/zit-hexagon-npg`, `zit-hexagon-private`, `hf_token`, `~/.secrets`: **no matches** in the published tree.
- No `.jks` / `.keystore` / `local.properties` in the tree.
- No `.safetensors` / `.gguf` / ReaZIT weights.
- Nested `.git` / `.gitmodules` stripped; **fresh squash history** (single commit) for the public repo.
- Upload scripts that read `~/.secrets/hf_token*` were **not** included.
- Signing: release APK uses the existing app signing cert; **keystore file not published**.

## Large binaries
- `libnph.so` and HTP skels tracked with **Git LFS**.
- Release APK attached via GitHub Releases only (not in git history).
