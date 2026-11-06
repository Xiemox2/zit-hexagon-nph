#!/bin/bash
# base under the CURRENT zit_model code (ref_p0.pt is from Oct 23 code, rms rewrite since) -> exact zero-LoRA check
cd /workspace/zit-qnn; PY=/workspace/pyenv/bin/python; L=val/v025/run.log
while ! grep -q VAL_DONE $L; do sleep 30; done
echo "[$(date -u '+%F %T')] gen base_now (merge mode, no LoRA = plain checkpoint, lora=None path)" >> $L
[ -f val/v025_base_now.pt ] || $PY py/gen_lora.py --mode merge --tag v025_base_now > val/v025/gen_base_now.log 2>&1
$PY py/vae_png.py val/v025_base_now.pt val/v025/base_now.png >> $L 2>&1
$PY - >> $L 2>&1 <<'PY'
import torch, numpy as np
from PIL import Image
L = lambda t: torch.load(f"/workspace/zit-qnn/val/{t}.pt")["lat"].double()
b, z, c, m = L("v025_base_now"), L("v025_zero"), L("v025_cpp_pixel"), L("v025_merge_pixel")
r = lambda x, y: float((x - y).norm() / y.norm())
print("[base_now] zero-LoRA vs base_now: bitwise %s max|d| %.3g" % (bool((z == b).all()), float((z - b).abs().max())))
print("[base_now] cpp pixel vs base_now rel %.4f ; merged vs base_now rel %.4f" % (r(c, b), r(m, b)))
I = lambda t: np.asarray(Image.open(f"/workspace/zit-qnn/val/v025/{t}.png")).astype(np.float64)
def psnr(x, y): mse = ((x - y) ** 2).mean(); return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
print("[base_now] PSNR zero vs base_now %.2f dB ; cpp_pixel vs base_now %.2f dB" % (psnr(I("zero"), I("base_now")), psnr(I("cpp_pixel"), I("base_now"))))
w = 512; im = [Image.open(f"/workspace/zit-qnn/val/v025/{t}.png").resize((w, w)) for t in ["base_now", "zero", "cpp_pixel", "merge_pixel"]]
s = Image.new("RGB", (4 * w, w)); [s.paste(x, (i * w, 0)) for i, x in enumerate(im)]; s.save("/workspace/zit-qnn/val/v025/compare2.png")
PY
echo "[$(date -u '+%F %T')] VAL2_DONE" >> $L
