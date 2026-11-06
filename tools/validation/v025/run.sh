#!/bin/bash
# 0.25 host validation queue (waits for the c5 ctxgen to finish so RAM is free)
cd /workspace/zit-qnn; PY=/workspace/pyenv/bin/python; L=val/v025/run.log
log(){ echo "[$(date -u '+%F %T')] $*" >> $L; }
while pgrep -f qnn-context-binary-generator >/dev/null || pgrep -f q1024.sh >/dev/null; do sleep 30; done
log start
for spec in "zero:--mode zero" "cpp_pixel:--mode cpp --dump b025/ldump/pixel96" "merge_pixel:--mode merge --merge b025/loras/pixel.safetensors:1.0"; do
  t=${spec%%:*}; args=${spec#*:}
  [ -f val/v025_$t.pt ] && { log "$t exists"; continue; }
  log "gen $t"; $PY py/gen_lora.py $args --tag v025_$t > val/v025/gen_$t.log 2>&1 || { log "FAIL $t"; exit 1; }
done
for t in zero cpp_pixel merge_pixel; do $PY py/vae_png.py val/v025_$t.pt val/v025/$t.png >> $L 2>&1; done
$PY py/vae_png.py val/ref_p0.pt val/v025/base_ref.png >> $L 2>&1
$PY - >> $L 2>&1 <<'PY'
import torch, numpy as np
from PIL import Image
L = lambda t: torch.load(f"/workspace/zit-qnn/val/{t}.pt")["lat"].double()
b, z, c, m = L("ref_p0"), L("v025_zero"), L("v025_cpp_pixel"), L("v025_merge_pixel")
r = lambda x, y: float((x - y).norm() / y.norm())
print("zero-LoRA latent vs base: max|d| %.3g rel %.3g bitwise %s" % (float((z - b).abs().max()), r(z, b), bool((z == b).all())))
print("cpp pixel vs base: rel %.4f (LoRA effect)" % r(c, b))
print("cpp pixel vs merged-weights pixel: rel %.3g" % r(c, m))
I = lambda t: np.asarray(Image.open(f"/workspace/zit-qnn/val/v025/{t}.png")).astype(np.float64)
def psnr(x, y): mse = ((x - y) ** 2).mean(); return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
print("PSNR zero vs base %.2f dB | cpp vs merged %.2f dB | cpp vs base %.2f dB" % (psnr(I("zero"), I("base_ref")), psnr(I("cpp_pixel"), I("merge_pixel")), psnr(I("cpp_pixel"), I("base_ref"))))
w = 512; im = [Image.open(f"/workspace/zit-qnn/val/v025/{t}.png").resize((w, w)) for t in ["base_ref", "zero", "cpp_pixel", "merge_pixel"]]
s = Image.new("RGB", (4 * w, w)); [s.paste(x, (i * w, 0)) for i, x in enumerate(im)]; s.save("/workspace/zit-qnn/val/v025/compare.png")
PY
log VAL_DONE
