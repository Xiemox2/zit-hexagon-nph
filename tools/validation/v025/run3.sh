#!/bin/bash
# style check with an explicit pixel-art prompt (SFW): base vs pixel LoRA (C++ R96 inputs)
cd /workspace/zit-qnn; PY=/workspace/pyenv/bin/python; L=val/v025/run.log
while ! grep -q VAL2_DONE $L; do sleep 30; done
echo "[$(date -u '+%F %T')] run3: te pixel prompt" >> $L
[ -f val/v025/te_pixel.pt ] || $PY py/te_one.py val/v025/te_pixel.pt "pixel art of a red fox standing in the snow in a pine forest, 16-bit retro video game style" >> $L 2>&1
for spec in "px_base:--mode merge" "px_lora:--mode cpp --dump b025/ldump/pixel96"; do t=${spec%%:*}; args=${spec#*:}
  [ -f val/v025_$t.pt ] || $PY py/gen_lora.py $args --te val/v025/te_pixel.pt --tag v025_$t > val/v025/gen_$t.log 2>&1
  $PY py/vae_png.py val/v025_$t.pt val/v025/$t.png >> $L 2>&1; done
$PY - >> $L 2>&1 <<'PY'
from PIL import Image
im=[Image.open(f"/workspace/zit-qnn/val/v025/{t}.png").resize((512,512)) for t in ["px_base","px_lora"]]
s=Image.new("RGB",(1024,512)); s.paste(im[0],(0,0)); s.paste(im[1],(512,0)); s.save("/workspace/zit-qnn/val/v025/compare_pixelprompt.png"); print("saved compare_pixelprompt")
PY
echo "[$(date -u '+%F %T')] VAL3_DONE" >> $L
