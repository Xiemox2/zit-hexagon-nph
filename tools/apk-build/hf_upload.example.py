#!/usr/bin/env python3
"""Example: upload an APK to HF. Token from env only — never commit secrets."""
import os, sys
from huggingface_hub import HfApi
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not tok:
    sys.exit("Set HF_TOKEN in the environment (do not commit it).")
api = HfApi(token=tok)
print("Edit repo_id/path, then call api.upload_file(...)")
