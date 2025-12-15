import os
import torch
from transformers import AutoTokenizer
from openrlhf.models.multitype_rmt import MultiTypeRewardModel  # 按你训练代码的 import 路径来

BASE_MODEL = "/hpc2hdd/home/jianmu/home/models/Qwen2.5-0.5B-Instruct"
RM_CKPT    = "outputs/gsm8k-multirmt-0.5b_llm_2/final.pt"
EXPORT_DIR = "outputs/models/rmt_model_1"

os.makedirs(EXPORT_DIR, exist_ok=True)

# 1) load checkpoint (注意：final.pt 是 {"model": ..., "cfg": ...})
ckpt = torch.load(RM_CKPT, map_location="cpu")
sd = ckpt["model"]
cfg = ckpt["cfg"]

# 2) rebuild the exact training wrapper
rm = MultiTypeRewardModel(
    model_name_or_path=cfg["model"]["pretrain"],
    type_specs=cfg["types"],
    use_flash_attn_2=cfg["model"].get("flash_attn2", True),
)

# 3) load strictly to ensure it REALLY restored
missing, unexpected = rm.load_state_dict(sd, strict=True)
print("[RESTORE] strict load ok. missing:", len(missing), "unexpected:", len(unexpected))

# 4) export ONLY the trained LM (this is what OpenCompass wants)
rm.lm.save_pretrained(EXPORT_DIR)
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tok.save_pretrained(EXPORT_DIR)

print("[DONE] exported lm-only to:", EXPORT_DIR)

