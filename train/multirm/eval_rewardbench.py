# train/multirm/eval_rewardbench.py
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from openrlhf.models.multitype_rm import MultiTypeRewardModel


@torch.inference_mode()
def score_one(model, tokenizer, prompt, resp, type_name, max_len, device):
    text = (prompt or "") + "\n\nAssistant: " + (resp or "")
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    emb = model.encode(enc["input_ids"], enc["attention_mask"])
    r = model.reward_scalar(emb, type_name)
    return float(r.item())


def eval_rewardbench(model, tokenizer, type_name, split="train", subset=None, max_len=2048, device="cuda"):
    ds = load_dataset("allenai/reward-bench", split=split)
    total = 0
    correct = 0
    by_subset = {}

    for row in ds:
        s = row["subset"]
        if subset is not None and s != subset:
            continue

        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        sc = score_one(model, tokenizer, prompt, chosen, type_name, max_len, device)
        sr = score_one(model, tokenizer, prompt, rejected, type_name, max_len, device)

        win = sc > sr
        total += 1
        correct += int(win)

        stat = by_subset.setdefault(s, {"total": 0, "correct": 0})
        stat["total"] += 1
        stat["correct"] += int(win)

    acc = correct / total if total > 0 else 0.0
    print(f"[RewardBench] overall acc (type={type_name}, split={split}): {acc:.4f} ({correct}/{total})")
    for s, stat in sorted(by_subset.items(), key=lambda kv: kv[0]):
        if stat["total"] == 0:
            continue
        a = stat["correct"] / stat["total"]
        print(f"  subset={s:<20} acc={a:.4f} ({stat['correct']}/{stat['total']})")


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint path (final.pt)")
    ap.add_argument("--config", type=str, required=True, help="same config yaml/json used for training")
    ap.add_argument("--type-name", type=str, default="overall", help="reward type to use")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--subset", type=str, default=None)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    model_name = cfg["model"]["pretrain"]
    tok_name = cfg["model"].get("tokenizer_name_or_path") or model_name

    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = MultiTypeRewardModel(
        model_name_or_path=model_name,
        type_specs=cfg["types"],
        use_flash_attn_2=cfg["model"].get("flash_attn2", True),
        lora=cfg["model"].get("lora", False),
        lora_r=cfg["model"].get("lora_r", 8),
        lora_alpha=cfg["model"].get("lora_alpha", 16),
        lora_dropout=cfg["model"].get("lora_dropout", 0.05),
    ).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    eval_rewardbench(
        model,
        tokenizer,
        type_name=args.type_name,
        split=args.split,
        subset=args.subset,
        max_len=args.max_len,
        device=args.device,
    )


if __name__ == "__main__":
    main()
