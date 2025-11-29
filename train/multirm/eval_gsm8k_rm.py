# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from openrlhf.models.multitype_rm import MultiTypeRewardModel


@torch.inference_mode()
def score_one(model, tokenizer, prompt, resp, type_name, max_len, device):
    """对单个 (prompt, response) 计算 reward 标量（和 eval_rewardbench 完全一致风格）"""
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


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def eval_gsm8k(
    model,
    tokenizer,
    type_name: str,
    data_path: str,
    max_len: int = 2048,
    device: str = "cuda",
):
    """
    在 GSM8K 上评估训练后的 RM：
    data_path 是我们预处理好的 gsm8k_test_multirm.jsonl，
    只使用 mode="pref" 且 type=type_name 的样本。
    """
    print(f"[GSM8K] Loading test data from {data_path}")
    ds = load_dataset("json", data_files={"data": data_path})["data"]

    total = 0
    correct = 0

    for idx, row in enumerate(ds):
        # 只用偏好对 + 指定的 reward type
        if row.get("mode") != "pref":
            continue
        if row.get("type") != type_name:
            continue

        prompt = row["prompt"]
        pos = row["pos"]   # 正确解答
        neg = row["neg"]   # 造出来的错误解答

        sc = score_one(model, tokenizer, prompt, pos, type_name, max_len, device)
        sr = score_one(model, tokenizer, prompt, neg, type_name, max_len, device)

        win = sc > sr
        total += 1
        correct += int(win)

        if (idx + 1) % 200 == 0:
            print(f"Processed {idx + 1} examples... (current used={total})")

    if total == 0:
        print("[GSM8K] No preference samples found for this type.")
        return None

    acc = correct / total
    result = {
        "type_name": type_name,
        "total": total,
        "correct": correct,
        "accuracy": acc,
    }

    print(f"\n[GSM8K] acc (type={type_name}): {acc:.4f} ({correct}/{total})")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="path to RM checkpoint, e.g. outputs/gsm8k-multirm-0.5b/final.pt")
    ap.add_argument("--config", type=str, required=True,
                    help="yaml/json config file (same as training, contains model.pretrain & types)")
    ap.add_argument("--data", type=str, required=True,
                    help="gsm8k test jsonl, e.g. data/gsm8k_test_multirm.jsonl")
    ap.add_argument("--type-name", type=str, default="math_correctness")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # 和 eval_rewardbench.py 一样：从 config 里拿 base model / tokenizer
    model_name = cfg["model"]["pretrain"]
    tok_name = cfg["model"].get("tokenizer_name_or_path") or model_name

    print(f"[GSM8K] Loading tokenizer from {tok_name}")
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[GSM8K] Building MultiTypeRewardModel from {model_name}")
    model = MultiTypeRewardModel(
        model_name_or_path=model_name,
        type_specs=cfg["types"],
        use_flash_attn_2=cfg["model"].get("flash_attn2", True),
    ).to(args.device)

    # 加载 checkpoint 权重（.pt 文件）
    print(f"[GSM8K] Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=args.device)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    result = eval_gsm8k(
        model=model,
        tokenizer=tokenizer,
        type_name=args.type_name,
        data_path=args.data,
        max_len=args.max_len,
        device=args.device,
    )

    if result is None:
        return

    # 保存 json 到 eval_gsm8k 目录
    save_dir = Path("eval_gsm8k")
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 用 ckpt 上一级目录名当作模型名的一部分
    ckpt_name = Path(args.ckpt).parent.name
    out_path = save_dir / f"gsm8k_{ckpt_name}_{args.type_name}_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] GSM8K results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
