# -*- coding: utf-8 -*-
"""
用 base LLM 在 GSM8K 偏好对上做 baseline 评估：
score = log p(model)(prompt+response)，比较 pos vs neg。
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.inference_mode()
def compute_loglikelihood(model, tokenizer, prompt, resp, max_len, device):
    """计算 log p(y|x)，返回标量（越大越好）"""

    text = (prompt or "") + "\n\nAssistant: " + (resp or "")
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

    # labels = input_ids -> 得到平均负对数似然
    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        labels=enc["input_ids"],
    )
    neg_loglik = out.loss.item()
    return -neg_loglik   # 负的 CE，当成 reward


def eval_gsm8k_baseline(
    model_name: str,
    data_path: str,
    type_name: str = "math_correctness",
    max_len: int = 2048,
    device: str = "cuda",
):

    print(f"[GSM8K-Baseline] Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"[GSM8K-Baseline] Loading data: {data_path}")
    ds = load_dataset("json", data_files={"data": data_path})["data"]

    total = 0
    correct = 0

    for idx, row in enumerate(ds):
        # 只用偏好对 & 指定 reward type
        if row.get("mode") != "pref":
            continue
        if row.get("type") != type_name:
            continue

        prompt = row["prompt"]
        pos = row["pos"]
        neg = row["neg"]

        s_pos = compute_loglikelihood(model, tokenizer, prompt, pos, max_len, device)
        s_neg = compute_loglikelihood(model, tokenizer, prompt, neg, max_len, device)

        win = s_pos > s_neg
        total += 1
        correct += int(win)

        if (idx + 1) % 200 == 0:
            print(f"Processed {idx + 1} examples... (used={total})")

    if total == 0:
        print("[GSM8K-Baseline] No matching preference samples found.")
        return None

    acc = correct / total
    print(f"\n[GSM8K-Baseline] accuracy (type={type_name}): {acc:.4f} ({correct}/{total})")

    # 保存结果
    save_dir = Path("eval_gsm8k")
    save_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = model_name.split("/")[-1]
    out_path = save_dir / f"gsm8k_baseline_{model_tag}_{type_name}_{ts}.json"

    result = {
        "model": model_name,
        "data": data_path,
        "type_name": type_name,
        "total": total,
        "correct": correct,
        "accuracy": acc,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[GSM8K-Baseline] results saved to: {out_path}\n")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="base LLM name or local path, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", type=str, required=True,
                    help="gsm8k test jsonl, e.g. data/gsm8k_test_multirm.jsonl")
    ap.add_argument("--type-name", type=str, default="math_correctness")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    eval_gsm8k_baseline(
        model_name=args.model,
        data_path=args.data,
        type_name=args.type_name,
        max_len=args.max_len,
        device=args.device,
    )


if __name__ == "__main__":
    main()
