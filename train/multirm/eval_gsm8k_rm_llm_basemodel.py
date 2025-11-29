# -*- coding: utf-8 -*-
"""
Evaluate base CausalLM as a preference baseline on GSM8K dataset.
Save results as JSON.
"""

import argparse
import json
import os
from datetime import datetime
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ------------------------------
# JSON 保存
# ------------------------------
def save_results(out_dir, stats, overall):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"gsm8k_basemodel_{ts}.json")

    payload = {
        "overall": overall,
        "types": stats,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] Baseline results saved to: {out_path}\n")


# ------------------------------
# Base LM 加载
# ------------------------------
def load_basemodel(model_path, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(device).eval()
    return model, tokenizer


# ------------------------------
# 计算 LM 的平均 logprob
# ------------------------------
@torch.inference_mode()
def avg_logprob(model, tokenizer, text, device="cuda", max_len=2048):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    input_ids = enc["input_ids"]
    att = enc["attention_mask"]

    logits = model(input_ids=input_ids, attention_mask=att).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = att[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size()) * shift_mask

    tok_count = shift_mask.sum().item()
    if tok_count == 0:
        return float("nan")

    nll = loss.sum().item() / tok_count
    return -nll


# ------------------------------
# 主评估：pref-only
# ------------------------------
def eval_gsm8k_pref_basemodel(model, tokenizer, data_file, device, max_len):
    stats = defaultdict(lambda: {"pref_total": 0, "pref_correct": 0})
    overall = {"pref_total": 0, "pref_correct": 0}

    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for ex in map(lambda l: json.loads(l.strip()), lines):
        if ex["mode"] != "pref":
            continue

        type_name = ex["type"]
        prompt = ex["prompt"]
        pos, neg = ex["pos"], ex["neg"]

        s_pos = avg_logprob(model, tokenizer, prompt + "\n" + pos, device, max_len)
        s_neg = avg_logprob(model, tokenizer, prompt + "\n" + neg, device, max_len)
        win = int(s_pos > s_neg)

        stats[type_name]["pref_total"] += 1
        stats[type_name]["pref_correct"] += win

        overall["pref_total"] += 1
        overall["pref_correct"] += win

    # 整理成可保存格式
    out_stats = {}
    for t, st in stats.items():
        if st["pref_total"] > 0:
            out_stats[t] = {
                "pref_acc": st["pref_correct"] / st["pref_total"]
            }

    out_overall = {}
    if overall["pref_total"] > 0:
        out_overall["pref_acc"] = overall["pref_correct"] / overall["pref_total"]

    return out_stats, out_overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-file", default="data/gsm8k_test_multirm_llm.jsonl")
    ap.add_argument("--out-dir", default="eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    model, tokenizer = load_basemodel(args.model, args.device)

    stats, overall = eval_gsm8k_pref_basemodel(
        model, tokenizer,
        data_file=args.data_file,
        device=args.device,
        max_len=args.max_len,
    )

    save_results(args.out_dir, stats, overall)


if __name__ == "__main__":
    main()
