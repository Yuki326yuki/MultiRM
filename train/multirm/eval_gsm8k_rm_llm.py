# -*- coding: utf-8 -*-
"""
Evaluate MultiType Reward Model on GSM8K multi-task dataset, and save results as JSON.
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import torch
from transformers import AutoTokenizer

from openrlhf.models.multitype_rm import MultiTypeRewardModel


# ------------------------------
# 保存 JSON 输出
# ------------------------------
def save_results(out_dir, all_stats, all_overall):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"gsm8k_rm_{ts}.json")

    payload = {
        "overall": all_overall,
        "types": all_stats
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] Evaluation results saved to: {out_path}\n")


# ------------------------------
# 模型加载（保持与你训练时一致）
# ------------------------------
def load_model_and_tokenizer(ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]

    tok_name = cfg["model"].get("tokenizer_name_or_path") or cfg["model"]["pretrain"]
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = MultiTypeRewardModel(
        model_name_or_path=cfg["model"]["pretrain"],
        type_specs=cfg["types"],
        use_flash_attn_2=cfg["model"].get("flash_attn2", True),
    )

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    return model, tokenizer, cfg


# ------------------------------
# encode
# ------------------------------
@torch.inference_mode()
def encode_text(model, tokenizer, prompt: str, response: str, device: str, max_len: int):
    text = prompt + "\n" + response
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)

    return model.encode(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )


# ------------------------------
# 三类任务：pref, cls, reg
# ------------------------------
@torch.inference_mode()
def score_pref(model, tokenizer, prompt, pos, neg, type_name, device, max_len):
    emb_pos = encode_text(model, tokenizer, prompt, pos, device, max_len)
    emb_neg = encode_text(model, tokenizer, prompt, neg, device, max_len)

    s_pos = model.logits(emb_pos, type_name).amax(dim=-1).item()
    s_neg = model.logits(emb_neg, type_name).amax(dim=-1).item()
    return s_pos, s_neg


@torch.inference_mode()
def score_cls(model, tokenizer, prompt, response, type_name, device, max_len):
    emb = encode_text(model, tokenizer, prompt, response, device, max_len)
    return int(torch.argmax(model.logits(emb, type_name), dim=-1).item())


@torch.inference_mode()
def score_reg(model, tokenizer, prompt, response, type_name, device, max_len):
    emb = encode_text(model, tokenizer, prompt, response, device, max_len)
    r = model.reward_scalar(emb, type_name)
    return float(r.view(-1)[0].item())


# ------------------------------
# 主评估
# ------------------------------
def eval_gsm8k_rm(model, tokenizer, data_file, device, max_len):
    stats = defaultdict(lambda: {
        "pref_total": 0, "pref_correct": 0,
        "cls_total": 0, "cls_correct": 0,
        "reg_total": 0, "reg_se": 0.0,
    })
    overall = {
        "pref_total": 0, "pref_correct": 0,
        "cls_total": 0, "cls_correct": 0,
        "reg_total": 0, "reg_se": 0.0
    }

    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for ex in map(lambda l: json.loads(l.strip()), lines):
        mode = ex["mode"]
        type_name = ex["type"]
        prompt = ex["prompt"]
        st = stats[type_name]

        if mode == "pref":
            pos, neg = ex["pos"], ex["neg"]
            s_pos, s_neg = score_pref(model, tokenizer, prompt, pos, neg, type_name, device, max_len)
            win = int(s_pos > s_neg)
            st["pref_total"] += 1
            st["pref_correct"] += win
            overall["pref_total"] += 1
            overall["pref_correct"] += win

        elif mode == "cls":
            resp, label = ex["response"], int(ex["label"])
            pred = score_cls(model, tokenizer, prompt, resp, type_name, device, max_len)
            correct = int(pred == label)
            st["cls_total"] += 1
            st["cls_correct"] += correct
            overall["cls_total"] += 1
            overall["cls_correct"] += correct

        elif mode == "reg":
            resp, target = ex["response"], float(ex["score"])
            r = score_reg(model, tokenizer, prompt, resp, type_name, device, max_len)
            se = (r - target) ** 2
            st["reg_total"] += 1
            st["reg_se"] += se
            overall["reg_total"] += 1
            overall["reg_se"] += se

    # 转化成可 JSON 保存的 summary
    output_stats = {}
    for t, st in stats.items():
        out = {}
        if st["pref_total"] > 0:
            out["pref_acc"] = st["pref_correct"] / st["pref_total"]
        if st["cls_total"] > 0:
            out["cls_acc"] = st["cls_correct"] / st["cls_total"]
        if st["reg_total"] > 0:
            out["reg_mse"] = st["reg_se"] / st["reg_total"]
        output_stats[t] = out

    # overall
    output_overall = {}
    if overall["pref_total"] > 0:
        output_overall["pref_acc"] = overall["pref_correct"] / overall["pref_total"]
    if overall["cls_total"] > 0:
        output_overall["cls_acc"] = overall["cls_correct"] / overall["cls_total"]
    if overall["reg_total"] > 0:
        output_overall["reg_mse"] = overall["reg_se"] / overall["reg_total"]

    return output_stats, output_overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-file", default="data/gsm8k_test_multirm_llm.jsonl")
    ap.add_argument("--out-dir", default="eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    model, tokenizer, cfg = load_model_and_tokenizer(args.ckpt, args.device)

    stats, overall = eval_gsm8k_rm(
        model, tokenizer,
        data_file=args.data_file,
        device=args.device,
        max_len=args.max_len,
    )

    save_results(args.out_dir, stats, overall)


if __name__ == "__main__":
    main()
