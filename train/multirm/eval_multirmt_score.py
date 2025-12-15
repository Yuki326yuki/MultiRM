# -*- coding: utf-8 -*-
"""
评测“打分能力”的统一脚本（基于你原来的 eval_multirmt_score.py 扩展）。

支持两种 scorer：
1) --scorer multirm
   使用 MultiTypeRewardModel.token_implicit_reward 得到序列分数（你原来的方式）
2) --scorer logprob
   使用任意 HF CausalLM（例如训练 RM 的 base model）对回答部分计算平均 logprob 作为分数
   用于测试“训练前 base 是否有打分能力”，以及对比“训练后是否学到东西”。

输出指标：
- PREF: pairwise accuracy
- CLS:  AUC + median-threshold accuracy（logprob 模式推荐）
       multirm 模式同时给 AUC + median-threshold（比 score>0 更稳）
- REG:  MSE + Pearson

并将所有样例写入 JSONL 文件（--out-path）。

用法示例：

(1) 评测训练后的 MultiRM:
python eval_multirmt_score.py \
  --scorer multirm \
  --rm-ckpt outputs/xxx/final.pt \
  --test-file data/gsm8k_test_multirm_llm.jsonl \
  --rm-type math_final_answer \
  --out-path outputs/rm_eval/multirm_score.jsonl

(2) 评测训练 RM 用的 base model（训练前）：
python eval_multirmt_score.py \
  --scorer logprob \
  --lm-model /path/to/Qwen2.5-0.5B-Instruct \
  --test-file data/gsm8k_test_multirm_llm.jsonl \
  --rm-type math_final_answer \
  --out-path outputs/rm_eval/base_logprob_score.jsonl
"""

import argparse
import json
import math
import os
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from openrlhf.models.multitype_rmt import MultiTypeRewardModel


# ============================================
# 构造输入（与训练一致）并 mask question 部分
# ============================================

def build_input_for_rm(tokenizer, prompt: str, response: str, max_len: int = 1024):
    """
    构造 input_ids / attention_mask / response_mask —— 与训练完全一致。
    prompt_text = prompt + "\n\nAssistant: "
    full_text  = prompt_text + response
    response_mask 在 prompt_text 之后设为 1（question 部分为 0）。
    """
    prompt = prompt.strip()
    resp = response.strip()

    prompt_text = prompt + "\n\nAssistant: "
    full_text = prompt_text + resp

    enc_prompt = tokenizer(prompt_text, return_tensors="pt",
                           truncation=True, max_length=max_len, padding=False)
    enc_full = tokenizer(full_text, return_tensors="pt",
                         truncation=True, max_length=max_len, padding=False)

    input_ids = enc_full["input_ids"]          # [1, T]
    attention_mask = enc_full["attention_mask"]  # [1, T]
    T = input_ids.size(1)

    len_prompt = enc_prompt["input_ids"].size(1)
    if len_prompt > T:
        len_prompt = T

    response_mask = torch.zeros_like(input_ids)
    if len_prompt < T:
        response_mask[:, len_prompt:] = 1
    response_mask = response_mask * attention_mask

    return input_ids, attention_mask, response_mask


# ============================================
# AUC（不依赖 sklearn）
# ============================================

def auc_from_scores(scores: List[float], labels: List[int]) -> float:
    """
    ROC-AUC (Mann–Whitney U) 简单实现。
    """
    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    rank_sum_pos = 0.0
    for i, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum_pos += i

    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def median_threshold_accuracy(scores: List[float], labels: List[int]) -> float:
    """
    用 score 的中位数作为阈值给一个“粗糙 accuracy”：
      score >= median -> 1
      score <  median -> 0
    """
    if not scores:
        return float("nan")
    ss = sorted(scores)
    med = ss[len(ss) // 2]
    pred = [1 if s >= med else 0 for s in scores]
    return sum(int(p == y) for p, y in zip(pred, labels)) / len(labels)


# ============================================
# 加载 MultiRM
# ============================================

def load_rm(rm_ckpt: str, device: str):
    ckpt = torch.load(rm_ckpt, map_location="cpu")

    cfg = ckpt["cfg"]
    state_dict = ckpt["model"]

    model_name = cfg["model"]["pretrain"]
    type_specs = cfg["types"]

    tokenizer_name = cfg["model"].get("tokenizer_name_or_path") or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rm = MultiTypeRewardModel(
        model_name_or_path=model_name,
        type_specs=type_specs,
        use_flash_attn_2=cfg["model"].get("flash_attn2", True)
    )

    missing, unexpected = rm.load_state_dict(state_dict, strict=False)
    print("[load_rm] missing keys:", missing)
    print("[load_rm] unexpected keys:", unexpected)

    rm.to(device)
    rm.eval()
    return rm, tokenizer, cfg


# ============================================
# 加载 base LM（logprob scorer）
# ============================================

def load_lm(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device).eval()

    return model, tokenizer


@torch.no_grad()
def seq_token_logprobs(model, input_ids, attention_mask):
    """
    返回每个位置 token 的 logprob（对齐到 input_ids 的 token）。
    position 0 没有预测，置 0。
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [B, T, V]
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logps = F.log_softmax(logits, dim=-1)
    token_logps = logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    pad0 = torch.zeros((input_ids.size(0), 1), device=input_ids.device, dtype=token_logps.dtype)
    token_logps = torch.cat([pad0, token_logps], dim=1)  # [B, T]
    return token_logps


@torch.no_grad()
def logprob_score(model, input_ids, attention_mask, response_mask):
    """
    分数 = 回答 token 的平均 logprob（mask 掉 question）。
    这用于衡量 base LM 是否“更偏好好答案”。
    """
    tlp = seq_token_logprobs(model, input_ids, attention_mask)  # [1, T]
    masked = tlp * response_mask
    denom = response_mask.sum().clamp(min=1.0)
    return (masked.sum() / denom).item()


# ============================================
# 统一评测逻辑：multirm 或 logprob
# ============================================

def eval_on_file(
    scorer: str,
    rm=None,
    lm=None,
    tokenizer=None,
    cfg: Dict[str, Any] = None,
    test_file: str = "",
    rm_type: str = "math_final_answer",
    device: str = "cuda",
    out_path: str = None,
    max_samples: int = -1,
    max_len: int = 1024,
):
    # multirm 模式用 beta（按 type）
    beta = 1.0
    if scorer == "multirm":
        beta = float(cfg["types"][rm_type].get("beta", 1.0))
        print(f"[INFO] scorer=multirm rm_type={rm_type} beta={beta}")
    else:
        print(f"[INFO] scorer=logprob rm_type={rm_type}")

    pref_total = pref_correct = 0

    cls_scores, cls_labels = [], []
    reg_scores, reg_targets = [], []

    reg_total = 0
    reg_mse_sum = 0.0

    results = []

    with open(test_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_samples > 0 and line_idx >= max_samples:
                break
            if not line.strip():
                continue

            obj = json.loads(line)
            if obj.get("type") != rm_type:
                continue

            mode = obj["mode"]
            prompt = obj["prompt"]

            def score_one(resp_text: str) -> float:
                in_ids, att, mask = build_input_for_rm(tokenizer, prompt, resp_text, max_len=max_len)
                in_ids, att, mask = in_ids.to(device), att.to(device), mask.to(device)

                if scorer == "multirm":
                    with torch.no_grad():
                        _, r_seq = rm.token_implicit_reward(in_ids, att, mask, beta=beta)
                    return r_seq.item()
                else:
                    return logprob_score(lm, in_ids, att, mask)

            # ===== PREF =====
            if mode == "pref":
                pos = obj["pos"]
                neg = obj["neg"]

                s_pos = score_one(pos)
                s_neg = score_one(neg)

                correct = (s_pos > s_neg)
                pref_total += 1
                pref_correct += int(correct)

                results.append({
                    "mode": "pref",
                    "prompt": prompt,
                    "pos": pos,
                    "neg": neg,
                    "score_pos": s_pos,
                    "score_neg": s_neg,
                    "correct": bool(correct),
                })

            # ===== CLS =====
            elif mode == "cls":
                resp = obj["response"]
                label = int(obj["label"])

                s = score_one(resp)

                cls_scores.append(s)
                cls_labels.append(label)

                results.append({
                    "mode": "cls",
                    "prompt": prompt,
                    "response": resp,
                    "score": s,
                    "label": label,
                })

            # ===== REG =====
            elif mode == "reg":
                resp = obj["response"]
                tgt = float(obj["score"])

                s = score_one(resp)

                reg_total += 1
                reg_mse_sum += (s - tgt) ** 2
                reg_scores.append(s)
                reg_targets.append(tgt)

                results.append({
                    "mode": "reg",
                    "prompt": prompt,
                    "response": resp,
                    "score": s,
                    "target": tgt,
                })

    summary: Dict[str, Any] = {"scorer": scorer, "rm_type": rm_type}

    if pref_total > 0:
        summary["pref_acc"] = pref_correct / pref_total
        summary["pref_total"] = pref_total
        summary["pref_correct"] = pref_correct

    # CLS: 用 AUC + median threshold acc（比 s>0 更稳）
    if cls_scores:
        summary["cls_total"] = len(cls_scores)
        summary["cls_auc"] = auc_from_scores(cls_scores, cls_labels)
        summary["cls_acc_median_th"] = median_threshold_accuracy(cls_scores, cls_labels)

    # REG
    if reg_total > 0:
        mse = reg_mse_sum / reg_total
        mean_s = sum(reg_scores) / reg_total
        mean_t = sum(reg_targets) / reg_total
        num = sum((s - mean_s) * (t - mean_t) for s, t in zip(reg_scores, reg_targets))
        den_s = math.sqrt(sum((s - mean_s) ** 2 for s in reg_scores) + 1e-8)
        den_t = math.sqrt(sum((t - mean_t) ** 2 for t in reg_targets) + 1e-8)
        corr = num / (den_s * den_t + 1e-8)

        summary["reg_total"] = reg_total
        summary["reg_mse"] = mse
        summary["reg_pearson"] = corr

    print("\n========== SCORING SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 写 jsonl
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as wf:
            for r in results:
                wf.write(json.dumps(r, ensure_ascii=False) + "\n")
            wf.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
        print("[INFO] Saved evaluation jsonl to:", out_path)


# ============================================
# MAIN
# ============================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--scorer", type=str, choices=["multirm", "logprob"], required=True,
                    help="multirm: 用训练后的 MultiRM implicit reward；logprob: 用 base LM 的回答 logprob 打分")

    # multirm scorer 需要
    ap.add_argument("--rm-ckpt", type=str, default=None,
                    help="(scorer=multirm) MultiRM final.pt 路径")

    # logprob scorer 需要
    ap.add_argument("--lm-model", type=str, default=None,
                    help="(scorer=logprob) HF CausalLM 路径（例如训练 RM 的 base 模型）")

    ap.add_argument("--test-file", type=str, required=True)
    ap.add_argument("--rm-type", type=str, default="math_final_answer")
    ap.add_argument("--out-path", type=str, default=None, help="保存结果 jsonl 文件路径")
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--max-len", type=int, default=1024)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if args.scorer == "multirm":
        if args.rm_ckpt is None:
            raise ValueError("scorer=multirm 需要提供 --rm-ckpt")
        rm, tokenizer, cfg = load_rm(args.rm_ckpt, device)
        eval_on_file(
            scorer="multirm",
            rm=rm,
            lm=None,
            tokenizer=tokenizer,
            cfg=cfg,
            test_file=args.test_file,
            rm_type=args.rm_type,
            device=device,
            out_path=args.out_path,
            max_samples=args.max_samples,
            max_len=args.max_len,
        )
    else:
        if args.lm_model is None:
            raise ValueError("scorer=logprob 需要提供 --lm-model")
        lm, tokenizer = load_lm(args.lm_model, device)
        eval_on_file(
            scorer="logprob",
            rm=None,
            lm=lm,
            tokenizer=tokenizer,
            cfg=None,
            test_file=args.test_file,
            rm_type=args.rm_type,
            device=device,
            out_path=args.out_path,
            max_samples=args.max_samples,
            max_len=args.max_len,
        )


if __name__ == "__main__":
    main()

