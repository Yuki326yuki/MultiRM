# train/multirm/prepare_multirm_data.py
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset


def bucket_score(x: float, n_bins: int = 3) -> int:
    """把 [0,1] 区间的连续分数分成 0..n_bins-1 的离散类别"""
    x = float(x)
    x = max(0.0, min(1.0, x))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        if x <= edges[i + 1] or i == n_bins - 1:
            return int(i)
    return n_bins - 1


def add_ultrafeedback_binarized(out_f, max_samples=None):
    """
    HuggingFaceH4/ultrafeedback_binarized:
      - train_prefs: 偏好对 + (score_chosen, score_rejected)
    我们生成:
      - mode=pref, type=overall
      - 同时用 chosen 生成一个 reg 和一个 cls 样本
    """
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    n = len(ds) if max_samples is None else min(len(ds), max_samples)

    def msgs_to_text(msgs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)

    for i in range(n):
        row = ds[i]
        prompt = row["prompt"]
        chosen = msgs_to_text(row["chosen"])
        rejected = msgs_to_text(row["rejected"])
        score_chosen = float(row.get("score_chosen", 0.0))

        # 偏好对
        pref = {
            "dataset": "ultrafeedback_binarized",
            "mode": "pref",
            "type": "overall",
            "prompt": prompt,
            "pos": chosen,
            "neg": rejected,
        }
        out_f.write(json.dumps(pref, ensure_ascii=False) + "\n")

        # 连续 & 离散（用 chosen）
        score_norm = max(0.0, min(1.0, score_chosen / 10.0))
        reg = {
            "dataset": "ultrafeedback_binarized",
            "mode": "reg",
            "type": "overall",
            "prompt": prompt,
            "response": chosen,
            "score": score_norm,
        }
        cls = {
            "dataset": "ultrafeedback_binarized",
            "mode": "cls",
            "type": "overall",
            "prompt": prompt,
            "response": chosen,
            "label": bucket_score(score_norm, n_bins=3),
        }
        out_f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        out_f.write(json.dumps(cls, ensure_ascii=False) + "\n")


def add_ultrafeedback(out_f, max_samples=None):
    """
    openbmb/UltraFeedback:
      每条 row：
        - "instruction": prompt
        - "completions": list[
            {
              "response": str,
              "overall_score": float(0~10),
              "fine-grained_score": float,
              "principle": "...",
              "annotations": {
                  "helpfulness": [ {"Rating": "3", ...}, ... ],
                  "truthfulness": [ {"Rating": "5", ...}, ... ],
                  "honesty": [...],
                  "instruction_following": [...]
              }
            }, ...
          ]
    我们从每个 completion 里抽：
      - overall_score 作为 overall 维度
      - annotations 里的各维度平均 Rating 作为 1~5 分，再归一到 [0,1]
      然后生成 reg + cls 样本
    """
    from datasets import load_dataset

    ds = load_dataset("openbmb/UltraFeedback", split="train")
    n = len(ds) if max_samples is None else min(len(ds), max_samples)

    for i in range(n):
        row = ds[i]
        prompt = row.get("instruction") or row.get("prompt") or ""
        completions = row["completions"]   # ✅ 正确字段名

        for c in completions:
            text = c.get("response") or ""
            annotations = c.get("annotations", {})

            # 1) overall_score 0~10，归一到 0~1
            overall_score = float(c.get("overall_score", 0.0))
            attrs = {}
            if overall_score is not None:
                attrs["overall"] = max(0.0, min(1.0, overall_score / 10.0))

            # 2) 四个细分维度：helpfulness / truthfulness / honesty / instruction_following
            for dim in ["helpfulness", "truthfulness", "honesty", "instruction_following"]:
                if dim not in annotations:
                    continue
                lst = annotations[dim]
                ratings = []
                for item in lst:
                    if "Rating" in item:
                        try:
                            ratings.append(float(item["Rating"]))
                        except Exception:
                            pass
                if ratings:
                    # Rating 是 1~5，简单除以 5.0 归一
                    val = sum(ratings) / len(ratings)
                    attrs[dim] = max(0.0, min(1.0, val / 5.0))

            # 3) 为每一个维度产生 reg + cls 样本
            for t, v in attrs.items():
                reg = {
                    "dataset": "UltraFeedback",
                    "mode": "reg",
                    "type": t,
                    "prompt": prompt,
                    "response": text,
                    "score": v,
                }
                cls = {
                    "dataset": "UltraFeedback",
                    "mode": "cls",
                    "type": t,
                    "prompt": prompt,
                    "response": text,
                    "label": bucket_score(v, n_bins=3),
                }
                out_f.write(json.dumps(reg, ensure_ascii=False) + "\n")
                out_f.write(json.dumps(cls, ensure_ascii=False) + "\n")



def add_helpsteer2(out_f, max_samples=None):
    """
    nvidia/HelpSteer2：
      - prompt, response, 多个属性 0~4/0~5 评分
    我们把这些属性都当作不同的 type。
    """
    ds = load_dataset("nvidia/HelpSteer2", split="train")
    n = len(ds) if max_samples is None else min(len(ds), max_samples)

    for i in range(n):
        row = ds[i]
        prompt = row["prompt"]
        response = row["response"]

        attrs = {}
        for key in ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]:
            if key in row:
                attrs[key] = float(row[key])

        def norm(s):
            # 简单除以 4.0，近似归一到 [0,1]
            return max(0.0, min(1.0, s / 4.0))

        for t, s in attrs.items():
            s_n = norm(s)
            reg = {
                "dataset": "HelpSteer2",
                "mode": "reg",
                "type": t,
                "prompt": prompt,
                "response": response,
                "score": s_n,
            }
            cls = {
                "dataset": "HelpSteer2",
                "mode": "cls",
                "type": t,
                "prompt": prompt,
                "response": response,
                "label": bucket_score(s_n, n_bins=3),
            }
            out_f.write(json.dumps(reg, ensure_ascii=False) + "\n")
            out_f.write(json.dumps(cls, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--max_ultra_bin", type=int, default=None)
    ap.add_argument("--max_ultra", type=int, default=None)
    ap.add_argument("--max_helpsteer2", type=int, default=None)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        add_ultrafeedback_binarized(f, args.max_ultra_bin)
        add_ultrafeedback(f, args.max_ultra)
        add_helpsteer2(f, args.max_helpsteer2)

    print("Wrote data to", out_path)


if __name__ == "__main__":
    main()
