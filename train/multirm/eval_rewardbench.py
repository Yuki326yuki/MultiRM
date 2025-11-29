# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from openrlhf.models.multitype_rm import MultiTypeRewardModel


@torch.inference_mode()
def score_one(model, tokenizer, prompt, resp, type_name, max_len, device):
    """对单个 (prompt, response) 计算 reward 标量"""
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


def eval_rewardbench(
    model,
    tokenizer,
    type_name,
    split="filtered",
    subset=None,
    max_len=2048,
    device="cuda",
    local_path=None,
):
    """
    如果 local_path 不为 None，则从本地 JSONL 加载 RewardBench
    否则从 HuggingFace 在线加载
    """
    if local_path is not None:
        print(f"[RewardBench] Offline mode: loading from {local_path}")
        ds = load_dataset("json", data_files={"data": local_path})["data"]
    else:
        print(f"[RewardBench] Online mode: loading allenai/reward-bench split={split}")
        ds = load_dataset("allenai/reward-bench", split=split)

    total = 0
    correct = 0
    by_subset = {}

    for idx, row in enumerate(ds):
        s = row.get("subset", "unknown")
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

        st = by_subset.setdefault(s, {"total": 0, "correct": 0})
        st["total"] += 1
        st["correct"] += int(win)

        # 每 200 条打印一次进度
        if (idx + 1) % 200 == 0:
            print(f"Processed {idx + 1}/{len(ds)} examples...")

    # 没有评估样本（子集为空时可能出现）
    if total == 0:
        print("[RewardBench] No samples evaluated.")
        return None

    # 汇总结果
    result = {
        "type_name": type_name,
        "total": total,
        "correct": correct,
        "overall_acc": correct / total,
        "subset_acc": {},
    }

    for s, st in by_subset.items():
        if st["total"] == 0:
            continue
        result["subset_acc"][s] = st["correct"] / st["total"]

    # 打印结果
    print(f"[RewardBench] overall acc (type={type_name}): {result['overall_acc']:.4f} ({correct}/{total})")
    for s, acc in result["subset_acc"].items():
        print(f"  subset={s:<20} acc={acc:.4f}")

    return result


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--type-name", type=str, default="overall")
    ap.add_argument("--split", type=str, default="filtered")
    ap.add_argument("--subset", type=str, default=None)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--local-data", type=str, default=None,
                    help="local RewardBench json/jsonl for offline evaluate")
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
    ).to(args.device)

    # 加载 checkpoint
    state = torch.load(args.ckpt, map_location=args.device)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    result = eval_rewardbench(
        model,
        tokenizer,
        type_name=args.type_name,
        split=args.split,
        subset=args.subset,
        max_len=args.max_len,
        device=args.device,
        local_path=args.local_data,
    )

    if result is None:
        return

    # 保存 json 到 eval 目录
    save_dir = Path("evaluation/eval_rewardbench/rb_rm")
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"result_{args.type_name}_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
