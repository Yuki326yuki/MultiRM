# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.inference_mode()
def compute_loglikelihood(model, tokenizer, text, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["input_ids"],
    )
    # Cross entropy loss: average negative loglikelihood
    neg_loglik = outputs.loss.item()
    return -neg_loglik  # higher = better


def evaluate_baseline(model_name, local_data, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"[RewardBench] Loading local data: {local_data}")
    ds = load_dataset("json", data_files={"data": local_data})["data"]

    total = 0
    correct = 0
    by_subset = {}

    for idx, row in enumerate(ds):
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]
        subset = row.get("subset", "unknown")

        txt_chosen = prompt + "\n\nAssistant: " + chosen
        txt_rejected = prompt + "\n\nAssistant: " + rejected

        s1 = compute_loglikelihood(model, tokenizer, txt_chosen, device)
        s2 = compute_loglikelihood(model, tokenizer, txt_rejected, device)

        win = s1 > s2
        total += 1
        correct += int(win)

        st = by_subset.setdefault(subset, {"total": 0, "correct": 0})
        st["total"] += 1
        st["correct"] += int(win)

        if (idx + 1) % 200 == 0:
            print(f"Processed {idx+1}/{len(ds)} examples...")

    result = {
        "model": model_name,
        "total": total,
        "correct": correct,
        "overall_acc": correct / total,
        "subset_acc": {},
    }

    for s, st in by_subset.items():
        result["subset_acc"][s] = st["correct"] / st["total"]

    # save result
    save_dir = Path("evaluation/eval_rewardbench/rb_basemode")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"basemode_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n===== Baseline Evaluation Finished =====")
    print(f"overall acc = {result['overall_acc']:.4f} ({correct}/{total})")
    print(f"saved to: {out_path}\n")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--local-data", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    evaluate_baseline(
        args.model,
        args.local_data,
        device=args.device,
    )


if __name__ == "__main__":
    main()
