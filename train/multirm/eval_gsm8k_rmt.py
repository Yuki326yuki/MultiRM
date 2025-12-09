import argparse
import json
import os
import re
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 这一条要保证和你工程路径一致
from openrlhf.models.multitype_rm import MultiTypeRewardModel


# --------- 工具函数：读取 GSM8K --------- #

def load_gsm8k(path: str) -> List[Tuple[str, str]]:
    """读取原生 GSM8K jsonl，返回 [(question, answer_str), ...]"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj["question"]
            a = obj["answer"]
            data.append((q, a))
    return data


# --------- 工具函数：从答案里解析数字 --------- #

def extract_gsm8k_answer(ans: str) -> str:
    """
    从 GSM8K 的答案字符串中抽取最终数值。
    - 先尝试找 '#### ' 后面的部分
    - 否则从最后一行里找数字
    返回纯文本数字（去掉逗号），找不到就返回空字符串
    """
    if "####" in ans:
        # 通常格式: "... #### 42"
        tail = ans.split("####")[-1].strip()
        # 去掉逗号，例如 "1,234"
        tail = tail.replace(",", "")
        # 取第一个 token 即可
        return tail.split()[0]

    # 否则从最后一行找
    lines = [l for l in ans.splitlines() if l.strip()]
    if not lines:
        return ""
    last = lines[-1]
    last = last.replace(",", "")
    # 找最后一个数字
    m = list(re.finditer(r"-?\d+(\.\d+)?", last))
    if not m:
        return ""
    return m[-1].group(0)


def equal_answer(a_pred: str, a_gold: str) -> bool:
    """
    判断两个数值是否相等。
    简单做法：尝试转换为 float；转换失败则直接字符串对比。
    """
    if a_pred == "" or a_gold == "":
        return False
    try:
        return float(a_pred) == float(a_gold)
    except Exception:
        return a_pred == a_gold


# --------- MultiRM / DPO 加载 --------- #

def load_multirm(ckpt_path: str, device: str):
    """
    从你训练好的 MultiRM checkpoint 加载模型和 tokenizer。
    ckpt 假定是 train_multirm.py 保存的 final.pt，包含 {'cfg', 'model'}。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    state_dict = ckpt["model"]

    model_name = cfg["model"]["pretrain"]
    type_specs = cfg["types"]

    tokenizer_name = cfg["model"].get("tokenizer_name_or_path") or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = MultiTypeRewardModel(
        model_name_or_path=model_name,
        type_specs=type_specs,
        use_flash_attn_2=cfg["model"].get("flash_attn2", True),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer


def load_dpo(model_path: str, device: str):
    """
    加载 DPO 模型：标准 AutoModelForCausalLM checkpoint。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


# --------- 评测循环：给定模型和 tokenizer --------- #

@torch.no_grad()
def evaluate_gsm8k(
    model,
    tokenizer,
    data: List[Tuple[str, str]],
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_samples: int = None,
) -> float:
    """
    返回 GSM8K accuracy。
    model: MultiTypeRewardModel 或 AutoModelForCausalLM
    tokenizer: 对应 tokenizer
    """
    if num_samples is not None:
        data = data[:num_samples]

    n_total = 0
    n_correct = 0

    for i, (q, gold) in enumerate(data, start=1):
        # 使用和你训练数据一致的 prompt 模板
        prompt = q.strip() + "\n\nAssistant:"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        # MultiRM 有 generate 方法；DPO 模型也有 generate（同接口）
        if hasattr(model, "generate"):
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            # 理论上 MultiTypeRewardModel 实现了 generate，这个分支几乎用不到
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 只取新生成的部分
        gen_text = tokenizer.decode(gen_ids[0][inputs["input_ids"].size(1):], skip_special_tokens=True)

        gold_ans = extract_gsm8k_answer(gold)
        pred_ans = extract_gsm8k_answer(gen_text)

        is_ok = equal_answer(pred_ans, gold_ans)
        n_total += 1
        n_correct += int(is_ok)

        if i % 50 == 0:
            print(
                f"[{i}/{len(data)}] ACC so far: {n_correct}/{n_total} = {n_correct / max(1, n_total):.4f}"
            )

    acc = n_correct / max(1, n_total)
    print(f"Final GSM8K Acc: {n_correct}/{n_total} = {acc:.4f}")
    return acc


# --------- CLI 主程序 --------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k_path", type=str, required=True, help="原生 GSM8K test.jsonl 路径")
    ap.add_argument("--model_type", type=str, choices=["multirm", "dpo"], required=True)
    ap.add_argument("--model_path", type=str, required=True,
                    help="multirm: final.pt; dpo: HF CausalLM checkpoint 目录")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--num_samples", type=int, default=None,
                    help="只评估前 num_samples 条，默认全部")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    data = load_gsm8k(args.gsm8k_path)
    print("Loaded GSM8K test size:", len(data))

    if args.model_type == "multirm":
        print("Loading MultiRM implicit PRM model...")
        model, tokenizer = load_multirm(args.model_path, device)
    else:
        print("Loading DPO model...")
        model, tokenizer = load_dpo(args.model_path, device)

    evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        data=data,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
