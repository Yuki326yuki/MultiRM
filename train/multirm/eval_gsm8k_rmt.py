import argparse
import json
import os
import re
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 按你工程结构调整 import 路径
from openrlhf.models.multitype_rmt import MultiTypeRewardModel


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
    - 如果 '####' 后面为空，就回退到“最后一行找数字”的逻辑
    - 再不行，就返回空字符串
    """
    if "####" in ans:
        tail = ans.split("####")[-1].strip()
        tail = tail.replace(",", "")
        parts = tail.split()
        if parts:  # 只有在确实有内容时才取第一个
            return parts[0]
        # 如果 '####' 后面啥也没有，继续往下走，用 fallback 逻辑

    # fallback：从最后一行里找数字
    lines = [l for l in ans.splitlines() if l.strip()]
    if not lines:
        return ""
    last = lines[-1].replace(",", "")
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


# --------- MultiRM / HF CausalLM 加载 --------- #

def load_multirm(ckpt_path: str, device: str):
    """
    从 MultiRM checkpoint 加载模型和 tokenizer。
    ckpt 假定是 train_multirmt.py 保存的 final.pt，包含 {'cfg', 'model'}。
    对 state_dict 使用 strict=False，以适配 ref_lm / 结构微调。
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
    # 对 decoder-only 建议左 padding
    tokenizer.padding_side = "left"

    model = MultiTypeRewardModel(
        model_name_or_path=model_name,
        type_specs=type_specs,
        use_flash_attn_2=cfg["model"].get("flash_attn2", True),
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[load_multirm] missing keys:", missing)
    print("[load_multirm] unexpected keys:", unexpected)

    model.to(device)
    model.eval()

    # MultiRM 的评测仍然使用默认前后缀
    prompt_prefix = ""
    prompt_suffix = "\n\nAssistant:"

    return model, tokenizer, prompt_prefix, prompt_suffix


def load_hf_causallm(model_path: str, device: str):
    """
    加载任意 HF CausalLM 模型：
      - 可以是没训练的基础模型 (e.g. Qwen2.5-7B-Instruct)
      - 也可以是 DPO / SFT / 其他 RL 之后的 checkpoint 目录
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(device)
    model.eval()

    prompt_prefix = ""
    prompt_suffix = "\n\nAssistant:"

    return model, tokenizer, prompt_prefix, prompt_suffix


def load_grpo_policy(base_model_path: str, ckpt_path: str, device: str):
    """
    加载 GRPO 训练后的 policy：
      - 先从 base_model_path 加载一个 HF CausalLM
      - 再用 ckpt 中的 policy_state_dict 覆盖
      - 从 ckpt["cfg"]["train"] 中恢复 prompt_prefix / prompt_suffix
    """
    print("[load_grpo_policy] loading tokenizer and base model from:", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    print("[load_grpo_policy] loading policy_state_dict from:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["policy_state_dict"]
    model.load_state_dict(state_dict, strict=False)

    cfg = ckpt.get("cfg", {})
    train_cfg = cfg.get("train", {})
    prompt_prefix = train_cfg.get("prompt_prefix", "")
    prompt_suffix = train_cfg.get("prompt_suffix", "\n\nAssistant:")

    model.to(device)
    model.eval()

    print("[load_grpo_policy] prompt_prefix:", repr(prompt_prefix))
    print("[load_grpo_policy] prompt_suffix:", repr(prompt_suffix))

    return model, tokenizer, prompt_prefix, prompt_suffix


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
    num_samples: Optional[int] = None,
    out_path: Optional[str] = None,
    model_tag: str = "",
    model_type: str = "",
    prompt_prefix: str = "",
    prompt_suffix: str = "\n\nAssistant:",
) -> float:
    """
    返回 GSM8K accuracy。
    model: MultiTypeRewardModel 或 AutoModelForCausalLM
    tokenizer: 对应 tokenizer
    out_path: 若不为 None，则将每条样本结果和最终 summary 保存到该 jsonl 文件中
    """
    if num_samples is not None:
        data = data[:num_samples]

    n_total = 0
    n_correct = 0

    results = []

    for i, (q, gold) in enumerate(data, start=1):
        prompt = f"{prompt_prefix}{q.strip()}{prompt_suffix}"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        # MultiRM / HF / GRPO policy 都有 generate 接口
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=tokenizer.eos_token_id,
        )

        # 只取新生成的部分
        gen_text = tokenizer.decode(
            gen_ids[0][inputs["input_ids"].size(1):],
            skip_special_tokens=True,
        )

        gold_ans = extract_gsm8k_answer(gold)
        pred_ans = extract_gsm8k_answer(gen_text)

        is_ok = equal_answer(pred_ans, gold_ans)
        n_total += 1
        n_correct += int(is_ok)

        # 记录到结果里（包含题目和模型回答）
        results.append(
            {
                "index": i - 1,
                "question": q,
                "gold_raw": gold,
                "gold_answer": gold_ans,
                "pred_raw": gen_text,
                "pred_answer": pred_ans,
                "correct": bool(is_ok),
                "model_type": model_type,
                "model_tag": model_tag,
            }
        )

        if i % 50 == 0:
            print(
                f"[{i}/{len(data)}] ACC so far: {n_correct}/{n_total} = {n_correct / max(1, n_total):.4f}"
            )

    acc = n_correct / max(1, n_total)
    print(f"Final GSM8K Acc: {n_correct}/{n_total} = {acc:.4f}")

    # 把最终的 summary 也写进 jsonl
    summary_record = {
        "index": "summary",
        "accuracy": acc,
        "n_correct": n_correct,
        "n_total": n_total,
        "model_type": model_type,
        "model_tag": model_tag,
    }
    results.append(summary_record)

    # 保存 jsonl 结果
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("Saved detailed results to:", out_path)

    return acc


# --------- CLI 主程序 --------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k_path", type=str, required=True, help="原生 GSM8K test.jsonl 路径")
    ap.add_argument(
        "--model_type",
        type=str,
        choices=["multirm", "hf", "grpo"],   # hf: 任意 HF CausalLM；grpo: GRPO policy ckpt
        required=True,
    )
    ap.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "multirm: final.pt (包含 cfg+state_dict); "
            "hf: HF CausalLM checkpoint 目录; "
            "grpo: GRPO policy ckpt (grpo_final.pt 等)"
        ),
    )
    ap.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="当 model_type=grpo 时，需要提供 base HF CausalLM 路径 (例如 Qwen2.5-0.5B-Instruct)",
    )
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="只评估前 num_samples 条，默认全部",
    )
    # 新增：只评估前 limit 条（和 num_samples 功能类似，用哪个都行）
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只加载前 N 条样本进行评估（快速调试用）。",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="保存详细评测结果 jsonl 的目录（可选）",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    data = load_gsm8k(args.gsm8k_path)
    print("Loaded GSM8K test size:", len(data))

    # 如果指定了 limit，就在这里先裁掉数据，进一步加速
    if args.limit is not None and args.limit > 0:
        data = data[:args.limit]
        print(f"[INFO] Only evaluating first {args.limit} samples (after limit).")

    # 给结果文件起一个 tag，方便区分不同模型
    model_tag = os.path.basename(args.model_path.rstrip("/"))
    model_type = args.model_type

    # out-dir -> 具体文件名
    out_path = None
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(
            args.out_dir,
            f"{model_tag}_gsm8k_{model_type}.jsonl",
        )

    # 根据 model_type 加载不同模型
    if model_type == "multirm":
        print("Loading MultiRM implicit PRM model...")
        model, tokenizer, prefix, suffix = load_multirm(args.model_path, device)
    elif model_type == "hf":
        print("Loading HF CausalLM model (base / DPO / etc.)...")
        model, tokenizer, prefix, suffix = load_hf_causallm(args.model_path, device)
    else:  # grpo
        if args.base_model_path is None:
            raise ValueError("When model_type='grpo', you must provide --base_model_path")
        print("Loading GRPO policy model...")
        model, tokenizer, prefix, suffix = load_grpo_policy(
            base_model_path=args.base_model_path,
            ckpt_path=args.model_path,
            device=device,
        )

    evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        data=data,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        out_path=out_path,
        model_tag=model_tag,
        model_type=model_type,
        prompt_prefix=prefix,
        prompt_suffix=suffix,
    )


if __name__ == "__main__":
    main()
