#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
离线版 GSM8K 多任务 Reward 数据构造脚本（ImplicitPRM + MultiTypeRM）

特点：
- 从本地 GSM8K jsonl 读取（不访问网络）
- 使用本地大模型（如 Qwen3-32B / Qwen2.5-7B）生成高质量错误推理
- 构造多种 reward type：
    * math_final_answer
    * math_reasoning_validity
    * step_correctness
    * reasoning_complexity
    * format_adherence
- 为每个 type 生成三类监督：
    * mode = "pref" / "cls" / "reg"
- 带断点续跑：
    * 输出文件存在时，自动跳过已处理 example
    * 每条样本包含 example_hash 字段，便于追踪 & 去重

输出：
  <output-dir>/gsm8k_train_multirm_llm.jsonl
  <output-dir>/gsm8k_test_multirm_llm.jsonl
"""

import argparse
import json
import os
import re
import hashlib
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ===================== 基础工具函数 =====================

def extract_final_answer(answer: str) -> Optional[str]:
    """从 answer 中提取 '#### 42' 这种最后答案。"""
    m = re.search(r"####\s*([-+]?\d+)", answer)
    return m.group(1) if m else None


def replace_final_answer(full_answer: str, new_ans: str) -> str:
    """把原来的最终答案替换为 new_ans。"""
    if "####" in full_answer:
        return re.sub(r"(####\s*)[-+]?\d+", r"\g<1>" + new_ans, full_answer)
    return full_answer.rstrip() + f"\n#### {new_ans}"


def split_steps(answer: str) -> List[str]:
    lines = [l.strip() for l in answer.split("\n")]
    return [l for l in lines if l]


def truncate_reasoning(answer: str) -> str:
    """砍掉最后一两步，生成不完整推理。"""
    steps = split_steps(answer)
    if len(steps) <= 2:
        return answer
    return "\n".join(steps[: max(1, len(steps) - 2)])


def introduce_intermediate_error(answer: str) -> str:
    """对含 '=' 的某行做 +1 扰动，制造中间步错误。"""
    steps = split_steps(answer)
    new_steps = steps[:]
    for i, s in enumerate(steps):
        if "=" in s:
            try:
                left, right = s.split("=", 1)
                m = re.search(r"([-+]?\d+)", right)
                if not m:
                    continue
                val = int(m.group(1))
                wrong_val = val + 1
                new_right = re.sub(r"([-+]?\d+)", str(wrong_val), right, count=1)
                new_steps[i] = left.rstrip() + " = " + new_right
                return "\n".join(new_steps)
            except Exception:
                continue
    return answer


def format_score(answer: str) -> float:
    """是否遵守 '####' 格式。"""
    return 1.0 if "####" in answer else 0.0


def reasoning_complexity_score(answer: str, max_steps: int = 30) -> float:
    """用步数近似推理复杂度。"""
    steps = split_steps(answer)
    return min(len(steps) / max_steps, 1.0)


def check_simple_step_correctness(step: str) -> Optional[int]:
    """尝试判断简单算式是否正确，返回 1/0 或 None。"""
    if "=" not in step:
        return None
    try:
        left, right = step.split("=", 1)

        def clean(expr: str) -> str:
            return "".join(ch for ch in expr if ch.isdigit() or ch in "+-*/(). ")

        lhs = eval(clean(left))
        rhs = eval(clean(right))
        return 1 if abs(lhs - rhs) < 1e-6 else 0
    except Exception:
        return None


# ===================== 本地 LLM 封装 =====================

class LocalLLM:
    """本地 HF 模型（如 Qwen3-32B / Qwen2.5-7B-Instruct）"""

    def __init__(self, model_path: str, device: str = "cuda", max_new_tokens: int = 256):
        print(f"[LLM] Loading local model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)
        ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        out_ids = ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()


def build_wrong_reasoning_prompt(question: str, gold_ans: str) -> str:
    """构造 prompt，让 LLM 生成“貌似合理但最终答案错误”的推理。"""
    return f"""You are a math tutor.

Write a detailed step-by-step solution that is plausible but leads to a WRONG final answer.

Rules:
- The reasoning should look logical and typical for math word problems.
- Do NOT say it is wrong.
- End with "#### <number>", and the number MUST NOT be {gold_ans}.

Question:
{question}

Now write your solution:
"""


def generate_llm_wrong_reasonings(
    llm: LocalLLM,
    question: str,
    gold_ans: str,
    num_samples: int = 2,
) -> List[str]:
    """用本地大模型生成若干条“貌似合理但错误”的推理。"""
    outs = []
    for _ in range(num_samples):
        prompt = build_wrong_reasoning_prompt(question, gold_ans)
        text = llm.generate(prompt)
        if "####" not in text:
            continue
        wrong = extract_final_answer(text)
        if wrong is None or wrong == gold_ans:
            continue
        outs.append(text)
    return outs


# ===================== 读本地 GSM8K =====================

def load_local_gsm8k(jsonl_path: str) -> List[dict]:
    """
    假设每行至少包含 question / answer 字段。
    如果你的 raw 文件字段名不一样，在这里调整。
    """
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if "question" in ex and "answer" in ex:
                data.append(ex)
    return data


# ===================== 多任务样本构造 =====================

def build_samples_for_example(
    ex: dict,
    llm: LocalLLM,
    max_llm_neg: int = 2,
) -> List[dict]:
    out = []

    prompt = ex["question"]
    full_ans = ex["answer"]
    gold = extract_final_answer(full_ans)
    if gold is None:
        return out

    # 1) LLM 生成的负样本
    llm_negs = generate_llm_wrong_reasonings(llm, prompt, gold, num_samples=max_llm_neg)

    # 2) 规则负样本
    if gold.lstrip("-").isdigit():
        wrong_final = replace_final_answer(full_ans, str(int(gold) + 1))
    else:
        wrong_final = replace_final_answer(full_ans, gold + "0")
    truncated = truncate_reasoning(full_ans)
    inter_err = introduce_intermediate_error(full_ans)

    # ========= Type: math_final_answer (pref/cls/reg) =========

    def add_cls_reg(sol: str, label: int, score: float):
        out.append({
            "dataset": "gsm8k",
            "mode": "cls",
            "type": "math_final_answer",
            "prompt": prompt,
            "response": sol,
            "label": label,
        })
        out.append({
            "dataset": "gsm8k",
            "mode": "reg",
            "type": "math_final_answer",
            "prompt": prompt,
            "response": sol,
            "score": score,
        })

    # 人类解 vs 规则错解
    out.append({
        "dataset": "gsm8k",
        "mode": "pref",
        "type": "math_final_answer",
        "prompt": prompt,
        "pos": full_ans,
        "neg": wrong_final,
    })
    add_cls_reg(full_ans, 1, 1.0)
    add_cls_reg(wrong_final, 0, 0.0)

    # 人类解 vs LLM 错解
    for neg in llm_negs:
        out.append({
            "dataset": "gsm8k",
            "mode": "pref",
            "type": "math_final_answer",
            "prompt": prompt,
            "pos": full_ans,
            "neg": neg,
        })
        add_cls_reg(neg, 0, 0.0)

    # ========= Type: math_reasoning_validity (pref/reg) =========

    # 完整推理 vs 截断推理
    if truncated != full_ans:
        out.append({
            "dataset": "gsm8k",
            "mode": "pref",
            "type": "math_reasoning_validity",
            "prompt": prompt,
            "pos": full_ans,
            "neg": truncated,
        })
        out.append({
            "dataset": "gsm8k",
            "mode": "reg",
            "type": "math_reasoning_validity",
            "prompt": prompt,
            "response": truncated,
            "score": 0.4,
        })

    # 完整推理 vs 中间步错误推理
    if inter_err != full_ans:
        out.append({
            "dataset": "gsm8k",
            "mode": "pref",
            "type": "math_reasoning_validity",
            "prompt": prompt,
            "pos": full_ans,
            "neg": inter_err,
        })
        out.append({
            "dataset": "gsm8k",
            "mode": "reg",
            "type": "math_reasoning_validity",
            "prompt": prompt,
            "response": inter_err,
            "score": 0.2,
        })

    # 完整推理 vs LLM 错解
    for neg in llm_negs:
        out.append({
            "dataset": "gsm8k",
            "mode": "pref",
            "type": "math_reasoning_validity",
            "prompt": prompt,
            "pos": full_ans,
            "neg": neg,
        })
        out.append({
            "dataset": "gsm8k",
            "mode": "reg",
            "type": "math_reasoning_validity",
            "prompt": prompt,
            "response": neg,
            "score": 0.3,
        })

    # ========= Type: step_correctness (cls) =========

    for step in split_steps(full_ans):
        c = check_simple_step_correctness(step)
        if c is not None:
            out.append({
                "dataset": "gsm8k",
                "mode": "cls",
                "type": "step_correctness",
                "prompt": prompt,
                "response": step,
                "label": int(c),
            })

    # ========= Type: reasoning_complexity (reg) =========

    out.append({
        "dataset": "gsm8k",
        "mode": "reg",
        "type": "reasoning_complexity",
        "prompt": prompt,
        "response": full_ans,
        "score": reasoning_complexity_score(full_ans),
    })
    for neg in llm_negs:
        out.append({
            "dataset": "gsm8k",
            "mode": "reg",
            "type": "reasoning_complexity",
            "prompt": prompt,
            "response": neg,
            "score": min(reasoning_complexity_score(neg), 0.9),
        })

    # ========= Type: format_adherence (reg) =========

    for sol in [full_ans, wrong_final] + llm_negs:
        out.append({
            "dataset": "gsm8k",
            "mode": "reg",
            "type": "format_adherence",
            "prompt": prompt,
            "response": sol,
            "score": format_score(sol),
        })

    return out


# ===================== 断点续跑支持 =====================

def hash_example(ex: dict) -> str:
    """根据 question + answer 生成稳定 hash，用于识别 example。"""
    h = hashlib.md5()
    h.update(ex["question"].encode("utf-8"))
    h.update(ex["answer"].encode("utf-8"))
    return h.hexdigest()


def load_processed_hashes(out_path: str) -> set:
    """从已有输出文件中读取 example_hash 集合。"""
    processed = set()
    if not os.path.exists(out_path):
        return processed
    print(f"[RESUME] Loading processed example_hash from {out_path}")
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                h = obj.get("example_hash", None)
                if h:
                    processed.add(h)
            except Exception:
                continue
    print(f"[RESUME] Found {len(processed)} processed examples")
    return processed


def process_split_with_resume(
    name: str,
    jsonl_path: str,
    out_path: str,
    llm: LocalLLM,
    max_llm_neg: int = 2,
):
    print(f"[GSM8K] Loading {name} from {jsonl_path}")
    data = load_local_gsm8k(jsonl_path)
    print(f"[GSM8K] {name} total: {len(data)} examples")

    processed_hashes = load_processed_hashes(out_path)

    # 追加模式（若文件不存在，'a' 会新建）
    f_out = open(out_path, "a", encoding="utf-8")

    total_written = 0
    for i, ex in enumerate(data):
        ex_hash = hash_example(ex)
        if ex_hash in processed_hashes:
            if (i + 1) % 50 == 0:
                print(f"  [SKIP] {i+1}/{len(data)} already processed")
            continue

        try:
            samples = build_samples_for_example(ex, llm=llm, max_llm_neg=max_llm_neg)
        except Exception as e:
            print(f"[ERROR] fail at example {i}, skipping. Err={e}")
            continue

        for s in samples:
            s["example_hash"] = ex_hash
            f_out.write(json.dumps(s, ensure_ascii=False) + "\n")
            total_written += 1

        processed_hashes.add(ex_hash)

        if (i + 1) % 20 == 0:
            print(f"  [WRITE] {i+1}/{len(data)}, total_samples={total_written}")

    f_out.close()
    print(f"[DONE] {name}: wrote {total_written} samples to {out_path}")


# ===================== main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", type=str, required=True,
                    help="本地 GSM8K 训练集 jsonl（含 question / answer）")
    ap.add_argument("--test-file", type=str, required=True,
                    help="本地 GSM8K 测试集 jsonl")
    ap.add_argument("--output-dir", type=str, required=True,
                    help="输出目录")
    ap.add_argument("--llm-model", type=str, required=True,
                    help="本地 HF 模型路径或名称，如 Qwen3-32B / Qwen2.5-7B-Instruct")
    ap.add_argument("--device", type=str, default="cuda",
                    help="设备，如 'cuda', 'cuda:0'")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="LLM 生成的最大 new tokens 数")
    ap.add_argument("--max-llm-neg", type=int, default=2,
                    help="每个样本最多生成多少条 LLM 错误推理")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    llm = LocalLLM(
        model_path=args.llm_model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    train_out = os.path.join(args.output_dir, "gsm8k_train_multirm_llm.jsonl")
    test_out = os.path.join(args.output_dir, "gsm8k_test_multirm_llm.jsonl")

    process_split_with_resume("train", args.train_file, train_out, llm, args.max_llm_neg)
    process_split_with_resume("test", args.test_file, test_out, llm, args.max_llm_neg)


if __name__ == "__main__":
    main()
