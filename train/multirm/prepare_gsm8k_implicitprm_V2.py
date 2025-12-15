#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GSM8K multi-type reward dataset builder (BATCH + 2nd round)

- Keep prompt text unchanged.
- Keep sample schema/fields unchanged.
- Keep filtering logic unchanged.
- Add second (and up to N rounds) generation for questions that didn't reach max_llm_neg.
- Still uses batch generation (sub-batch for the remaining questions).

This prevents "empty neg" / "not enough neg" without fabricating fake negatives.

[Resume/Checkpoint Protection Added]
- Adds a .progress file per output jsonl, storing next_batch index.
- Does NOT change prompt, filtering, schema, or sample generation logic.
- Only changes where the outer loop resumes after an interruption.
"""

import os
import re
import json
import argparse
import hashlib
import tempfile
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Resume / checkpoint helpers
# =========================

def load_progress(progress_path: str) -> int:
    """
    Read next batch index from progress file.
    If missing/corrupted, return 0 (start from beginning).
    """
    if not os.path.exists(progress_path):
        return 0
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return int(obj.get("next_batch", 0))
    except Exception:
        return 0


def save_progress_atomic(progress_path: str, next_batch: int):
    """
    Atomic write to avoid corruption if process is killed during write.
    Writes {"next_batch": <int>} to progress_path.
    """
    os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(progress_path) or ".",
        prefix=os.path.basename(progress_path) + ".tmp.",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"next_batch": int(next_batch)}, f, ensure_ascii=False)
        os.replace(tmp_path, progress_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# =========================
# Answer parsing utilities
# =========================

FINAL_ANS_RE = re.compile(r"####\s*([-+]?\d+(\.\d+)?)")

def extract_final_answer(answer: str) -> Optional[str]:
    m = FINAL_ANS_RE.search(answer)
    return m.group(1).strip() if m else None

def replace_final_answer(full_answer: str, new_ans: str) -> str:
    if "####" not in full_answer:
        return full_answer.rstrip() + f"\n#### {new_ans}\n"
    return FINAL_ANS_RE.sub(f"#### {new_ans}", full_answer, count=1)

def format_score(answer: str) -> float:
    return 1.0 if extract_final_answer(answer) is not None else 0.0


# =========================
# Prompt (UNCHANGED)
# =========================

WRONG_REASONING_PROMPT = """Write a detailed step-by-step solution that contains a subtle but realistic mistake in the reasoning process (e.g., a wrong assumption, an incorrect calculation, or a misapplied formula), so that the conclusion is incorrect.

Rules:
- The reasoning should look logical and typical for math word problems.
- The mistake MUST appear before the final answer line.
- Do NOT intentionally change only the final answer.
- Do NOT say that the solution is wrong.
- End with "#### <number>", and the number MUST NOT be {gold_ans}.

Question:
{question}

Now write your solution:
"""

def build_wrong_reasoning_prompt(question: str, gold_ans: str) -> str:
    return WRONG_REASONING_PROMPT.format(question=question, gold_ans=gold_ans)


# =========================
# Local LLM with BATCH GENERATE
# =========================

class LocalLLM:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 256,
        attn_impl: str = "auto",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        extra = {}
        if attn_impl != "auto":
            extra["attn_implementation"] = attn_impl  # e.g. "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=dtype,
            trust_remote_code=True,
            **extra,
        ).eval()

        self.device = device
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def generate_many_batch(
        self,
        prompts: List[str],
        num_return_sequences: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[List[str]]:
        """
        prompts: [B]
        returns: List[B][num_return_sequences]
        IMPORTANT BUG FIX:
          Decode ONLY newly generated tokens; do NOT include prompt.
        """
        B = len(prompts)
        if B == 0:
            return []

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        if isinstance(self.device, str) and self.device.startswith("cuda"):
            inputs = inputs.to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        # ids: [B * R, prompt_len + gen_len]
        gen_ids = ids[:, prompt_len:]  # only new tokens
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # regroup to [B][R] in the exact generation order
        out = [[] for _ in range(B)]
        idx = 0
        for b in range(B):
            for _ in range(num_return_sequences):
                t = (texts[idx] or "").strip()
                marker = "Now write your solution:"
                if marker in t:
                    t = t.split(marker, 1)[-1].strip()
                out[b].append(t)
                idx += 1
        return out


# =========================
# Batch negative generation WITH ROUNDS (still batch)
# =========================

def _filter_and_append_negs(
    raw_texts: List[str],
    gold: str,
    want: int,
    existing: List[str],
    seen_hash: set,
) -> None:
    """
    Keep EXACT SAME filtering logic as before:
      - must contain '####'
      - extracted final answer exists and != gold
      - de-dup by full text
    Append until len(existing) reaches want.
    """
    if want <= 0:
        return

    for t in raw_texts:
        if len(existing) >= want:
            return
        if not t:
            continue
        if "####" not in t:
            continue
        wrong = extract_final_answer(t)
        if wrong is None or wrong == gold:
            continue
        h = hashlib.md5(t.encode("utf-8")).hexdigest()
        if h in seen_hash:
            continue
        seen_hash.add(h)
        existing.append(t)


def generate_llm_wrong_reasonings_batch_with_rounds(
    llm: LocalLLM,
    questions: List[str],
    gold_answers: List[str],
    num_samples: int,
    oversample: int,
    max_rounds: int,
    sub_batch_size: int,
) -> List[List[str]]:
    """
    Batch generate negatives for a batch of questions, with up to max_rounds.

    Logic (NO change in semantics):
      - Each question wants num_samples negatives.
      - Each round: generate R = max(2, num_samples * oversample) candidates per question.
      - Filter candidates with EXACT SAME filters.
      - If some questions still have < num_samples, run another round ONLY on those questions
        (still in batch, potentially chunked by sub_batch_size).
      - Never fabricate negatives.
    """
    B = len(questions)
    want = max(0, int(num_samples))
    if B == 0 or want == 0:
        return [[] for _ in range(B)]

    results: List[List[str]] = [[] for _ in range(B)]
    seen_hashes: List[set] = [set() for _ in range(B)]

    # Indices within this batch that still need negs
    remaining = [i for i in range(B) if gold_answers[i] is not None]
    # If some gold is None, they will simply end up with [] (no negs)
    # This matches "no fake fill" principle.

    R = max(2, want * max(2, int(oversample)))

    for _round in range(max(1, int(max_rounds))):
        if not remaining:
            break

        # To keep GPU utilization, we still batch them; if remaining is huge, chunk it.
        for start in range(0, len(remaining), max(1, int(sub_batch_size))):
            idxs = remaining[start:start + max(1, int(sub_batch_size))]

            prompts = []
            for j in idxs:
                g = gold_answers[j]
                if g is None:
                    prompts.append("")  # won't be used meaningfully
                else:
                    prompts.append(build_wrong_reasoning_prompt(questions[j], g))

            # Generate candidates (still batch)
            raw = llm.generate_many_batch(prompts, num_return_sequences=R)

            # Filter & append to each corresponding question
            for local_k, j in enumerate(idxs):
                g = gold_answers[j]
                if g is None:
                    continue
                _filter_and_append_negs(
                    raw_texts=raw[local_k],
                    gold=g,
                    want=want,
                    existing=results[j],
                    seen_hash=seen_hashes[j],
                )

        # Recompute remaining (only those still short)
        remaining = [i for i in remaining if len(results[i]) < want]

    return results


# =========================
# Sample builder (unchanged schema/logic)
# =========================

def build_samples_for_example(
    ex: dict,
    llm_negs: List[str],
    types: List[str],
    include_wrong_final: bool,
    include_reg_for_final: bool,
) -> List[dict]:
    out = []
    question = ex["question"]
    full_ans = ex["answer"]
    gold = extract_final_answer(full_ans)
    if gold is None:
        return out

    wrong_final = None
    if include_wrong_final:
        if gold.lstrip("-").isdigit():
            wrong_final = replace_final_answer(full_ans, str(int(gold) + 1))
        else:
            wrong_final = replace_final_answer(full_ans, gold + "0")

    neg_pool = []
    if wrong_final:
        neg_pool.append(wrong_final)
    neg_pool.extend(llm_negs)

    if "math_final_answer" in types:
        for neg in neg_pool:
            out.append({
                "dataset": "gsm8k",
                "mode": "pref",
                "type": "math_final_answer",
                "prompt": question,
                "pos": full_ans,
                "neg": neg,
                "meta": {"gold": gold},
            })

        out.append({
            "dataset": "gsm8k",
            "mode": "cls",
            "type": "math_final_answer",
            "prompt": question,
            "response": full_ans,
            "label": 1,
            "meta": {"gold": gold},
        })
        for neg in neg_pool:
            out.append({
                "dataset": "gsm8k",
                "mode": "cls",
                "type": "math_final_answer",
                "prompt": question,
                "response": neg,
                "label": 0,
                "meta": {"gold": gold},
            })

        if include_reg_for_final:
            out.append({
                "dataset": "gsm8k",
                "mode": "reg",
                "type": "math_final_answer",
                "prompt": question,
                "response": full_ans,
                "score": 1.0,
                "meta": {"gold": gold},
            })
            for neg in neg_pool:
                out.append({
                    "dataset": "gsm8k",
                    "mode": "reg",
                    "type": "math_final_answer",
                    "prompt": question,
                    "response": neg,
                    "score": 0.0,
                    "meta": {"gold": gold},
                })

    if "format_adherence" in types:
        candidates = [full_ans]
        if wrong_final:
            candidates.append(wrong_final)
        candidates.extend(llm_negs)

        uniq = []
        seen = set()
        for c in candidates:
            h = hashlib.md5(c.encode("utf-8")).hexdigest()
            if h not in seen:
                uniq.append(c)
                seen.add(h)

        for sol in uniq:
            out.append({
                "dataset": "gsm8k",
                "mode": "reg",
                "type": "format_adherence",
                "prompt": question,
                "response": sol,
                "score": float(format_score(sol)),
                "meta": {"gold": gold},
            })

    return out


# =========================
# Main with BATCH + ROUNDS
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-model", required=True)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-llm-neg", type=int, default=2)
    parser.add_argument("--neg-oversample", type=int, default=8)

    # NEW (batch control)
    parser.add_argument("--gen-batch-size", type=int, default=8,
                        help="Number of questions per outer batch generation")

    # NEW (rounds, still batch)
    parser.add_argument("--neg-max-rounds", type=int, default=2,
                        help="How many rounds to try for missing negatives (batch sub-rounds).")
    parser.add_argument("--neg-round-batch-size", type=int, default=0,
                        help="Sub-batch size for round generation on remaining questions. "
                             "0 means use gen-batch-size.")

    parser.add_argument("--types", default="math_final_answer,format_adherence")
    parser.add_argument("--no-wrong-final", action="store_true")
    parser.add_argument("--final-reg", action="store_true")

    # Optional attention impl knob (pure speed, no logic change)
    parser.add_argument("--attn-impl", type=str, default="auto",
                        help='Optional: "flash_attention_2" if available, else "auto".')

    args = parser.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    include_wrong_final = not args.no_wrong_final

    sub_bs = args.neg_round_batch_size if args.neg_round_batch_size and args.neg_round_batch_size > 0 else args.gen_batch_size

    llm = LocalLLM(
        model_path=args.llm_model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.attn_impl,
    )

    def process_split(split: str, in_path: str, out_path: str):
        with open(in_path, "r", encoding="utf-8") as f:
            data = [json.loads(l) for l in f if l.strip()]

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # ===== Resume logic (does not touch generation/filtering/schema) =====
        progress_path = out_path + ".progress"
        start_batch = load_progress(progress_path)
        total_batches = (len(data) + args.gen_batch_size - 1) // args.gen_batch_size

        if start_batch > 0:
            print(f"[{split}] resume enabled: start_batch={start_batch} / total_batches={total_batches}")
        else:
            print(f"[{split}] start from beginning: total_batches={total_batches}")

        with open(out_path, "a", encoding="utf-8") as f_out:
            for batch_idx in range(start_batch, total_batches):
                i = batch_idx * args.gen_batch_size
                batch = data[i:i + args.gen_batch_size]
                questions = [ex["question"] for ex in batch]
                golds = [extract_final_answer(ex["answer"]) for ex in batch]

                llm_negs_batch = generate_llm_wrong_reasonings_batch_with_rounds(
                    llm=llm,
                    questions=questions,
                    gold_answers=golds,
                    num_samples=args.max_llm_neg,
                    oversample=args.neg_oversample,
                    max_rounds=args.neg_max_rounds,
                    sub_batch_size=sub_bs,
                )

                for ex, llm_negs in zip(batch, llm_negs_batch):
                    samples = build_samples_for_example(
                        ex=ex,
                        llm_negs=llm_negs,
                        types=types,
                        include_wrong_final=include_wrong_final,
                        include_reg_for_final=args.final_reg,
                    )
                    for s in samples:
                        f_out.write(json.dumps(s, ensure_ascii=False) + "\n")

                # Ensure output is on disk before marking progress
                f_out.flush()

                # Update progress ONLY after finishing this batch
                save_progress_atomic(progress_path, batch_idx + 1)

                if batch_idx % 10 == 0:
                    print(f"[{split}] batch {batch_idx}/{total_batches}")

    train_out = os.path.join(args.output_dir, "gsm8k_train_multirm_V2.jsonl")
    test_out = os.path.join(args.output_dir, "gsm8k_test_multirm_V2.jsonl")

    process_split("train", args.train_file, train_out)
    process_split("test", args.test_file, test_out)


if __name__ == "__main__":
    main()
