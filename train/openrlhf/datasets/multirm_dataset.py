# -*- coding: utf-8 -*-
import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class MultiTypeJSONLDataset(Dataset):
    """
    JSONL 每行样例：
      - 分类: {"type":"safety","mode":"cls","prompt":"P","response":"R","target":1}
      - 偏好: {"type":"help","mode":"pref","prompt":"Q","pos":"A+","neg":"A-"}
      - 回归: {"type":"humanity","mode":"reg","prompt":"X","response":"Y","r_true":0.82}
    """
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def build_dataloader(
    model_name_or_path: str,
    path: str,
    batch_size: int,
    max_len: int,
    shuffle: bool = True,
    use_chat_template: bool = False,
):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = MultiTypeJSONLDataset(path)

    def _encode(prompt: str, resp: str):
        if use_chat_template and hasattr(tok, "apply_chat_template"):
            msgs = [{"role": "user", "content": prompt or ""}, {"role": "assistant", "content": resp or ""}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        else:
            text = (prompt or "") + ("\n" if prompt else "") + (resp or "")
        enc = tok(text, truncation=True, max_length=max_len, padding=False, return_tensors="pt")
        return enc["input_ids"][0], enc["attention_mask"][0]

    def collate(batch):
        out = {"cls": [], "pref": [], "reg": []}
        for ex in batch:
            t = ex["type"]; m = ex["mode"]; prompt = ex.get("prompt", "")

            if m == "cls":
                y = ex["response"]; target = int(ex["target"])
                ids, attn = _encode(prompt, y)
                out["cls"].append({"type": t, "input_ids": ids, "attention_mask": attn, "target": target})

            elif m == "pref":
                pos, neg = ex["pos"], ex["neg"]
                pos_ids, pos_attn = _encode(prompt, pos)
                neg_ids, neg_attn = _encode(prompt, neg)
                out["pref"].append({
                    "type": t,
                    "pos_input_ids": pos_ids, "pos_attention_mask": pos_attn,
                    "neg_input_ids": neg_ids, "neg_attention_mask": neg_attn
                })

            elif m == "reg":
                y = ex["response"]; r_true = float(ex["r_true"])
                ids, attn = _encode(prompt, y)
                out["reg"].append({"type": t, "input_ids": ids, "attention_mask": attn, "r_true": r_true})

        def _pad(stack, pad_val):
            T = max(x.size(0) for x in stack)
            padded = []
            for x in stack:
                if x.size(0) < T:
                    pad = torch.full((T - x.size(0),), pad_val, dtype=torch.long)
                    x = torch.cat([x, pad], dim=0)
                padded.append(x)
            return torch.stack(padded, dim=0)

        batch_out = {}

        if out["cls"]:
            ids  = _pad([o["input_ids"] for o in out["cls"]], tok.pad_token_id)
            attn = _pad([o["attention_mask"] for o in out["cls"]], 0)
            target = torch.tensor([o["target"] for o in out["cls"]], dtype=torch.long)
            types  = [o["type"] for o in out["cls"]]
            batch_out["cls"] = {"input_ids": ids, "attention_mask": attn, "target": target, "types": types}

        if out["pref"]:
            pos_ids  = _pad([o["pos_input_ids"] for o in out["pref"]], tok.pad_token_id)
            pos_attn = _pad([o["pos_attention_mask"] for o in out["pref"]], 0)
            neg_ids  = _pad([o["neg_input_ids"] for o in out["pref"]], tok.pad_token_id)
            neg_attn = _pad([o["neg_attention_mask"] for o in out["pref"]], 0)
            types    = [o["type"] for o in out["pref"]]
            batch_out["pref"] = {
                "pos_input_ids": pos_ids, "pos_attention_mask": pos_attn,
                "neg_input_ids": neg_ids, "neg_attention_mask": neg_attn,
                "types": types
            }

        if out["reg"]:
            ids  = _pad([o["input_ids"] for o in out["reg"]], tok.pad_token_id)
            attn = _pad([o["attention_mask"] for o in out["reg"]], 0)
            r    = torch.tensor([o["r_true"] for o in out["reg"]], dtype=torch.float32)
            types = [o["type"] for o in out["reg"]]
            batch_out["reg"] = {"input_ids": ids, "attention_mask": attn, "r_true": r, "types": types}

        return batch_out

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
