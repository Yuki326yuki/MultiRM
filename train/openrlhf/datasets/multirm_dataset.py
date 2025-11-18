# train/openrlhf/datasets/multirm_dataset.py
# -*- coding: utf-8 -*-
import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


class MultiTypeJSONLDataset(Dataset):
    """
    每行 JSON 示例（prepare_multirm_data.py 会生成）：
      - 偏好:
        {"dataset":"ultra_bin","mode":"pref","type":"overall",
         "prompt":"...", "pos":"...", "neg":"..."}

      - 回归:
        {"dataset":"UltraFeedback","mode":"reg","type":"helpfulness",
         "prompt":"...", "response":"...", "score":0.73}

      - 分类:
        {"dataset":"HelpSteer2","mode":"cls","type":"correctness",
         "prompt":"...", "response":"...", "label":2}
    """
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


def build_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    path: str,
    batch_size: int,
    max_len: int,
    shuffle: bool = True,
):
    ds = MultiTypeJSONLDataset(path)

    def _encode(prompt: str, resp: str):
        text = (prompt or "") + ("\n\nAssistant: " if prompt else "") + (resp or "")
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors="pt",
        )
        return enc["input_ids"][0], enc["attention_mask"][0]

    def collate(batch: List[Dict]):
        out = {"cls": [], "pref": [], "reg": []}

        for ex in batch:
            mode = ex["mode"]
            t = ex["type"]
            prompt = ex.get("prompt", "")

            if mode == "cls":
                resp = ex["response"]
                label = int(ex["label"])
                ids, attn = _encode(prompt, resp)
                out["cls"].append(
                    {"type": t, "input_ids": ids, "attention_mask": attn, "target": label}
                )

            elif mode == "pref":
                pos = ex["pos"]
                neg = ex["neg"]
                pos_ids, pos_attn = _encode(prompt, pos)
                neg_ids, neg_attn = _encode(prompt, neg)
                out["pref"].append(
                    {
                        "type": t,
                        "pos_input_ids": pos_ids,
                        "pos_attention_mask": pos_attn,
                        "neg_input_ids": neg_ids,
                        "neg_attention_mask": neg_attn,
                    }
                )

            elif mode == "reg":
                resp = ex["response"]
                score = float(ex["score"])
                ids, attn = _encode(prompt, resp)
                out["reg"].append(
                    {"type": t, "input_ids": ids, "attention_mask": attn, "r_true": score}
                )

        def _pad(tensors: List[torch.Tensor], pad_val: int):
            if not tensors:
                return None
            T = max(x.size(0) for x in tensors)
            padded = []
            for x in tensors:
                if x.size(0) < T:
                    pad = torch.full(
                        (T - x.size(0),),
                        pad_val,
                        dtype=torch.long,
                    )
                    x = torch.cat([x, pad], dim=0)
                padded.append(x)
            return torch.stack(padded, dim=0)

        batch_out: Dict[str, Dict] = {}

        if out["cls"]:
            ids = _pad([o["input_ids"] for o in out["cls"]], tokenizer.pad_token_id)
            attn = _pad([o["attention_mask"] for o in out["cls"]], 0)
            target = torch.tensor([o["target"] for o in out["cls"]], dtype=torch.long)
            types = [o["type"] for o in out["cls"]]
            batch_out["cls"] = {
                "input_ids": ids,
                "attention_mask": attn,
                "target": target,
                "types": types,
            }

        if out["pref"]:
            pos_ids = _pad([o["pos_input_ids"] for o in out["pref"]], tokenizer.pad_token_id)
            pos_attn = _pad([o["pos_attention_mask"] for o in out["pref"]], 0)
            neg_ids = _pad([o["neg_input_ids"] for o in out["pref"]], tokenizer.pad_token_id)
            neg_attn = _pad([o["neg_attention_mask"] for o in out["pref"]], 0)
            types = [o["type"] for o in out["pref"]]
            batch_out["pref"] = {
                "pos_input_ids": pos_ids,
                "pos_attention_mask": pos_attn,
                "neg_input_ids": neg_ids,
                "neg_attention_mask": neg_attn,
                "types": types,
            }

        if out["reg"]:
            ids = _pad([o["input_ids"] for o in out["reg"]], tokenizer.pad_token_id)
            attn = _pad([o["attention_mask"] for o in out["reg"]], 0)
            r_true = torch.tensor([o["r_true"] for o in out["reg"]], dtype=torch.float32)
            types = [o["type"] for o in out["reg"]]
            batch_out["reg"] = {
                "input_ids": ids,
                "attention_mask": attn,
                "r_true": r_true,
                "types": types,
            }

        return batch_out

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
