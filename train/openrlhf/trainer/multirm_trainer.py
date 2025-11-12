# -*- coding: utf-8 -*-
import os, math, json
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from openrlhf.models.multitype_rm import MultiTypeRewardModel
from openrlhf.datasets.multirm_dataset import build_dataloader


class MultiTypeRMTrainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # dataloader
        self.train_dl = build_dataloader(
            cfg["model"]["pretrain"],
            cfg["data"]["train_path"],
            cfg["train"]["batch_size"],
            cfg["train"]["max_len"],
            shuffle=True,
            use_chat_template=cfg["data"].get("use_chat_template", False),
        )

        # model
        self.model = MultiTypeRewardModel(
            model_name_or_path=cfg["model"]["pretrain"],
            type_specs=cfg["types"],
            use_flash_attn_2=cfg["model"].get("flash_attn2", True),
            lora=cfg["model"].get("lora", False),
            lora_r=cfg["model"].get("lora_r", 8),
            lora_alpha=cfg["model"].get("lora_alpha", 16),
            lora_dropout=cfg["model"].get("lora_dropout", 0.05),
        ).to(self.device)

        # optim & sched
        self.opt = AdamW(self.model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.0))
        t_total = cfg["train"]["steps"]
        self.sched = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=int(0.03 * t_total), num_training_steps=t_total)

        os.makedirs(cfg["train"]["out_dir"], exist_ok=True)

    def _compute_losses(self, batch: Dict):
        device = self.device
        total = torch.tensor(0., device=device)
        log = {}

        # 分类 CE
        if "cls" in batch:
            ids, attn = batch["cls"]["input_ids"].to(device), batch["cls"]["attention_mask"].to(device)
            emb = self.model.encode(ids, attn)
            targets = batch["cls"]["target"].to(device)
            types = batch["cls"]["types"]

            loss_ce = torch.tensor(0., device=device)
            for i in range(emb.size(0)):
                k = types[i]
                logits = self.model.logits(emb[i:i+1], k)  # [1, m_k]
                loss_ce = loss_ce + F.cross_entropy(logits, targets[i:i+1])
            loss_ce = loss_ce / max(1, emb.size(0))
            total = total + loss_ce
            log["loss_ce"] = float(loss_ce.detach())

        # 偏好 CE（pos vs neg 拼成二维）
        if "pref" in batch:
            pos_ids = batch["pref"]["pos_input_ids"].to(device)
            pos_attn = batch["pref"]["pos_attention_mask"].to(device)
            neg_ids = batch["pref"]["neg_input_ids"].to(device)
            neg_attn = batch["pref"]["neg_attention_mask"].to(device)
            types = batch["pref"]["types"]

            pos_emb = self.model.encode(pos_ids, pos_attn)
            neg_emb = self.model.encode(neg_ids, neg_attn)

            loss_pref = torch.tensor(0., device=device)
            for i in range(pos_emb.size(0)):
                k = types[i]
                s_pos = self.model.logits(pos_emb[i:i+1], k).amax(dim=-1, keepdim=True)  # [1,1]
                s_neg = self.model.logits(neg_emb[i:i+1], k).amax(dim=-1, keepdim=True)  # [1,1]
                two = torch.cat([s_pos, s_neg], dim=-1)                                   # [1,2]
                target = torch.zeros(1, dtype=torch.long, device=device)                  # 0=pos更优
                loss_pref = loss_pref + F.cross_entropy(two, target)
            loss_pref = loss_pref / max(1, pos_emb.size(0))
            total = total + loss_pref
            log["loss_pref"] = float(loss_pref.detach())

        # 回归 MSE（保留 softmax 输出供分析/可选KL）
        if "reg" in batch:
            ids, attn = batch["reg"]["input_ids"].to(device), batch["reg"]["attention_mask"].to(device)
            emb = self.model.encode(ids, attn)
            r_true = batch["reg"]["r_true"].to(device)
            types = batch["reg"]["types"]

            loss_mse = torch.tensor(0., device=device)
            for i in range(emb.size(0)):
                k = types[i]
                r_hat = self.model.reward_scalar(emb[i:i+1], k)  # [1]
                loss_mse = loss_mse + F.mse_loss(r_hat, r_true[i:i+1])
            loss_mse = loss_mse / max(1, emb.size(0))
            total = total + loss_mse
            log["loss_mse"] = float(loss_mse.detach())

        return total, log

    def train(self):
        steps = self.cfg["train"]["steps"]
        log_every = self.cfg["train"]["log_every"]
        save_every = self.cfg["train"]["save_every"]
        out_dir = self.cfg["train"]["out_dir"]

        self.model.train()
        step = 0
        while step < steps:
            for batch in self.train_dl:
                step += 1
                loss, log = self._compute_losses(batch)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"].get("grad_clip", 1.0))
                self.opt.step()
                self.sched.step()

                if step % log_every == 0:
                    print(f"[step {step}/{steps}] loss={loss.item():.4f} log={log}")

                if step % save_every == 0:
                    path = os.path.join(out_dir, f"step{step}.pt")
                    torch.save({"model": self.model.state_dict(), "cfg": self.cfg}, path)
                    print("saved:", path)

                if step >= steps:
                    break

        final_path = os.path.join(out_dir, "final.pt")
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg}, final_path)
        print("saved:", final_path)
