# train/openrlhf/trainer/multirmt_trainer.py
# -*- coding: utf-8 -*-
import os
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from openrlhf.models.multitype_rmt import MultiTypeRewardModel
from openrlhf.datasets.multirmt_dataset import build_dataloader


class MultiTypeRMTrainer:
    """
    Multi-type implicit PRM trainer with:
      - token-level implicit rewards for all modes (cls/pref/reg)
      - optional LM loss to preserve generation ability
      - separate learning rates for lm vs. other parameters
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # tokenizer
        tok_name = cfg["model"].get("tokenizer_name_or_path") or cfg["model"]["pretrain"]
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # dataloader
        self.train_dl: DataLoader = build_dataloader(
            self.tokenizer,
            cfg["data"]["train_path"],
            cfg["train"]["batch_size"],
            cfg["train"]["max_len"],
            shuffle=True,
        )
        self.iterator = iter(self.train_dl)

        # model
        self.model = MultiTypeRewardModel(
            model_name_or_path=cfg["model"]["pretrain"],
            type_specs=cfg["types"],
            use_flash_attn_2=cfg["model"].get("flash_attn2", True),
        ).to(self.device)

        # optimizer with different lr for lm vs. others
        base_lr = cfg["train"]["lr"]
        lr_lm = cfg["train"].get("lr_lm", base_lr)
        weight_decay = cfg["train"].get("weight_decay", 0.0)

        lm_params = []
        other_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("lm."):
                lm_params.append(p)
            else:
                other_params.append(p)

        param_groups = []
        if lm_params:
            param_groups.append({"params": lm_params, "lr": lr_lm, "weight_decay": weight_decay})
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})

        self.opt = AdamW(param_groups)

        t_total = cfg["train"]["steps"]
        self.sched = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=int(0.03 * t_total),
            num_training_steps=t_total,
        )

        os.makedirs(cfg["train"]["out_dir"], exist_ok=True)

    # ------------------------------------------------------------------
    def _next_batch(self) -> Dict:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.train_dl)
            batch = next(self.iterator)
        return batch

    # ------------------------------------------------------------------
    def _compute_multirm_losses(self, batch: Dict[str, Dict]) -> (torch.Tensor, Dict[str, float]):
        """
        Core multi-type implicit PRM losses:
          - cls : multi-class CE on sequence-level implicit reward r_seq
          - pref: pairwise CE on [r_pos_seq, r_neg_seq]
          - reg : MSE on r_seq vs. scalar score
        All three share the token-level implicit reward pipeline.
        """
        device = self.device
        total = torch.tensor(0.0, device=device)
        log: Dict[str, float] = {}
        beta_kl = float(self.cfg["train"].get("beta_kl", 0.0))

        # ========= 1) CLS: use implicit sequence reward + type-specific head =========
        loss_ce = torch.tensor(0.0, device=device)
        if "cls" in batch:
            data = batch["cls"]
            ids = data["input_ids"].to(device)           # [B, T]
            attn = data["attention_mask"].to(device)     # [B, T]
            resp_mask = data["response_mask"].to(device) # [B, T]
            targets = data["target"].to(device)          # [B]
            types = data["types"]
            B = ids.size(0)

            for i in range(B):
                k = types[i]
                beta_k = float(self.cfg["types"][k].get("beta", 1.0))

                # sequence-level implicit reward from token-level r_t
                _, r_seq = self.model.token_implicit_reward(
                    ids[i : i + 1],
                    attn[i : i + 1],
                    resp_mask[i : i + 1],
                    beta=beta_k,
                )  # [1]

                # map scalar reward to multi-class logits via per-type head
                logits_k = self.model.cls_logits_from_reward(r_seq, k)  # [1, m_k]

                alpha = float(self.cfg["types"][k].get("alpha", 1.0))
                loss_ce = loss_ce + alpha * F.cross_entropy(logits_k, targets[i : i + 1])

                # optional KL to some reference label distribution (if provided)
                if beta_kl > 0 and "ref_prob_cls" in data:
                    pk = torch.softmax(logits_k, dim=-1)
                    pref = data["ref_prob_cls"][i : i + 1].to(device)
                    loss_kl = F.kl_div((pk + 1e-12).log(), pref, reduction="batchmean")
                    total = total + beta_kl * loss_kl

            loss_ce = loss_ce / max(1, B)
            total = total + loss_ce
            log["loss_ce"] = float(loss_ce.detach())

        # ========= 2) PREF: pairwise implicit PRM CE =========
        loss_pref = torch.tensor(0.0, device=device)
        if "pref" in batch:
            data = batch["pref"]
            pos_ids = data["pos_input_ids"].to(device)
            pos_attn = data["pos_attention_mask"].to(device)
            pos_rm = data["pos_response_mask"].to(device)

            neg_ids = data["neg_input_ids"].to(device)
            neg_attn = data["neg_attention_mask"].to(device)
            neg_rm = data["neg_response_mask"].to(device)
            types = data["types"]

            B = pos_ids.size(0)
            for i in range(B):
                k = types[i]
                beta_k = float(self.cfg["types"][k].get("beta", 1.0))

                _, r_pos_seq = self.model.token_implicit_reward(
                    pos_ids[i : i + 1],
                    pos_attn[i : i + 1],
                    pos_rm[i : i + 1],
                    beta=beta_k,
                )
                _, r_neg_seq = self.model.token_implicit_reward(
                    neg_ids[i : i + 1],
                    neg_attn[i : i + 1],
                    neg_rm[i : i + 1],
                    beta=beta_k,
                )

                two = torch.stack([r_pos_seq, r_neg_seq], dim=-1)  # [1,2], logit for pos/neg
                target = torch.zeros(1, dtype=torch.long, device=device)

                alpha = float(self.cfg["types"][k].get("alpha", 1.0))
                loss_pref = loss_pref + alpha * F.cross_entropy(two, target)

                if beta_kl > 0 and "ref_prob_pref" in data:
                    pk = torch.softmax(two, dim=-1)
                    pref = data["ref_prob_pref"][i : i + 1].to(device)
                    loss_kl = F.kl_div((pk + 1e-12).log(), pref, reduction="batchmean")
                    total = total + beta_kl * loss_kl

            loss_pref = loss_pref / max(1, B)
            total = total + loss_pref
            log["loss_pref"] = float(loss_pref.detach())

        # ========= 3) REG: implicit PRM regression on r_seq =========
        loss_mse = torch.tensor(0.0, device=device)
        if "reg" in batch:
            data = batch["reg"]
            ids = data["input_ids"].to(device)
            attn = data["attention_mask"].to(device)
            resp_mask = data["response_mask"].to(device)
            r_true = data["r_true"].to(device)
            types = data["types"]

            B = ids.size(0)
            for i in range(B):
                k = types[i]
                beta_k = float(self.cfg["types"][k].get("beta", 1.0))

                _, r_seq = self.model.token_implicit_reward(
                    ids[i : i + 1],
                    attn[i : i + 1],
                    resp_mask[i : i + 1],
                    beta=beta_k,
                )

                mu = float(self.cfg["types"][k].get("mu", 1.0))
                loss_mse = loss_mse + mu * F.mse_loss(r_seq, r_true[i : i + 1])

                if beta_kl > 0 and "ref_prob_reg" in data:
                    # example: KL on type-specific logits if such labels exist
                    emb = self.model.encode(ids[i : i + 1], attn[i : i + 1])
                    logits_k = self.model.logits(emb, k)
                    pk = torch.softmax(logits_k, dim=-1)
                    pref = data["ref_prob_reg"][i : i + 1].to(device)
                    loss_kl = F.kl_div((pk + 1e-12).log(), pref, reduction="batchmean")
                    total = total + beta_kl * loss_kl

            loss_mse = loss_mse / max(1, B)
            total = total + loss_mse
            log["loss_mse"] = float(loss_mse.detach())

        return total, log

    # ------------------------------------------------------------------
    def _compute_lm_loss(self, batch: Dict[str, Dict]) -> torch.Tensor:
        """
        LM loss to preserve generation ability.

        在 cls / pref / reg 三类样本上都施加标准 LM loss：
          - 确保整个训练始终围绕“语言建模”这个根本目标；
          - 真正控制 LM 更新强度的手段是 lr_lm 和 lambda_lm。
        """
        device = self.device
        total_lm = torch.tensor(0.0, device=device)
        n_parts = 0

        # cls
        if "cls" in batch:
            data = batch["cls"]
            ids = data["input_ids"].to(device)
            attn = data["attention_mask"].to(device)
            labels = ids.clone()
            labels[attn == 0] = -100
            out_cls = self.model.lm(input_ids=ids, attention_mask=attn, labels=labels)
            total_lm = total_lm + out_cls.loss
            n_parts += 1

        # pref: pos + neg
        if "pref" in batch:
            data = batch["pref"]

            ids = data["pos_input_ids"].to(device)
            attn = data["pos_attention_mask"].to(device)
            labels = ids.clone()
            labels[attn == 0] = -100
            out_pos = self.model.lm(input_ids=ids, attention_mask=attn, labels=labels)
            total_lm = total_lm + out_pos.loss
            n_parts += 1

            ids = data["neg_input_ids"].to(device)
            attn = data["neg_attention_mask"].to(device)
            labels = ids.clone()
            labels[attn == 0] = -100
            out_neg = self.model.lm(input_ids=ids, attention_mask=attn, labels=labels)
            total_lm = total_lm + out_neg.loss
            n_parts += 1

        # reg
        if "reg" in batch:
            data = batch["reg"]
            ids = data["input_ids"].to(device)
            attn = data["attention_mask"].to(device)
            labels = ids.clone()
            labels[attn == 0] = -100
            out_reg = self.model.lm(input_ids=ids, attention_mask=attn, labels=labels)
            total_lm = total_lm + out_reg.loss
            n_parts += 1

        if n_parts == 0:
            return torch.tensor(0.0, device=device)
        return total_lm / n_parts

    # ------------------------------------------------------------------
    def train(self):
        steps = self.cfg["train"]["steps"]
        log_every = self.cfg["train"]["log_every"]
        save_every = self.cfg["train"]["save_every"]
        out_dir = self.cfg["train"]["out_dir"]
        l2_lambda = float(self.cfg["train"].get("l2_lambda", 0.0))
        lambda_lm = float(self.cfg["train"].get("lambda_lm", 1.0))

        self.model.train()
        step = 0

        while step < steps:
            batch = self._next_batch()
            step += 1

            # multi-type implicit PRM losses
            loss_rm, log = self._compute_multirm_losses(batch)

            # LM loss
            loss_lm = self._compute_lm_loss(batch)
            if lambda_lm > 0:
                loss = loss_rm + lambda_lm * loss_lm
            else:
                loss = loss_rm
            log["loss_lm"] = float(loss_lm.detach())
            log["loss_total"] = float(loss.detach())

            # explicit L2 regularization
            if l2_lambda > 0:
                l2 = torch.tensor(0.0, device=self.device)
                for p in self.model.parameters():
                    if p.requires_grad:
                        l2 = l2 + (p ** 2).sum()
                loss = loss + l2_lambda * l2
                log["loss_l2"] = float((l2_lambda * l2).detach())

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
