# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


class MultiTypeRewardModel(nn.Module):
    """
    多类型 Reward Model：
      - base 编码器：HF AutoModel（即你的 ImplicitPRM 主干；保持原样加载）
      - 每类型独立 softmax 头：heads[type]: Linear(D -> m_type)
      - 每类型标量奖励投影：reward_proj[type]: Linear(D -> 1)（回归MSE用）
      - 温度 τ_type：logits / tau
    """
    def __init__(
        self,
        model_name_or_path: str,
        type_specs: dict,
        use_flash_attn_2: bool = True,
        lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.type_specs = type_specs
        self.types = sorted(type_specs.keys())

        # 1) 编码器（ImplicitPRM主干）：从HF加载，与现有管线一致
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if use_flash_attn_2:
            cfg.attn_implementation = "flash_attention_2"
        self.base = AutoModel.from_pretrained(model_name_or_path, config=cfg, trust_remote_code=True)

        hidden_size = getattr(self.base.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.base.config, "hidden_sizes"):
            hidden_size = self.base.config.hidden_sizes[-1]
        assert hidden_size is not None, "Cannot infer hidden_size from base model config."

        # 2) LoRA（可选）
        if lora and PEFT_AVAILABLE:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            peft_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=target_modules, task_type="CAUSAL_LM"
            )
            self.base = get_peft_model(self.base, peft_cfg)

        # 3) 多类型独立头 + 标量投影
        self.heads = nn.ModuleDict()
        self.reward_proj = nn.ModuleDict()
        for k in self.types:
            m = int(type_specs[k]["m"])
            self.heads[k] = nn.Linear(hidden_size, m, bias=True)
            self.reward_proj[k] = nn.Linear(hidden_size, 1, bias=True)

        # 4) 简单池化策略
        self.pooling = "last_token"  # 可改为 "mean"

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        将 (prompt + response) 拼接后的序列编码为 [B, D] 表征
        """
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [B, T, D]
        if self.pooling == "last_token":
            idx = attention_mask.long().sum(dim=1) - 1
            idx = torch.clamp(idx, min=0)
            emb = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), idx]
        else:
            emb = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return emb  # [B, D]

    def logits(self, emb: torch.Tensor, k: str) -> torch.Tensor:
        tau = float(self.type_specs[k].get("tau", 1.0))
        return self.heads[k](emb) / tau  # [B, m_k]

    def probs(self, emb: torch.Tensor, k: str) -> torch.Tensor:
        return F.softmax(self.logits(emb, k), dim=-1)  # 分类和偏好

    def reward_scalar(self, emb: torch.Tensor, k: str) -> torch.Tensor:
        return self.reward_proj[k](emb).squeeze(-1)  # [B] 回归
