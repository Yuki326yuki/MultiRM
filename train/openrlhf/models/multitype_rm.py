# train/openrlhf/models/multitype_rm.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class MultiTypeRewardModel(nn.Module):
    """
    多类型 Reward Model（无 LoRA，纯全参）:
      - base: HF AutoModel (你的 ImplicitPRM 编码器主干)
      - 每个类型 k：
          heads[k]:       Linear(D -> m_k)      softmax 分布 p^{(k)}
          reward_proj[k]: Linear(D -> 1)        标量奖励 r^{(k)}
      - 温度 tau_k: logits / tau_k
    """

    def __init__(
        self,
        model_name_or_path: str,
        type_specs: dict,
        use_flash_attn_2: bool = True,
    ):
        super().__init__()
        self.type_specs = type_specs
        self.types = sorted(type_specs.keys())

        # 1) 编码器：复用 ImplicitPRM 的 HF 主干
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if use_flash_attn_2:
            cfg.attn_implementation = "flash_attention_2"
        self.base = AutoModel.from_pretrained(
            model_name_or_path,
            config=cfg,
            trust_remote_code=True,
        )

        hidden_size = getattr(self.base.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.base.config, "hidden_sizes"):
            hidden_size = self.base.config.hidden_sizes[-1]
        assert hidden_size is not None, "Cannot infer hidden_size from base model config"

        # 2) 每类型独立 softmax 头 + 标量投影
        self.heads = nn.ModuleDict()
        self.reward_proj = nn.ModuleDict()
        for k in self.types:
            m = int(type_specs[k]["m"])
            self.heads[k] = nn.Linear(hidden_size, m, bias=True)
            self.reward_proj[k] = nn.Linear(hidden_size, 1, bias=True)
        

        # 3) 简单池化（默认最后一个非 PAD token）
        self.pooling = "last_token"

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        输入 (prompt+response) token 序列，输出 [B, D] 表征 h
        """
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [B, T, D]

        if self.pooling == "last_token":
            idx = attention_mask.long().sum(dim=1) - 1
            idx = torch.clamp(idx, min=0)
            emb = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), idx]
        else:
            emb = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )
        return emb  # [B, D]

    def logits(self, emb: torch.Tensor, k: str) -> torch.Tensor:
        tau = float(self.type_specs[k].get("tau", 1.0))
        return self.heads[k](emb) / tau

    def probs(self, emb: torch.Tensor, k: str) -> torch.Tensor:
        return F.softmax(self.logits(emb, k), dim=-1)

    def reward_scalar(self, emb: torch.Tensor, k: str) -> torch.Tensor:
        return self.reward_proj[k](emb).squeeze(-1)
