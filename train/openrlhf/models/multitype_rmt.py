# train/openrlhf/models/multitype_rm.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


class MultiTypeRewardModel(nn.Module):
    """
    多类型 Implicit PRM + ORM 统一模型：

      - lm:        AutoModelForCausalLM, 既能生成又能提供 token logits
      - ref_lm:    冻结的参考模型，用于 implicit reward 的 logprob ratio
      - 每个类型 k：
          heads[k]:       Linear(D -> m_k)      softmax 分布 p^{(k)}  (分类 / 多类标签)
        （可选）reward_proj[k]: Linear(D -> 1)  若你仍想在句向量上做额外回归可继续用

      - encode(): 基于 hidden_states[-1] 做 pooling，得到句向量 h(x,y)
      - token_implicit_reward(): 计算 token-level + 序列级隐式奖励
    """

    def __init__(
        self,
        model_name_or_path: str,
        type_specs: dict,
        use_flash_attn_2: bool = True,
        ref_model_name_or_path: str = None,
        freeze_ref: bool = True,
    ):
        super().__init__()
        self.type_specs = type_specs
        self.types = sorted(type_specs.keys())

        # 1) Causal LM 主干 (policy / ORM / implicit PRM 同体)
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        # 为了能拿到 hidden_states 来做 encode
        cfg.output_hidden_states = True
        if use_flash_attn_2:
            cfg.attn_implementation = "flash_attention_2"

        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=cfg,
            trust_remote_code=True,
        )

        # 2) 冻结参考模型 ref_lm（可与 lm 同结构同初始化）
        if ref_model_name_or_path is None:
            ref_model_name_or_path = model_name_or_path

        ref_cfg = AutoConfig.from_pretrained(ref_model_name_or_path, trust_remote_code=True)
        ref_cfg.output_hidden_states = True
        if use_flash_attn_2:
            ref_cfg.attn_implementation = "flash_attention_2"

        self.ref_lm = AutoModelForCausalLM.from_pretrained(
            ref_model_name_or_path,
            config=ref_cfg,
            trust_remote_code=True,
        )
        if freeze_ref:
            for p in self.ref_lm.parameters():
                p.requires_grad = False

        hidden_size = getattr(self.lm.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.lm.config, "hidden_sizes"):
            hidden_size = self.lm.config.hidden_sizes[-1]
        assert hidden_size is not None, "Cannot infer hidden_size from lm model config"

        # 3) 每类型独立 softmax 头 + （可选）标量投影
        self.heads = nn.ModuleDict()
        self.reward_proj = nn.ModuleDict()
        for k in self.types:
            m = int(type_specs[k]["m"])
            self.heads[k] = nn.Linear(hidden_size, m, bias=True)
            self.reward_proj[k] = nn.Linear(hidden_size, 1, bias=True)

        # 4) 句向量池化策略
        self.pooling = "last_token"

    # ========== 句向量编码 (用于多类型 CE / 分类 / 辅助 head) ==========

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        输入 (prompt+response) token 序列，输出 [B, D] 表征 h(x,y)
        使用 lm 的 hidden_states[-1]，不影响 logits 用途。
        """
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = out.hidden_states[-1]  # [B, T, D]

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
        """
        多类型 softmax logits，对应你公式里的 s^{(k)}(x,y) / τ_k。
        """
        tau = float(self.type_specs[k].get("tau", 1.0))
        return self.heads[k](emb) / tau

    def probs(self, emb: torch.Tensor, k: str) -> torch.Tensor:
        return F.softmax(self.logits(emb, k), dim=-1)

    def reward_scalar_head(self, emb: torch.Tensor, k: str) -> torch.Tensor:
        """
        若你仍想基于句向量做一个显式标量回归，可以继续使用这个投影。
        在 token-level implicit PRM 版本里，我们更多会用 token_implicit_reward 聚合后的 r_seq。
        """
        return self.reward_proj[k](emb).squeeze(-1)

    # ========== token-level implicit reward ==========

    @torch.no_grad()
    def _ref_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        计算参考模型在真实 token 上的 log p_ref(y_t | x, y_<t)，shape: [B, T]
        （no grad）
        """
        out = self.ref_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = out.logits  # [B, T, V]
        logp = logits.log_softmax(dim=-1)  # [B, T, V]
        # gather 到真实 token 上
        token_logp_ref = logp.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
        return token_logp_ref

    def _lm_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        当前模型在真实 token 上的 log p(y_t | x, y_<t)，shape: [B, T]
        """
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = out.logits  # [B, T, V]
        logp = logits.log_softmax(dim=-1)
        token_logp = logp.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
        return token_logp

    def token_implicit_reward(
        self,
        input_ids: torch.Tensor,       # [B, T]
        attention_mask: torch.Tensor,  # [B, T]
        response_mask: torch.Tensor,   # [B, T], 1=属于 response
        beta: float = 1.0,
    ):
        """
        根据 PRIME / Implicit PRM:
          r_φ(y_t) = β ( log π_φ(y_t | y_<t) - log π_ref(y_t | y_<t) )

        返回:
          - r_tokens: [B, T]，仅在 response_mask==1 且非 pad 上非零
          - r_seq:    [B]，对 response 区间聚合后的标量 reward
        """
        # 当前模型 logp
        token_logp = self._lm_logprobs(input_ids, attention_mask)      # [B, T]
        # 参考模型 logp（无梯度）
        token_logp_ref = self._ref_logprobs(input_ids, attention_mask) # [B, T]

        # implicit token-level reward
        r_tokens = beta * (token_logp - token_logp_ref)  # [B, T]

        # 只在 response 区间聚合
        mask = (attention_mask * response_mask).float()  # [B, T]
        r_tokens = r_tokens * mask

        lengths = mask.sum(dim=-1).clamp(min=1.0)        # 防止除 0
        r_seq = r_tokens.sum(dim=-1) / lengths           # [B]

        return r_tokens, r_seq

    # ========== 生成接口（方便之后直接用这个模型做 generation） ==========

    def generate(self, *args, **kwargs):
        """
        代理到 self.lm.generate，方便直接用这个类生成内容。
        """
        return self.lm.generate(*args, **kwargs)

