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

        # 1) 主模型 lm，用于生成和打分
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
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
            self.ref_lm.eval()

        # 3) 每类型独立 softmax 头 + （可选）标量投影
        hidden_size = getattr(self.lm.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.lm.config, "hidden_sizes"):
            hidden_size = self.lm.config.hidden_sizes[-1]
        assert hidden_size is not None, "Cannot infer hidden_size from lm model config"

        # 3) 每类型独立 softmax 头 + （可选）标量投影
        self.heads = nn.ModuleDict()
        self.reward_proj = nn.ModuleDict()
        # 基于序列隐式奖励 r_seq 的多类分类 head
        self.cls_heads_from_reward = nn.ModuleDict()

        for k in self.types:
            m = int(type_specs[k]["m"])
            # 句向量上的多类 softmax（如需直接用 emb 分类）
            self.heads[k] = nn.Linear(hidden_size, m, bias=True)
            # 句向量上的标量回归（可选）
            self.reward_proj[k] = nn.Linear(hidden_size, 1, bias=True)
            # 序列隐式奖励 r_seq: [B] → [B, m_k] 的多类分类头
            self.cls_heads_from_reward[k] = nn.Linear(1, m, bias=True)
        # 4) 句向量池化策略
        self.pooling = "last_token"

    # ========== 句向量编码 (用于多类型 CE / 分类 / 辅助 head) ==========

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        返回句向量 h(x,y)，目前采用 last_token pooling。
        """
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1]  # [B, T, D]

        if self.pooling == "last_token":
            # 取每行最后一个非 padding token 的 hidden state
            lengths = attention_mask.sum(dim=-1)  # [B]
            idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))  # [B,1,D]
            emb = last_hidden.gather(dim=1, index=idx).squeeze(1)  # [B, D]
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

    def cls_logits_from_reward(self, r_seq: torch.Tensor, k: str) -> torch.Tensor:
        """基于序列隐式奖励 r_seq 的多类分类 head:
        r_seq: [B] → logits: [B, m_k]
        """
        tau = float(self.type_specs[k].get("tau", 1.0))
        # 先扩展到 [B, 1] 再线性映射到 [B, m_k]
        return self.cls_heads_from_reward[k](r_seq.unsqueeze(-1)) / tau

    # ========== token-level implicit reward ==========

    @torch.no_grad()
    def _ref_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        计算参考模型 ref_lm 的 token-level log P_ref(y_t | x, y_{<t})
        返回形状为 [B, T] 的 logprobs（只保留对应 input_ids 的位置）。
        """
        out = self.ref_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = out.logits  # [B, T, V]
        log_probs = torch.log_softmax(logits, dim=-1)
        # 取对应 token 的 logprob
        token_logp = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
        # padding 位置置 0
        token_logp = token_logp * attention_mask
        return token_logp

    def _lm_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        计算当前 lm 的 token-level log P_lm(y_t | x, y_{<t})
        返回形状为 [B, T] 的 logprobs。
        """
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = out.logits  # [B, T, V]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logp = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
        token_logp = token_logp * attention_mask
        return token_logp

    def token_implicit_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        beta: float = 1.0,
    ):
        """
        计算 token-level 隐式奖励 r_t 以及序列级奖励 r_seq。

        输入:
          - input_ids:    [B, T]
          - attention_mask: [B, T]
          - response_mask:  [B, T]  (prompt 之后到最后 token 为 1，其他为 0)

        输出:
          - r_tokens: [B, T]，在 response 位置为非零，其余为 0
          - r_seq:    [B]，为 response 区间 r_t 的平均（或和 / 归一化）
        """
        # 1) lm & ref_lm 的 token-level logprobs
        with torch.no_grad():
            logp_ref = self._ref_logprobs(input_ids, attention_mask)  # [B, T]

        logp_lm = self._lm_logprobs(input_ids, attention_mask)  # [B, T]

        # 2) log-ratio 作为原始隐式 reward signal
        #    r_t = (log p_lm - log p_ref) * response_mask
        r_tokens = (logp_lm - logp_ref) * response_mask  # [B, T]

        # 3) 归一化 / 平均为序列级奖励
        mask = (response_mask > 0).float()
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

