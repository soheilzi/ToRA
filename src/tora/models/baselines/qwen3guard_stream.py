"""Qwen3Guard-Stream (baseline A6), reimplemented on qwen3_min.

The guard is a full fine-tuned Qwen3 whose final normed hidden state feeds two small heads (response
risk and query risk), each Linear(hidden -> inner) -> RMSNorm -> Linear(inner -> classes). Only the
checkpoint's safetensors are loaded; the repo's remote code is not executed. Reimplemented from
Qwen3ForGuardModel in the checkpoint's modeling_qwen3_guard.py and QwenLM/Qwen3Guard eval/eval_stream.py.

Risk levels: 0 Safe, 1 Unsafe, 2 Controversial. "strict" counts Controversial as unsafe, "loose" does not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tora.models.qwen3_min import Qwen3Min, RMSNorm

SAFE, UNSAFE, CONTROVERSIAL = 0, 1, 2


class GuardHead(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int, n_risk: int, n_category: int, eps: float):
        super().__init__()
        self.pre = nn.Linear(hidden_size, inner_size, bias=False)
        self.norm = RMSNorm(inner_size, eps)
        self.risk = nn.Linear(inner_size, n_risk, bias=False)
        self.category = nn.Linear(inner_size, n_category, bias=False)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        x = self.norm(self.pre(h))
        return self.risk(x), self.category(x)


# checkpoint key -> (head, parameter)
_HEAD_KEYS = {
    "risk_level_category_pre.weight": ("response", "pre.weight"),
    "risk_level_category_layernorm.weight": ("response", "norm.weight"),
    "risk_level_head.weight": ("response", "risk.weight"),
    "category_head.weight": ("response", "category.weight"),
    "query_risk_level_category_pre.weight": ("query", "pre.weight"),
    "query_risk_level_category_layernorm.weight": ("query", "norm.weight"),
    "query_risk_level_head.weight": ("query", "risk.weight"),
    "query_category_head.weight": ("query", "category.weight"),
}


@dataclass
class GuardOutput:
    response_risk_logits: Tensor  # [B, T, 3]
    query_risk_logits: Tensor  # [B, T, 3]


class Qwen3GuardStream(nn.Module):
    def __init__(self, backbone: Qwen3Min, response_head: GuardHead, query_head: GuardHead):
        super().__init__()
        self.backbone = backbone
        self.response_head = response_head
        self.query_head = query_head

    @classmethod
    def from_pretrained(cls, repo_id: str, *, device="cuda", dtype=torch.bfloat16, max_seq_len: int = 4096, hf_home=None) -> Qwen3GuardStream:
        backbone, extras = Qwen3Min.from_pretrained_with_extras(
            repo_id, device=device, dtype=dtype, max_seq_len=max_seq_len, hf_home=hf_home, verbose=False
        )
        raw = json.loads((backbone.source_path / "config.json").read_text())
        expected = {"Safe": SAFE, "Unsafe": UNSAFE, "Controversial": CONTROVERSIAL}
        for key in ("response_risk_level_map", "query_risk_level_map"):
            if {v: int(k) for k, v in raw[key].items()} != expected:
                raise ValueError(f"{repo_id}: {key}={raw[key]} differs from the assumed {expected}")
        h, inner, eps = raw["hidden_size"], raw["guard_inner_size"], raw["rms_norm_eps"]
        heads = {
            "response": GuardHead(h, inner, raw["num_risk_level"], raw["num_category"], eps),
            "query": GuardHead(h, inner, raw["num_query_risk_level"], raw["num_query_category"], eps),
        }
        if set(extras) != set(_HEAD_KEYS):
            raise ValueError(f"{repo_id}: unexpected head tensors {sorted(set(extras) ^ set(_HEAD_KEYS))}")
        for name, head in heads.items():
            head.load_state_dict({param: extras[k] for k, (which, param) in _HEAD_KEYS.items() if which == name}, strict=True)
        model = cls(backbone, heads["response"].to(device, dtype), heads["query"].to(device, dtype))
        return model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, input_ids: Tensor) -> GuardOutput:
        h = self.backbone(input_ids, logits="none").final_hidden
        return GuardOutput(self.response_head(h)[0], self.query_head(h)[0])


def official_response_decision(risk_argmax: list[int]) -> tuple[int, int]:
    """eval_stream.py's consecutive rule over response tokens -> (risk level, flag index or -1).

    Unsafe if two consecutive tokens are Unsafe; else Controversial if two consecutive are Controversial.
    (A mixed Unsafe/Controversial pair does not count.)
    """
    for level in (UNSAFE, CONTROVERSIAL):
        for i in range(len(risk_argmax) - 1):
            if risk_argmax[i] == level and risk_argmax[i + 1] == level:
                return level, i + 1
    return SAFE, -1


def first_unsafe_token(risk_argmax: list[int]) -> tuple[int, int]:
    """Where a stream stopped by the released rule (strict: Controversial counts too) first stops, read in token order
    -> (index of the first token of the earliest two-in-a-row Unsafe or Controversial pair, its level), or (-1, SAFE).
    The onset of the teacher labels for ToRA's per-token training (scripts/teacher_onsets.py)."""
    for i in range(len(risk_argmax) - 1):
        if risk_argmax[i] != SAFE and risk_argmax[i] == risk_argmax[i + 1]:
            return i, risk_argmax[i]
    return -1, SAFE
