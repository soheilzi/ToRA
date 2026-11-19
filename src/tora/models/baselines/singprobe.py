"""SingProbe off the shelf (inclusionAI/Qwen3-0.6B-singprobe), reimplemented on qwen3_min.

Features: for each tapped layer l (here 8, 17, 26), the residual stream entering layer l -- our
hidden_states[l] -- RMS-normalized WITHOUT a weight in fp32 (eps 1e-6), cast to the model dtype, and
concatenated in tap order. (SGLang token-probe `tap_into` kernel: rmsnorm(hidden_states + residual) at the
layer input.) Probe: causal multi-query attention over the token sequence, query-residual, RMSNorm,
linear classifier (SingProbeAttnModel in the checkpoint's modeling_sing_probe.py). Only the
safetensors are loaded; the remote code is not executed.

Outputs (10 logits per token): 0-6 query risk classes A-G (multi-label), 7 query safe (H),
8 response unsafe, 9 hallucination. Probabilities are sigmoids; the report's decision threshold is 0.5.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

from tora.models.qwen3_min import Qwen3Min
from tora.utils.hf import download_model

QUERY_RISK = slice(0, 7)
QUERY_SAFE, RESPONSE_UNSAFE, HALLUCINATION = 7, 8, 9


def tap_norm(h: Tensor, eps: float = 1e-6) -> Tensor:
    x = h.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)).to(h.dtype)


class SingProbeAttn(nn.Module):
    def __init__(self, in_dim: int, num_heads: int, head_dim: int, num_labels: int, window: int | None):
        super().__init__()
        self.num_heads, self.head_dim, self.window = num_heads, head_dim, window
        q_dim = num_heads * head_dim
        self.proj_q = nn.Linear(in_dim, q_dim, bias=False)
        self.proj_k = nn.Linear(in_dim, head_dim, bias=False)
        self.proj_v = nn.Linear(in_dim, head_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.norm = nn.RMSNorm(q_dim, eps=1e-6)
        self.classifier = nn.Linear(q_dim, num_labels)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, in_dim] -> logits [B, T, num_labels]."""
        b, t, _ = x.shape
        q_feat = self.proj_q(x)
        q = q_feat.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(x).view(b, t, 1, self.head_dim).transpose(1, 2)
        v = self.proj_v(x).view(b, t, 1, self.head_dim).transpose(1, 2)
        if self.window and self.window < t:
            pos = torch.arange(t, device=x.device)
            rel = pos[:, None] - pos[None, :]
            mask = (rel >= 0) & (rel < self.window)
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self.num_heads > 1)
        else:
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.num_heads > 1)
        ctx = ctx.transpose(1, 2).reshape(b, t, -1)
        return self.classifier(self.norm(self.o_proj(ctx) + q_feat))


class SingProbe(nn.Module):
    def __init__(self, backbone: Qwen3Min, probe: SingProbeAttn, taps: list[int]):
        super().__init__()
        self.backbone, self.probe, self.taps = backbone, probe, taps

    @classmethod
    def from_pretrained(
        cls, probe_repo: str, *, base_repo: str | None = None, device="cuda", dtype=torch.bfloat16, max_seq_len: int = 4096, hf_home=None
    ) -> SingProbe:
        path = download_model(probe_repo, hf_home=hf_home)
        raw = json.loads((Path(path) / "config.json").read_text())
        if raw.get("model_type") != "sing_probe_attn":
            raise NotImplementedError(f"{probe_repo}: model_type={raw.get('model_type')!r} (only sing_probe_attn)")
        base_repo = base_repo or raw["base_model_name"]
        if base_repo != raw["base_model_name"]:
            raise ValueError(f"{probe_repo} was trained on {raw['base_model_name']}, not {base_repo}")
        backbone = Qwen3Min.from_pretrained(base_repo, device=device, dtype=dtype, max_seq_len=max_seq_len, hf_home=hf_home, verbose=False)
        if raw["hidden_size"] != backbone.cfg.hidden_size:
            raise ValueError("probe hidden_size does not match the base model")
        taps = list(raw["base_model_layer_ids"])
        probe = SingProbeAttn(raw["hidden_size"] * len(taps), raw["num_attention_heads"], raw["head_dim"], raw["num_labels"], raw.get("sliding_window"))
        probe.load_state_dict(load_file(Path(path) / "model.safetensors"), strict=True)
        return cls(backbone, probe.to(device, dtype), taps).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, input_ids: Tensor) -> Tensor:
        """-> per-token logits [B, T, 10]. Right padding is safe: the probe attention is causal too."""
        hs = self.backbone(input_ids, logits="none", output_hidden_states=True).hidden_states
        return self.probe(torch.cat([tap_norm(hs[i]) for i in self.taps], dim=-1))
