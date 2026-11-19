"""Classification heads shared by the frozen-feature probe and the verifiers (ToRA, own KV).

A head maps a feature vector to logits for HEAD_OUTPUTS. Features are a concatenation of taps:
"final" is the final-normed hidden state (the verifiers), an integer i the residual stream entering
layer i (the probe; see qwen3_min's hidden-state convention). The raw residual stream grows ~25x in
RMS from layer 8 to layer 26 and a few dimensions are hundreds of times the median, so multi-tap
heads normalize their input (FeatureNorm).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tora.utils.config import HeadConfig

HEAD_OUTPUTS = ("prompt_harm", "response_harm")


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_out: int = len(HEAD_OUTPUTS)):
        super().__init__()
        self.proj = nn.Linear(in_dim, n_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_out: int = len(HEAD_OUTPUTS), dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FeatureNorm(nn.Module):
    """Input normalization in front of a head; features are `n_taps` taps of `hidden` dims, concatenated.

    none         identity
    rms          each tap divided by its RMS, no weight (SingProbe's tap_norm)
    standardize  per-feature (x - mean) / std with mean / std from `fit` on training features. They start
                 as NaN, so a head used before fit() outputs NaN instead of silently skipping the norm.
    """

    def __init__(self, kind: str, n_taps: int, hidden: int, eps: float = 1e-6):
        super().__init__()
        if kind not in ("none", "rms", "standardize"):
            raise ValueError(f"head norm {kind!r}: expected none, rms or standardize")
        self.kind, self.n_taps, self.hidden, self.eps = kind, n_taps, hidden, eps
        init = float("nan") if kind == "standardize" else 0.0
        self.register_buffer("mean", torch.full((n_taps * hidden,), init))
        self.register_buffer("std", torch.full((n_taps * hidden,), init))

    @torch.no_grad()
    def fit(self, x: Tensor, chunk: int = 65536) -> None:
        """Per-feature mean / std over the rows of x [N, F] (fp64 accumulation); a no-op unless standardize."""
        if self.kind != "standardize":
            return
        total = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
        sq = torch.zeros_like(total)
        for start in range(0, x.shape[0], chunk):
            part = x[start : start + chunk].double()
            total += part.sum(0)
            sq += part.pow(2).sum(0)
        mean = total / x.shape[0]
        var = (sq / x.shape[0] - mean.pow(2)).clamp_min(0)
        self.mean.copy_(mean)
        self.std.copy_(var.sqrt().clamp_min(self.eps))

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        if self.kind == "rms":
            t = x.unflatten(-1, (self.n_taps, self.hidden))
            return (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.eps)).flatten(-2)
        if self.kind == "standardize":
            return (x - self.mean) / self.std
        return x


class Head(nn.Module):
    """FeatureNorm -> LinearHead | MLPHead, in fp32. Input: concatenated taps [..., n_taps * hidden]."""

    def __init__(self, norm: FeatureNorm, body: nn.Module):
        super().__init__()
        self.norm, self.body = norm, body

    def forward(self, x: Tensor) -> Tensor:
        return self.body(self.norm(x))


def build_head(cfg: HeadConfig, hidden_size: int) -> Head:
    in_dim = hidden_size * len(cfg.taps)
    if cfg.kind == "linear":
        body: nn.Module = LinearHead(in_dim)
    elif cfg.kind == "mlp":
        body = MLPHead(in_dim, cfg.hidden_dim, dropout=cfg.dropout)
    else:
        raise ValueError(f"head kind {cfg.kind!r}: expected 'linear' or 'mlp'")
    return Head(FeatureNorm(cfg.norm, len(cfg.taps), hidden_size), body)
