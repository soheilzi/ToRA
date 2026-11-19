"""FFN-LoRA for the verifier row: y = W x + (alpha / r) * B(A x) on gate_proj, up_proj and down_proj.

The adapters hold no base weights; they are applied to a frozen qwen3_min MLP passed in at call time,
so one backbone serves the generator row (plain MLP) and the verifier row (MLP + LoRA). Only verifier rows
ever go through LoRAFFN; in a dual-row batch the caller routes rows (decode_ref computes them separately).

Numerics: the base projection runs in the model dtype exactly as in the generator; the LoRA term is
computed in fp32 and added in fp32 before rounding back. With B = 0 the result is therefore bitwise the
generator's MLP output (S3).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tora.models.qwen3_min import MLP, Qwen3Config


class LoRA(nn.Module):
    """delta(x) = scale * B(A x) in fp32. A: kaiming-uniform (as PEFT), B: zeros, so training starts at the base."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * F.linear(F.linear(x.float(), self.A), self.B)


def lora_linear(base: nn.Linear, lora: LoRA, x: Tensor) -> Tensor:
    y = base(x)
    return (y.float() + lora(x)).to(y.dtype)


class LoRAFFN(nn.Module):
    def __init__(self, cfg: Qwen3Config, rank: int, alpha: float):
        super().__init__()
        self.gate = LoRA(cfg.hidden_size, cfg.intermediate_size, rank, alpha)
        self.up = LoRA(cfg.hidden_size, cfg.intermediate_size, rank, alpha)
        self.down = LoRA(cfg.intermediate_size, cfg.hidden_size, rank, alpha)

    def forward(self, mlp: MLP, x: Tensor) -> Tensor:
        """Same op order as MLP.forward, with LoRA on each projection."""
        h = F.silu(lora_linear(mlp.gate_proj, self.gate, x)) * lora_linear(mlp.up_proj, self.up, x)
        return lora_linear(mlp.down_proj, self.down, h)
