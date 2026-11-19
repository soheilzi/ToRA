"""PyTorch references for the dual-row kernels (kernels/dual_lora.py), with the op order of the trained model
(models/lora_ffn.py): each projection's base output is rounded to the model dtype, the LoRA term is computed and added
in fp32, then rounded again; SiLU and the gate * up product run in the model dtype as in qwen3_min's MLP.

Rows: x [M, d_in] with M = 1 (the generator alone) or 2 (row 0 the generator's, row 1 the verifier's). LoRA applies to
row 1 only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dual_lora_linear_ref(x: Tensor, w: Tensor, lora_a: Tensor | None = None, lora_b: Tensor | None = None, scale: float = 1.0) -> Tensor:
    """x [M, d_in], w [d_out, d_in], lora_a [r, d_in], lora_b [d_out, r] (fp32) -> [M, d_out]."""
    y = F.linear(x, w)
    if x.shape[0] == 2 and lora_a is not None:
        y1 = (y[1].float() + scale * F.linear(F.linear(x[1].float(), lora_a), lora_b)).to(y.dtype)
        y = torch.stack([y[0], y1])
    return y


def dual_lora_gate_up_ref(
    x: Tensor, w_gate: Tensor, w_up: Tensor, gate_ab: tuple[Tensor, Tensor] | None = None, up_ab: tuple[Tensor, Tensor] | None = None,
    scale: float = 1.0,
) -> Tensor:
    """silu(gate(x)) * up(x) for both rows, LoRA on row 1's gate and up -> [M, d_ff]."""
    return F.silu(dual_lora_linear_ref(x, w_gate, *(gate_ab or (None, None)), scale)) * dual_lora_linear_ref(x, w_up, *(up_ab or (None, None)), scale)
