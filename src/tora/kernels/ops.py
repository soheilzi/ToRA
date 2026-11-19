"""The dual-row kernels as torch custom ops (tora::dual_linear, tora::dual_lora_ffn), so that torch.compile treats
them as opaque kernels: their Python launch code reads the autotuner's state, which dynamo cannot trace. Eager callers
go through the same ops.
"""

from typing import Optional

import torch
from torch import Tensor

from tora.kernels.dual_lora import dual_lora_ffn, dual_lora_linear


@torch.library.custom_op("tora::dual_linear", mutates_args=())
def dual_linear(x: Tensor, w: Tensor) -> Tensor:
    """K1 without LoRA: x [M, d_in] -> [M, d_out]."""
    return dual_lora_linear(x, w)


@dual_linear.register_fake
def _(x: Tensor, w: Tensor) -> Tensor:
    return x.new_empty(x.shape[0], w.shape[0])


@torch.library.custom_op("tora::dual_lora_ffn", mutates_args=())
def lora_ffn(
    x: Tensor, w_gate: Tensor, w_up: Tensor, w_down: Tensor, a_gate: Optional[Tensor], b_gate: Optional[Tensor],
    a_up: Optional[Tensor], b_up: Optional[Tensor], a_down: Optional[Tensor], b_down: Optional[Tensor], scale: float,
) -> Tensor:
    """K2 + K1: the FFN of x [M, hidden]; the LoRA factors are all given or all None."""
    loras = (None,) * 3 if a_gate is None else ((a_gate, b_gate), (a_up, b_up), (a_down, b_down))
    return dual_lora_ffn(x, w_gate, w_up, w_down, *loras, scale)


@lora_ffn.register_fake
def _(x, w_gate, w_up, w_down, a_gate, b_gate, a_up, b_up, a_down, b_down, scale):
    return x.new_empty(x.shape[0], w_down.shape[0])
