"""The frozen-feature control of the ablations, with the verifier's training interface: module(backbone, captures,
positions) -> logits [B, P, len(HEAD_OUTPUTS)], plus `first`, `lora` (empty) and `head`. train_verifier trains it
under exactly ToRA's recipe (data order, labels, optimizer, schedule, val selection), sharing the generator pass.

The probe is a heads.Head (MLP, per-tap RMS norm) on the generator's residual streams entering SingProbe's three tap
layers, read at the label positions. Taps are layer indices l: the residual stream entering layer l
(= captures[l].h_in = hidden_states[l]).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tora.models.heads import build_head
from tora.models.qwen3_min import LayerCapture, Qwen3Config, Qwen3Min
from tora.models.verifier import gather_positions
from tora.utils.config import ProbeConfig


class GeneratorProbe(nn.Module):
    def __init__(self, cfg: Qwen3Config, pcfg: ProbeConfig):
        super().__init__()
        taps = pcfg.head.taps
        if not all(t.isdigit() and int(t) < cfg.num_layers for t in taps):
            raise ValueError(f"probe taps must be layer indices < {cfg.num_layers} (residual stream entering the layer), got {taps}")
        if pcfg.head.norm == "standardize":
            raise NotImplementedError("probe heads support norm none / rms (standardize needs fitted statistics)")
        self.taps = [int(t) for t in taps]
        self.first = min(self.taps)
        self.lora = nn.ModuleList()  # nothing adapted: only the probe trains
        self.head = build_head(pcfg.head, cfg.hidden_size)

    def forward(self, backbone: Qwen3Min, captures: dict[int, LayerCapture], positions: Tensor | None = None) -> Tensor:
        x = torch.cat([captures[t].h_in for t in self.taps], dim=-1)
        return self.head(x if positions is None else gather_positions(x, positions))
