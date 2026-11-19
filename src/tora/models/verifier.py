"""The ToRA verifier (shared KV) and the own-KV verifier of the cache ablation: teacher-forced training / scoring mode.

Both read one frozen generator pass (qwen3_min forward with capture_layers = verifier.capture_layers):
  captures[first].h_mid   residual after layer `first`'s attention, first = L - K. The verifier's attention
                          input there is the generator's (LoRA is FFN-only), so it branches at that FFN.
  captures[j].k / .v      the generator's post-RoPE K/V of layer j (shared mode reads j = first+1 .. L-1).

shared (ToRA)  In layers first+1 .. L-1 the verifier query at position t attends the generator's K/V at positions
             s < t plus its own (k, v) at t; its k/v are never visible to any other position (the paper's Eq. 2).
             A verifier row therefore depends only on generator states and its own state at t, so rows can be
             computed at any subset of positions and equal computing every row and picking. Training uses
             only the label positions (`shared_kv_attention`, scores materialized per query). Every position
             at once goes through FlexAttention over KV = concat(generator K/V, verifier K/V) with the
             mask: query t sees generator key s iff s < t, and verifier key s iff s == t.
  + sink S   Used only by the rank ablation (S = 4). The verifier's own K/V replace the generator's at positions
             s < S (attention sinks). Sink rows depend only on sink rows, so label-position training stays cheap:
             the S sink rows run first (causal among themselves), then the label rows read them.
own          Standard LoRA verifier with its own KV (the cache ablation): from layer first+1 on, it computes K/V at
             every position from its own hidden states and attends causally among verifier tokens (needs every position).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from tora.models.heads import build_head
from tora.models.lora_ffn import LoRAFFN
from tora.models.qwen3_min import LayerCapture, Qwen3Config, Qwen3Min
from tora.utils.config import VerifierConfig

# Inductor's default flex tiles for head_dim 128 need 104 KB (bf16) / 112 KB (fp32 at 64x64) of shared
# memory; sm_86 (A6000) has 99 KB.
FLEX_KERNEL_OPTIONS = {
    torch.bfloat16: {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 2},
    torch.float16: {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 2},
    torch.float32: {"BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 1},
}
FLEX_PAD = 128  # sequences are padded to a multiple of the block-mask block (shorter ones fail to lower)
# Flex is compiled with dynamic shapes: training batches vary in size and length, and a static compile recompiles
# for every new batch size (~6 s each). Dynamic shapes still specialize sizes 0 / 1, so inputs are padded to
# batch >= 2 and >= 2 blocks; what remains is 3 graphs (tests/test_verifier.py). Should recompiles
# still pile up, hitting the limit raises instead of silently falling back to eager flex attention (which
# materializes every score).
torch._dynamo.config.fail_on_recompile_limit_hit = True
_compiled_flex = None


def _flex():
    global _compiled_flex
    if _compiled_flex is None:
        _compiled_flex = torch.compile(flex_attention, dynamic=True)
    return _compiled_flex


def shared_kv_attention(
    q: Tensor, k_gen: Tensor, v_gen: Tensor, k_self: Tensor, v_self: Tensor, q_pos: Tensor, scale: float,
    k_sink: Tensor | None = None, v_sink: Tensor | None = None,
) -> Tensor:
    """The shared-KV attention at any subset of positions; fp32 scores and softmax.

    q [B, Hq, P, D] and the verifier's own k_self / v_self [B, Hkv, P, D] are at absolute positions
    q_pos [B, P]; k_gen / v_gen [B, Hkv, T, D] are the generator's. Query p sees generator keys
    s < q_pos[b, p] and its own key. With sinks (the verifier's own K/V at positions 0 .. S-1, [B, Hkv, S, D];
    every q_pos must be >= S) those replace the generator's keys s < S. Returns [B, Hq, P, D] in q's dtype.
    """
    b, hq, p, d = q.shape
    hkv, t = k_gen.shape[1], k_gen.shape[2]
    n_sink = 0 if k_sink is None else k_sink.shape[2]
    qg = q.float().view(b, hkv, hq // hkv, p, d)
    s_gen = torch.einsum("bhgpd,bhtd->bhgpt", qg, k_gen.float()) * scale
    s_idx = torch.arange(t, device=q.device)
    visible = (s_idx < q_pos[..., None]) & (s_idx >= n_sink)  # [B, P, T]
    s_gen = s_gen.masked_fill(~visible[:, None, None], float("-inf"))
    s_self = (qg * k_self.float()[:, :, None]).sum(-1, keepdim=True) * scale
    parts = [s_gen, s_self]
    if n_sink:
        parts.append(torch.einsum("bhgpd,bhsd->bhgps", qg, k_sink.float()) * scale)
    probs = torch.cat(parts, dim=-1).softmax(-1)
    out = torch.einsum("bhgpt,bhtd->bhgpd", probs[..., :t], v_gen.float()) + probs[..., t : t + 1] * v_self.float()[:, :, None]
    if n_sink:
        out = out + torch.einsum("bhgps,bhsd->bhgpd", probs[..., t + 1 :], v_sink.float())
    return out.reshape(b, hq, p, d).to(q.dtype)


def flex_length(t: int) -> int:
    return max(2, -(-t // FLEX_PAD)) * FLEX_PAD


def shared_kv_attention_all(
    q: Tensor, k_gen: Tensor, v_gen: Tensor, k_self: Tensor, v_self: Tensor, scale: float, sink: int = 0,
) -> Tensor:
    """Every position 0..T-1 (q, k_self, v_self: [B, H, T, D]) with compiled FlexAttention: query t sees
    generator key s iff sink <= s < t, and verifier key s iff s == t or s < sink.

    Zero padding (to flex_length, and a second batch row when B = 1) is invisible to real queries: padded generator
    keys sit at s >= T > t - 1, padded verifier keys at s > t, and batch rows never mix.
    """
    b, t = q.shape[0], q.shape[2]
    tp = flex_length(t)
    q, k_gen, v_gen, k_self, v_self = (F.pad(x, (0, 0, 0, tp - t, 0, 0, 0, 1 if b == 1 else 0)) for x in (q, k_gen, v_gen, k_self, v_self))

    def mask_mod(b, h, q_idx, kv_idx):
        own = kv_idx - tp
        gen = (kv_idx < tp) & (kv_idx >= sink) & (kv_idx <= q_idx - 1)
        return gen | ((own >= 0) & (own <= q_idx) & ((own < sink) | (own > q_idx - 1)))

    block_mask = create_block_mask(mask_mod, None, None, tp, 2 * tp, device=q.device)
    k, v = torch.cat([k_gen, k_self], dim=2), torch.cat([v_gen, v_self], dim=2)
    out = _flex()(q, k, v, block_mask=block_mask, scale=scale, enable_gqa=True, kernel_options=FLEX_KERNEL_OPTIONS[q.dtype])
    return out[:b, :, :t]


def gather_positions(x: Tensor, positions: Tensor) -> Tensor:
    """x [B, T, ...], positions [B, P] -> [B, P, ...]."""
    index = positions.view(*positions.shape, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:])
    return x.gather(1, index)


class Verifier(nn.Module):
    """FFN-LoRA on the last K layers + a head on the verifier's final hidden state. Holds no backbone weights:
    the frozen backbone is passed to every call, so the state dict is just the LoRA and the head."""

    def __init__(self, cfg: Qwen3Config, vcfg: VerifierConfig):
        super().__init__()
        if not 1 <= vcfg.k_layers <= cfg.num_layers:
            raise ValueError(f"k_layers={vcfg.k_layers} not in [1, {cfg.num_layers}]")
        if vcfg.kv not in ("shared", "own"):
            raise ValueError(f"kv={vcfg.kv!r}: expected shared or own")
        if vcfg.sink < 0 or (vcfg.kv != "shared" and vcfg.sink):
            raise ValueError(f"sink={vcfg.sink}: need sink >= 0, and kv=shared to set it")
        if vcfg.head.taps != ["final"]:
            raise NotImplementedError(f"verifier heads read the verifier's final hidden state (taps: [final]), got {vcfg.head.taps}")
        self.num_layers, self.k_layers, self.kv = cfg.num_layers, vcfg.k_layers, vcfg.kv
        self.sink = vcfg.sink
        self.first = cfg.num_layers - vcfg.k_layers
        alpha = vcfg.alpha if vcfg.alpha is not None else 2.0 * vcfg.rank
        self.lora = nn.ModuleList(LoRAFFN(cfg, vcfg.rank, alpha) for _ in range(vcfg.k_layers))
        self.head = build_head(vcfg.head, cfg.hidden_size)

    @property
    def capture_layers(self) -> range:
        return range(self.first, self.num_layers)

    def ffn(self, backbone: Qwen3Min, j: int, h: Tensor) -> Tensor:
        """Residual FFN block of adapted layer first + j for verifier rows h [B, *, hidden]."""
        layer = backbone.layers[self.first + j]
        return h + self.lora[j](layer.mlp, layer.post_attention_layernorm(h))

    def final_hidden(self, backbone: Qwen3Min, captures: dict[int, LayerCapture], positions: Tensor | None = None) -> Tensor:
        """Verifier hidden state after the shared final norm at positions [B, P] -> [B, P, hidden];
        positions=None: every position -> [B, T, hidden]."""
        if self.kv == "own":
            h = self._own_kv(backbone, captures)
            return h if positions is None else gather_positions(h, positions)
        if positions is None:
            return self._all_positions(backbone, captures)
        if self.sink and int(positions.min()) < self.sink:
            raise ValueError(f"label positions must be >= sink={self.sink} (got {int(positions.min())})")
        h = self.ffn(backbone, 0, gather_positions(captures[self.first].h_mid, positions))
        cos, sin = backbone.rope_cos[positions][:, None], backbone.rope_sin[positions][:, None]  # [B, 1, P, D]
        if self.sink:  # sink rows: causal among themselves
            hs = self.ffn(backbone, 0, captures[self.first].h_mid[:, : self.sink])
            cs, ss = backbone.rope(0, self.sink)
        for j in range(1, self.k_layers):
            layer, cap = backbone.layers[self.first + j], captures[self.first + j]
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(h), cos, sin)
            k_sink = v_sink = None
            if self.sink:
                qs, k_sink, v_sink = layer.self_attn.project_qkv(layer.input_layernorm(hs), cs, ss)
                out_s = F.scaled_dot_product_attention(qs, k_sink, v_sink, is_causal=True, scale=layer.self_attn.scale, enable_gqa=True)
                hs = self.ffn(backbone, j, hs + layer.self_attn.project_out(out_s))
            out = shared_kv_attention(q, cap.k, cap.v, k, v, positions, layer.self_attn.scale, k_sink, v_sink)
            h = self.ffn(backbone, j, h + layer.self_attn.project_out(out))
        return backbone.norm(h)

    def _all_positions(self, backbone: Qwen3Min, captures: dict[int, LayerCapture]) -> Tensor:
        h = captures[self.first].h_mid
        cos, sin = backbone.rope(0, h.shape[1])
        h = self.ffn(backbone, 0, h)
        for j in range(1, self.k_layers):
            layer, cap = backbone.layers[self.first + j], captures[self.first + j]
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(h), cos, sin)
            out = shared_kv_attention_all(q, cap.k, cap.v, k, v, layer.self_attn.scale, self.sink)
            h = self.ffn(backbone, j, h + layer.self_attn.project_out(out))
        return backbone.norm(h)

    def _own_kv(self, backbone: Qwen3Min, captures: dict[int, LayerCapture]) -> Tensor:
        h = captures[self.first].h_mid
        cos, sin = backbone.rope(0, h.shape[1])
        h = self.ffn(backbone, 0, h)
        for j in range(1, self.k_layers):
            h_mid, _, _ = backbone.layers[self.first + j].attention_block(h, cos, sin, layer_idx=self.first + j, cache=None, start_pos=0)
            h = self.ffn(backbone, j, h_mid)
        return backbone.norm(h)

    def forward(self, backbone: Qwen3Min, captures: dict[int, LayerCapture], positions: Tensor | None = None) -> Tensor:
        """Head logits [B, P, len(HEAD_OUTPUTS)] (or [B, T, ...] for positions=None), fp32."""
        return self.head(self.final_hidden(backbone, captures, positions))


@torch.no_grad()
def generator_captures(backbone: Qwen3Min, input_ids: Tensor, first: int) -> dict[int, LayerCapture]:
    """One frozen generator pass over right-padded input_ids [B, T]; captures layers first .. L-1, no LM head."""
    return backbone(input_ids, logits="none", capture_layers=range(first, backbone.cfg.num_layers)).captures
