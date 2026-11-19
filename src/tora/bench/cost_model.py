"""Analytic cost per decoded token (batch 1, context length T): FLOPs and bytes moved, for the generator and the
extra cost of the verifier row on top of it. Device-independent; scripts/cost_table.py prints the numbers the paper
cites (extra FLOPs by depth K, extra bytes with a shared or an own KV cache).

Conventions: a multiply-add is 2 FLOPs; weights and KV are bf16 (2 bytes); norms, RoPE and softmax are ignored
(small next to the GEMMs). "Extra bytes" assume the fused dual-row kernels for the shared-KV verifier: each
weight tile and KV tile is loaded once for both rows, so the verifier adds only its LoRA and head weights.
`unfused` instead re-reads the adapted layers' weights and KV (decode_ref / two separate calls).
"""

from __future__ import annotations

from dataclasses import dataclass

from tora.models.qwen3_min import Qwen3Config

BYTES = 2


@dataclass(frozen=True)
class Cost:
    flops: float
    bytes: float

    def __add__(self, other: Cost) -> Cost:
        return Cost(self.flops + other.flops, self.bytes + other.bytes)


def layer_params(c: Qwen3Config) -> tuple[int, int]:
    """(attention projection params, FFN params) of one decoder layer."""
    attn = c.hidden_size * c.head_dim * (2 * c.num_heads + 2 * c.num_kv_heads)
    ffn = 3 * c.hidden_size * c.intermediate_size
    return attn, ffn


def attention_flops(c: Qwen3Config, t: int) -> float:
    return 4.0 * c.num_heads * c.head_dim * t  # QK^T and PV for one query over t keys


def kv_bytes(c: Qwen3Config, t: int, layers: int) -> float:
    return 2.0 * layers * c.num_kv_heads * c.head_dim * t * BYTES


def generator(c: Qwen3Config, t: int) -> Cost:
    attn, ffn = layer_params(c)
    flops = c.num_layers * (2.0 * (attn + ffn) + attention_flops(c, t)) + 2.0 * c.hidden_size * c.vocab_size
    weights = c.num_layers * (attn + ffn) + c.hidden_size * c.vocab_size  # LM head (tied: the embedding matrix)
    return Cost(flops, weights * BYTES + kv_bytes(c, t, c.num_layers))


def head_params(kind: str, in_dim: int, hidden_dim: int, n_out: int = 2) -> int:
    return in_dim * n_out if kind == "linear" else in_dim * hidden_dim + hidden_dim * n_out


def verifier_extra(
    c: Qwen3Config, t: int, k: int, rank: int, head: str = "linear", head_hidden: int = 1024, own_kv: bool = False,
    unfused: bool = False,
) -> Cost:
    """ToRA (shared KV) or the own-KV verifier on top of the generator: layer L-K's FFN + K-1 full layers for one more
    row, LoRA on gate/up/down of K layers, final norm + head; no LM head."""
    attn, ffn = layer_params(c)
    lora = k * rank * 3 * (c.hidden_size + c.intermediate_size)
    hp = head_params(head, c.hidden_size, head_hidden)
    flops = 2.0 * ffn + (k - 1) * (2.0 * (attn + ffn) + attention_flops(c, t + 1)) + 2.0 * (lora + hp)
    moved = (lora + hp) * BYTES
    if own_kv:  # its own cache for layers L-K+1 .. L-1: read every step (weights still shared in a batch-2 GEMM)
        moved += kv_bytes(c, t, k - 1)
    if unfused:
        moved += (ffn + (k - 1) * (attn + ffn)) * BYTES + (0 if own_kv else kv_bytes(c, t, k - 1))
    return Cost(flops, moved)
