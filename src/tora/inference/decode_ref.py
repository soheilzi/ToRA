"""Reference dual-row decode in pure PyTorch: a generator row and a verifier row per step.

Per step at position t:
  generator row  qwen3_min's own decode call (the same call as generator-only decode); its K/V go into the
                 static cache. It is never stacked with the verifier row into one GEMM, so its logits are
                 bitwise those of generator-only decode (generator invariance).
  verifier row   layers < L-K and layer L-K's attention are the generator's (reused, not recomputed). Layer
                 L-K's FFN gets LoRA. In layers L-K+1 .. L-1 its query attends the generator's cached K/V at
                 positions < t plus its own (k, v) at t, which are temporary and never written to the cache.
                 Then the shared final norm and the head; no LM head.

The prompt is prefilled in one generator call, and its verifier rows come from the parallel shared-KV pass. This
decoder is the readable reference the tests check the teacher-forced verifier (models/verifier.py) and the fused
decode step (inference/decode_static.py) against.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from tora.models.qwen3_min import Qwen3Min, StaticKVCache
from tora.models.verifier import Verifier


@dataclass
class StepOutput:
    logits: Tensor  # generator next-token logits [B, V]
    verifier_logits: Tensor  # [B, len(HEAD_OUTPUTS)], fp32


class DualRowDecoder:
    def __init__(self, backbone: Qwen3Min, verifier: Verifier, batch_size: int = 1, max_len: int | None = None):
        if verifier.kv != "shared" or verifier.sink:
            raise ValueError("decode_ref implements the shared-KV verifier (ToRA, no sinks); the own-KV verifier keeps its own cache")
        self.backbone, self.verifier = backbone, verifier
        self.cache: StaticKVCache = backbone.new_cache(batch_size, max_len)

    @torch.no_grad()
    def step(self, tokens: Tensor) -> StepOutput:
        """tokens [B, 1] at position cache.length; advances the cache by one."""
        m, v = self.backbone, self.verifier
        t = self.cache.length
        out = m(tokens, cache=self.cache, logits="last", capture_layers=[v.first])
        return StepOutput(out.logits[:, -1], self.verifier_row(out.captures[v.first].h_mid, t)[:, -1])

    def verifier_row(self, branch: Tensor, t: int) -> Tensor:
        """branch: the generator's residual after layer L-K's attention at position t [B, 1, hidden]."""
        m, v = self.backbone, self.verifier
        cos, sin = m.rope(t, 1)
        h = v.ffn(m, 0, branch)
        for j in range(1, v.k_layers):
            layer = m.layers[v.first + j]
            q, k, val = layer.self_attn.project_qkv(layer.input_layernorm(h), cos, sin)
            k_gen, v_gen = self.cache.view(v.first + j, t)  # generator positions < t only
            attn = F.scaled_dot_product_attention(
                q, torch.cat([k_gen, k], dim=2), torch.cat([v_gen, val], dim=2), scale=layer.self_attn.scale, enable_gqa=True
            )
            h = v.ffn(m, j, h + layer.self_attn.project_out(attn))
        return v.head(m.norm(h))

    @torch.no_grad()
    def full_verifier_row(self, token: Tensor, t: int) -> Tensor:
        """Reference for skipping the shared layers: the verifier row at position t computed through every layer,
        reusing nothing from the generator row: from token [B, 1]'s embedding, base weights below L-K, LoRA FFN from
        layer L-K on, and in every layer its query attends the generator's cached K/V at positions < t plus its own
        (k, v) at t. Call after step() has consumed `token` at t. -> head logits [B, 1, C]."""
        m, v = self.backbone, self.verifier
        cos, sin = m.rope(t, 1)
        h = m.embed_tokens(token)
        for i, layer in enumerate(m.layers):
            q, k, val = layer.self_attn.project_qkv(layer.input_layernorm(h), cos, sin)
            k_gen, v_gen = self.cache.view(i, t)
            attn = F.scaled_dot_product_attention(
                q, torch.cat([k_gen, k], dim=2), torch.cat([v_gen, val], dim=2), scale=layer.self_attn.scale, enable_gqa=True
            )
            h = h + layer.self_attn.project_out(attn)
            h = v.ffn(m, i - v.first, h) if i >= v.first else layer.mlp_block(h)
        return v.head(m.norm(h))

    @torch.no_grad()
    def teacher_forced(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Feed input_ids [B, T] one token per step from position 0 -> (logits [B, T, V], verifier logits [B, T, C])."""
        if self.cache.length != 0:
            raise ValueError("teacher_forced starts from an empty cache")
        steps = [self.step(input_ids[:, i : i + 1]) for i in range(input_ids.shape[1])]
        return torch.stack([s.logits for s in steps], 1), torch.stack([s.verifier_logits for s in steps], 1)

    @torch.no_grad()
    def prefill(self, prompt_ids: Tensor) -> StepOutput:
        """The prompt in one generator call (as greedy_generate does), verifier rows at every prompt position (the
        parallel shared-KV pass). -> generator logits at the last prompt token [B, V], verifier logits [B, T, C]."""
        if self.cache.length != 0:
            raise ValueError("prefill starts from an empty cache")
        m, v = self.backbone, self.verifier
        out = m(prompt_ids, cache=self.cache, logits="last", capture_layers=v.capture_layers)
        return StepOutput(out.logits[:, -1], v(m, out.captures))

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, max_new_tokens: int, eos_token_ids: Iterable[int] = ()) -> tuple[Tensor, Tensor]:
        """Batch-1 greedy decode: prefill, then one dual-row step per token. -> (new tokens [n], verifier logits
        [len(prompt) + n - 1, C] at every position whose token the generator has consumed). The generator calls are
        exactly greedy_generate's, so the tokens must match it (generator invariance)."""
        eos = set(eos_token_ids)
        first = self.prefill(prompt_ids)
        verifier = [first.verifier_logits[0]]
        tokens = [first.logits.argmax(-1, keepdim=True)]
        while len(tokens) < max_new_tokens and tokens[-1].item() not in eos:
            s = self.step(tokens[-1])
            verifier.append(s.verifier_logits)
            tokens.append(s.logits.argmax(-1, keepdim=True))
        return torch.cat(tokens, dim=1)[0], torch.cat(verifier, dim=0)
