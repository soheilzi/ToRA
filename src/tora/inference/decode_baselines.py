"""Static-shape batch-1 decode steps of the generator with a baseline guard (the latency comparison against
ToRA on the fused kernels), under decode_static.StaticDecoder's rules: static shapes, the position as a device tensor,
one zero-argument step to capture in a CUDA graph or compile, `fill_random` for timing, `teacher_forced` for checks.

  SingProbeDecoder    the generator (gen mode) plus SingProbe's probe on its hidden states: the residual stream
                      entering layers 8 / 17 / 26, RMS-normed without weight, concatenated; then the probe's own
                      causal multi-query attention over the token stream (its own K/V cache, sliding window 2048).
                      The probe reads the generator's states, so it needs no second model.
  GuardStreamDecoder  the generator plus Qwen3Guard-Stream: a second 0.6B model that reads every token with its own
                      KV cache, then its response-risk head (no LM head). Both models run in the same step, so one
                      graph holds both.

Both step the generator exactly as StaticDecoder's gen mode does (cuBLAS, FlexAttention decode), so the generator's
share of the step is the same in every compared method. Given ToRA's verifier, they step it as fused_gen does instead
(cuBLAS below the verifier's L - K, our kernels from there on: the generator's share of ToRA's own fused step), and
Qwen3Guard-Stream's model runs the same way; the verifier only fixes L - K, its LoRA and head are not used. Returns
(generator logits [1, V], guard logits [1, C]).
"""

from __future__ import annotations

import torch
from torch import Tensor

from tora.inference.decode_static import StaticDecoder, _flex, block_mask
from tora.models.baselines.qwen3guard_stream import Qwen3GuardStream
from tora.models.baselines.singprobe import SingProbe, tap_norm
from tora.models.qwen3_min import Qwen3Min
from tora.models.verifier import Verifier


def teacher_forced(decoder, step, input_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Feed input_ids [1, T] one token per step from position 0 through `step` (the decoder's step as executed, e.g.
    decode_static.runner(decoder.step, "compile+graph")) -> (generator logits [1, T, V], guard logits [1, T, C]), for
    checks against the baselines' own full-sequence forward."""
    if input_ids.shape[0] != 1 or input_ids.shape[1] > decoder.max_len:
        raise ValueError(f"input_ids {tuple(input_ids.shape)}: batch 1, at most max_len={decoder.max_len} tokens")
    decoder.reset()
    logits, guard = [], []
    for t in range(input_ids.shape[1]):
        decoder.pos.fill_(t)
        decoder.token.copy_(input_ids[:, t : t + 1])
        lg, gl = step()
        logits.append(lg.clone())
        guard.append(gl.clone())
    return torch.stack(logits, 1), torch.stack(guard, 1)


class SingProbeDecoder:
    def __init__(self, model: SingProbe, max_len: int, verifier: Verifier | None = None):
        self.gen = StaticDecoder(model.backbone, verifier, "gen" if verifier is None else "fused_gen", max_len)
        self.probe, self.taps, self.max_len = model.probe, set(model.taps), max_len
        self.pos, self.token = self.gen.pos, self.gen.token
        p, dev = self.probe, self.pos.device
        shape = (1, 1, self.gen.slots, p.head_dim)  # one K/V head (multi-query)
        self.k = torch.zeros(shape, device=dev, dtype=model.backbone.dtype)
        self.v = torch.zeros_like(self.k)
        pos, window = self.pos, p.window or self.gen.slots

        def causal_window(b, h, q, kv):  # SingProbeAttn: 0 <= pos - kv < window
            return (kv <= pos[0]) & (kv > pos[0] - window)

        self.mask = block_mask(p.num_heads, self.gen.slots, causal_window, dev)

    def reset(self) -> None:
        for x in (*self.gen.k, *self.gen.v, self.k, self.v):
            x.zero_()

    def fill_random(self, length: int, seed: int = 0) -> None:
        """Random generator and probe K/V at positions < length (timing) and pos = length."""
        self.gen.fill_random(length, seed)
        g = torch.Generator(device=self.pos.device).manual_seed(seed + 1)
        for x in (self.k, self.v):
            x.zero_()
            x[:, :, :length].normal_(generator=g)

    def probe_step(self, x: Tensor) -> Tensor:
        """SingProbeAttn.forward for the token at pos (x [1, 1, in_dim]); its K/V go into the probe cache."""
        p = self.probe
        q_feat = p.proj_q(x)
        q = q_feat.view(1, 1, p.num_heads, p.head_dim).transpose(1, 2)
        self.k.index_copy_(2, self.pos, p.proj_k(x).view(1, 1, 1, p.head_dim))
        self.v.index_copy_(2, self.pos, p.proj_v(x).view(1, 1, 1, p.head_dim))
        ctx = _flex(q, self.k, self.v, self.mask, p.head_dim**-0.5).transpose(1, 2).reshape(1, 1, -1)
        return p.classifier(p.norm(p.o_proj(ctx) + q_feat))

    @torch.no_grad()
    def step(self) -> tuple[Tensor, Tensor]:
        """-> (generator logits [1, V], probe logits [1, 10]; singprobe.RESPONSE_UNSAFE is the streaming score)."""
        m, cos, sin = self.gen.backbone, *self.gen.rope()
        h, taps = m.embed_tokens(self.token), []
        for i in range(m.cfg.num_layers):
            if i in self.taps:
                taps.append(tap_norm(h))
            h = self.gen.layer_alone(i, h, cos, sin)
        return m.lm_logits(m.norm(h))[:, -1], self.probe_step(torch.cat(taps, dim=-1))[:, -1]


class GuardStreamDecoder:
    def __init__(self, generator: Qwen3Min, guard: Qwen3GuardStream, max_len: int, verifier: Verifier | None = None):
        mode = "gen" if verifier is None else "fused_gen"
        self.gen = StaticDecoder(generator, verifier, mode, max_len)
        self.guard_dec = StaticDecoder(guard.backbone, verifier, mode, max_len)  # its own cache, pos and token
        self.guard, self.max_len = guard, max_len
        self.pos, self.token = self.gen.pos, self.gen.token

    def reset(self) -> None:
        for d in (self.gen, self.guard_dec):
            for x in (*d.k, *d.v):
                x.zero_()

    def fill_random(self, length: int, seed: int = 0) -> None:
        self.gen.fill_random(length, seed)
        self.guard_dec.fill_random(length, seed + 1)

    @torch.no_grad()
    def step(self) -> tuple[Tensor, Tensor]:
        """-> (generator logits [1, V], guard response-risk logits [1, 3]: Safe, Unsafe, Controversial)."""
        logits = self.gen.step()[0]
        g = self.guard_dec
        g.pos.copy_(self.pos)  # the guard reads the token the generator consumes, at the same position
        g.token.copy_(self.token)
        m, cos, sin = g.backbone, *g.rope()
        h = m.embed_tokens(g.token)
        for i in range(m.cfg.num_layers):
            h = g.layer_alone(i, h, cos, sin)
        risk, _category = self.guard.response_head(m.norm(h))
        return logits, risk[:, -1]

