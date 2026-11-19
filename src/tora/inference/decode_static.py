"""Static-shape batch-1 decode step: the generator alone or with the ToRA verifier row, as one function of
fixed-shape tensors, so that it can be captured in a CUDA graph or compiled.

Modes (the paper's execution ablation, Fig. 4, compares them):
  gen        the generator alone (G, the floor).
  separate   the verifier row by separate calls, as decode_ref (S): in the adapted layers every projection runs once
             per row (M = 1), so their weights are read twice, and the verifier's attention reads the KV again.
  batched    the naive dual-row step (N): from layer L-K's FFN on, both rows go through each projection as one M = 2
             batch (one weight read); LoRA is separate small matmuls on row 1; one attention call for both rows.
  fused      batched with every projection of the adapted layers on the dual-row Triton kernels (kernels/dual_lora.py):
             the FFN on K2 + K1 with the LoRA inside, q / k / v / o on K1 (the deployed step).
  fused_gen  the generator alone through fused's kernels (the verifier only fixes L - K): its generator logits must
             equal fused's bitwise (S2), since each kernel's row 0 does not depend on row 1.

Below layer L-K and in layer L-K's attention the verifier row is the generator's (decode_ref, S5), so every mode
runs the generator alone there.

Cache and attention: the cache holds `slots` = round_up(max_len + 1, 128) positions per layer; the last slot holds the
verifier row's own (k, v) at the current position, rewritten every step and never read by the generator. The
position is the device tensor `pos`, read by FlexAttention's mask_mod; every KV block is marked partial, so the block
mask never changes with it. The generator row sees slots <= pos, the verifier row slots < pos and its own slot
(decode_ref's keys). In batched mode the two rows are query positions 0 and 1 of one call, so each KV tile is read
once for both.

`step` decodes `token` at position `pos`; it writes the generator's K/V at `pos` but does not advance it. `prefill`
fills the cache from a prompt in one generator call, as deployment does before decoding the response.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from tora.bench.timing import cuda_graph
from tora.kernels import ops
from tora.models.lora_ffn import LoRA, LoRAFFN
from tora.models.qwen3_min import MLP, Attention, Qwen3Min, StaticKVCache, apply_rope
from tora.models.verifier import Verifier

Mode = Literal["gen", "separate", "batched", "fused", "fused_gen"]
MODES: tuple[Mode, ...] = ("gen", "separate", "batched", "fused", "fused_gen")
Execution = Literal["eager", "graph", "compile", "compile+graph"]
EXECS: tuple[Execution, ...] = ("eager", "graph", "compile", "compile+graph")
BLOCK = 128  # FlexAttention's KV block; `slots` is a multiple of it
_compiled_flex = None


def _flex(q: Tensor, k: Tensor, v: Tensor, block_mask: BlockMask, scale: float) -> Tensor:
    """q [1, H, R, D] over the cache k, v [1, Hkv, slots, D] -> [1, H, R, D]. The G = H / Hkv query heads that share a
    KV head are packed as rows by hand ([1, Hkv, G * R, D], row g * R + r; the masks take r = q_idx % R) instead of
    enable_gqa=True. Same kernel work (the GQA decode path packs them the same way), but with R = 2 rows and q built
    inside a compiled step, torch 2.9.1's compiled flex with q [1, H, 2, D] returned wrong values for every row (1.1
    relative error against an exact masked softmax, with or without enable_gqa, while the same q passed in from
    outside the graph was right); packed it is within 2e-3 of the exact softmax either way."""
    _, h, r, d = q.shape
    qp = q.reshape(1, k.shape[1], h // k.shape[1] * r, d)
    if torch.compiler.is_compiling():  # inside a compiled step: traced with it
        return flex_attention(qp, k, v, block_mask=block_mask, scale=scale).view(1, h, r, d)
    global _compiled_flex
    if _compiled_flex is None:  # eager flex materializes every score; the compiled one is the decode kernel
        # Static shapes, one kernel per row count and cache size: after a size change (e.g. one query row, then two)
        # dynamo would otherwise switch to a dynamic-shape kernel.
        _compiled_flex = torch.compile(flex_attention, fullgraph=True, dynamic=False)
    with torch._dynamo.config.patch(recompile_limit=64):  # one graph per dtype, query rows, mask and cache size
        return _compiled_flex(qp, k, v, block_mask=block_mask, scale=scale).view(1, h, r, d)


def runner(step: Callable[[], tuple], exe: Execution) -> Callable[[], tuple]:
    """A decoder's zero-argument step as executed: eager, replayed as a CUDA graph, compiled (fullgraph), or compiled
    and then replayed as a CUDA graph (the fastest). Compiling resets dynamo first (one compiled step at a time, so
    recompile limits do not build up) and runs the step once, so the first timed call does not compile."""
    if exe not in EXECS:
        raise ValueError(f"execution {exe!r}: expected one of {EXECS}")
    if exe == "eager":
        return step
    if exe == "graph":
        return cuda_graph(step)
    torch._dynamo.reset()
    compiled = torch.compile(step, fullgraph=True)
    compiled()
    return compiled if exe == "compile" else cuda_graph(compiled)


def block_mask(q_rows: int, slots: int, mask_mod, device) -> BlockMask:
    """A static decode block mask over `slots` cache positions (a multiple of BLOCK) for q_rows packed query rows:
    every KV block is marked partial, so mask_mod (reading the device position) decides and the mask never changes."""
    n = slots // BLOCK
    return BlockMask.from_kv_blocks(
        torch.full((1, 1, 1), n, dtype=torch.int32, device=device),
        torch.arange(n, dtype=torch.int32, device=device).view(1, 1, 1, n),
        BLOCK_SIZE=(BLOCK, BLOCK), mask_mod=mask_mod, seq_lengths=(q_rows, slots),
    )


def lora_row1(base: torch.nn.Linear, lora: LoRA, x: Tensor) -> Tensor:
    """x [1, 2, d_in] (row 0 the generator's, row 1 the verifier's) -> one M = 2 GEMM, LoRA added to row 1 as in
    lora_ffn.lora_linear."""
    y = base(x)
    return torch.cat([y[:, :1], (y[:, 1:].float() + lora(x[:, 1:])).to(y.dtype)], dim=1)


def ffn_batched(mlp: MLP, lora: LoRAFFN, x: Tensor) -> Tensor:
    """The FFN of both rows (normed input x [1, 2, hidden]); LoRA on row 1. Same op order as LoRAFFN."""
    h = F.silu(lora_row1(mlp.gate_proj, lora.gate, x)) * lora_row1(mlp.up_proj, lora.up, x)
    return lora_row1(mlp.down_proj, lora.down, h)


def k1(linear: torch.nn.Linear, x: Tensor) -> Tensor:
    """x [1, R, d_in] -> [1, R, d_out] on K1 (no LoRA)."""
    return ops.dual_linear(x[0], linear.weight)[None]


def project_qkv_fused(attn: Attention, x: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """qwen3_min's Attention.project_qkv with q / k / v on K1."""
    shape = (*x.shape[:-1], -1, attn.head_dim)
    q = attn.q_norm(k1(attn.q_proj, x).view(shape)).transpose(1, 2)
    k = attn.k_norm(k1(attn.k_proj, x).view(shape)).transpose(1, 2)
    v = k1(attn.v_proj, x).view(shape).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    return q, k, v


def ffn_fused(mlp: MLP, lora: LoRAFFN | None, x: Tensor) -> Tensor:
    """The FFN of the rows in x [1, R, hidden] on K2 + K1, LoRA on row 1 (lora=None: the generator row alone)."""
    factors = (None,) * 6 if lora is None else tuple(t for p in (lora.gate, lora.up, lora.down) for t in (p.A, p.B))
    scale = 1.0 if lora is None else lora.gate.scale
    return ops.lora_ffn(x[0], mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight, *factors, scale)[None]


class StaticDecoder:
    def __init__(self, backbone: Qwen3Min, verifier: Verifier | None, mode: Mode, max_len: int):
        if mode not in MODES:
            raise ValueError(f"mode={mode!r}: expected one of {MODES}")
        if (verifier is None) != (mode == "gen"):
            raise ValueError("mode gen runs without a verifier; the other modes need one")
        if verifier is not None and len({p.scale for lora in verifier.lora for p in (lora.gate, lora.up, lora.down)}) != 1:
            raise ValueError("the fused FFN takes one LoRA scale for gate, up and down")
        if verifier is not None and (verifier.kv != "shared" or verifier.sink):
            raise ValueError("the static decoder implements ToRA (shared KV, no sinks)")
        if max_len > backbone.max_seq_len:
            raise ValueError(f"max_len {max_len} > the backbone's RoPE table ({backbone.max_seq_len})")
        c, dev = backbone.cfg, backbone.device
        self.backbone, self.verifier, self.mode, self.max_len = backbone, verifier, mode, max_len
        self.scores = mode not in ("gen", "fused_gen")  # whether step returns verifier logits
        self.first = c.num_layers if verifier is None else verifier.first
        self.slots = -(-(max_len + 1) // BLOCK) * BLOCK
        self.vslot = self.slots - 1
        # one tensor per layer: compiled, a write into a view of one shared buffer copies the whole buffer
        shape = (1, c.num_kv_heads, self.slots, c.head_dim)
        self.k = [torch.zeros(shape, device=dev, dtype=backbone.dtype) for _ in range(c.num_layers)]
        self.v = [torch.zeros_like(k) for k in self.k]
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.vslot_index = torch.tensor([self.vslot], device=dev)  # index_copy_, not slice assignment: compiled, the
        # latter copies the whole layer cache
        self.token = torch.zeros(1, 1, dtype=torch.long, device=dev)
        pos, vslot = self.pos, self.vslot
        self.group = c.num_heads // c.num_kv_heads  # query heads per KV head, packed as rows (_flex)

        def gen(b, h, q, kv):
            return kv <= pos[0]

        def ver(b, h, q, kv):
            return (kv < pos[0]) | (kv == vslot)

        def dual(b, h, q, kv):  # packed row q = g * 2 + r: r = 0 the generator, r = 1 the verifier
            return torch.where(q % 2 == 0, gen(b, h, q, kv), ver(b, h, q, kv))

        self.mask_gen, self.mask_ver, self.mask_dual = (self._block_mask(rows, m) for rows, m in ((1, gen), (1, ver), (2, dual)))
        self._prefill_cache: StaticKVCache | None = None

    def _block_mask(self, rows: int, mask_mod) -> BlockMask:
        return block_mask(self.group * rows, self.slots, mask_mod, self.pos.device)

    def rope(self) -> tuple[Tensor, Tensor]:
        m = self.backbone
        return m.rope_cos.index_select(0, self.pos), m.rope_sin.index_select(0, self.pos)

    def gen_layer(self, i: int, h: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Layer i for the generator row alone (h [1, 1, hidden]), on cuBLAS: what every mode runs below L - K."""
        return self.backbone.layers[i].mlp_block(self._attention(i, h, cos, sin, "gen"))

    def layer_alone(self, i: int, h: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Layer i for the generator row alone as this decoder's step runs it: gen on cuBLAS; fused_gen on cuBLAS below
        L - K and in layer L - K's attention, on fused's kernels from there on. For callers that step the generator
        layer by layer (decode_baselines), so that they run the same generator as gen or fused_gen."""
        if i < self.first:  # every layer in gen mode (first = num_layers)
            return self.gen_layer(i, h, cos, sin)
        if self.mode != "fused_gen":
            raise ValueError(f"mode {self.mode!r} runs two rows from layer {self.first} on; only gen and fused_gen run one")
        layer = self.backbone.layers[i]
        h = self._attention(i, h, cos, sin, "gen", fused=i > self.first)
        return h + ffn_fused(layer.mlp, None, layer.post_attention_layernorm(h))

    @torch.no_grad()
    def prefill(self, prompt_ids: Tensor) -> None:
        """The generator over prompt_ids [1, P] in one qwen3_min call (as decode_ref.prefill and greedy_generate); its
        K/V go into the cache at positions < P and pos = P, so that `step` decodes the next token. Positions >= P keep
        whatever they held: every mask reads only slots <= pos (and the verifier's own slot, rewritten each step)."""
        p = prompt_ids.shape[1]
        if prompt_ids.shape[0] != 1 or not 0 < p <= self.max_len - 1:
            raise ValueError(f"prompt_ids {tuple(prompt_ids.shape)}: batch 1, 1 to max_len - 1 = {self.max_len - 1} tokens")
        if self._prefill_cache is None:
            self._prefill_cache = self.backbone.new_cache(1, self.max_len)
        cache = self._prefill_cache
        cache.reset()
        self.backbone(prompt_ids, cache=cache, logits="none")
        for i in range(self.backbone.cfg.num_layers):
            self.k[i][:, :, :p].copy_(cache.k[i, :, :, :p])
            self.v[i][:, :, :p].copy_(cache.v[i, :, :, :p])
        self.pos.fill_(p)

    def fill_random(self, length: int, seed: int = 0) -> None:
        """Random K/V at positions < length (timing: the values do not matter) and pos = length."""
        if not 0 <= length <= self.max_len - 1:
            raise ValueError(f"length {length} leaves no room for the step (max_len {self.max_len})")
        g = torch.Generator(device=self.pos.device).manual_seed(seed)
        for x in (*self.k, *self.v):
            x.zero_()
            x[:, :, :length].normal_(generator=g)
        self.pos.fill_(length)

    def _attention(self, i: int, h: Tensor, cos: Tensor, sin: Tensor, rows: Literal["gen", "ver", "dual"], fused: bool = False) -> Tensor:
        """Residual attention block of layer i for h [1, R, hidden] (R = 2 for dual: generator, verifier); fused:
        the projections on K1.

        The one-row (gen, ver) and two-row (dual) calls run different FlexAttention kernels, which inductor generates
        per query shape; the kernel that combines the KV splits can then sum them in a different order for the
        generator row (seen: row 0 differed from the second KV block on, in a process that had compiled the fp32
        model first). So S2 is measured for the whole step, not guaranteed by construction as for the GEMV kernels.
        Running every call as two rows would guarantee it, but cost ~3% per token in every mode."""
        layer = self.backbone.layers[i]
        x = layer.input_layernorm(h)
        q, k, v = project_qkv_fused(layer.self_attn, x, cos, sin) if fused else layer.self_attn.project_qkv(x, cos, sin)
        if rows != "ver":  # the generator's K/V go into the cache at pos
            self.k[i].index_copy_(2, self.pos, k[:, :, :1])
            self.v[i].index_copy_(2, self.pos, v[:, :, :1])
        if rows != "gen":  # the verifier's own K/V at pos, in its slot
            self.k[i].index_copy_(2, self.vslot_index, k[:, :, -1:])
            self.v[i].index_copy_(2, self.vslot_index, v[:, :, -1:])
        mask = {"gen": self.mask_gen, "ver": self.mask_ver, "dual": self.mask_dual}[rows]
        out = _flex(q, self.k[i], self.v[i], mask, layer.self_attn.scale)
        if fused:
            return h + k1(layer.self_attn.o_proj, out.transpose(1, 2).reshape(*h.shape[:2], -1))
        return h + layer.self_attn.project_out(out)

    @torch.no_grad()
    def step(self) -> tuple[Tensor, Tensor | None]:
        """-> (generator logits [1, V], verifier logits [1, C] or None)."""
        m, ver = self.backbone, self.verifier
        cos, sin = self.rope()
        h = m.embed_tokens(self.token)
        for i in range(self.first):
            h = self.gen_layer(i, h, cos, sin)
        if ver is None:
            return m.lm_logits(m.norm(h))[:, -1], None
        h_mid = self._attention(self.first, h, cos, sin, "gen")  # the branch point: both rows start here
        if self.mode == "separate":
            hg, hv = m.layers[self.first].mlp_block(h_mid), ver.ffn(m, 0, h_mid)
            for j in range(1, ver.k_layers):
                i = self.first + j
                hg = m.layers[i].mlp_block(self._attention(i, hg, cos, sin, "gen"))
                hv = ver.ffn(m, j, self._attention(i, hv, cos, sin, "ver"))
        elif self.mode == "batched":
            h = torch.cat([h_mid, h_mid], dim=1)
            for j in range(ver.k_layers):
                i = self.first + j
                layer = m.layers[i]
                if j:
                    h = self._attention(i, h, cos, sin, "dual")
                h = h + ffn_batched(layer.mlp, ver.lora[j], layer.post_attention_layernorm(h))
            hg, hv = h[:, :1], h[:, 1:]
        else:  # fused / fused_gen
            two = self.mode == "fused"
            h = torch.cat([h_mid, h_mid], dim=1) if two else h_mid
            for j in range(ver.k_layers):
                i = self.first + j
                layer = m.layers[i]
                if j:
                    h = self._attention(i, h, cos, sin, "dual" if two else "gen", fused=True)
                h = h + ffn_fused(layer.mlp, ver.lora[j] if two else None, layer.post_attention_layernorm(h))
            if not two:
                return m.lm_logits(m.norm(h))[:, -1], None
            hg, hv = h[:, :1], h[:, 1:]
        return m.lm_logits(m.norm(hg))[:, -1], ver.head(m.norm(hv))[:, -1]

    @torch.no_grad()
    def teacher_forced(self, input_ids: Tensor, step=None) -> tuple[Tensor, Tensor | None]:
        """Feed input_ids [1, T] one token per step from position 0 (S2 / S4 checks against decode_ref).
        step: the callable to run per token (default self.step; e.g. its CUDA graph). -> logits [1, T, V],
        verifier logits [1, T, C] or None."""
        if input_ids.shape[0] != 1 or input_ids.shape[1] > self.max_len:
            raise ValueError(f"input_ids {tuple(input_ids.shape)}: batch 1, at most max_len={self.max_len} tokens")
        step = step or self.step
        for x in (*self.k, *self.v):
            x.zero_()
        logits, vlogits = [], []
        for t in range(input_ids.shape[1]):
            self.pos.fill_(t)
            self.token.copy_(input_ids[:, t : t + 1])
            lg, vl = step()
            logits.append(lg.clone())
            vlogits.append(None if vl is None else vl.clone())
        return torch.stack(logits, 1), torch.stack(vlogits, 1) if self.scores else None
