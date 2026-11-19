"""Minimal Qwen3 decoder that loads HF weights: prefill, static-KV-cache decode, capture hooks.

Numerics mirror transformers' Qwen3 + SDPA path op for op (fp32 RMSNorm, fp32 RoPE tables cast to
the model dtype, SDPA with native GQA), so prefill logits and greedy decode match HF (tests S1/S1b).
Each decoder layer exposes its attention half and its MLP half separately so the verifier can reuse
the generator's computation and splice LoRA into the FFN.

Hidden-state convention (used by probe taps and the verifier):
  hidden_states[i]  residual stream entering layer i (i = 0..L-1; hidden_states[0] = token embedding)
  hidden_states[L]  output of the last layer, before the final RMSNorm
  final_hidden      norm(hidden_states[L]), the input of the LM head
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

from tora.utils.hf import download_model


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    bos_token_id: int | None
    eos_token_id: int | list[int] | None

    @classmethod
    def from_hf_json(cls, path: str | Path) -> Qwen3Config:
        d = json.loads(Path(path).read_text())
        if d.get("model_type") != "qwen3":
            raise ValueError(f"{path}: model_type={d.get('model_type')!r}, expected 'qwen3'")
        # Features this implementation does not have: refuse instead of computing something else.
        if d.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(f"hidden_act={d['hidden_act']!r}")
        if d.get("attention_bias", False):
            raise NotImplementedError("attention_bias=True")
        if d.get("use_sliding_window", False):
            raise NotImplementedError("use_sliding_window=True")
        rope_theta = d.get("rope_theta")
        rope_params = d.get("rope_parameters") or {}
        if rope_theta is None:
            rope_theta = rope_params["rope_theta"]
        if d.get("rope_scaling") is not None or rope_params.get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope scaling {d.get('rope_scaling') or rope_params}")
        if "head_dim" not in d:
            raise ValueError(f"{path}: no head_dim (Qwen3 sets it explicitly; refusing to infer it)")
        if d["num_attention_heads"] % d["num_key_value_heads"]:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_layers=d["num_hidden_layers"],
            num_heads=d["num_attention_heads"],
            num_kv_heads=d["num_key_value_heads"],
            head_dim=d["head_dim"],
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=rope_theta,
            max_position_embeddings=d["max_position_embeddings"],
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            bos_token_id=d.get("bos_token_id"),
            eos_token_id=d.get("eos_token_id"),
        )

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    def summary(self) -> str:
        rows = [(f.name, getattr(self, f.name)) for f in fields(self)]
        rows += [("num_kv_groups", self.num_kv_groups), ("qk_norm", "RMSNorm over head_dim (q and k)")]
        width = max(len(name) for name, _ in rows)
        return "\n".join(f"  {name:<{width}}  {value}" for name, value in rows)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """q, k: [B, H, T, D]; cos, sin: [T, D]."""
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def attend(q: Tensor, k: Tensor, v: Tensor, *, scale: float, q_offset: int = 0) -> Tensor:
    """Causal GQA attention. q: [B, Hq, Tq, D] at absolute positions q_offset.., k/v: [B, Hkv, Tk, D] from 0.

    Makes the same SDPA call transformers makes when no padding mask is needed.
    """
    tq, tk = q.shape[2], k.shape[2]
    if tq == 1:  # decode: the query sees every key
        return F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True)
    if q_offset == 0 and tk == tq:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
    # Chunked prefill: query i (absolute q_offset + i) sees keys j <= q_offset + i.
    rows = q_offset + torch.arange(tq, device=q.device)
    mask = torch.arange(tk, device=q.device)[None, :] <= rows[:, None]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True)


class StaticKVCache:
    """Preallocated post-RoPE K/V, [L, B, Hkv, max_len, D]. Positions [0, length) are valid."""

    def __init__(self, cfg: Qwen3Config, batch_size: int, max_len: int, device: torch.device, dtype: torch.dtype):
        shape = (cfg.num_layers, batch_size, cfg.num_kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.length = 0

    def write(self, layer: int, start: int, k: Tensor, v: Tensor) -> None:
        end = start + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: writing up to {end} > max_len {self.max_len}")
        self.k[layer, :, :, start:end] = k
        self.v[layer, :, :, start:end] = v

    def view(self, layer: int, end: int) -> tuple[Tensor, Tensor]:
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def reset(self) -> None:
        self.length = 0


class Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim**-0.5
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def project_qkv(self, x: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """x: normed hidden [B, T, hidden] -> q [B, Hq, T, D], k, v [B, Hkv, T, D] (QK-norm and RoPE applied)."""
        shape = (*x.shape[:-1], -1, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(shape)).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        return q, k, v

    def project_out(self, attn: Tensor) -> Tensor:
        """attn: [B, Hq, T, D] -> [B, T, hidden]."""
        b, _, t, _ = attn.shape
        return self.o_proj(attn.transpose(1, 2).reshape(b, t, -1))


class MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def attention_block(
        self,
        h: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        layer_idx: int,
        cache: StaticKVCache | None,
        start_pos: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (h + attn(h), k, v) where k, v are this chunk's post-RoPE keys/values."""
        q, k, v = self.self_attn.project_qkv(self.input_layernorm(h), cos, sin)
        k_all, v_all = k, v
        if cache is not None:
            cache.write(layer_idx, start_pos, k, v)
            if start_pos > 0:
                k_all, v_all = cache.view(layer_idx, start_pos + k.shape[2])
        out = attend(q, k_all, v_all, scale=self.self_attn.scale, q_offset=start_pos)
        return h + self.self_attn.project_out(out), k, v

    def mlp_block(self, h: Tensor) -> Tensor:
        return h + self.mlp(self.post_attention_layernorm(h))


@dataclass
class LayerCapture:
    h_in: Tensor  # residual stream entering the layer [B, T, hidden]
    h_mid: Tensor  # after the attention residual add [B, T, hidden]
    k: Tensor  # post-RoPE keys of this chunk [B, Hkv, T, D]
    v: Tensor  # values of this chunk [B, Hkv, T, D]


@dataclass
class ForwardOutput:
    final_hidden: Tensor  # [B, T, hidden]
    logits: Tensor | None = None  # [B, T, V], or [B, 1, V] for logits="last"
    hidden_states: list[Tensor] | None = None  # L + 1 tensors, see module docstring
    captures: dict[int, LayerCapture] = field(default_factory=dict)


class Qwen3Min(nn.Module):
    def __init__(self, cfg: Qwen3Config, max_seq_len: int):
        super().__init__()
        self.cfg = cfg
        self.max_seq_len = max_seq_len
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = None if cfg.tie_word_embeddings else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.source_path: Path | None = None

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 4096,
        revision: str | None = None,
        hf_home: str | None = None,
        verbose: bool = True,
    ) -> Qwen3Min:
        model, extras = cls.from_pretrained_with_extras(
            repo_or_path,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
            revision=revision,
            hf_home=hf_home,
            verbose=verbose,
        )
        if extras:
            raise ValueError(f"checkpoint has non-backbone tensors {sorted(extras)}; use from_pretrained_with_extras")
        return model

    @classmethod
    def from_pretrained_with_extras(
        cls,
        repo_or_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 4096,
        revision: str | None = None,
        hf_home: str | None = None,
        verbose: bool = True,
    ) -> tuple[Qwen3Min, dict[str, Tensor]]:
        """Load the backbone (`model.*`, `lm_head.weight`); return every other tensor (e.g. guard heads) on device."""
        path = Path(repo_or_path)
        if not path.is_dir():
            path = download_model(str(repo_or_path), revision=revision, hf_home=hf_home)
        cfg = Qwen3Config.from_hf_json(path / "config.json")
        if verbose:
            print(f"Qwen3 config from {path / 'config.json'}:\n{cfg.summary()}")
        if max_seq_len > cfg.max_position_embeddings:
            raise ValueError(f"max_seq_len {max_seq_len} > max_position_embeddings {cfg.max_position_embeddings}")

        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no *.safetensors in {path}")
        state: dict[str, Tensor] = {}
        for shard in shards:
            state.update(load_file(shard))
        extras = {k: v for k, v in state.items() if not k.startswith("model.") and k != "lm_head.weight"}
        state = {k.removeprefix("model."): v for k, v in state.items() if k not in extras}
        if cfg.tie_word_embeddings:
            lm_head = state.pop("lm_head.weight", None)
            if lm_head is not None and not torch.equal(lm_head, state["embed_tokens.weight"]):
                raise ValueError("tie_word_embeddings=True but the checkpoint's lm_head differs from embed_tokens")

        with torch.device("meta"):
            model = cls(cfg, max_seq_len)
        model.load_state_dict(state, strict=True, assign=True)
        model = model.to(device=device, dtype=dtype)
        model._init_rope()
        model.source_path = path
        extras = {k: v.to(device=device, dtype=dtype) for k, v in extras.items()}
        return model.eval().requires_grad_(False), extras

    def _init_rope(self) -> None:
        # Same recipe as HF: inv_freq on CPU in fp32, angles in fp32 on device, cos/sin cast to model dtype.
        d = self.cfg.head_dim
        inv_freq = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float) / d))
        weight = self.embed_tokens.weight
        pos = torch.arange(self.max_seq_len, device=weight.device)
        freqs = pos[:, None].float() * inv_freq.to(device=weight.device, dtype=torch.float)[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos().to(weight.dtype), persistent=False)
        self.register_buffer("rope_sin", emb.sin().to(weight.dtype), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    @property
    def lm_head_weight(self) -> Tensor:
        return self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight

    def lm_logits(self, final_hidden: Tensor) -> Tensor:
        return F.linear(final_hidden, self.lm_head_weight)

    def rope(self, start: int, length: int) -> tuple[Tensor, Tensor]:
        if start + length > self.max_seq_len:
            raise ValueError(f"positions up to {start + length} exceed max_seq_len {self.max_seq_len}")
        return self.rope_cos[start : start + length], self.rope_sin[start : start + length]

    def new_cache(self, batch_size: int = 1, max_len: int | None = None) -> StaticKVCache:
        return StaticKVCache(self.cfg, batch_size, max_len or self.max_seq_len, self.device, self.dtype)

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: StaticKVCache | None = None,
        start_pos: int | None = None,
        logits: Literal["all", "last", "none"] = "all",
        output_hidden_states: bool = False,
        capture_layers: Iterable[int] = (),
    ) -> ForwardOutput:
        """input_ids [B, T] at absolute positions start_pos.. (default: cache.length, or 0 without a cache).

        With a cache, this chunk's K/V are written at [start_pos, start_pos + T) and cache.length advances.
        Batches must be right-padded (or unpadded): causal attention then never lets a real token see a pad.
        """
        _, t = input_ids.shape
        if start_pos is None:
            start_pos = cache.length if cache is not None else 0
        if cache is None and start_pos != 0:
            raise ValueError("start_pos > 0 needs a KV cache that holds the earlier positions")
        capture = set(capture_layers)
        if capture and not capture <= set(range(self.cfg.num_layers)):
            raise ValueError(f"capture_layers {sorted(capture)} out of range [0, {self.cfg.num_layers})")
        cos, sin = self.rope(start_pos, t)

        h = self.embed_tokens(input_ids)
        hidden_states = [] if output_hidden_states else None
        captures: dict[int, LayerCapture] = {}
        for i, layer in enumerate(self.layers):
            if hidden_states is not None:
                hidden_states.append(h)
            h_mid, k, v = layer.attention_block(h, cos, sin, layer_idx=i, cache=cache, start_pos=start_pos)
            if i in capture:
                captures[i] = LayerCapture(h_in=h, h_mid=h_mid, k=k, v=v)
            h = layer.mlp_block(h_mid)
        if hidden_states is not None:
            hidden_states.append(h)
        if cache is not None:
            cache.length = start_pos + t

        final = self.norm(h)
        out = ForwardOutput(final_hidden=final, hidden_states=hidden_states, captures=captures)
        if logits == "all":
            out.logits = self.lm_logits(final)
        elif logits == "last":
            out.logits = self.lm_logits(final[:, -1:])
        elif logits != "none":
            raise ValueError(f"logits={logits!r}")
        return out


def eos_ids(cfg: Qwen3Config, generation_config_path: Path | None = None) -> list[int]:
    """EOS ids from generation_config.json if present (Qwen3 lists two there), else from config.json."""
    eos = cfg.eos_token_id
    if generation_config_path is not None and generation_config_path.exists():
        eos = json.loads(generation_config_path.read_text()).get("eos_token_id", eos)
    if eos is None:
        return []
    return [eos] if isinstance(eos, int) else list(eos)


@torch.no_grad()
def greedy_generate(
    model: Qwen3Min,
    prompt_ids: Tensor,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] = (),
    cache: StaticKVCache | None = None,
) -> Tensor:
    """Batch-1 greedy decode with the static cache. Returns the new tokens [n], EOS included if hit."""
    if prompt_ids.dim() != 2 or prompt_ids.shape[0] != 1:
        raise ValueError(f"prompt_ids must be [1, T], got {tuple(prompt_ids.shape)}")
    eos = set(eos_token_ids)
    if cache is None:
        cache = model.new_cache(1, prompt_ids.shape[1] + max_new_tokens)
    tok = model(prompt_ids, cache=cache, logits="last").logits[:, -1].argmax(-1, keepdim=True)
    generated = [tok]
    for _ in range(max_new_tokens - 1):
        if tok.item() in eos:
            break
        tok = model(tok, cache=cache, logits="last").logits[:, -1].argmax(-1, keepdim=True)
        generated.append(tok)
    return torch.cat(generated, dim=1)[0]
