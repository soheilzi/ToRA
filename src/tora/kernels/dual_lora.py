"""Dual-row LoRA kernels (Triton): the generator row and the verifier row through one pass over the weights,
with the verifier's LoRA applied inside the kernel.

K1 dual_lora_linear   x [M, d_in] -> y [M, d_out]. Each program owns BLOCK_N output columns and streams its W rows
                      once, multiplying every tile by both rows. The epilogue adds scale * B (A x1) to row 1 (the
                      verifier) for its columns. A x1 comes either from the same loop (A is r x d_in) or, for the FFN's
                      down projection, from partial sums K2 already wrote (below). W + sBA is never formed: no second
                      weight copy is read or stored.
K2 dual_lora_gate_up  the same for gate and up at once (one pass over both), with silu(gate) * up in the epilogue.
                      Given the down projection's A, each program also writes A_down h1 over its own BLOCK_N columns:
                      the partial sums K1 adds up instead of re-reading A_down (r x d_ff fp32, as many bytes as W_down
                      at r = 8 when every K1 program reads it). One writer per partial, summed in a fixed order: the
                      result is deterministic.
dual_lora_ffn         K2 then K1: the FFN of both rows.

Numerics follow kernels/reference.py (the trained model's op order): fp32 accumulation, base output rounded to the
model dtype before the fp32 LoRA term is added, SiLU and the product rounded as in PyTorch.

Generator invariance: each row has its own accumulator, and the tile configuration depends only on the projection's
shape: autotuning is keyed on (N, K) and always run on two rows, so the one-row call (the generator alone) reuses the
config the two-row call uses. With bf16 / fp16 inputs (every product exact in fp32) the generator row is then bitwise
the same with the verifier row present or not (tests/test_kernels.py, test_decode_static.py); with fp32 inputs the
one- and two-row builds may contract multiply-adds differently, and it is not.

M = 1 or 2 rows (batch-1 decode). The LoRA rank must be a power of two (tl.arange).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

NO_LORA, LORA_A, LORA_PARTIALS = 0, 1, 2  # where K1 gets A x1 from


def _configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_N": bn, "BLOCK_K": bk, "DEFER": defer}, num_warps=w, num_stages=s)
        for bn in (8, 16, 32, 64)
        for bk in (128, 256, 512)
        for w in (2, 4, 8)
        for s in (1, 2)
        for defer in (0, 1)
        if bn * bk <= (8192 if defer else 16384)
    ]


@triton.jit
def _dot_rows(w_ptr, x_ptr, a_ptr, offs_n, mask_n, k0, K, stride_w, stride_x, acc0, acc1, ax,
              TWO_ROWS: tl.constexpr, IN_LOOP_A: tl.constexpr, R: tl.constexpr, BLOCK_K: tl.constexpr, DEFER: tl.constexpr):
    """One BLOCK_K step: both rows against one W tile, and A x1 alongside. DEFER: accumulate [BLOCK_N, BLOCK_K]
    products and reduce once after the loop, instead of a cross-lane reduction every step."""
    offs_k = k0 + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    w = tl.load(w_ptr + offs_n[:, None] * stride_w + offs_k[None, :], mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
    x0 = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    if DEFER:
        acc0 += w * x0[None, :]
    else:
        acc0 += tl.sum(w * x0[None, :], axis=1)
    if TWO_ROWS:
        x1 = tl.load(x_ptr + stride_x + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        if DEFER:
            acc1 += w * x1[None, :]
        else:
            acc1 += tl.sum(w * x1[None, :], axis=1)
        if IN_LOOP_A:
            a = tl.load(a_ptr + tl.arange(0, R)[:, None] * K + offs_k[None, :], mask=mask_k[None, :], other=0.0)
            if DEFER:
                ax += a * x1[None, :]
            else:
                ax += tl.sum(a * x1[None, :], axis=1)
    return acc0, acc1, ax


@triton.jit
def _lora_epilogue(y1, b_ptr, offs_n, mask_n, ax, scale, R: tl.constexpr):
    """Row 1: round the base output to the model dtype, add scale * B (A x1) in fp32, round again."""
    b = tl.load(b_ptr + offs_n[:, None] * R + tl.arange(0, R)[None, :], mask=mask_n[:, None], other=0.0)
    return (y1.to(tl.float32) + scale * tl.sum(b * ax[None, :], axis=1)).to(y1.dtype)


@triton.autotune(configs=_configs(), key=["N", "K"])
@triton.jit
def _dual_lora_linear_kernel(x_ptr, w_ptr, a_ptr, b_ptr, y_ptr, N, K, G, scale, stride_x, stride_w, stride_y,
                             TWO_ROWS: tl.constexpr, LORA: tl.constexpr, R: tl.constexpr,
                             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, DEFER: tl.constexpr):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    in_loop_a: tl.constexpr = LORA == 1
    if DEFER:
        acc0 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
        ax = tl.zeros([R, BLOCK_K], dtype=tl.float32)
    else:
        acc0 = tl.zeros([BLOCK_N], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_N], dtype=tl.float32)
        ax = tl.zeros([R], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        acc0, acc1, ax = _dot_rows(w_ptr, x_ptr, a_ptr, offs_n, mask_n, k0, K, stride_w, stride_x, acc0, acc1, ax,
                                   TWO_ROWS, in_loop_a, R, BLOCK_K, DEFER)
    if DEFER:
        acc0 = tl.sum(acc0, axis=1)
        acc1 = tl.sum(acc1, axis=1)
        ax = tl.sum(ax, axis=1)
    dt = y_ptr.dtype.element_ty
    tl.store(y_ptr + offs_n, acc0.to(dt), mask=mask_n)
    if TWO_ROWS:
        y1 = acc1.to(dt)
        if LORA == 2:  # A x1 = the sum of K2's G partials [G, R] at a_ptr
            ax = tl.zeros([R], dtype=tl.float32)
            for g0 in range(0, G, 128):
                gs = g0 + tl.arange(0, 128)
                ax += tl.sum(tl.load(a_ptr + gs[:, None] * R + tl.arange(0, R)[None, :], mask=gs[:, None] < G, other=0.0), axis=0)
        if LORA != 0:
            y1 = _lora_epilogue(y1, b_ptr, offs_n, mask_n, ax, scale, R)
        tl.store(y_ptr + stride_y + offs_n, y1, mask=mask_n)


@triton.jit
def _silu_mul(g, u):
    """silu(g) * u as PyTorch computes it on model-dtype tensors: silu in fp32, rounded; the product in fp32, rounded."""
    gf = g.to(tl.float32)
    s = (gf / (1.0 + libdevice.exp(-gf))).to(g.dtype)
    return (s.to(tl.float32) * u.to(tl.float32)).to(g.dtype)


@triton.autotune(configs=_configs(), key=["N", "K"])
@triton.jit
def _dual_lora_gate_up_kernel(x_ptr, wg_ptr, wu_ptr, ag_ptr, bg_ptr, au_ptr, bu_ptr, ad_ptr, p_ptr, h_ptr, N, K, scale,
                              stride_x, stride_w, stride_h, TWO_ROWS: tl.constexpr, HAS_LORA: tl.constexpr, PARTIALS: tl.constexpr,
                              R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, DEFER: tl.constexpr):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    if DEFER:
        g0 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
        g1 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
        u0 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
        u1 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
        axg = tl.zeros([R, BLOCK_K], dtype=tl.float32)
        axu = tl.zeros([R, BLOCK_K], dtype=tl.float32)
    else:
        g0 = tl.zeros([BLOCK_N], dtype=tl.float32)
        g1 = tl.zeros([BLOCK_N], dtype=tl.float32)
        u0 = tl.zeros([BLOCK_N], dtype=tl.float32)
        u1 = tl.zeros([BLOCK_N], dtype=tl.float32)
        axg = tl.zeros([R], dtype=tl.float32)
        axu = tl.zeros([R], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        g0, g1, axg = _dot_rows(wg_ptr, x_ptr, ag_ptr, offs_n, mask_n, k0, K, stride_w, stride_x, g0, g1, axg, TWO_ROWS, HAS_LORA, R, BLOCK_K, DEFER)
        u0, u1, axu = _dot_rows(wu_ptr, x_ptr, au_ptr, offs_n, mask_n, k0, K, stride_w, stride_x, u0, u1, axu, TWO_ROWS, HAS_LORA, R, BLOCK_K, DEFER)
    if DEFER:
        g0 = tl.sum(g0, axis=1)
        g1 = tl.sum(g1, axis=1)
        u0 = tl.sum(u0, axis=1)
        u1 = tl.sum(u1, axis=1)
        axg = tl.sum(axg, axis=1)
        axu = tl.sum(axu, axis=1)
    dt = h_ptr.dtype.element_ty
    tl.store(h_ptr + offs_n, _silu_mul(g0.to(dt), u0.to(dt)), mask=mask_n)
    if TWO_ROWS:
        gr, ur = g1.to(dt), u1.to(dt)
        if HAS_LORA:
            gr = _lora_epilogue(gr, bg_ptr, offs_n, mask_n, axg, scale, R)
            ur = _lora_epilogue(ur, bu_ptr, offs_n, mask_n, axu, scale, R)
        h1 = _silu_mul(gr, ur)
        tl.store(h_ptr + stride_h + offs_n, h1, mask=mask_n)
        if PARTIALS:  # A_down h1 over this program's columns -> row program_id of p [programs, R]
            ad = tl.load(ad_ptr + tl.arange(0, R)[:, None] * N + offs_n[None, :], mask=mask_n[None, :], other=0.0)
            part = tl.sum(ad * tl.where(mask_n, h1.to(tl.float32), 0.0)[None, :], axis=1)
            tl.store(p_ptr + tl.program_id(0) * R + tl.arange(0, R), part)


def _check(x: Tensor, ws: list[Tensor], loras: list[tuple[Tensor, Tensor] | None]) -> tuple[int, int, int, bool]:
    if x.dim() != 2 or x.shape[0] not in (1, 2) or x.stride(1) != 1:
        raise ValueError(f"x must be [1 or 2, d_in] with unit stride in d_in, got {tuple(x.shape)} strides {x.stride()}")
    n, k = ws[0].shape
    for w in ws:
        if w.shape != (n, k) or not w.is_contiguous() or w.dtype != x.dtype or k != x.shape[1]:
            raise ValueError(f"weights must be contiguous [{n}, {x.shape[1]}] in {x.dtype}, got {tuple(w.shape)} {w.dtype}")
    if all(lora is None for lora in loras):
        return n, k, 8, False
    if any(lora is None for lora in loras):
        raise ValueError("give LoRA factors for every projection or for none")
    r = loras[0][0].shape[0]
    if r & (r - 1):
        raise ValueError(f"LoRA rank {r} is not a power of two")
    for a, b in loras:
        if a.shape != (r, k) or b.shape != (n, r) or a.dtype != torch.float32 or b.dtype != torch.float32 or not (a.is_contiguous() and b.is_contiguous()):
            raise ValueError(f"LoRA factors must be contiguous fp32 A [{r}, {k}], B [{n}, {r}], got {tuple(a.shape)} {a.dtype}, {tuple(b.shape)} {b.dtype}")
    return n, k, r, True


_tuned: set[tuple] = set()
_placeholders: dict[torch.device, Tensor] = {}


def _placeholder(device: torch.device) -> Tensor:
    """fp32 stand-in for absent LoRA / partials pointers (never read). The autotuner's cache key includes every
    tensor argument's dtype, so a stand-in of another dtype would give the no-LoRA call its own tuned config, and the
    generator row could then differ between the one-row and the two-row call."""
    if device not in _placeholders:
        _placeholders[device] = torch.zeros(1, device=device, dtype=torch.float32)
    return _placeholders[device]


def _tune_on_two_rows(kernel_name: str, x: Tensor, n: int, k: int, run) -> None:
    """The first call for a shape tunes the kernel; make it a two-row call, so the generator-alone (one-row) call uses
    the config of the two-row call (generator invariance)."""
    key = (kernel_name, n, k, x.dtype)
    if key not in _tuned:
        if x.shape[0] == 1:
            run(torch.zeros(2, k, device=x.device, dtype=x.dtype))
        _tuned.add(key)


def dual_lora_linear(x: Tensor, w: Tensor, lora: tuple[Tensor, Tensor] | None = None, scale: float = 1.0,
                     ax_partials: Tensor | None = None) -> Tensor:
    """K1. x [M, d_in] (M = 1: the generator row; 2: + the verifier row), w [d_out, d_in] -> y [M, d_out] in x's dtype.
    lora = (A [r, d_in], B [d_out, r]) fp32 for row 1; with ax_partials ([G, r] fp32, rows summing to A x1, from
    dual_lora_gate_up) A is not read and may be None."""
    if ax_partials is not None:
        if lora is None or x.shape[0] != 2:
            raise ValueError("ax_partials needs two rows and lora=(A or None, B)")
        b = lora[1]
        n, k, _, _ = _check(x, [w], [None])
        r = b.shape[-1]
        if b.shape != (n, r) or b.dtype != torch.float32 or not b.is_contiguous() or r & (r - 1):
            raise ValueError(f"B must be contiguous fp32 [{n}, r] with r a power of two, got {tuple(b.shape)} {b.dtype}")
        if ax_partials.shape[1] != r or ax_partials.dtype != torch.float32 or not ax_partials.is_contiguous():
            raise ValueError(f"ax_partials must be contiguous fp32 [G, {r}], got {tuple(ax_partials.shape)} {ax_partials.dtype}")
        mode, a, g = LORA_PARTIALS, ax_partials, ax_partials.shape[0]
    else:
        n, k, r, has_lora = _check(x, [w], [lora])
        mode, (a, b), g = (LORA_A, lora, 0) if has_lora else (NO_LORA, (_placeholder(x.device),) * 2, 0)

    def run(xx: Tensor) -> Tensor:
        y = torch.empty(xx.shape[0], n, device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)  # noqa: E731
        _dual_lora_linear_kernel[grid](xx, w, a, b, y, n, k, g, float(scale), xx.stride(0), w.stride(0), y.stride(0),
                                       TWO_ROWS=xx.shape[0] == 2, LORA=mode if xx.shape[0] == 2 else NO_LORA, R=r)
        return y

    _tune_on_two_rows("linear", x, n, k, run)
    return run(x)


def dual_lora_gate_up(
    x: Tensor, w_gate: Tensor, w_up: Tensor, lora_gate: tuple[Tensor, Tensor] | None = None, lora_up: tuple[Tensor, Tensor] | None = None,
    scale: float = 1.0, down_a: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """K2. silu(gate(x)) * up(x) for both rows in one pass over w_gate and w_up [d_ff, d_in], LoRA on row 1
    -> h [M, d_ff]; with down_a (A_down [r, d_ff] fp32, two rows) -> (h, partials [programs, r] for K1)."""
    n, k, r, has_lora = _check(x, [w_gate, w_up], [lora_gate, lora_up])
    partials = down_a is not None
    if partials and (not has_lora or x.shape[0] != 2 or down_a.shape != (r, n) or down_a.dtype != torch.float32 or not down_a.is_contiguous()):
        raise ValueError(f"down_a needs two rows, LoRA and contiguous fp32 A_down [{r}, {n}]")
    ph = _placeholder(x.device)
    (ag, bg), (au, bu) = (lora_gate, lora_up) if has_lora else ((ph, ph), (ph, ph))
    p = torch.empty(triton.cdiv(n, 8), r, device=x.device, dtype=torch.float32) if partials else ph  # up to one row per program

    def run(xx: Tensor) -> Tensor:
        h = torch.empty(xx.shape[0], n, device=x.device, dtype=x.dtype)
        two = xx.shape[0] == 2
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)  # noqa: E731
        _dual_lora_gate_up_kernel[grid](xx, w_gate, w_up, ag, bg, au, bu, down_a if partials else ph, p, h, n, k, float(scale),
                                        xx.stride(0), w_gate.stride(0), h.stride(0), TWO_ROWS=two, HAS_LORA=has_lora and two,
                                        PARTIALS=partials and two, R=r)
        return h

    _tune_on_two_rows("gate_up", x, n, k, run)
    h = run(x)
    if not partials:
        return h
    return h, p[: triton.cdiv(n, _dual_lora_gate_up_kernel.best_config.kwargs["BLOCK_N"])]  # the rows this launch wrote


def dual_lora_ffn(x: Tensor, w_gate: Tensor, w_up: Tensor, w_down: Tensor, lora_gate: tuple[Tensor, Tensor] | None = None,
                  lora_up: tuple[Tensor, Tensor] | None = None, lora_down: tuple[Tensor, Tensor] | None = None, scale: float = 1.0) -> Tensor:
    """down(silu(gate(x)) * up(x)) for both rows, LoRA on row 1's three projections: K2 (which also writes the
    A_down h1 partials) then K1 on them. x [M, d_in] -> [M, d_in]."""
    if x.shape[0] == 2 and lora_down is not None:
        h, p = dual_lora_gate_up(x, w_gate, w_up, lora_gate, lora_up, scale, down_a=lora_down[0])
        return dual_lora_linear(h, w_down, (None, lora_down[1]), scale, ax_partials=p)
    loras = (lora_gate, lora_up, lora_down) if x.shape[0] == 2 else (None, None, None)
    return dual_lora_linear(dual_lora_gate_up(x, w_gate, w_up, loras[0], loras[1], scale), w_down, loras[2], scale)
