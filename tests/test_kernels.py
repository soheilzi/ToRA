"""S9: the dual-row LoRA kernels (kernels/dual_lora.py) against their PyTorch references and fp32, over a
shape grid (Qwen3-0.6B / 1.7B projections, sizes that are not powers of two), LoRA ranks 8 / 16, bf16 / fp16, and a
fixed set of tile configs that covers every code path (per-step vs deferred reduction, narrow vs wide tiles; the
autotuner normally picks one per shape).

Tolerance: the kernels and the references round at the same points, so their error against an fp32 computation of
the same op order is rounding-level; it must stay within REL_SLACK of the reference's own (cuBLAS) error, plus an
absolute floor of one rounding unit of the dtype. Measured (bf16): ours 2.1-4.9e-3, cuBLAS reference 2.1-5.0e-3.

Exact: the generator row (row 0) is bitwise the same with and without the verifier row, for every config (S2 at the
kernel level).
"""

import pytest
import torch
import torch.nn.functional as F
import triton

from conftest import rel_err
from tora.kernels import dual_lora as K
from tora.kernels.reference import dual_lora_gate_up_ref, dual_lora_linear_ref

pytestmark = pytest.mark.gpu

REL_SLACK = 1.5
FLOOR = {torch.bfloat16: 2**-8, torch.float16: 2**-11}
CONFIGS = [
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 512, "DEFER": 1}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 128, "DEFER": 0}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_N": 16, "BLOCK_K": 256, "DEFER": 0}, num_warps=4, num_stages=2),
]
LINEAR_SHAPES = [(2048, 1024), (1024, 1024), (1024, 3072), (2048, 6144), (1000, 777)]  # (d_out, d_in)
FFN_SHAPES = [(1024, 3072), (2048, 6144), (777, 1000)]  # (hidden, d_ff)


@pytest.fixture(params=range(len(CONFIGS)), ids=lambda i: f"cfg{i}")
def config(request):
    """Pin both kernels to one config (and forget earlier tuning) for the test, then restore the full search."""
    kernels = (K._dual_lora_linear_kernel, K._dual_lora_gate_up_kernel)
    saved = [kern.configs for kern in kernels]
    for kern in kernels:
        kern.configs = [CONFIGS[request.param]]
        kern.cache.clear()
    K._tuned.clear()
    yield CONFIGS[request.param]
    for kern, cfgs in zip(kernels, saved):
        kern.configs = cfgs
        kern.cache.clear()
    K._tuned.clear()


def lora(d_out: int, d_in: int, r: int, device, g):
    return (torch.randn(r, d_in, device=device, generator=g) * d_in**-0.5, torch.randn(d_out, r, device=device, generator=g) * 0.05)


def exact_linear(x, w, ab, scale):
    """fp32 with the op order of the reference: each row's base product rounded to the dtype, then LoRA in fp32."""
    base = (x.float() @ w.float().T).to(x.dtype).float()
    a, b = ab
    return torch.stack([base[0], base[1] + scale * b @ (a @ x[1].float())])


def check(out, ref, exact, dtype, what):
    ours, theirs = rel_err(out, exact).max().item(), rel_err(ref, exact).max().item()
    print(f"\n{what}: rel err vs fp32 {ours:.2e} (reference {theirs:.2e})")
    assert ours <= REL_SLACK * theirs + FLOOR[dtype], f"{what}: {ours:.2e} vs reference {theirs:.2e}"


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("r", [8, 16])
@pytest.mark.parametrize("d_out,d_in", LINEAR_SHAPES)
def test_k1_dual_lora_linear(d_out, d_in, r, dtype, config, device):
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(2, d_in, device=device, generator=g).to(dtype)
    w = (torch.randn(d_out, d_in, device=device, generator=g) * d_in**-0.5).to(dtype)
    ab, scale = lora(d_out, d_in, r, device, g), 2.0
    out = K.dual_lora_linear(x, w, ab, scale)
    check(out, dual_lora_linear_ref(x, w, *ab, scale), exact_linear(x, w, ab, scale), dtype, f"K1 {d_out}x{d_in} r={r} {dtype}")
    assert torch.equal(K.dual_lora_linear(x[:1], w)[0], out[0]), "the generator row depends on the verifier row"
    no_lora = K.dual_lora_linear(x, w)
    assert torch.equal(no_lora[0], out[0]) and rel_err(no_lora, F.linear(x, w).float()).max() <= REL_SLACK * FLOOR[dtype]


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("r", [8, 16])
@pytest.mark.parametrize("hidden,d_ff", FFN_SHAPES)
def test_k2_and_ffn(hidden, d_ff, r, dtype, config, device):
    g = torch.Generator(device=device).manual_seed(1)
    x = torch.randn(2, hidden, device=device, generator=g).to(dtype)
    wg, wu = ((torch.randn(d_ff, hidden, device=device, generator=g) * hidden**-0.5).to(dtype) for _ in range(2))
    wd = (torch.randn(hidden, d_ff, device=device, generator=g) * d_ff**-0.5).to(dtype)
    lg, lu, ld = lora(d_ff, hidden, r, device, g), lora(d_ff, hidden, r, device, g), lora(hidden, d_ff, r, device, g)
    scale = 2.0
    # K2 alone
    h = K.dual_lora_gate_up(x, wg, wu, lg, lu, scale)
    h_exact = F.silu(exact_linear(x, wg, lg, scale).to(dtype)).float() * exact_linear(x, wu, lu, scale).to(dtype).float()
    check(h, dual_lora_gate_up_ref(x, wg, wu, lg, lu, scale), h_exact, dtype, f"K2 {d_ff}x{hidden} r={r} {dtype}")
    assert torch.equal(K.dual_lora_gate_up(x[:1], wg, wu)[0], h[0])
    # K2 with the down partials, then K1 on them: the whole FFN
    y = K.dual_lora_ffn(x, wg, wu, wd, lg, lu, ld, scale)
    ref = dual_lora_linear_ref(dual_lora_gate_up_ref(x, wg, wu, lg, lu, scale), wd, *ld, scale)
    hr = dual_lora_gate_up_ref(x, wg, wu, lg, lu, scale)
    check(y, ref, exact_linear(hr, wd, ld, scale), dtype, f"FFN {hidden}x{d_ff} r={r} {dtype}")
    assert torch.equal(K.dual_lora_ffn(x[:1], wg, wu, wd)[0], y[0]), "the generator row depends on the verifier row"


def test_partials_sum_to_a_down_h(device):
    g = torch.Generator(device=device).manual_seed(2)
    x = torch.randn(2, 1024, device=device, generator=g).bfloat16()
    wg, wu = ((torch.randn(3072, 1024, device=device, generator=g) * 1024**-0.5).bfloat16() for _ in range(2))
    lg, lu, ld = lora(3072, 1024, 8, device, g), lora(3072, 1024, 8, device, g), lora(1024, 3072, 8, device, g)
    h, p = K.dual_lora_gate_up(x, wg, wu, lg, lu, 2.0, down_a=ld[0])
    assert p.shape[1] == 8 and p.shape[0] == triton.cdiv(3072, K._dual_lora_gate_up_kernel.best_config.kwargs["BLOCK_N"])
    torch.testing.assert_close(p.sum(0), ld[0] @ h[1].float(), rtol=1e-4, atol=1e-4)


def test_rejects_bad_inputs(device):
    x = torch.randn(2, 64, device=device).bfloat16()
    w = torch.randn(32, 64, device=device).bfloat16()
    with pytest.raises(ValueError, match="power of two"):
        K.dual_lora_linear(x, w, (torch.randn(12, 64, device=device), torch.randn(32, 12, device=device)))
    with pytest.raises(ValueError, match="1 or 2"):
        K.dual_lora_linear(torch.randn(3, 64, device=device).bfloat16(), w)
    with pytest.raises(ValueError, match="contiguous"):
        K.dual_lora_linear(x, w.T.contiguous().T)
    with pytest.raises(ValueError, match="fp32"):
        K.dual_lora_linear(x, w, (torch.randn(8, 64, device=device).bfloat16(), torch.randn(32, 8, device=device)))
    with pytest.raises(ValueError, match="every projection or for none"):
        K.dual_lora_gate_up(x, w, w, (torch.randn(8, 64, device=device), torch.randn(32, 8, device=device)), None)
