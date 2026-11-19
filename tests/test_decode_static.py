"""Static decode step (inference/decode_static.py) against decode_ref, on real Aegis sequences, K = 14.

Tolerances as in test_verifier.py. fp32 is the logic check: every mode equals decode_ref up to summation order
(measured <= 5e-6 in logits and probabilities). In bf16 the static step uses FlexAttention's decode kernel where
decode_ref uses SDPA, and batched mode M = 2 GEMMs, so the rows drift like the generator's own prefill-vs-decode drift
(measured with the trained K = 14 checkpoint: |d prob| <= 0.039 vs decode_ref; generator logits 3.5-5.6% relative).
Prefilling the prompt in one generator call and then stepping matches stepping from position 0 (same tolerances).
Exact: a CUDA-graph replay equals the eager step (every dtype), and the generator row is unchanged by the verifier row
(S2): in separate mode (vs gen, every dtype) and in fused mode (vs fused_gen, the same Triton kernels on the generator
row alone) in bf16, the deployment dtype. With bf16 / fp16 inputs every product in the kernels is exact in fp32, so
the one- and two-row kernel builds sum identical terms in the same order; with fp32 inputs they need not (whether
the compiler fuses a multiply into the add then changes the rounding), and measured they differ (rel. 2-3e-6).
batched mode's M = 2 cuBLAS GEMMs keep S2 in no dtype (cuBLAS is not batch-invariant: M = 1 and M = 2 pick different kernels).
fused's S2 is by construction in the GEMV kernels but measured for the whole step: its attention (FlexAttention, one
query row vs two) runs kernels inductor generates per shape (decode_static._attention). Each dtype therefore starts
from a fresh compile state, as a deployment process with one model does: after the fp32 model had been compiled in
the same process, the bf16 generator row differed from the second KV block on.

Compiled (torch.compile of the whole step, bf16; the benchmark's fastest execution): inductor keeps intermediates in
fp32 inside its fused kernels and fuses the one- and two-row graphs differently, so S2 is not bitwise there. The
test bounds both drifts instead: against decode_ref (measured with the trained checkpoint: compiled gen alone 4.0-7.6%
in logits, every mode 4.0-9.3%; verifier |d prob| <= 0.049), and fused vs fused_gen compiled, which must not exceed
what compiling itself moves fused_gen (compiled vs eager).
"""

import pytest
import torch

from conftest import aegis2_sample, rel_err
from tora.bench.timing import cuda_graph
from tora.data.formatting import ChatFormatter, load_tokenizer
from tora.inference.decode_ref import DualRowDecoder
from tora.inference import decode_static
from tora.inference.decode_static import MODES, StaticDecoder
from tora.models.qwen3_min import Qwen3Min
from tora.models.verifier import Verifier
from tora.utils.config import HeadConfig, VerifierConfig

pytestmark = pytest.mark.gpu

FP32_ATOL = 1e-4
BF16_PROB_ATOL = 0.05
BF16_LOGIT_REL_TOL = 0.08
COMPILED_LOGIT_REL_TOL = 0.12


@pytest.fixture(scope="module", params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def backbone(request, device):
    torch._dynamo.reset()  # a fresh compile state per dtype (module docstring)
    decode_static._compiled_flex = None
    model = Qwen3Min.from_pretrained("Qwen/Qwen3-0.6B", device=device, dtype=request.param, max_seq_len=1024, verbose=False)
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def sequences(backbone):
    fmt = ChatFormatter(load_tokenizer(backbone.source_path))
    examples = aegis2_sample(2, seed=1)
    return [fmt.render(e.prompt, e.response, max_len=160).input_ids for e in examples]


@pytest.fixture(scope="module")
def verifier(backbone):
    torch.manual_seed(0)
    v = Verifier(backbone.cfg, VerifierConfig(k_layers=14, rank=8, head=HeadConfig(kind="mlp"))).to(backbone.device)
    with torch.no_grad():
        for name, p in v.lora.named_parameters():
            if name.endswith(".B"):
                p.normal_(0, 0.02)
    return v.eval()


@pytest.mark.slow
@torch.no_grad()
def test_static_step_matches_decode_ref(backbone, verifier, sequences, device):
    fp32 = backbone.dtype == torch.float32
    for seq in sequences:
        ids = torch.tensor([seq], device=device)
        ref_logits, ref_v = DualRowDecoder(backbone, verifier, max_len=ids.shape[1]).teacher_forced(ids)
        out = {}
        for mode in MODES:
            decoder = StaticDecoder(backbone, None if mode == "gen" else verifier, mode, max_len=ids.shape[1])
            out[mode] = decoder.teacher_forced(ids)
            graphed = decoder.teacher_forced(ids, step=cuda_graph(decoder.step))
            assert torch.equal(out[mode][0], graphed[0]), f"{mode}: graph replay differs from eager"
            assert out[mode][1] is None or torch.equal(out[mode][1], graphed[1])
        assert torch.equal(out["separate"][0], out["gen"][0]), "separate mode changed the generator row (S2)"
        if not fp32:
            assert torch.equal(out["fused"][0], out["fused_gen"][0]), "the fused kernels' verifier row changed the generator row (S2)"
        for mode in MODES:
            logit_err = rel_err(out[mode][0][0], ref_logits[0]).max().item()
            prob_err = 0.0 if out[mode][1] is None else (out[mode][1].sigmoid() - ref_v.sigmoid()).abs().max().item()
            print(f"\n{backbone.dtype} {mode}: generator logits rel err {logit_err:.2e}, verifier |d prob| {prob_err:.2e}")
            assert logit_err <= (FP32_ATOL if fp32 else BF16_LOGIT_REL_TOL)
            assert prob_err <= (FP32_ATOL if fp32 else BF16_PROB_ATOL)


@pytest.mark.slow
@torch.no_grad()
def test_compiled_step_matches_decode_ref(backbone, verifier, sequences, device):
    if backbone.dtype != torch.bfloat16:
        pytest.skip("the compiled step is checked in the deployment dtype")
    ids = torch.tensor([sequences[0]], device=device)
    ref_logits, ref_v = DualRowDecoder(backbone, verifier, max_len=ids.shape[1]).teacher_forced(ids)
    out = {}
    for mode in MODES:
        decoder = StaticDecoder(backbone, None if mode == "gen" else verifier, mode, max_len=ids.shape[1])
        torch._dynamo.reset()
        out[mode] = decoder.teacher_forced(ids, step=torch.compile(decoder.step, fullgraph=True))
    eager = StaticDecoder(backbone, verifier, "fused_gen", max_len=ids.shape[1]).teacher_forced(ids)[0]
    s2_gap = rel_err(out["fused"][0][0], out["fused_gen"][0][0]).max().item()
    compile_gap = rel_err(out["fused_gen"][0][0], eager[0]).max().item()
    print(f"\ncompiled fused vs fused_gen generator logits rel {s2_gap:.2e}; compiled vs eager fused_gen {compile_gap:.2e}")
    assert s2_gap <= compile_gap, "compiled, the verifier row moves the generator row more than compiling does"
    for mode in MODES:
        logit_err = rel_err(out[mode][0][0], ref_logits[0]).max().item()
        prob_err = 0.0 if out[mode][1] is None else (out[mode][1].sigmoid() - ref_v.sigmoid()).abs().max().item()
        print(f"\ncompiled {mode}: generator logits rel err {logit_err:.2e}, verifier |d prob| {prob_err:.2e}")
        assert logit_err <= COMPILED_LOGIT_REL_TOL and prob_err <= BF16_PROB_ATOL


@pytest.mark.slow
@torch.no_grad()
def test_prefill_then_step_matches_teacher_forced(backbone, verifier, sequences, device):
    """prefill (the first half in one generator call, as deployed and as eval/fused_scorer.py runs) and then one fused
    step per token gives the verifier scores of stepping every token from position 0, up to the generator's own
    prefill-vs-decode drift in bf16."""
    ids = torch.tensor([sequences[0]], device=device)
    p = ids.shape[1] // 2
    decoder = StaticDecoder(backbone, verifier, "fused", max_len=ids.shape[1])
    ref = decoder.teacher_forced(ids)[1][0, p:]
    decoder.prefill(ids[:, :p])
    got = []
    for t in range(p, ids.shape[1]):
        decoder.pos.fill_(t)
        decoder.token.copy_(ids[:, t : t + 1])
        got.append(decoder.step()[1][0].clone())
    err = (torch.stack(got).sigmoid() - ref.sigmoid()).abs().max().item()
    print(f"\nprefill + step vs stepped from 0: verifier |d prob| {err:.2e}")
    assert err <= (FP32_ATOL if backbone.dtype == torch.float32 else BF16_PROB_ATOL)


def test_static_decoder_rejects_bad_setups(backbone, verifier):
    with pytest.raises(ValueError, match="mode gen"):
        StaticDecoder(backbone, verifier, "gen", max_len=16)
    with pytest.raises(ValueError, match="need one"):
        StaticDecoder(backbone, None, "batched", max_len=16)
    with pytest.raises(ValueError, match="RoPE"):
        StaticDecoder(backbone, verifier, "batched", max_len=backbone.max_seq_len + 1)
    with pytest.raises(ValueError, match="no room"):
        StaticDecoder(backbone, verifier, "batched", max_len=16).fill_random(16)
    with pytest.raises(ValueError, match="1 to max_len - 1"):
        StaticDecoder(backbone, verifier, "batched", max_len=16).prefill(torch.zeros(1, 16, dtype=torch.long, device=backbone.device))
