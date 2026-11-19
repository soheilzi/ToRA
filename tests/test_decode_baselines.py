"""The baselines' static decode steps (inference/decode_baselines.py, the latency comparison) against their own
full-sequence forward (models/baselines), on real Aegis sequences, bf16 (their released dtype). Stepped one token at a
time, eager and compiled under a CUDA graph: every token's guard probabilities within the S4 tolerance (0.05) of the
full forward. Eager, the generator logits equal StaticDecoder's gen mode bitwise (the same generator step, so the
latency comparison times the same generator in every method). SingProbe also with its sliding window cut to 16
positions, so that the window mask is exercised (its released window, 2048, exceeds these sequences). Both also on
our kernels (given a K = 14 verifier, which only fixes L - K): the generator logits then equal fused_gen's bitwise."""

import pytest
import torch

from conftest import aegis2_sample
from tora.data.formatting import ChatFormatter, load_tokenizer
from tora.inference.decode_baselines import GuardStreamDecoder, SingProbeDecoder, teacher_forced
from tora.inference.decode_static import StaticDecoder, runner
from tora.models.baselines.qwen3guard_stream import Qwen3GuardStream
from tora.models.baselines.singprobe import RESPONSE_UNSAFE, SingProbe
from tora.models.qwen3_min import Qwen3Min
from tora.models.verifier import Verifier
from tora.utils.config import HeadConfig, VerifierConfig
from tora.utils.hf import download_model

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

PROB_ATOL = 0.05
MAX_LEN = 256


@pytest.fixture(scope="module")
def sequences():
    fmt = ChatFormatter(load_tokenizer(download_model("Qwen/Qwen3-0.6B")))
    examples = aegis2_sample(2, seed=1)
    return [fmt.render(e.prompt, e.response, max_len=160).input_ids for e in examples]


def k14(generator: Qwen3Min) -> Verifier:
    """A K = 14 verifier on the generator: on our kernels it fixes L - K (its LoRA and head are not used)."""
    return Verifier(generator.cfg, VerifierConfig(k_layers=14, rank=8, head=HeadConfig(kind="mlp"))).to(generator.device).eval()


def check(dec, generator: Qwen3Min, reference, sequences, device, verifier: Verifier | None = None) -> None:
    """reference(ids) -> the guard's probabilities [T, C] from its full forward; probs(guard logits) the same from
    the decoder's. verifier: the decoder runs on our kernels, so its generator must equal fused_gen's."""
    gen_only = StaticDecoder(generator, verifier, "gen" if verifier is None else "fused_gen", MAX_LEN)
    for exe in ("eager", "compile+graph"):
        step = runner(dec.step, exe)
        for seq in sequences:
            ids = torch.tensor([seq], device=device)
            logits, guard = teacher_forced(dec, step, ids)
            ref, got = reference(ids, None), reference(ids, guard[0])
            assert (got - ref).abs().max().item() <= PROB_ATOL, exe
            if exe == "eager":
                assert torch.equal(logits, gen_only.teacher_forced(ids)[0])


@pytest.mark.parametrize("window, kernels", [(None, False), (16, False), (None, True)], ids=["released_window", "window16", "kernels"])
@torch.no_grad()
def test_singprobe_step_matches_forward(sequences, device, window, kernels):
    torch._dynamo.reset()
    model = SingProbe.from_pretrained("inclusionAI/Qwen3-0.6B-singprobe", device=device, max_seq_len=MAX_LEN)
    if window is not None:
        model.probe.window = window

    def probs(ids, guard):
        return (model(ids)[0] if guard is None else guard)[:, RESPONSE_UNSAFE].float().sigmoid()

    verifier = k14(model.backbone) if kernels else None
    check(SingProbeDecoder(model, MAX_LEN, verifier), model.backbone, probs, sequences, device, verifier)


@pytest.mark.parametrize("kernels", [False, True], ids=["cublas", "kernels"])
@torch.no_grad()
def test_guard_stream_step_matches_forward(sequences, device, kernels):
    torch._dynamo.reset()
    generator = Qwen3Min.from_pretrained("Qwen/Qwen3-0.6B", device=device, max_seq_len=MAX_LEN, verbose=False)
    guard = Qwen3GuardStream.from_pretrained("Qwen/Qwen3Guard-Stream-0.6B", device=device, max_seq_len=MAX_LEN)

    def probs(ids, logits):
        return (guard(ids).response_risk_logits[0] if logits is None else logits).float().softmax(-1)

    verifier = k14(generator) if kernels else None
    check(GuardStreamDecoder(generator, guard, MAX_LEN, verifier), generator, probs, sequences, device, verifier)
