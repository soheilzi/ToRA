"""Reimplemented baselines: loading, causality, padding invariance. Accuracy vs. published numbers is S8
(run_eval with configs/eval/baselines.yaml), not a unit test."""

import pytest
import torch

from tora.data.formatting import ChatFormatter, collate_right_padded, load_tokenizer
from tora.models.baselines.qwen3guard_stream import Qwen3GuardStream
from tora.models.baselines.singprobe import SingProbe, SingProbeAttn
from tora.utils.hf import download_model

pytestmark = pytest.mark.gpu

PAIRS = [("How do I bake bread?", "Mix flour, water, salt and yeast, then knead."), ("Hi", "Hello! How can I help?")]


@pytest.fixture(scope="module")
def fmt():
    return ChatFormatter(load_tokenizer(download_model("Qwen/Qwen3-0.6B")))


def padded_matches_single(model_fn, fmt, device):
    seqs = [fmt.render(p, r).input_ids for p, r in PAIRS]
    ids, lengths = collate_right_padded(seqs, fmt.pad_id)
    batch = model_fn(ids.to(device)).float()
    for i, s in enumerate(seqs):
        single = model_fn(torch.tensor([s], device=device)).float()[0]
        torch.testing.assert_close(batch[i, : lengths[i]], single, atol=0.15, rtol=0.05)  # bf16, different GEMM shapes


def test_singprobe_attn_window_and_causality(device):
    torch.manual_seed(0)
    probe = SingProbeAttn(in_dim=48, num_heads=4, head_dim=8, num_labels=10, window=None).to(device)
    x = torch.randn(2, 40, 48, device=device)
    full = probe(x)
    probe.window = 64  # window >= T: same as full causal
    torch.testing.assert_close(probe(x), full)
    x2 = x.clone()
    x2[:, 30:] = torch.randn_like(x2[:, 30:])
    torch.testing.assert_close(probe(x2)[:, :30], full[:, :30])  # causal
    probe.window = 8  # banded: position t sees t-7..t only
    banded = probe(x)
    x3 = x.clone()
    x3[:, :20] = torch.randn_like(x3[:, :20])
    torch.testing.assert_close(probe(x3)[:, 27:], banded[:, 27:])


def test_singprobe_loads_and_pads(device, fmt):
    model = SingProbe.from_pretrained("inclusionAI/Qwen3-0.6B-singprobe", device=device, max_seq_len=512)
    assert model.taps == [8, 17, 26]
    ids = torch.tensor([fmt.render(*PAIRS[0]).input_ids], device=device)
    assert model(ids).shape == (1, ids.shape[1], 10)
    padded_matches_single(model, fmt, device)


def test_qwen3guard_stream_loads_and_pads(device, fmt):
    model = Qwen3GuardStream.from_pretrained("Qwen/Qwen3Guard-Stream-0.6B", device=device, max_seq_len=512)
    ids = torch.tensor([fmt.render(*PAIRS[0]).input_ids], device=device)
    out = model(ids)
    assert out.response_risk_logits.shape == out.query_risk_logits.shape == (1, ids.shape[1], 3)
    padded_matches_single(lambda x: model(x).response_risk_logits, fmt, device)



def test_mlp_probe_reads_singprobe_taps(device, fmt):
    """The frozen-feature probe (models/probes.py) reads the residual streams entering SingProbe's tap layers: its
    forward on the verifier trainer's generator captures equals its head on the backbone's hidden states."""
    from tora.models.probes import GeneratorProbe
    from tora.models.verifier import generator_captures
    from tora.utils.config import ProbeConfig

    model = SingProbe.from_pretrained("inclusionAI/Qwen3-0.6B-singprobe", device=device, max_seq_len=512)
    pcfg = ProbeConfig()
    assert [int(t) for t in pcfg.head.taps] == list(model.taps)
    torch.manual_seed(0)
    probe = GeneratorProbe(model.backbone.cfg, pcfg).to(device).eval()
    ids, lengths = collate_right_padded([fmt.render(p, r).input_ids for p, r in PAIRS], fmt.pad_id)
    ids = ids.to(device)
    with torch.no_grad():
        hs = model.backbone(ids, logits="none", output_hidden_states=True).hidden_states
        ref = probe.head(torch.cat([hs[t] for t in model.taps], dim=-1))
        ours = probe(model.backbone, generator_captures(model.backbone, ids, probe.first))
        positions = torch.tensor([[0, n - 1] for n in lengths.tolist()], device=device)
        picked = probe(model.backbone, generator_captures(model.backbone, ids, probe.first), positions)
    for i, n in enumerate(lengths.tolist()):
        assert torch.equal(ours[i, :n], ref[i, :n])
        torch.testing.assert_close(picked[i], ref[i, [0, n - 1]], atol=1e-5, rtol=0)  # fp32 GEMMs of another shape
