"""Verifier: attention paths, S3 (B = 0 gives the generator), S4 (training pass == decode_ref), S5 (skipping
layers < L-K changes nothing), S2 (generator invariance), own-KV checks, and S7 (overfits 128 examples). ToRA, plus
the attention sinks of the rank ablation on the training paths (decode_ref implements ToRA only).

Tolerances. fp32 checks are logic checks: the paths differ only in summation order (measured <= 8e-6), so
any masking, position or caching bug shows up as O(1) error against REL_TOL_FP32 / FP32_PROB_ATOL. In bf16
the paths use different attention kernels and GEMM shapes (fp32-softmax einsum, FlexAttention, SDPA), whose
roundings compound over the adapted layers, exactly like the generator's own prefill-vs-decode drift. Measured
(Qwen3-0.6B, 10 Aegis sequences, random head): S3 final-hidden rel. error 2.3% (K=8) / 4.7% (K=28), |d prob| <=
0.032; S4 |d prob| 0.048 vs decode_ref, while the generator's own prefill-vs-decode drift through the same head
is 0.040. So S4 in bf16 is self-calibrating: the verifier may drift at most BF16_DRIFT_FACTOR times the
generator's drift, measured in the same test. S2 is exact: decode_ref's generator row is the generator-only
decode call.
"""

import pytest
import torch

from conftest import aegis2_sample, rel_err
from tora.data.formatting import ChatFormatter, collate_right_padded, load_tokenizer
from tora.inference.decode_ref import DualRowDecoder
from tora.models.lora_ffn import LoRAFFN
from tora.models.qwen3_min import MLP, Qwen3Config, Qwen3Min, greedy_generate
from tora.models.verifier import Verifier, generator_captures, shared_kv_attention, shared_kv_attention_all
from tora.train.train_verifier import (
    TrainVerifierConfig,
    capture_chunks,
    chunk_accuracy,
    class_pos_weight,
    fit_full_batch,
    label_positions,
    training_sequences,
)
from tora.utils.config import HeadConfig, VerifierConfig, load_config
from tora.utils.paths import CONFIGS_ROOT

pytestmark = pytest.mark.gpu

REL_TOL_FP32 = 1e-4
FP32_PROB_ATOL = 1e-4
BF16_HIDDEN_REL_TOL = 0.06  # S3 in bf16: relative L2 of the final hidden state (generator decode vs prefill: ~4%)
BF16_PROB_ATOL = 0.05  # S3 in bf16: |delta probability| per output and position
BF16_DRIFT_FACTOR = 2.0  # S4 in bf16: verifier drift <= this x the generator's own prefill-vs-decode drift
MAX_LEN = 192


def tiny_cfg() -> Qwen3Config:
    return Qwen3Config(1000, 64, 160, 4, 4, 2, 16, 1e-6, 1e6, 512, True, None, None)


def random_lora(verifier: Verifier, std: float) -> Verifier:
    """Nonzero B so the verifier really differs from the generator."""
    with torch.no_grad():
        for name, p in verifier.lora.named_parameters():
            if name.endswith(".B"):
                p.normal_(0, std)
    return verifier


def test_lora_zero_b_is_bitwise_base_mlp(device):
    torch.manual_seed(0)
    cfg = tiny_cfg()
    mlp = MLP(cfg).to(device, torch.bfloat16)
    lora = LoRAFFN(cfg, rank=8, alpha=16.0).to(device)
    x = torch.randn(3, 7, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    assert torch.equal(lora(mlp, x), mlp(x))


def reference_attention(q, k_gen, v_gen, k_own, v_own, q_pos, scale, sink=0):
    """Per-query loop. The query at position t reads the verifier's own key at s = t and at s < sink, and the
    generator's key at every other s < t (ToRA, sink 0: generator keys < t, then its own key). k_own / v_own are
    indexed by absolute position."""
    b, hq, p, d = q.shape
    g = hq // k_gen.shape[1]
    out = torch.empty(b, hq, p, d, device=q.device)
    for i in range(b):
        for j in range(p):
            t = int(q_pos[i, j])
            own = [s for s in range(t + 1) if s < sink or s == t]
            gen = [s for s in range(t) if s not in own]
            k = torch.cat([k_gen[i, :, gen], k_own[i, :, own]], dim=1).float().repeat_interleave(g, 0)
            v = torch.cat([v_gen[i, :, gen], v_own[i, :, own]], dim=1).float().repeat_interleave(g, 0)
            att = (q[i, :, j : j + 1].float() @ k.transpose(-1, -2) * scale).softmax(-1)
            out[i, :, j] = (att @ v)[:, 0]
    return out


@pytest.mark.parametrize("sink", [0, 3], ids=["tora", "sink3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shared_kv_attention_paths(dtype, sink, device):
    torch.manual_seed(0)
    b, hq, hkv, t, d = 2, 4, 2, 37, 128
    scale = d**-0.5
    q = torch.randn(b, hq, t, d, device=device, dtype=dtype)
    k_gen, v_gen, k_self, v_self = (torch.randn(b, hkv, t, d, device=device, dtype=dtype) for _ in range(4))
    all_pos = torch.arange(t, device=device).expand(b, t)
    ref = reference_attention(q, k_gen, v_gen, k_self, v_self, all_pos, scale, sink)
    atol = 1e-5 if dtype == torch.float32 else 2e-2
    flex = shared_kv_attention_all(q, k_gen, v_gen, k_self, v_self, scale, sink).float()
    torch.testing.assert_close(flex, ref, atol=atol, rtol=0)
    one = shared_kv_attention_all(q[1:], k_gen[1:], v_gen[1:], k_self[1:], v_self[1:], scale, sink).float()  # B = 1 is padded to 2 rows
    torch.testing.assert_close(one, ref[1:], atol=atol, rtol=0)
    sinks = (k_self[:, :, :sink], v_self[:, :, :sink]) if sink else (None, None)
    dense = shared_kv_attention(q[:, :, sink:], k_gen, v_gen, k_self[:, :, sink:], v_self[:, :, sink:], all_pos[:, sink:], scale, *sinks).float()
    torch.testing.assert_close(dense, ref[:, :, sink:], atol=atol, rtol=0)
    # A subset of query positions (different per row) gives the same rows as computing every position.
    pos = torch.tensor([[sink, 5 + sink, 36], [3 + sink, 3 + sink, 20]], device=device)
    sub_q = q.gather(2, pos[:, None, :, None].expand(-1, hq, -1, d))
    sub_k, sub_v = (x.gather(2, pos[:, None, :, None].expand(-1, hkv, -1, d)) for x in (k_self, v_self))
    sub = shared_kv_attention(sub_q, k_gen, v_gen, sub_k, sub_v, pos, scale, *sinks).float()
    torch.testing.assert_close(sub, ref.gather(2, pos[:, None, :, None].expand(-1, hq, -1, d)), atol=atol, rtol=0)


@pytest.mark.parametrize("sink", [0, 4], ids=["tora", "sink4"])
def test_flex_compiles_a_bounded_number_of_graphs(sink, device):
    """Training batches vary in size and length (incl. B = 1 and T <= 128) and eval runs without grad: the flex
    attention must stay at a fixed handful of compiled graphs, or long jobs hit the recompile limit. Expected: the
    first call, one upgrade of the block-mask sizes to dynamic on the first new length, one no-grad graph."""
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    for b, t, grad in [(32, 250, True), (1, 1900, True), (64, 100, True), (1, 100, True), (7, 700, True),
                       (32, 250, False), (1, 1900, False), (200, 120, False), (3, 700, False)]:
        q = torch.randn(b, 4, t, 128, device=device, dtype=torch.bfloat16, requires_grad=grad)
        kv = [torch.randn(b, 2, t, 128, device=device, dtype=torch.bfloat16, requires_grad=grad) for _ in range(4)]
        with torch.set_grad_enabled(grad):
            out = shared_kv_attention_all(q, *kv, 128**-0.5, sink=sink)
            if grad:
                out.float().sum().backward()
        assert out.shape == q.shape
    assert counters["stats"]["unique_graphs"] <= 3, dict(counters["stats"])


@pytest.fixture(scope="module", params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def backbone(request, device):
    model = Qwen3Min.from_pretrained("Qwen/Qwen3-0.6B", device=device, dtype=request.param, max_seq_len=2048, verbose=False)
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def fmt(backbone):
    return ChatFormatter(load_tokenizer(backbone.source_path))


@pytest.fixture(scope="module")
def sequences(fmt):
    """10 real Aegis prompt + response pairs, cut to MAX_LEN tokens."""
    examples = aegis2_sample(10, seed=1)
    return [fmt.render(e.prompt, e.response, max_len=MAX_LEN) for e in examples]


def batch(sequences, fmt, device):
    ids, lengths = collate_right_padded(sequences, fmt.pad_id)
    return ids.to(device), lengths


def make_verifier(backbone, k_layers, kv="shared", std=0.0, kind="linear", sink=0):
    torch.manual_seed(0)
    vcfg = VerifierConfig(k_layers=k_layers, rank=16, kv=kv, sink=sink, head=HeadConfig(kind=kind))
    v = Verifier(backbone.cfg, vcfg).to(backbone.device)
    return random_lora(v, std).eval() if std else v.eval()


@torch.no_grad()
@pytest.mark.parametrize("k_layers,sink", [(1, 0), (8, 0), (14, 0), (28, 0), (8, 4)])
def test_s3_zero_lora_verifier_equals_generator(k_layers, sink, backbone, fmt, sequences, device):
    ids, lengths = batch(sequences, fmt, device)
    gen = backbone(ids, logits="none")
    v = make_verifier(backbone, k_layers, sink=sink)
    captures = generator_captures(backbone, ids, v.first)
    positions = label_positions(sequences).to(device)
    fp32 = backbone.dtype == torch.float32
    worst_h, worst_p = 0.0, 0.0
    for path, h in (("flex, every position", v.final_hidden(backbone, captures)), ("dense, label positions", None)):
        if h is None:
            h = v.final_hidden(backbone, captures, positions)
            ref = gen.final_hidden.gather(1, positions[..., None].expand(-1, -1, h.shape[-1]))
            rows = [h[i] for i in range(len(sequences))], [ref[i] for i in range(len(sequences))]
        else:
            rows = [h[i, : lengths[i]] for i in range(len(sequences))], [gen.final_hidden[i, : lengths[i]] for i in range(len(sequences))]
        err = max(rel_err(a, b).max().item() for a, b in zip(*rows))
        dprob = max((v.head(a).sigmoid() - v.head(b).sigmoid()).abs().max().item() for a, b in zip(*rows))  # same head on both
        print(f"\nS3 K={k_layers} sink={sink} {backbone.dtype} {path}: max rel err of final hidden {err:.2e}, max |d prob| {dprob:.2e}")
        worst_h, worst_p = max(worst_h, err), max(worst_p, dprob)
    if fp32:
        assert worst_h <= REL_TOL_FP32
    else:
        assert worst_h <= BF16_HIDDEN_REL_TOL and worst_p <= BF16_PROB_ATOL


@torch.no_grad()
def test_s4_training_pass_matches_decode_ref(backbone, fmt, sequences, device):
    v = make_verifier(backbone, 8, std=0.02)
    ids, lengths = batch(sequences, fmt, device)
    captures = generator_captures(backbone, ids, v.first)
    flex = v(backbone, captures).sigmoid()
    every = torch.arange(ids.shape[1], device=device).expand(ids.shape[0], -1)
    dense = v(backbone, captures, every).sigmoid()
    # the LoRA must matter, or this test would only re-check S3
    moved = (v.head(v.final_hidden(backbone, captures)) - v.head(backbone(ids, logits="none").final_hidden)).sigmoid().abs().max().item()
    assert moved > 0.05, f"random LoRA barely changes the verifier (max |d logit-sigmoid| {moved:.3f})"
    worst = {"flex": 0.0, "dense": 0.0}
    generator_drift = 0.0  # the generator's own prefill-vs-decode drift, read through the same head
    for i, r in enumerate(sequences):
        one = torch.tensor([r.input_ids], device=device)
        decoder = DualRowDecoder(backbone, v, max_len=len(r.input_ids))
        _, step_logits = decoder.teacher_forced(one)
        ref = step_logits[0].sigmoid()
        n = lengths[i]
        for name, scores in (("flex", flex), ("dense", dense)):
            worst[name] = max(worst[name], (scores[i, :n] - ref).abs().max().item())
        cache = backbone.new_cache(1, n)
        decoded = torch.cat([backbone(one[:, j : j + 1], cache=cache, logits="none").final_hidden[0] for j in range(n)])
        prefill = backbone(one, logits="none").final_hidden[0]
        generator_drift = max(generator_drift, (v.head(prefill).sigmoid() - v.head(decoded).sigmoid()).abs().max().item())
    print(f"\nS4 {backbone.dtype}: max |d prob| over every position of {len(sequences)} sequences vs decode_ref: {worst}; "
          f"generator prefill-vs-decode drift through the same head: {generator_drift:.2e}")
    tol = FP32_PROB_ATOL if backbone.dtype == torch.float32 else BF16_DRIFT_FACTOR * generator_drift
    assert max(worst.values()) <= tol


@torch.no_grad()
@pytest.mark.parametrize("k_layers", [8, 14])
def test_s5_skipping_early_layers_matches_computing_them(k_layers, backbone, fmt, sequences, device):
    """S5: decode_ref's verifier row (layers < L-K and layer L-K's attention taken from the generator row) equals the
    row computed through all L layers, bitwise, at every position (measured: 1,769 / 1,769 positions of the 10
    sequences in fp32 and bf16, K = 8 and 14; the same kernels see the same inputs). 3 sequences here for time."""
    v = make_verifier(backbone, k_layers, std=0.02)
    worst, exact, n = 0.0, 0, 0
    for r in sequences[:3]:
        ids = torch.tensor([r.input_ids], device=device)
        decoder = DualRowDecoder(backbone, v, max_len=ids.shape[1])
        for t in range(ids.shape[1]):
            reused = decoder.step(ids[:, t : t + 1]).verifier_logits
            full = decoder.full_verifier_row(ids[:, t : t + 1], t)[:, -1]
            exact += int(torch.equal(reused, full))
            n += 1
            worst = max(worst, (reused.sigmoid() - full.sigmoid()).abs().max().item())
    print(f"\nS5 K={k_layers} {backbone.dtype}: {exact}/{n} positions bitwise equal, max |d prob| {worst:.2e}")
    assert exact == n, f"{n - exact} positions differ (max |d prob| {worst:.2e})"


@torch.no_grad()
def test_s2_generator_logits_unchanged(backbone, fmt, sequences, device):
    if backbone.dtype != torch.bfloat16:
        pytest.skip("S2 is checked in the deployment dtype")
    v = make_verifier(backbone, 8, std=0.02)
    for r in sequences[:3]:
        ids = torch.tensor([r.input_ids], device=device)
        decoder = DualRowDecoder(backbone, v, max_len=ids.shape[1])
        logits, _ = decoder.teacher_forced(ids)
        cache = backbone.new_cache(1, ids.shape[1])
        alone = torch.stack([backbone(ids[:, i : i + 1], cache=cache, logits="last").logits[:, -1] for i in range(ids.shape[1])], 1)
        assert torch.equal(logits, alone)
        assert torch.equal(decoder.cache.k, cache.k) and torch.equal(decoder.cache.v, cache.v)


@torch.no_grad()
def test_own_kv_matches_generator_and_shared(backbone, fmt, sequences, device):
    ids, _ = batch(sequences, fmt, device)
    gen = backbone(ids, logits="none").final_hidden
    own0 = make_verifier(backbone, 8, kv="own")
    assert torch.equal(own0.final_hidden(backbone, generator_captures(backbone, ids, own0.first)), gen)  # B = 0: it is the generator
    # K = 1: no attention after the branch point, so own and shared KV are the same computation
    own1, shared1 = make_verifier(backbone, 1, kv="own", std=0.02), make_verifier(backbone, 1, std=0.02)
    captures = generator_captures(backbone, ids, own1.first)
    assert torch.equal(own1(backbone, captures), shared1(backbone, captures))


@torch.no_grad()
def test_prefill_and_generate(backbone, fmt, sequences, device):
    """prefill scores the prompt in one parallel pass: in fp32 it equals teacher-forced decode; generate() still
    produces greedy_generate's tokens (the generator calls are the same)."""
    v = make_verifier(backbone, 8, std=0.02)
    for r in sequences[:2]:
        prompt = torch.tensor([r.input_ids[: r.prompt_end + 1]], device=device)
        if backbone.dtype == torch.float32:
            pre = DualRowDecoder(backbone, v, max_len=prompt.shape[1]).prefill(prompt).verifier_logits
            _, forced = DualRowDecoder(backbone, v, max_len=prompt.shape[1]).teacher_forced(prompt)
            torch.testing.assert_close(pre.sigmoid(), forced.sigmoid(), atol=FP32_PROB_ATOL, rtol=0)
        tokens, scores = DualRowDecoder(backbone, v, max_len=prompt.shape[1] + 24).generate(prompt, 24)
        assert torch.equal(tokens, greedy_generate(backbone, prompt, 24))
        assert scores.shape[0] == prompt.shape[1] + len(tokens) - 1


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_s7_verifier_overfits_128_examples(kind, backbone, fmt, device):
    if backbone.dtype != torch.bfloat16:
        pytest.skip("S7 is run in the training dtype")
    cfg = load_config(TrainVerifierConfig, CONFIGS_ROOT / "train" / "tora.yaml").train.base
    cfg.verifier.head.kind = kind
    examples = aegis2_sample(128, seed=0)
    rendered = [fmt.render(e.prompt, e.response, max_len=2048) for e in examples]
    keep, labels, _ = training_sequences(examples, dedupe_prompts=False)
    assert keep == list(range(128))
    torch.manual_seed(0)
    v = Verifier(backbone.cfg, cfg.verifier).to(device)
    chunks = capture_chunks(backbone, v.first, rendered, labels, fmt.pad_id, batch_tokens=16384)
    fit_full_batch(v, backbone, chunks, steps=150, lr=cfg.lr, head_lr=cfg.head_lr, pos_weight=class_pos_weight(labels).to(device))
    acc = chunk_accuracy(v, backbone, chunks)
    print(f"\nS7 verifier ({kind} head, K={cfg.verifier.k_layers}, r={cfg.verifier.rank}): train accuracy prompt {acc[0]:.4f}, response {acc[1]:.4f}")
    assert min(acc) >= 0.99
