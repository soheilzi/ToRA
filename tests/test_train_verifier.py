"""Verifier trainer bookkeeping (CPU): label rows, sweep expansion, batching, loss weighting, per-token labels."""

import math

import numpy as np
import pytest
import torch

from tora.data.formatting import Rendered
from tora.data.schema import Example
from tora.models.baselines.qwen3guard_stream import CONTROVERSIAL, SAFE, UNSAFE, first_unsafe_token, official_response_decision
from tora.train.train_verifier import (
    TrainVerifierConfig,
    epoch_batches,
    expand_variants,
    micro_batches,
    prefix_loss,
    response_spans,
    sample_prefix_positions,
    teacher_onsets,
    training_sequences,
    verifier_loss,
)
from tora.utils.config import load_config
from tora.utils.paths import CONFIGS_ROOT


def test_training_sequences_dedupe_prompt_labels():
    exs = [
        Example("a", "p1", "r1", 1, 0, "unit", "train"),
        Example("b", "p1", "r2", 0, 1, "unit", "train"),  # duplicate prompt, conflicting label: no prompt label
        Example("c", "p2", None, 0, None, "unit", "train"),
        Example("d", "p1", None, 1, None, "unit", "train"),  # duplicate prompt-only: no label left -> dropped
        Example("e", "p3", "r3", None, 1, "unit", "train"),
    ]
    keep, labels, conflicts = training_sequences(exs, dedupe_prompts=True)
    assert keep == [0, 1, 2, 4] and conflicts == 1
    nan = math.nan
    torch.testing.assert_close(labels, torch.tensor([[1, 0], [nan, 1], [0, nan], [nan, 1]]), equal_nan=True)
    keep, labels, conflicts = training_sequences(exs, dedupe_prompts=False)
    assert keep == [0, 1, 2, 3, 4] and conflicts == 0 and labels[:, 0].isnan().tolist() == [False] * 4 + [True]  # "e" has none


def test_sweep_expansion():
    cfg = load_config(TrainVerifierConfig, CONFIGS_ROOT / "train" / "tora.yaml", ["train.variants={}"])
    cfg.train.sweep = {"verifier.k_layers": [14, 28], "verifier.rank": [4, 8], "verifier.head.kind": ["linear", "mlp"]}
    variants = expand_variants(cfg.train)
    assert len(variants) == 8 and "k28_r8_mlp" in variants
    v = variants["k14_r4_linear"]
    assert (v.verifier.k_layers, v.verifier.rank, v.verifier.head.kind, v.lr) == (14, 4, "linear", cfg.train.base.lr)
    cfg.train.sweep = {"verifier.nope": [1]}
    with pytest.raises(ValueError, match="not in train.base"):
        expand_variants(cfg.train)
    cfg.train.sweep = {}
    assert list(expand_variants(cfg.train)) == ["base"]


def test_epoch_batches_cover_every_sequence_once():
    lengths = list(np.random.default_rng(0).integers(5, 300, size=1000))
    batches = epoch_batches(lengths, batch_tokens=2048, pool_size=128, rng=np.random.default_rng(1))
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(1000))
    assert all(len(b) * max(lengths[i] for i in b) <= 2048 or len(b) == 1 for b in batches)


def test_micro_batch_losses_add_up_to_the_batch_loss():
    torch.manual_seed(0)
    lengths = [10, 20, 30, 40, 50, 60]
    idx = list(range(6))
    logits = torch.randn(6, 2, 2)
    y = torch.tensor([[1, 0], [0, math.nan], [math.nan, 1], [1, 1], [0, 0], [1, math.nan]])
    pos_weight = torch.tensor([1.5, 3.0])
    full = verifier_loss(logits, y, pos_weight)
    parts = micro_batches(idx, lengths, 70)
    assert len(parts) > 2 and [i for p in parts for i in p] == idx
    n_valid = int((~y.isnan()).sum())
    acc = sum(verifier_loss(logits[p], y[p], pos_weight) * int((~y[p].isnan()).sum()) / n_valid for p in parts)
    torch.testing.assert_close(acc, full)
    assert micro_batches(idx, lengths, None) == [idx]
    assert micro_batches(idx, lengths, None, 4) == [[0, 1, 2, 3], [4, 5]]
    assert micro_batches(idx, lengths, 70, 1) == [[i] for i in idx]
    assert micro_batches(idx, lengths, 1000, 2) == [[0, 1], [2, 3], [4, 5]]


def test_named_variants_merge_onto_base():
    cfg = load_config(TrainVerifierConfig, CONFIGS_ROOT / "train" / "tora.yaml",
                      ["train.variants={probe: {probe: {}, head_lr: 3.0e-4}, safe: {prefix_labels: safe}}"])
    variants = expand_variants(cfg.train)
    probe, safe = variants["probe"], variants["safe"]
    assert (probe.probe.head.kind, probe.probe.head.taps, probe.head_lr, probe.lr) == ("mlp", ["8", "17", "26"], 3e-4, cfg.train.base.lr)
    assert safe.probe is None and (safe.prefix_labels, safe.verifier.k_layers, safe.verifier.head.kind) == ("safe", 14, "mlp")
    cfg.train.sweep = {"lr": [1e-3]}
    with pytest.raises(ValueError, match="not both"):
        expand_variants(cfg.train)


def test_prefix_positions_and_loss():
    rendered = [Rendered(list(range(20)), prompt_end=5, response_end=15), Rendered(list(range(8)), prompt_end=5, response_end=7),
                Rendered(list(range(6)), prompt_end=5, response_end=None)]
    labels = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, math.nan]])
    pre = sample_prefix_positions(rendered, labels, 4, np.random.default_rng(0))
    assert pre.shape == (3, 4) and all(6 <= p < 15 for p in pre[0].tolist()) and len(set(pre[0].tolist())) == 4
    assert pre[1].tolist() == [6, -1, -1, -1] and pre[2].tolist() == [-1] * 4  # before the last token only; no response
    logits, y, valid, pw = torch.zeros(3, 4), labels[:, 1], pre >= 0, torch.ones(2)
    bce0 = math.log(2.0)  # BCE of logit 0 against either label
    assert prefix_loss(logits, y, valid, "safe", 1.0, pw).item() == pytest.approx(bce0)  # one unit of weight, the safe response only
    assert prefix_loss(logits, y, valid, "none", 1.0, pw).item() == 0.0
    # prefix_positions = -1: every candidate, read off a [B, T] row of all positions; equals sampling them all
    spans = response_spans(rendered, labels)
    assert spans.tolist() == [[6, 15], [6, 7], [0, 0]]
    every = torch.arange(20).expand(3, -1)
    every_valid = (every >= spans[:, :1]) & (every < spans[:, 1:])
    assert [every[i][every_valid[i]].tolist() for i in range(3)] == [list(range(6, 15)), [6], []]
    full = sample_prefix_positions(rendered, labels, 20, np.random.default_rng(0))
    assert [r[r >= 0].tolist() for r in full] == [every[i][every_valid[i]].tolist() for i in range(3)]
    torch.manual_seed(0)
    dense = torch.randn(3, 20)
    sampled = dense.gather(1, full.clamp(min=0))
    assert prefix_loss(dense, y, every_valid, "safe", 1.0, pw).item() == pytest.approx(prefix_loss(sampled, y, full >= 0, "safe", 1.0, pw).item())
    onset = torch.tensor([9, -1, -1])
    assert prefix_loss(dense, y, every_valid, "teacher", 1.0, pw, every, onset).item() == pytest.approx(
        prefix_loss(sampled, y, full >= 0, "teacher", 1.0, pw, full, onset).item())


def test_teacher_prefix_labels():
    """Safe responses -> every prefix 0; unsafe -> 1 from the teacher's onset on, 0 before; unsafe and never flagged
    -> every prefix 0; the last token keeps y."""
    rendered = [Rendered(list(range(20)), prompt_end=5, response_end=15)] * 3
    labels = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    onsets = {"a": 3, "b": -1, "c": 7}  # relative: "a" flags at position 9; "c" is safe, so its onset is ignored
    onset = teacher_onsets(rendered, labels, ["a", "b", "c"], onsets)
    assert onset.tolist() == [9, -1, -1]
    assert teacher_onsets(rendered[:1], labels[:1], ["a"], {"a": 9}).tolist() == [-1]  # at the last token: too late
    with pytest.raises(ValueError, match="no teacher onset"):
        teacher_onsets(rendered[:1], labels[:1], ["x"], onsets)
    prefix = torch.tensor([[6, 8, 10, 12]] * 3)
    valid, y, pw = prefix >= 0, labels[:, 1], torch.ones(2)
    logits = torch.ones(3, 4)
    pos, neg = math.log1p(math.exp(-1.0)), math.log1p(math.exp(1.0))  # BCE at logit 1 against label 1 / 0
    expect = (2 * neg + 2 * pos) / 4 + neg + neg  # "a": 0 0 1 1; "b": all 0; "c": all 0
    assert prefix_loss(logits, y, valid, "teacher", 1.0, pw, prefix, onset).item() == pytest.approx(expect)
    with pytest.raises(ValueError, match="needs the prefix positions"):
        prefix_loss(logits, y, valid, "teacher", 1.0, pw)


def test_first_unsafe_token_is_where_the_stream_stops():
    S, U, C = SAFE, UNSAFE, CONTROVERSIAL
    assert first_unsafe_token([S, U, S, C, C, U, U]) == (3, C)  # earliest pair, in token order (not Unsafe first)
    assert first_unsafe_token([S, U, C, U, S]) == (-1, S)  # mixed pairs do not count, as in the released rule
    assert first_unsafe_token([U, U]) == (0, U)
    assert official_response_decision([S, U, S, C, C, U, U]) == (U, 6)  # the official verdict searches Unsafe first
