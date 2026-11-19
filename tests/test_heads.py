"""Heads and their input norms."""

import pytest
import torch

from tora.models.baselines.singprobe import tap_norm
from tora.models.heads import FeatureNorm, build_head
from tora.utils.config import HeadConfig


def test_rms_norm_matches_singprobe():
    x = torch.randn(5, 3 * 16) * torch.tensor([0.5, 2.0, 20.0]).repeat_interleave(16)
    ref = torch.cat([tap_norm(t) for t in x.split(16, dim=-1)], dim=-1)
    torch.testing.assert_close(FeatureNorm("rms", 3, 16)(x), ref)


def test_standardize_needs_fit():
    norm = FeatureNorm("standardize", 2, 8)
    x = torch.randn(1000, 16) * 7 + 3
    assert norm(x).isnan().all()  # unfitted: loud, not identity
    norm.fit(x.bfloat16())
    y = norm(x)
    assert y.mean(0).abs().max() < 0.1 and (y.std(0) - 1).abs().max() < 0.05
    with pytest.raises(ValueError):
        FeatureNorm("layer", 1, 8)


def test_build_head_shapes():
    for kind in ("linear", "mlp"):
        head = build_head(HeadConfig(kind=kind, taps=["8", "17"], norm="rms"), hidden_size=16)
        assert head(torch.randn(4, 32)).shape == (4, 2)
