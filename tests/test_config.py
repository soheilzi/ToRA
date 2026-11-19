from dataclasses import dataclass, field

import pytest

from tora.train.train_verifier import TrainVerifierConfig
from tora.utils.config import HeadConfig, apply_overrides, from_dict, load_config, load_yaml
from tora.utils.paths import CONFIGS_ROOT

TORA = CONFIGS_ROOT / "train" / "tora.yaml"


def test_tora_config_merges_defaults():
    cfg = load_config(TrainVerifierConfig, TORA)
    assert cfg.model.repo_id == "Qwen/Qwen3-0.6B"  # from model/qwen3_0p6b.yaml
    assert cfg.train.base.verifier.k_layers == 14
    assert cfg.train.base.verifier.head.taps == ["final"]


def test_model_configs_load():
    for name, repo in [("qwen3_0p6b", "Qwen/Qwen3-0.6B"), ("qwen3_1p7b", "Qwen/Qwen3-1.7B")]:
        assert load_yaml(CONFIGS_ROOT / "model" / f"{name}.yaml")["repo_id"] == repo


def test_cli_overrides():
    cfg = load_config(TrainVerifierConfig, TORA, ["train.epochs=7", "train.base.verifier.head.taps=[final, 8]", "run.hf_home=null"])
    assert cfg.train.epochs == 7
    assert cfg.train.base.verifier.head.taps == ["final", "8"]
    assert cfg.run.hf_home is None


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(TrainVerifierConfig, TORA, ["train.epohcs=7"])


def test_missing_required_raises():
    with pytest.raises(ValueError, match="missing required key"):
        from_dict(TrainVerifierConfig, {"model": {"repo_id": "x"}})


def test_type_mismatch_raises():
    with pytest.raises(TypeError):
        load_config(TrainVerifierConfig, TORA, ["train.epochs=lots"])


def test_bad_override_syntax():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_nested_default_factory():
    @dataclass
    class Outer:
        head: HeadConfig = field(default_factory=HeadConfig)

    assert from_dict(Outer, {}).head.kind == "linear"
    assert from_dict(Outer, {"head": {"kind": "mlp", "dropout": 0}}).head.dropout == 0.0
