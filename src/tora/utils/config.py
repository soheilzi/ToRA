"""Plain-dataclass configs loaded from YAML.

A YAML file may start with a `defaults` mapping `section: relative/path.yaml` (relative to
`configs/`). Each referenced file is loaded into that section, then the file's own values are
deep-merged on top. CLI overrides look like `model.max_seq_len=1024` (the value is parsed as YAML).

`from_dict` is strict: unknown keys and missing required keys raise, so typos never pass silently.
"""

from __future__ import annotations

import argparse
import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from tora.utils.paths import CONFIGS_ROOT

T = TypeVar("T")


@dataclass
class RunConfig:
    name: str
    phase: str
    seed: int = 0
    results_dir: str = "results"  # relative to the code root
    data_dir: str = "data_cache"  # processed datasets (relative to the code root)
    hf_home: str | None = None  # None -> $HF_HOME, else the huggingface default
    wandb: bool = False
    wandb_project: str = "tora"


@dataclass
class ModelConfig:
    repo_id: str
    revision: str | None = None
    dtype: str = "bfloat16"
    max_seq_len: int = 4096  # length of the RoPE table and the default static KV cache


@dataclass
class HeadConfig:
    kind: str = "linear"  # linear | mlp
    hidden_dim: int = 256  # mlp only
    dropout: float = 0.0  # mlp only
    taps: list[str] = field(default_factory=lambda: ["final"])  # "final" or layer indices as strings
    norm: str = "none"  # none | rms (per tap, SingProbe) | standardize (per feature, fit on training features)


@dataclass
class VerifierConfig:
    k_layers: int = 8  # adapted layers are the last K (indices L-K .. L-1)
    rank: int = 16
    alpha: float | None = None  # LoRA scale = alpha / rank; None -> 2 * rank
    kv: str = "shared"  # shared (ToRA: reads the generator's K/V) | own (ablation: own K/V, causal among verifier tokens)
    # shared only: the first `sink` positions read the verifier's own K/V instead of the generator's (attention sinks).
    # Used only by the rank ablation, which was run with sink = 4; ToRA itself has sink = 0.
    sink: int = 0
    head: HeadConfig = field(default_factory=HeadConfig)  # taps must be [final]


@dataclass
class ProbeConfig:
    """The frozen-feature control of the ablations: an MLP head on generator hidden states at the label positions,
    trained under the verifier recipe (models/probes.py). Taps are SingProbe's three layers."""

    head: HeadConfig = field(default_factory=lambda: HeadConfig(kind="mlp", hidden_dim=1024, dropout=0.1, taps=["8", "17", "26"], norm="rms"))


@dataclass
class DataConfig:
    name: str  # short id used in file names and results (e.g. "aegis2")
    loader: str  # key into tora.data.loaders.LOADERS
    repo_id: str
    splits: dict[str, str]  # source split -> unified split (train | val | test)
    revision: str | None = None
    val_fraction: float = 0.0  # > 0: hold out this much of train as val, by unique prompt, stratified
    keep_redacted: bool = False  # aegis2: keep the 'REDACTED' prompts (their text is not in the dataset)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path, configs_root: Path = CONFIGS_ROOT) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    defaults = raw.pop("defaults", None) or {}
    merged: dict[str, Any] = {}
    for section, rel in defaults.items():
        merged[section] = load_yaml(configs_root / rel, configs_root)
    return _deep_merge(merged, raw)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override must look like a.b.c=value, got {item!r}")
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {item!r}: {part!r} is not a section")
        node[leaf] = yaml.safe_load(value)
    return cfg


def from_dict(cls: type[T], data: dict, where: str | None = None) -> T:
    where = where or cls.__name__
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    if not isinstance(data, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"{where}: unknown config keys {unknown}; valid keys are {sorted(fields)}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, f in fields.items():
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], f"{where}.{name}")
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ValueError(f"{where}: missing required key {name!r}")
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value, where)
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ValueError(f"{where}: null is not allowed")
        non_none = [a for a in args if a is not type(None)]
        return _coerce(non_none[0], value, where) if len(non_none) == 1 else value
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        (inner,) = typing.get_args(tp) or (Any,)
        return [_coerce(inner, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected a mapping, got {value!r}")
        key_tp, value_tp = typing.get_args(tp) or (Any, Any)
        return {_coerce(key_tp, k, where): _coerce(value_tp, v, f"{where}[{k}]") for k, v in value.items()}
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp is str and isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if tp in (int, float, str, bool) and not isinstance(value, tp):
        raise TypeError(f"{where}: expected {tp.__name__}, got {value!r}")
    return value


def to_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg)


def load_config(cls: type[T], path: str | Path, overrides: list[str] | None = None) -> T:
    return from_dict(cls, apply_overrides(load_yaml(path), overrides or []))


def config_from_cli(cls: type[T], argv: list[str] | None = None, default_config: str | None = None) -> T:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config, required=default_config is None)
    parser.add_argument("overrides", nargs="*", help="a.b.c=value overrides")
    args = parser.parse_args(argv)
    return load_config(cls, args.config, args.overrides)
