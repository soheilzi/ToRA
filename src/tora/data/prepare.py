"""Build the processed datasets and run the data checks (S6a-S6d).

    python -m tora.data.prepare [--config configs/prepare.yaml] [prepare.datasets=[aegis2,beavertails] ...]

Writes data_cache/processed/<dataset>/<split>.jsonl, results/data/<run_id>.jsonl and a length histogram.
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from tora.data import checks
from tora.data.formatting import ChatFormatter, load_tokenizer
from tora.data.loaders import load_examples
from tora.data.schema import Example
from tora.utils.config import DataConfig, ModelConfig, RunConfig, config_from_cli, load_config, to_dict
from tora.utils.hf import download_model
from tora.utils.paths import CONFIGS_ROOT, resolve
from tora.utils.runlog import RunLogger, make_run_id
from tora.utils.seeding import seed_everything


@dataclass
class PrepareSection:
    datasets: list[str] = field(default_factory=list)  # names of configs/data/<name>.yaml
    max_len: int = 2048
    n_label_checks: int = 100
    n_print: int = 5
    drop_train_in_val: bool = False  # also drop train examples whose prompt is in ANY dataset's val split (on for the paper's training mixture)


@dataclass
class PrepareConfig:
    run: RunConfig
    model: ModelConfig
    prepare: PrepareSection = field(default_factory=PrepareSection)


def data_config(name: str) -> DataConfig:
    return load_config(DataConfig, CONFIGS_ROOT / "data" / f"{name}.yaml")


def processed_path(data_dir: str, name: str, split: str) -> Path:
    return resolve(data_dir) / "processed" / name / f"{split}.jsonl"


def save_examples(examples: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for e in examples:
            fh.write(json.dumps(dataclasses.asdict(e)) + "\n")


def load_processed(data_dir: str, name: str, split: str) -> list[Example]:
    path = processed_path(data_dir, name, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m tora.data.prepare` first")
    with path.open() as fh:
        return [Example(**json.loads(line)) for line in fh]


def plot_lengths(lengths: dict[str, list[int]], max_len: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, ls in lengths.items():
        ax.hist(ls, bins=100, histtype="step", label=f"{name} (n={len(ls)})", log=True)
    ax.axvline(max_len, color="k", linestyle="--", label=f"max_len={max_len}")
    ax.set_xlabel("tokens (prompt part + response)")
    ax.set_ylabel("examples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    cfg = config_from_cli(PrepareConfig, argv, default_config=str(CONFIGS_ROOT / "prepare.yaml"))
    seed_everything(cfg.run.seed)
    p = cfg.prepare
    if not p.datasets:
        raise ValueError("prepare.datasets is empty")
    data_cfgs = {name: data_config(name) for name in p.datasets}
    fmt = ChatFormatter(load_tokenizer(download_model(cfg.model.repo_id, cfg.model.revision, cfg.run.hf_home)))
    report: dict = {}

    with RunLogger(cfg.run.phase, make_run_id(cfg.run.name), to_dict(cfg), cfg.run.seed, cfg.run.results_dir) as log:
        datasets: dict[str, list[Example]] = {}
        for name, dc in data_cfgs.items():
            exs = load_examples(dc, cfg.run.hf_home)
            if dc.val_fraction > 0:
                exs = checks.holdout_by_prompt(exs, dc.val_fraction, cfg.run.seed)
            datasets[name] = exs

        print("\n== S6b leakage: prompts shared between train/val and any test split")
        before = {
            "train_test_exact": checks.cross_split_duplicates(datasets, "train", "test", exact=True),
            "train_test_normalized": checks.cross_split_duplicates(datasets, "train", "test", exact=False),
            "val_test_normalized": checks.cross_split_duplicates(datasets, "val", "test", exact=False),
            "train_val_normalized": checks.cross_split_duplicates(datasets, "train", "val", exact=False),
        }
        datasets, removed = checks.remove_test_overlap(datasets)
        after = {
            "train_test_exact": checks.cross_split_duplicates(datasets, "train", "test", exact=True),
            "train_test_normalized": checks.cross_split_duplicates(datasets, "train", "test", exact=False),
            "val_test_normalized": checks.cross_split_duplicates(datasets, "val", "test", exact=False),
        }
        print(f"  shared prompts before: {before}\n  removed examples: {removed}\n  shared prompts after: {after}")
        if after["train_test_exact"] or after["train_test_normalized"] or after["val_test_normalized"]:
            raise AssertionError(f"S6b failed: {after}")
        report["S6b"] = {"before": before, "removed": removed, "after": after}
        if p.drop_train_in_val:
            datasets, removed = checks.remove_overlap(datasets, "val", ("train",))
            after = checks.cross_split_duplicates(datasets, "train", "val", exact=False)
            print(f"  train examples whose prompt is in any val split, removed: {removed}\n  shared train/val prompts after: {after}")
            if after:
                raise AssertionError(f"S6b train/val failed: {after} shared prompts")
            report["S6b"]["train_in_val_removed"] = removed

        print("\n== S6a per-split counts and label balance")
        report["S6a"] = {}
        for name, exs in datasets.items():
            stats = checks.split_stats(exs)
            report["S6a"][name] = stats
            print(checks.format_stats(name, stats))

        print(f"\n== S6c label positions ({p.n_label_checks} random examples per dataset)")
        shown = []
        for name, exs in datasets.items():
            records = checks.check_label_positions(fmt, exs, p.n_label_checks, cfg.run.seed)
            n_ref = checks.check_reference_tokens(fmt, exs)
            print(f"  {name}: {len(records)} examples OK" + (f"; {n_ref} equal the dataset's own input_ids" if n_ref else ""))
            shown += records
        report["S6c"] = {"checked_per_dataset": p.n_label_checks, "passed": True}
        for rec in random.Random(cfg.run.seed).sample(shown, min(p.n_print, len(shown))):
            print(f"  [{rec['id']}] prompt_end token {rec['prompt_end_token']!r} (after {rec['before_prompt_end']!r})")
            if "response_end_token" in rec:
                print(f"      response_end token {rec['response_end_token']!r}; decoded tail {rec['response_tail']!r}; text tail {rec['response_text_tail']!r}")

        print(f"\n== S6d lengths (prompt part + response, untruncated) and truncation at max_len={p.max_len}")
        lengths, report["S6d"] = {}, {}
        for name, exs in datasets.items():
            full = [fmt.render_example(e) for e in exs]
            lengths[name] = [len(r.input_ids) for r in full]
            trunc = [fmt.render_example(e, max_len=p.max_len) for e in exs]
            report["S6d"][name] = checks.length_stats(lengths[name], p.max_len) | checks.truncation_stats(trunc)
            print(f"  {name}: {report['S6d'][name]}")
        all_lengths = [x for ls in lengths.values() for x in ls]
        print(checks.text_histogram(all_lengths, [0, 128, 256, 512, 1024, 2048, 4096, 8192]))
        plot = log.path.with_suffix(".lengths.png")
        plot_lengths(lengths, p.max_len, plot)
        print(f"  histogram plot: {plot}")

        for name, exs in datasets.items():
            for split in sorted({e.split for e in exs}):
                save_examples([e for e in exs if e.split == split], processed_path(cfg.run.data_dir, name, split))
        print(f"\nprocessed data -> {resolve(cfg.run.data_dir) / 'processed'}")
        log.log(report)
    return report


if __name__ == "__main__":
    main()
