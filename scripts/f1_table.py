"""Main-results tables: unsafe-class F1 per benchmark and its average, measured through our harness, next to the
published numbers; then two checks of the decision rule and the threshold-free AUPRC.

Metric. F1 of the unsafe class (positive = harmful), x100: F1 = 2PR / (P + R) over the labeled test items. "Avg." is
the unweighted arithmetic mean of the per-benchmark F1 values of one task (prompt harm or response harm), every
benchmark counting once (SingProbe's "macro"); prompt and response are averaged separately, and a metric is only
ever averaged with itself. Subsets (WildGuard adversarial) are not benchmarks and never enter an average.
Benchmarks: every public test set of the Qwen3Guard and SingProbe reports (their union). "Avg. Q3G 7" restricts the
mean to Qwen3Guard's own 7 per task, so it compares directly with Qwen3Guard's published Avg.

Decision rules (one per method, the same on every benchmark):
  ours        score >= t, with one t per task: the F1-maximizing threshold on the pooled WildGuard + Aegis 2.0 val
              splits (the other benchmarks have no val split). Test sets never set anything.
  SingProbe   its released rule, p >= 0.5. Response: max over response tokens. Prompt: its query protocol, the
              query-risk scores averaged over a greedy generator answer (method singprobe_qwen3_0p6b_query).
  Qwen3Guard  its released rule, strict (controversial = unsafe; response: two consecutive tokens at one risk level).
Checks: ours at a fixed 0.5 (no tuning); the baselines with our rule on their continuous scores.

Seeds: mean ± sample std (ddof = 1). Published (unsafe-class F1): SingProbe tech report (arXiv 2608.30703v2) Tables
33 / 34, Qwen3-0.6B-singprobe, only where its benchmark is the same data as ours; Qwen3Guard tech report (arXiv
2510.14276v1) Tables 11 / 12, Qwen3Guard-0.6B-Stream strict (its English benchmarks are WildGuard's suite, as ours).

    python scripts/f1_table.py --baselines results/eval/baselines-<...> \
        --method "ToRA (deployed path)=results/eval/deployed_seed*-*/tora_s*" --method "ToRA (offline)=results/train/tora_seed*/tora"
Method names must not contain "=" (NAME=GLOB[,GLOB...]).
"""

import argparse
import glob
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from tora.eval.metrics import best_f1_threshold, classification_metrics, prf_at

TESTS = {  # Qwen3Guard's 7 per task first, then SingProbe's other public ones
    "response": ["wildguard", "aegis2", "beavertails", "saferlhf", "harmbench_response", "think", "xstest_response", "expguard"],
    "prompt": ["wildguard", "aegis2", "toxicchat", "openai_mod", "aegis1", "simplesafetytests", "harmbench_prompt", "xstest_prompt", "expguard"],
}
Q3G_TESTS = {task: tests[:7] for task, tests in TESTS.items()}
VALS = ["wildguard", "aegis2"]
TRAIN_SOURCES = {"wildguard", "aegis2", "beavertails", "saferlhf", "toxicchat", "aegis1", "expguard"}  # configs/train/tora.yaml
PUBLISHED = {  # (task, benchmark) -> unsafe-class F1 x100, copied from the papers' tables
    "SingProbe": {("response", "beavertails"): 82.53, ("response", "aegis2"): 78.71, ("response", "wildguard"): 75.89,
                  ("response", "xstest_response"): 84.47, ("response", "expguard"): 88.31,
                  ("prompt", "aegis2"): 84.09, ("prompt", "openai_mod"): 68.10, ("prompt", "xstest_prompt"): 87.96,
                  ("prompt", "expguard"): 86.95},
    "Qwen3Guard-Stream": {
        ("response", "wildguard"): 76.3, ("response", "aegis2"): 81.4, ("response", "beavertails"): 84.5, ("response", "saferlhf"): 62.8,
        ("response", "harmbench_response"): 83.1, ("response", "think"): 81.6, ("response", "xstest_response"): 84.8,
        ("prompt", "wildguard"): 87.1, ("prompt", "aegis2"): 84.9, ("prompt", "toxicchat"): 72.0, ("prompt", "openai_mod"): 68.3,
        ("prompt", "aegis1"): 85.2, ("prompt", "simplesafetytests"): 98.0, ("prompt", "harmbench_prompt"): 97.2,
    },
}
PUBLISHED_AVG = {"SingProbe": {"response": (83.56, 8), "prompt": (84.00, 6)}, "Qwen3Guard-Stream": {"response": (79.2, 7), "prompt": (86.3, 7)}}
BASELINES = {  # name -> {task: (eval method, official decision key)}
    "SingProbe": {"response": ("singprobe_qwen3_0p6b", "response"), "prompt": ("singprobe_qwen3_0p6b_query", "prompt")},
    "Qwen3Guard-Stream": {"response": ("qwen3guard_stream_0p6b", "response_strict"), "prompt": ("qwen3guard_stream_0p6b", "prompt_strict")},
}


def load(path: str, task: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Labels, continuous scores and official decisions of the items labeled for `task`."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    rows = [r for r in rows if r[f"{task}_label"] is not None]
    return np.array([r[f"{task}_label"] for r in rows]), np.array([r[f"{task}_score"] for r in rows]), [r["decisions"] for r in rows]


def val_threshold(path: Callable[[str, str], str], task: str) -> float:
    val = [load(path(d, "val"), task) for d in VALS]
    return best_f1_threshold(np.concatenate([v[0] for v in val]), np.concatenate([v[1] for v in val]))


def score_run(path: Callable[[str, str, str], str], official: dict | None = None) -> dict:
    """{(task, dataset): {main, check, auprc}} for one run; path(task, dataset, split) -> its score file. `main` is
    F1 under the method's own rule (ours: val threshold; a baseline: its official decision key per task), `check`
    the other rule. AUPRC needs both classes, so it is None on all-harmful benchmarks."""
    out = {}
    for task, tests in TESTS.items():
        t = val_threshold(lambda d, split: path(task, d, split), task)
        for d in tests:
            y, s, dec = load(path(task, d, "test"), task)
            val_f1, auprc = prf_at(y, s, t)["f1"], classification_metrics(y, s).get("auprc")
            if official is None:
                out[(task, d)] = {"main": val_f1, "check": prf_at(y, s, 0.5)["f1"], "auprc": auprc}
            else:
                flags = np.array([x[official[task]] for x in dec], dtype=float)
                out[(task, d)] = {"main": prf_at(y, flags, 0.5)["f1"], "check": val_f1, "auprc": auprc}
    return out


def baseline_file(dirs: list[str], method: str, dataset: str, split: str) -> str:
    hits = [f for d in dirs for f in glob.glob(f"{d}/{method}__{dataset}__{split}.jsonl")]
    if len(hits) != 1:
        raise SystemExit(f"{method} {dataset}/{split}: {len(hits)} score files in {dirs}, expected 1")
    return hits[0]


def fmt(vals: list[float | None]) -> str:
    if any(v is None for v in vals):
        return "–"
    return f"{100 * np.mean(vals):.1f}" + (f"±{100 * np.std(vals, ddof=1):.1f}" if len(vals) > 1 else "")


def table(methods: dict, task: str, key: str, title: str, published: bool) -> None:
    cols, groups = TESTS[task], (TESTS[task], Q3G_TESTS[task])
    print(f"\n{title}\n\n| Method | seeds | " + " | ".join(cols) + " | Avg. | Avg. Q3G 7 |\n" + "|---" * (len(cols) + 4) + "|")
    for name, runs in methods.items():
        cells = [fmt([r[(task, d)][key] for r in runs]) for d in cols]
        for group in groups:
            defined = [d for d in group if runs[0][(task, d)][key] is not None]  # AUPRC: not on all-harmful benchmarks
            cells.append(fmt([np.mean([r[(task, d)][key] for d in defined]) for r in runs]))  # Avg. per seed, then mean ± std
        print(f"| {name} | {len(runs)} | " + " | ".join(cells) + " |")
    if published:
        for name, cells in PUBLISHED.items():
            vals = [[cells.get((task, d)) for d in group] for group in groups]
            avgs = [f"{np.mean(v):.1f}" if all(x is not None for x in v) else "–" for v in vals]
            full, n = PUBLISHED_AVG[name][task]
            print(f"| {name}, published | | " + " | ".join(f"{v:.1f}" if v is not None else "–" for v in vals[0])
                  + f" | {avgs[0]} | {avgs[1]} (paper's own Avg. over {n} benchmarks: {full:.1f}) |")


def split_table(methods: dict, task: str) -> None:
    """Main-rule Avg. F1 over the benchmarks whose official train split is in the training mixture vs the rest
    (no training data from their source)."""
    seen = [d for d in TESTS[task] if d in TRAIN_SOURCES]
    held = [d for d in TESTS[task] if d not in TRAIN_SOURCES]
    print(f"\nMain-rule Avg. F1 by whether the benchmark's train split is in the training mixture\n\n"
          f"| Method | seeds | train split used ({', '.join(seen)}) | no train data ({', '.join(held)}) |\n|---|---|---|---|")
    for name, runs in methods.items():
        cells = [fmt([np.mean([r[(task, d)]["main"] for d in group]) for r in runs]) for group in (seen, held)]
        print(f"| {name} | {len(runs)} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", nargs="+", required=True, help="eval run dirs with the released baselines' score files")
    parser.add_argument("--method", action="append", default=[], help="NAME=GLOB[,GLOB...] of run_dir/variant prefixes (seeds)")
    args = parser.parse_args()

    ours: dict[str, list[dict]] = {}
    for spec in args.method:
        name, _, globs = spec.partition("=")
        prefixes = sorted(p.removesuffix("__wildguard__test.jsonl") for g in globs.split(",") for p in glob.glob(g + "__wildguard__test.jsonl"))
        if not prefixes:
            raise SystemExit(f"{name}: no run matches {globs}")
        ours[name] = [score_run(lambda task, d, split, p=p: f"{p}__{d}__{split}.jsonl") for p in prefixes]
    base = {
        name: [score_run(lambda task, d, split, m=m: baseline_file(args.baselines, m[task][0], d, split), {t: m[t][1] for t in m})]
        for name, m in BASELINES.items()
    }

    for task in TESTS:
        print(f"\n## {task} harm (unsafe-class F1 x100 unless noted)")
        table(ours | base, task, "main", "Main: ours at the val-chosen threshold, baselines by their released rule (all measured)", True)
        split_table(ours | base, task)
        table(ours, task, "check", "Check: ours at a fixed 0.5", False)
        table(base, task, "check", "Check: baselines at our val-chosen threshold on their continuous scores", False)
        table(ours | base, task, "auprc", "AUPRC x100 (threshold-free; Avg. = unweighted mean over the benchmarks with both classes)", False)


if __name__ == "__main__":
    main()
