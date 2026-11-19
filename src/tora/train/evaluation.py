"""Shared evaluation helpers of the trainer: the validation criterion for checkpoint selection (mean AUPRC), the test
metrics per (dataset, subset, task), mean / std over seeds and the printed summary.

Test metrics are as in run_eval: AUROC, AUPRC, FPR@95TPR, and F1 at the threshold that maximizes F1 on that
dataset's val split.
"""

from __future__ import annotations

import numpy as np

from tora.data.schema import Example
from tora.eval import metrics as M
from tora.eval.run_eval import task_metrics
from tora.eval.scorers import Scored
from tora.utils.config import ModelConfig

TASKS = ("prompt", "response")  # index = head output (HEAD_OUTPUTS)


def model_tag(model: ModelConfig) -> str:
    """The backbone a checkpoint was trained on, as stored in it."""
    return model.repo_id.replace("/", "__") + (f"@{model.revision}" if model.revision else "")


def mean_auprc(splits: list[list[tuple[Example, Scored]]]) -> float:
    """Mean AUPRC over (split, task) cells that have both classes (the val selection criterion)."""
    cells = []
    for scored in splits:
        for task in TASKS:
            pairs = [(getattr(e, f"{task}_label"), getattr(s, f"{task}_score")) for e, s in scored]
            pairs = [(lab, sc) for lab, sc in pairs if lab is not None]
            m = M.classification_metrics(*zip(*pairs)) if pairs else {}
            if "auprc" in m:
                cells.append(m["auprc"])
    if not cells:
        raise ValueError("no val split with both classes; cannot select an epoch")
    return float(np.mean(cells))


def evaluate_scored(scored: dict[tuple[str, str], list[tuple[Example, Scored]]], eval_datasets: list[str], subsets: dict[str, str]) -> list[dict]:
    """Test metrics per (dataset, subset, task); F1 at the threshold chosen on the dataset's full val split."""
    out = []
    for name in eval_datasets:
        test, val = scored[(name, "test")], scored.get((name, "val"))
        groups: list[tuple[str, list]] = [("all", test)]
        if name in subsets:
            key = subsets[name]
            groups += [(f"{key}={v}", [(e, s) for e, s in test if e.meta.get(key) is v]) for v in (True, False)]
        for subset, pairs in groups:
            for task in TASKS:
                res = task_metrics(task, pairs, val)
                if res is not None:
                    out.append({"dataset": name, "subset": subset, "task": task, **res})
    return out


SUMMARY_KEYS = ("f1@val", "auroc", "auprc", "fpr@95tpr")


def flat_metrics(rec: dict) -> dict[str, float]:
    out = {k: rec[k] for k in ("auroc", "auprc", "fpr@95tpr") if k in rec}
    if "at_val_threshold" in rec:
        out["f1@val"] = rec["at_val_threshold"]["f1"]
    return out


def summarize(per_seed: list[list[dict]]) -> list[dict]:
    """Mean / std (ddof=1) over seeds for each (dataset, subset, task)."""
    out = []
    for i, first in enumerate(per_seed[0]):
        vals = [flat_metrics(run[i]) for run in per_seed]
        rec = {k: first[k] for k in ("dataset", "subset", "task", "n")}
        for key in SUMMARY_KEYS:
            xs = [v[key] for v in vals if key in v]
            if xs:
                rec[key] = {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, "values": xs}
        out.append(rec)
    return out


def print_summary(name: str, summary: list[dict], n_seeds: int) -> None:
    def cell(r, k):
        return f"{100 * r[k]['mean']:5.1f} ± {100 * r[k]['std']:3.1f}" if k in r else "      -     "

    print(f"\n{name}: mean ± std over {n_seeds} seed(s), test splits (F1 at the val-chosen threshold)")
    print(f"{'dataset':12} {'subset':18} {'task':8} {'n':>5}   {'F1@val':12} {'AUROC':12} {'AUPRC':12} {'FPR@95TPR':12}")
    for r in summary:
        print(f"{r['dataset']:12} {r['subset']:18} {r['task']:8} {r['n']:5d}   " + " ".join(cell(r, k) for k in SUMMARY_KEYS))
