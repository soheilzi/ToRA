"""Streaming table from run_eval runs (scripts/stream_eval.sh, scripts/deployed_eval.py): one row per method, mean ± std over seeds
(methods named NAME_s0, NAME_s1, ... are one method's seeds).

    python scripts/stream_table.py results/eval/stream-*.jsonl > results/logs/stream_table.txt
    python scripts/stream_table.py RUNS... --methods tora qwen3guard_stream_0p6b singprobe_qwen3_0p6b   # only these, in this order

Qwen3GuardTest response_loc (813 unsafe responses, first unsafe sentence = token span), at thresholds that flag at
most 5% / 1% of the safe WildGuard + Aegis 2.0 val responses: detected (flagged anywhere), exact hit (first flag
inside the span), within 128 (first flag no later than span end + 128, flags before the span included, as in Qwen's
eval), median delay (first flag - span start, detected responses only), and the FPR those thresholds give on the
safe BeaverTails test responses. SingStreamBench: R-AUC = AUROC of the streaming-consistent response score (max over
tokens; Qwen3Guard: its two-in-a-row run), T-AUC = token-level AUROC against "at or after the unsafe onset".
Last, the same thresholds on each method's end-of-response score (ours: the last response token, i.e. the verdict
after the fact, as in the F1 tables; the baselines: their streaming score, so equal to above).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from tora.eval.metrics import classification_metrics, threshold_at_fpr


def records(path: Path) -> list[dict]:
    return [r for r in map(json.loads, path.read_text().splitlines()) if r.get("type") == "metrics" and "task" in r]


def r_auc(run_dir: Path, method: str) -> float | None:
    f = run_dir / f"{method}__singstreambench__test.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(line) for line in f.read_text().splitlines()]
    rows = [r for r in rows if r["response_label"] is not None]
    return classification_metrics([r["response_label"] for r in rows], [r["alt"].get("response_max", r["response_score"]) for r in rows])["auroc"]


def per_example(run_dir: Path, method: str, dataset: str, split: str) -> list[dict]:
    return [json.loads(line) for line in (run_dir / f"{method}__{dataset}__{split}.jsonl").read_text().splitlines()]


def final_verdict(run_dir: Path, method: str) -> dict[str, float]:
    """Detection on Qwen3GuardTest, safe-BeaverTails FPR and SingStreamBench R-AUC of the end-of-response score."""
    cal = [r["response_score"] for name in ("wildguard", "aegis2") for r in per_example(run_dir, method, name, "val") if r["response_label"] == 0]
    unsafe = [r["response_score"] for r in per_example(run_dir, method, "qwen3guardtest", "test")]
    safe = [r["response_score"] for r in per_example(run_dir, method, "beavertails", "test") if r["response_label"] == 0]
    ss = [r for r in per_example(run_dir, method, "singstreambench", "test") if r["response_label"] is not None]
    out = {"r_auc": classification_metrics([r["response_label"] for r in ss], [r["response_score"] for r in ss])["auroc"]}
    for fpr in (0.05, 0.01):
        thr = threshold_at_fpr(cal, fpr)
        out[f"detected_{fpr}"] = float(np.mean(np.array(unsafe) >= thr))
        out[f"fpr_safe_{fpr}"] = float(np.mean(np.array(safe) >= thr))
    return out


def fmt(vals: list[float | None], scale: float = 100.0, digits: int = 1) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    a = np.array(vals, dtype=float) * scale
    return f"{a.mean():.{digits}f}" if len(a) == 1 else f"{a.mean():.{digits}f}±{a.std(ddof=1):.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path, help="results/eval/<run_id>.jsonl")
    ap.add_argument("--methods", nargs="+", help="report only these methods (seed suffix _sN removed), in this order")
    args = ap.parse_args()
    by_method: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    order: list[str] = []
    for path in args.runs:
        for r in records(path):
            method = re.sub(r"_s\d+$", "", r["method"])
            if method not in order:
                order.append(method)
            m = by_method[method]
            if r["dataset"] == "qwen3guardtest" and r["task"] == "stream":
                for fpr in ("0.05", "0.01"):
                    m[fpr].append(r[f"at_calib_fpr_{fpr}"])
                m["final"].append(final_verdict(path.with_suffix(""), r["method"]))
                if "official" in r:
                    m["official"].append(r["official"])
            elif r["dataset"] == "singstreambench" and r["task"] == "token":
                m["t_auc"].append(r["t_auc"])
            elif r["dataset"] == "singstreambench" and r["task"] == "response":
                m["r_auc"].append(r_auc(path.with_suffix(""), r["method"]))
    if args.methods:
        missing = [m for m in args.methods if m not in by_method]
        if missing:
            raise ValueError(f"--methods {missing} not in the runs (found {order})")
        order = args.methods

    for fpr, label in (("0.05", "5%"), ("0.01", "1%")):
        print(f"\nQwen3GuardTest response_loc, threshold: {label} of safe WildGuard + Aegis 2.0 val responses flagged\n")
        print("| Method | seeds | Detected | Exact hit | Within 128 | Median delay (tokens) | Safe BeaverTails flagged |")
        print("|---|---|---|---|---|---|---|")
        for method in order:
            rs = by_method[method][fpr]
            if rs:
                print(f"| {method} | {len(rs)} | {fmt([r['detection_rate'] for r in rs])} | {fmt([r['exact_hit_rate'] for r in rs])} | "
                      f"{fmt([r['within_128_rate'] for r in rs])} | {fmt([r['delay_from_span_start_median'] for r in rs], 1.0)} | {fmt([r.get('fpr_safe') for r in rs])} |")
    print("\nOfficial rules (Qwen3Guard-Stream: two consecutive unsafe / controversial tokens)\n")
    print("| Method | Detected | Exact hit | Within 128 | Median delay |")
    print("|---|---|---|---|---|")
    for method in order:
        for r in by_method[method]["official"]:
            print(f"| {method} | {fmt([r['detection_rate']])} | {fmt([r['exact_hit_rate']])} | {fmt([r['within_128_rate']])} | {fmt([r['delay_from_span_start_median']], 1.0)} |")
    print("\nSingStreamBench (210 responses, 122 unsafe)\n")
    print("| Method | seeds | R-AUC (max over tokens) | T-AUC |")
    print("|---|---|---|---|")
    for method in order:
        m = by_method[method]
        if m["r_auc"] or m["t_auc"]:
            print(f"| {method} | {len(m['r_auc'])} | {fmt(m['r_auc'])} | {fmt(m['t_auc'])} |")
    print("\nEnd-of-response score (after the fact for ours), same calibration\n")
    print("| Method | seeds | Detected @5% | Safe BeaverTails flagged @5% | Detected @1% | Safe BeaverTails flagged @1% | SingStreamBench R-AUC |")
    print("|---|---|---|---|---|---|---|")
    for method in order:
        f = by_method[method]["final"]
        if f:
            print(f"| {method} | {len(f)} | " + " | ".join(fmt([x[k] for x in f]) for k in ("detected_0.05", "fpr_safe_0.05", "detected_0.01", "fpr_safe_0.01", "r_auc")) + " |")


if __name__ == "__main__":
    main()
