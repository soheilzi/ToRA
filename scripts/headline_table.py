"""Headline table (the paper's Table I): per method, its accuracy (the F1 tables' averages and the Qwen3GuardTest
streaming metrics) next to its measured per-token latency and the memory it adds to the generator.

    python scripts/headline_table.py --baselines results/eval/baselines-<...> \\
        --stream results/eval/stream-*.jsonl results/eval/deployed_seed*-*.jsonl \\
        --latency "results/bench/decode_guards_0p6b-*.jsonl" \\
        --row "ToRA" "results/eval/deployed_seed*-*/tora_s*" tora fused \\
        --row "SingProbe" baseline:SingProbe singprobe_qwen3_0p6b singprobe ...

A row is LABEL F1 STREAM LATENCY:
  F1       glob of score-file prefixes, one per seed (as f1_table.py --method), or baseline:NAME (f1_table.BASELINES,
           from --baselines); accuracy follows each method's main rule, as in the main table
  STREAM   a run_eval method in the --stream runs: NAME_sN is that seed alone, NAME every seed
  LATENCY  a mode of the --latency run (decode_bench.py)
Any of the three may be "-" (not reported). Accuracy is mean ± std over seeds.

Overhead is given twice: over the stock generator (gen), and over the generator the row itself runs on (fused_gen for
ToRA's fused step and the baselines on our kernels, the *_kernels modes; gen otherwise), so that rows on the same
kernels compare like for like.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from f1_table import BASELINES, TESTS, baseline_file, score_run  # noqa: E402
from stream_table import records  # noqa: E402


def mean_std(vals: list[float], scale: float = 100.0, digits: int = 1) -> str:
    a = np.array(vals, dtype=float) * scale
    return f"{a.mean():.{digits}f}" if len(a) == 1 else f"{a.mean():.{digits}f}±{a.std(ddof=1):.{digits}f}"


def f1_averages(spec: str, baselines: list[str]) -> tuple[int, dict[str, str]]:
    """-> (seeds, {task: Avg. F1 over the main table's benchmarks under the method's main rule})."""
    if spec.startswith("baseline:"):
        m = BASELINES[spec.removeprefix("baseline:")]
        runs = [score_run(lambda task, d, split: baseline_file(baselines, m[task][0], d, split), {t: m[t][1] for t in m})]
    else:
        prefixes = sorted(p.removesuffix("__wildguard__test.jsonl") for p in glob.glob(spec + "__wildguard__test.jsonl"))
        if not prefixes:
            raise SystemExit(f"no score files match {spec}")
        runs = [score_run(lambda task, d, split, p=p: f"{p}__{d}__{split}.jsonl") for p in prefixes]
    return len(runs), {task: mean_std([np.mean([r[(task, d)]["main"] for d in tests]) for r in runs]) for task, tests in TESTS.items()}


def stream_cells(spec: str, recs: list[dict]) -> tuple[int, dict[str, str]]:
    exact = re.search(r"_s\d+$", spec) is not None
    rows = [r for r in recs if r["dataset"] == "qwen3guardtest" and r["task"] == "stream"
            and (r["method"] == spec if exact else re.sub(r"_s\d+$", "", r["method"]) == spec)]
    if not rows:
        raise SystemExit(f"no streaming records for {spec}")
    out = {}
    for fpr in ("0.05", "0.01"):
        at = [r[f"at_calib_fpr_{fpr}"] for r in rows]
        out[f"det_{fpr}"] = mean_std([a["detection_rate"] for a in at])
        out[f"exact_{fpr}"] = mean_std([a["exact_hit_rate"] for a in at])
        out[f"delay_{fpr}"] = mean_std([a["delay_from_span_start_median"] for a in at], 1.0)
    return len(rows), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", required=True, help="eval run dirs with the released baselines' score files")
    ap.add_argument("--stream", nargs="+", required=True, type=Path, help="run_eval runs with streaming records")
    ap.add_argument("--latency", required=True, help="glob of one decode_bench.py run")
    ap.add_argument("--row", nargs=4, action="append", required=True, metavar=("LABEL", "F1", "STREAM", "LATENCY"))
    ap.add_argument("--exec", default="compile+graph", help="the execution of the headline latency column")
    args = ap.parse_args()

    runs = glob.glob(args.latency)
    if len(runs) != 1:
        raise SystemExit(f"--latency {args.latency!r} matches {len(runs)} runs, expected 1")
    lines = [json.loads(line) for line in Path(runs[0]).read_text().splitlines()]
    lat = {(r["context"], r["exec"], r["mode"]): r for r in lines if r.get("type") == "metrics"}
    meta = next(r for r in lines if r.get("type") == "meta")
    contexts = sorted({c for c, _, _ in lat})
    execs = meta["config"]["execs"]
    recs = [r for path in args.stream for r in records(path)]

    def floor(mode: str) -> str:
        return "fused_gen" if mode == "fused" or mode.endswith("_kernels") else "gen"

    def overhead(ctx: int, exe: str, mode: str, base: str = "gen") -> str:
        return f"{100 * (lat[(ctx, exe, mode)]['median_ms'] / lat[(ctx, exe, base)]['median_ms'] - 1):+.1f}%"

    mid = contexts[len(contexts) // 2]
    print(f"\nHeadline: accuracy (main rules; Qwen3GuardTest streaming at the 5% / 1% calibration budgets) and per-token "
          f"latency, batch 1, {meta['config']['model']}, {args.exec}, one {meta['gpu']['gpus'][0]['name']}\n")
    cs = " / ".join(map(str, contexts))
    print("| Method | seeds (F1 / stream) | Response Avg. F1 | Prompt Avg. F1 | Detected @5% | Exact hit @5% | Delay @5% | Detected @1% | "
          f"ms / token at {mid} | Over the generator on the same kernels ({cs}) | Over the stock generator ({cs}) | "
          f"Added memory: weights + caches at {contexts[-1]} |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, f1_spec, stream_spec, mode in args.row:
        n_f1, f1 = f1_averages(f1_spec, args.baselines) if f1_spec != "-" else (0, {})
        n_st, st = stream_cells(stream_spec, recs) if stream_spec != "-" else (0, {})
        cells = [f1.get("response", "–"), f1.get("prompt", "–"), st.get("det_0.05", "–"), st.get("exact_0.05", "–"),
                 st.get("delay_0.05", "–"), st.get("det_0.01", "–")]
        if mode == "-":
            cells += ["–"] * 4
        else:
            r = lat[(contexts[-1], args.exec, mode)]
            cells += [f"{lat[(mid, args.exec, mode)]['median_ms']:.3f}",
                      "–" if mode in ("gen", "fused_gen") else " / ".join(overhead(c, args.exec, mode, floor(mode)) for c in contexts),
                      "–" if mode == "gen" else " / ".join(overhead(c, args.exec, mode) for c in contexts),
                      "–" if mode in ("gen", "fused_gen") else f"{r['extra_param_bytes'] / 2**20:.1f} MiB + {r['extra_cache_bytes'] / 2**20:.1f} MiB"]
        print(f"| {label} | {n_f1 or '–'} / {n_st or '–'} | " + " | ".join(cells) + " |")

    modes = list(dict.fromkeys(m for _, _, m in lat))
    print("\nLatency, ms per token (median); (%) over the stock generator (gen) under the same execution; [%] over the "
          "generator on the same kernels (fused_gen), for the rows on our kernels\n")
    print("| Context | Execution | " + " | ".join(modes) + " |\n|---|---|" + "---|" * len(modes))
    for ctx in contexts:
        for exe in execs:
            cells = [f"{lat[(ctx, exe, m)]['median_ms']:.3f}" + ("" if m == "gen" else f" ({overhead(ctx, exe, m)})")
                     + (f" [{overhead(ctx, exe, m, 'fused_gen')}]" if floor(m) == "fused_gen" else "") for m in modes]
            print(f"| {ctx} | {exe} | " + " | ".join(cells) + " |")
    shared = [k for k, r in lat.items() if r["gpu_shared"]]
    if shared:
        print(f"\nWARNING: another process shared the GPU in rows {shared}")


if __name__ == "__main__":
    main()
