"""Every benchmark scored through the deployed decode step (the numbers of the paper's tables): ToRA run
token by token as deployed (eval/fused_scorer.py: prompt prefilled, every later token one fused dual-row step,
compiled under a CUDA graph), to check that the accuracy tables hold on the kernels the latency tables time.

  score    one shard of the (dataset, split) items of configs/eval/deployed.yaml (its test sets, the val splits
           of eval.stream_calibration, the test splits of eval.stream_fpr_eval). Writes
           results/eval/<run_id>/<method>__<dataset>__<split>.jsonl (full-precision token scores; never overwritten)
           and the shard's log results/deployed/<run_id>-shard<i>.jsonl. Items go to shards by greedy balancing of the
           number of decoded steps.
  metrics  after every shard: run_eval's metrics from those files -> results/eval/<run_id>.jsonl, whose run dir then
           holds its per-example scores, so f1_table.py and stream_table.py read it as any run_eval run; plus the
           agreement with the teacher-forced scores of the same checkpoint (--reference prefixes, first match wins).

    S=0; RUN=deployed_seed$S-$(date +%Y%m%d-%H%M%S)
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/deployed_eval.py score --run-id $RUN --shard $i \\
        --shards 4 --method tora_s$S --ckpt "results/train/tora_seed$S-*/tora.pt" & done; wait
    CUDA_VISIBLE_DEVICES=0 python scripts/deployed_eval.py metrics --run-id $RUN --method tora_s$S \\
        --reference "results/train/tora_seed$S-*/tora" "results/eval/stream-*/tora_tf_s$S"
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from tora.data.formatting import ChatFormatter, load_tokenizer
from tora.data.prepare import load_processed
from tora.eval.run_eval import EvalConfig, load_scores, method_metrics, print_summary, save_scores
from tora.eval.scorers import Scorer
from tora.utils.config import load_config, to_dict
from tora.utils.gpu_info import require_visible_devices
from tora.utils.hf import download_model
from tora.utils.paths import resolve
from tora.utils.runlog import RunLogger


def items(cfg: EvalConfig) -> list[tuple[str, str]]:
    ev = cfg.eval
    return sorted({(d, "test") for d in ev.datasets + ev.stream_fpr_eval} | {(d, "val") for d in ev.stream_calibration})


def decoded_steps(fmt: ChatFormatter, max_len: int, exs) -> int:
    """Fused steps to score exs: the prompt's last token, then every response token."""
    rs = [fmt.render_example(e, max_len) for e in exs]
    return sum(1 if r.response_end is None else r.response_end - r.prompt_end + 1 for r in rs)


def shard_items(costs: dict[tuple[str, str], int], shards: int) -> list[list[tuple[str, str]]]:
    """Largest item first onto the least-loaded shard (ties: the lower index); deterministic."""
    load, out = [0] * shards, [[] for _ in range(shards)]
    for item in sorted(costs, key=lambda k: (-costs[k], k)):
        i = min(range(shards), key=lambda j: (load[j], j))
        out[i].append(item)
        load[i] += costs[item]
    return out


def score(args, cfg: EvalConfig, fmt: ChatFormatter) -> None:
    from tora.eval.fused_scorer import FusedScorer

    require_visible_devices()
    ckpts = sorted(glob.glob(args.ckpt))
    if len(ckpts) != 1:
        raise FileNotFoundError(f"--ckpt {args.ckpt!r} matches {len(ckpts)} files, expected 1")
    data = {it: load_processed(cfg.run.data_dir, *it) for it in items(cfg)}
    costs = {it: decoded_steps(fmt, cfg.eval.max_len, exs) for it, exs in data.items()}
    mine = shard_items(costs, args.shards)[args.shard]
    out_dir = resolve(cfg.run.results_dir) / "eval" / args.run_id
    config = {**to_dict(cfg), "ckpt_path": ckpts[0], "method": args.method, "shard": args.shard, "shards": args.shards, "items": mine}
    with RunLogger("deployed", f"{args.run_id}-shard{args.shard}", config, cfg.run.seed, cfg.run.results_dir) as log:
        log.console.info("shard %d/%d: %s (%d decoded steps)", args.shard, args.shards, mine, sum(costs[it] for it in mine))
        scorer = FusedScorer(args.method, Path(ckpts[0]), fmt, cfg.eval.max_len, cfg.eval.tokenizer_repo, execution=args.execution)
        for name, split in mine:
            path = out_dir / f"{args.method}__{name}__{split}.jsonl"
            if path.exists():
                raise FileExistsError(f"{path} exists; score files are never overwritten")
            exs, t0 = data[(name, split)], time.perf_counter()
            scored = []
            for lo in range(0, len(exs), 250):
                scored += scorer.score(exs[lo : lo + 250])
                log.console.info("%s/%s: %d / %d examples, %.0f s", name, split, len(scored), len(exs), time.perf_counter() - t0)
            save_scores(path, list(zip(exs, scored)), with_tokens=True, token_digits=None)
            sec = time.perf_counter() - t0
            log.log({"dataset": name, "split": split, "n": len(exs), "decoded_steps": costs[(name, split)], "seconds": sec,
                     "ms_per_step": 1e3 * sec / costs[(name, split)], "execution": args.execution})


def agreement(ours: list, ref_path: str) -> dict:
    """|ours - reference| over one split's examples: the prompt score, the response score (last token), every token
    score where both files have them, and how many verdicts flip at 0.5."""
    rows = [json.loads(line) for line in Path(ref_path).read_text().splitlines()]
    if [r["id"] for r in rows] == [e.id for e, _ in ours]:  # the same split in the same order: match by position
        refs = rows
    else:  # a filtered copy (a calibration pool keeps only responses): match by id, which must then be unique
        by_id = {r["id"]: r for r in rows}
        if len(by_id) != len(rows) or len({e.id for e, _ in ours}) != len(ours):
            raise ValueError(f"{ref_path}: different examples and repeated ids, so no unambiguous match")
        refs = [by_id[e.id] for e, _ in ours]
    out: dict = {"reference": ref_path, "n": len(ours)}
    for key in ("prompt_score", "response_score"):
        pairs = np.array([(getattr(s, key), r[key]) for (e, s), r in zip(ours, refs) if getattr(s, key) is not None], dtype=float)
        if len(pairs):
            d = np.abs(pairs[:, 0] - pairs[:, 1])
            out[key] = {"n": len(d), "max_abs": d.max(), "mean_abs": d.mean(), "flips_at_0.5": int(((pairs[:, 0] >= 0.5) != (pairs[:, 1] >= 0.5)).sum())}
    tok = [np.abs(s.response_token_scores - np.array(r["response_token_scores"])) for (e, s), r in zip(ours, refs)
           if s.response_token_scores is not None and "response_token_scores" in r]
    if tok:
        d = np.concatenate(tok)
        out["token_scores"] = {"n": len(d), "max_abs": d.max(), "mean_abs": d.mean(), "p99_abs": np.quantile(d, 0.99)}
    return out


def metrics(args, cfg: EvalConfig, fmt: ChatFormatter) -> None:
    run_dir = resolve(cfg.run.results_dir) / "eval" / args.run_id
    scorer = Scorer(fmt, cfg.eval.max_len, cfg.eval.batch_tokens, append_turn_end=False)  # FusedScorer's rendering
    scored = {it: load_scores(run_dir / f"{args.method}__{it[0]}__{it[1]}.jsonl", load_processed(cfg.run.data_dir, *it)) for it in items(cfg)}
    config = {**to_dict(cfg), "method": args.method, "reference": args.reference}
    with RunLogger("eval", args.run_id, config, cfg.run.seed, cfg.run.results_dir) as log:
        summary = method_metrics(args.method, scorer, scored, cfg.eval)
        for rec in summary:
            log.log(rec)
        print_summary(summary)
        print("\nAgreement with the teacher-forced scores (|fused - reference|)")
        for (name, split), pairs in scored.items():
            refs = [f for prefix in args.reference for f in glob.glob(f"{prefix}__{name}__{split}.jsonl")]
            if not refs:
                raise FileNotFoundError(f"no reference score file for {name}/{split} under {args.reference}")
            rec = {"method": args.method, "dataset": name, "split": split, "task": "agreement", **agreement(pairs, refs[0])}
            log.log(rec)
            cells = [f"{k} max {rec[k]['max_abs']:.4f} mean {rec[k]['mean_abs']:.5f}" + (f" flips {rec[k]['flips_at_0.5']}" if "flips_at_0.5" in rec[k] else "")
                     for k in ("prompt_score", "response_score", "token_scores") if k in rec]
            print(f"  {name}/{split} (n {rec['n']}): " + "; ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["score", "metrics"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default="configs/eval/deployed.yaml")
    ap.add_argument("--method", default="tora_s0", help="name in the score files (NAME_sN: seed N, as stream_table expects)")
    ap.add_argument("--ckpt", help="score: glob of one ToRA checkpoint (<variant>.pt)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--execution", default="compile+graph", choices=["eager", "graph", "compile", "compile+graph"])
    ap.add_argument("--reference", nargs="+", default=[], help="metrics: prefixes of the teacher-forced score files")
    ap.add_argument("overrides", nargs="*", help="a.b.c=value overrides of --config")
    args = ap.parse_args()
    cfg = load_config(EvalConfig, resolve(args.config), args.overrides)
    fmt = ChatFormatter(load_tokenizer(download_model(cfg.eval.tokenizer_repo, hf_home=cfg.run.hf_home)))
    if args.command == "score":
        if args.ckpt is None or not 0 <= args.shard < args.shards:
            raise SystemExit("score needs --ckpt and 0 <= --shard < --shards")
        score(args, cfg, fmt)
    else:
        if not args.reference:
            raise SystemExit("metrics needs --reference")
        metrics(args, cfg, fmt)


if __name__ == "__main__":
    main()
