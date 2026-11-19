"""Any method x any dataset -> results/eval/<run_id>.jsonl (+ per-example scores in results/eval/<run_id>/).

    CUDA_VISIBLE_DEVICES=0 python -m tora.eval.run_eval --config configs/eval/baselines.yaml

Per method and dataset (test split):
  prompt / response   AUROC, AUPRC, FPR@95TPR; P/R/F1 of the method's official decisions; and F1 at the
                      threshold that maximizes F1 on the dataset's val split, when it has one
  stream              on datasets with unsafe spans: the official streaming rule, plus a threshold set so that
                      at most `target_fpr` of calibration safe responses (val splits) are flagged, with the
                      resulting FPR on held-out safe test responses
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from tora.data.formatting import ChatFormatter, load_tokenizer
from tora.data.prepare import load_processed, processed_path
from tora.data.schema import Example
from tora.eval import metrics as M
from tora.eval.scorers import METHODS, Scored, Scorer
from tora.utils.config import RunConfig, config_from_cli, to_dict
from tora.utils.gpu_info import require_visible_devices
from tora.utils.hf import download_model
from tora.utils.paths import CONFIGS_ROOT, resolve
from tora.utils.runlog import RunLogger, make_run_id
from tora.utils.seeding import seed_everything


@dataclass
class EvalSection:
    methods: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    tokenizer_repo: str = "Qwen/Qwen3-0.6B"
    max_len: int = 2048
    batch_tokens: int = 32768
    max_examples: int | None = None  # per split, deterministic prefix (quick runs only)
    stream_target_fprs: list[float] = field(default_factory=lambda: [0.05, 0.01])
    stream_calibration: list[str] = field(default_factory=lambda: ["aegis2"])  # safe responses of their val split
    stream_fpr_eval: list[str] = field(default_factory=lambda: ["beavertails"])  # safe responses of their test split
    generated: str = "data_cache/generated/qwen3-0.6b_greedy_512"  # generator answers for singprobe_qwen3_0p6b_query


@dataclass
class EvalConfig:
    run: RunConfig
    eval: EvalSection = field(default_factory=EvalSection)


def _load(cfg: EvalConfig, name: str, split: str) -> list[Example] | None:
    if not processed_path(cfg.run.data_dir, name, split).exists():
        return None
    exs = load_processed(cfg.run.data_dir, name, split)
    return exs[: cfg.eval.max_examples] if cfg.eval.max_examples else exs


def task_metrics(task: str, test: list[tuple[Example, Scored]], val: list[tuple[Example, Scored]] | None) -> dict | None:
    label = (lambda e: e.prompt_label) if task == "prompt" else (lambda e: e.response_label)
    score = (lambda s: s.prompt_score) if task == "prompt" else (lambda s: s.response_score)
    rows = [(label(e), score(s), s) for e, s in test if label(e) is not None and score(s) is not None]
    if not rows:
        return None
    y, sc = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    out = M.classification_metrics(y, sc)
    for key in rows[0][2].decisions:
        if key.startswith(task):
            dec = np.array([r[2].decisions[key] for r in rows])
            out[f"official[{key}]"] = M.prf_at(y, dec, 0.5)
    if val:
        vrows = [(label(e), score(s)) for e, s in val if label(e) is not None and score(s) is not None]
        vy = np.array([r[0] for r in vrows])
        if 0 < vy.sum() < len(vy):
            thr = M.best_f1_threshold(vy, np.array([r[1] for r in vrows]))
            out["at_val_threshold"] = M.prf_at(y, sc, thr)
    return out


def stream_metrics(scorer: Scorer, test: list[tuple[Example, Scored]], calib: list[Scored], heldout_safe: list[Scored], fprs: list[float]) -> dict:
    unsafe = [(e, s) for e, s in test if e.response_label == 1 and "unsafe_start_index" in e.meta]
    spans, scores, official = [], [], []
    for e, s in unsafe:
        prepared = scorer.prepare(e)
        common = min(len(prepared.ids), len(e.meta["input_ids"]))
        if prepared.ids[:common] != e.meta["input_ids"][:common]:
            raise ValueError(f"{e.id}: our token stream differs from the dataset's input_ids; unsafe spans would misalign")
        start = prepared.rendered.response_start
        spans.append((e.meta["unsafe_start_index"] - start, e.meta["unsafe_end_index"] - start))
        scores.append(s.response_token_scores)
        official.append(s.official_flag)
    out: dict = {"n_calibration_safe": len(calib), "n_heldout_safe": len(heldout_safe)}
    if all(f is not None for f in official):
        out["official"] = M.flag_metrics(official, spans)
    calib_scores = np.array([M.response_score(s.response_token_scores, scorer.consecutive) for s in calib])
    for fpr in fprs:
        thr = M.threshold_at_fpr(calib_scores, fpr)
        out[f"at_calib_fpr_{fpr}"] = M.streaming_metrics(
            scores, spans, thr, scorer.consecutive, safe_scores=[s.response_token_scores for s in heldout_safe]
        )
    return out


def token_auc(scorer: Scorer, test: list[tuple[Example, Scored]]) -> dict:
    """T-AUC (SingProbe report): every scored response token vs. 'at or after the unsafe onset'.

    Token labels come from the response's character offsets; safe responses are all 0. Appended turn-closing
    tokens inherit the label of the last response token.
    """
    labels, scores = [], []
    for e, s in test:
        enc = scorer.fmt.tokenizer(e.response, add_special_tokens=False, return_offsets_mapping=True)
        lab = [int(end > e.meta["unsafe_char_start"]) for _, end in enc["offset_mapping"]]
        lab = lab[scorer.prepare(e).rendered.response_trimmed :]
        lab += [lab[-1]] * (len(s.response_token_scores) - len(lab))
        labels += lab
        scores += s.response_token_scores.tolist()
    return {"t_auc": M.classification_metrics(labels, scores)["auroc"], "n_tokens": len(labels), "n_unsafe_tokens": int(sum(labels))}


def method_metrics(method: str, scorer: Scorer, scored: dict[tuple[str, str], list[tuple[Example, Scored]]], ev: EvalSection) -> list[dict]:
    """The metric records of one method from its scored splits {(dataset, split): [(example, scored)]}: prompt and
    response metrics per dataset, T-AUC where unsafe onsets are marked, streaming metrics where unsafe spans are."""
    out = []
    for name in ev.datasets:
        test, val = scored[(name, "test")], scored.get((name, "val"))
        for task in ("prompt", "response"):
            res = task_metrics(task, test, val)
            if res is not None:
                out.append({"method": method, "dataset": name, "task": task, **res})
        if scorer.token_level and any("unsafe_char_start" in e.meta for e, _ in test):
            out.append({"method": method, "dataset": name, "task": "token", **token_auc(scorer, test)})
        if any("unsafe_start_index" in e.meta for e, _ in test):
            calib = [s for d in ev.stream_calibration for e, s in scored.get((d, "val"), []) if e.response_label == 0]
            heldout = [s for d in ev.stream_fpr_eval for e, s in scored.get((d, "test"), []) if e.response_label == 0]
            out.append({"method": method, "dataset": name, "task": "stream", **stream_metrics(scorer, test, calib, heldout, ev.stream_target_fprs)})
    return out


def save_scores(path, pairs: list[tuple[Example, Scored]], with_tokens: bool, token_digits: int | None = 5) -> None:
    """One JSON line per example; token scores rounded to token_digits (None: full precision, so load_scores gives
    back exactly what was scored)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for e, s in pairs:
            rec = {"id": e.id, "prompt_label": e.prompt_label, "response_label": e.response_label} | {
                k: v for k, v in asdict(s).items() if k != "response_token_scores"
            }
            if with_tokens and s.response_token_scores is not None:
                tok = s.response_token_scores
                rec["response_token_scores"] = (tok if token_digits is None else np.round(tok, token_digits)).tolist()
            fh.write(json.dumps(rec) + "\n")


def load_scores(path, examples: list[Example]) -> list[tuple[Example, Scored]]:
    """A save_scores file of `examples` back as (example, Scored) pairs: line i must be examples[i] (save_scores keeps
    their order; matched by position, since a split may repeat an example id, e.g. two duplicate rows in Aegis 1.0 test)."""
    recs = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if [r["id"] for r in recs] != [e.id for e in examples]:
        raise ValueError(f"{path}: its {len(recs)} example ids are not the {len(examples)} given, in order")
    pairs = []
    for e, r in zip(examples, recs):
        tok = r.get("response_token_scores")
        s = Scored(prompt_score=r["prompt_score"], response_score=r["response_score"], response_token_scores=None if tok is None else np.array(tok),
                   decisions=r["decisions"], alt=r["alt"], official_flag=r["official_flag"])
        pairs.append((e, s))
    return pairs


def make_scorer(method: str, fmt: ChatFormatter, ev: EvalSection, hf_home) -> tuple[str, Scorer]:
    """-> (name used in results and file names, scorer). `ckpt:NAME=PATH` scores one of our trained checkpoints in one
    teacher-forced pass, `fused:NAME=PATH` a ToRA checkpoint through the fused (deployed) decode step."""
    if method.startswith("fused:"):  # ToRA through the fused decode step, one token per step
        name, sep, path = method.removeprefix("fused:").partition("=")
        if not sep or not name or "/" in name:
            raise ValueError(f"{method!r}: fused methods are fused:NAME=PATH/TO/<variant>.pt")
        from tora.eval.fused_scorer import FusedScorer

        return name, FusedScorer(name, resolve(path), fmt, ev.max_len, ev.tokenizer_repo, hf_home=hf_home)
    if method.startswith("ckpt:"):
        name, sep, path = method.removeprefix("ckpt:").partition("=")
        if not sep or not name or "/" in name:
            raise ValueError(f"{method!r}: checkpoint methods are ckpt:NAME=PATH/TO/<variant>.pt")
        from tora.eval.checkpoint_scorer import CheckpointScorer  # it imports train_verifier, which imports this module

        return name, CheckpointScorer(name, resolve(path), fmt, ev.max_len, ev.batch_tokens, ev.tokenizer_repo, hf_home=hf_home)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; known: {sorted(METHODS)}, ckpt:NAME=PATH or fused:NAME=PATH")
    return method, METHODS[method](fmt, ev, hf_home)


def main(argv: list[str] | None = None) -> list[dict]:
    cfg = config_from_cli(EvalConfig, argv, default_config=str(CONFIGS_ROOT / "eval" / "baselines.yaml"))
    require_visible_devices()
    seed_everything(cfg.run.seed)
    ev = cfg.eval
    fmt = ChatFormatter(load_tokenizer(download_model(ev.tokenizer_repo, hf_home=cfg.run.hf_home)))
    summary = []
    with RunLogger(cfg.run.phase, make_run_id(cfg.run.name), to_dict(cfg), cfg.run.seed, cfg.run.results_dir) as log:
        for spec in ev.methods:
            method, scorer = make_scorer(spec, fmt, ev, cfg.run.hf_home)
            scored: dict[tuple[str, str], list[tuple[Example, Scored]]] = {}
            needed = {(d, "test") for d in ev.datasets} | {(d, "val") for d in ev.datasets + ev.stream_calibration}
            needed |= {(d, "test") for d in ev.stream_fpr_eval}
            for name, split in sorted(needed):
                exs = _load(cfg, name, split)
                if exs is None:
                    if split == "test":
                        raise FileNotFoundError(f"{name}/test not prepared; run tora.data.prepare")
                    continue
                if split != "test" or name not in ev.datasets:  # calibration / FPR pools only need responses
                    exs = [e for e in exs if e.response is not None or name in ev.datasets]
                scored[(name, split)] = list(zip(exs, scorer.score(exs)))
                with_tokens = any("unsafe_start_index" in e.meta for e in exs[:1])
                save_scores(log.path.with_suffix("") / f"{method}__{name}__{split}.jsonl", scored[(name, split)], with_tokens)
                print(f"[{method}] scored {name}/{split}: {len(exs)} examples")

            for rec in method_metrics(method, scorer, scored, ev):
                log.log(rec)
                summary.append(rec)
            del scorer
            torch.cuda.empty_cache()
        print_summary(summary)
    return summary


def print_summary(summary: list[dict]) -> None:
    def pct(x):
        return f"{100 * x:5.1f}" if isinstance(x, float) else "   - "

    print("\nmethod                     dataset         task      n    AUROC AUPRC  F1(official: strict/loose or 0.5)  F1@val-thr")
    for r in summary:
        if r["task"] in ("stream", "token"):
            continue
        off = " / ".join(pct(v["f1"]) for k, v in r.items() if k.startswith("official["))
        print(
            f"{r['method']:26} {r['dataset']:15} {r['task']:8} {r['n']:5d}  {pct(r.get('auroc'))} {pct(r.get('auprc'))}  {off:33}  "
            f"{pct(r.get('at_val_threshold', {}).get('f1'))}"
        )
    for r in summary:
        if r["task"] == "token":
            print(f"{r['method']} on {r['dataset']}: T-AUC {pct(r['t_auc'])} over {r['n_tokens']} tokens")
    for r in summary:
        if r["task"] == "stream":
            print(f"\n{r['method']} on {r['dataset']} (streaming):")
            if "official" in r:
                o = r["official"]
                print(f"  official rule: detected {pct(o['detection_rate'])}%  exact-hit {pct(o['exact_hit_rate'])}%  within-128 {pct(o['within_128_rate'])}%  bins {o['bins']}")
            for k, v in r.items():
                if k.startswith("at_calib_fpr_"):
                    print(
                        f"  {k}: thr {v['threshold']:.3f}  detected {pct(v['detection_rate'])}%  exact-hit {pct(v['exact_hit_rate'])}%  "
                        f"within-128 {pct(v['within_128_rate'])}%  median delay {v['delay_from_span_start_median']}  held-out safe FPR {pct(v.get('fpr_safe'))}%"
                    )


if __name__ == "__main__":
    main()
