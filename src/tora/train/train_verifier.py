"""Verifiers (ToRA, the own-KV ablation) and the frozen-feature probe, trained on the frozen backbone.

    CUDA_VISIBLE_DEVICES=0 python -m tora.train.train_verifier --config configs/train/tora.yaml
    CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh tora 0        # the paper's runs: see scripts/train.sh

One job trains several variants at once: `sweep` is a cartesian product of dotted overrides on `base`
(e.g. verifier.rank: [4, 16]), or `variants` names them. The variants share each batch's frozen generator pass --
by far the dominant cost -- and differ only in their LoRA, head, optimizer and learning rates; they also share the
data order. Labels: BCE on prompt harm at the last prompt token and on response harm at the last response token,
one prompt label per unique prompt, positives weighted by training balance; with `prefix_positions`, also response
labels at earlier response positions (the per-token labels). Shared-KV verifiers compute rows only at the label
positions (exact; see models/verifier.py). Every variant is scored on the val splits `evals_per_epoch` times per
epoch and keeps its best mean-val-AUPRC state, which is then scored on the test splits. One seed per job.
"""

from __future__ import annotations

import copy
import glob
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from tora.data.formatting import ChatFormatter, Rendered, collate_right_padded, load_tokenizer, token_budget_batches
from tora.data.prepare import load_processed, processed_path
from tora.data.schema import Example
from tora.eval.run_eval import save_scores
from tora.eval.scorers import Scored
from tora.models.probes import GeneratorProbe
from tora.models.qwen3_min import Qwen3Config, Qwen3Min
from tora.models.verifier import Verifier, gather_positions, generator_captures
from tora.train.evaluation import TASKS, evaluate_scored, mean_auprc, model_tag, print_summary, summarize
from tora.utils.config import ModelConfig, ProbeConfig, RunConfig, VerifierConfig, _deep_merge, config_from_cli, from_dict, to_dict
from tora.utils.gpu_info import require_visible_devices
from tora.utils.paths import CONFIGS_ROOT, resolve
from tora.utils.runlog import RunLogger, make_run_id
from tora.utils.seeding import seed_everything


@dataclass
class VariantConfig:
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    probe: ProbeConfig | None = None  # set: train this probe on generator states instead of a verifier (frozen-feature control)
    lr: float = 3e-4  # LoRA
    head_lr: float = 1e-3  # the head, or the whole probe
    weight_decay: float = 0.0
    # Labels at the job's earlier response positions (train.prefix_positions != 0): teacher (ToRA) = safe responses 0;
    # unsafe responses 1 from the teacher's onset (train.prefix_teacher) on and 0 before it, all 0 if the teacher never
    # flags; safe (the no-teacher pilot) = only safe responses' positions, label 0 (every prefix of a safe response is
    # safe); none = unused.
    prefix_labels: str = "teacher"
    prefix_weight: float = 1.0  # total loss weight of one response's prefix positions, relative to its last token


@dataclass
class TrainVerifierSection:
    train_datasets: list[str] = field(default_factory=lambda: ["wildguard", "aegis2"])
    eval_datasets: list[str] = field(default_factory=lambda: ["wildguard", "aegis2", "beavertails"])
    subsets: dict[str, str] = field(default_factory=lambda: {"wildguard": "adversarial"})
    epochs: int = 2
    batch_tokens: int = 32768  # padded tokens per training batch (one optimizer step)
    micro_batch_tokens: int | None = None  # split each batch into micro-batches of at most this many padded tokens
    # and accumulate gradients (same steps and batches, less memory)
    micro_batch_seqs: int | None = None  # and of at most this many sequences: with prefix positions the verifier runs
    # prefix_positions + 2 rows per sequence, so a batch of up to 256 short sequences is large however few its tokens
    eval_batch_tokens: int = 32768
    pool_size: int = 8192  # bucketing: shuffle, then sort by length within pools of this many sequences
    warmup_frac: float = 0.03  # linear warmup, then cosine decay to 0
    grad_clip: float = 1.0
    evals_per_epoch: int = 2
    train_fraction: float = 1.0  # < 1: seeded subset of the training sequences (pilots)
    dedupe_prompts: bool = True
    max_len: int = 2048
    prefix_positions: int = 0  # > 0: also train the response head at this many earlier positions per response (sampled
    # once per run, seeded, from the response's tokens before its last); -1: at every one of those tokens (all rows at
    # once with FlexAttention, as Qwen3Guard-Stream's per-token loss); labels per variant (prefix_labels). Default off.
    prefix_teacher: str = ""  # glob of unsafe-onset files (scripts/teacher_onsets.py); needed by prefix_labels teacher*
    log_every_s: float = 60.0
    save_states: bool = True
    save_scores: bool = True  # per-example val / test scores of every variant
    base: VariantConfig = field(default_factory=VariantConfig)
    sweep: dict[str, list] = field(default_factory=dict)  # dotted key in `base` -> values; empty = base only
    variants: dict[str, dict] = field(default_factory=dict)  # or named variants: nested overrides deep-merged onto `base`


@dataclass
class TrainVerifierConfig:
    run: RunConfig
    model: ModelConfig
    train: TrainVerifierSection = field(default_factory=TrainVerifierSection)


SHORT = {
    "verifier.k_layers": "k", "verifier.rank": "r", "verifier.head.kind": "", "verifier.kv": "", "verifier.alpha": "a",
    "verifier.sink": "s", "head_lr": "hlr",
}


# Fields of earlier experiments that this code no longer has. Checkpoints saved by the original code carry them, with
# these values for every model in the paper.
LEGACY_VERIFIER_FIELDS = {"window": 1, "own_prompt": False, "invocation": ""}
LEGACY_PROBE_FIELDS = {"kind": "head", "taps": None, "num_heads": None, "head_dim": None, "window": None}


def variant_from_checkpoint(d: dict) -> VariantConfig:
    """A checkpoint's saved variant config -> VariantConfig, dropping the legacy fields (and refusing a model that used them)."""
    d = copy.deepcopy(d)
    for key, inert in LEGACY_VERIFIER_FIELDS.items():
        if key in d["verifier"] and d["verifier"].pop(key) != inert:
            raise ValueError(f"checkpoint uses verifier.{key}, which this code does not implement")
    if d.get("probe") is not None:
        if "taps" in d["probe"] and d["probe"]["taps"] != d["probe"].get("head", {}).get("taps"):
            raise ValueError("checkpoint's probe.taps differ from probe.head.taps, which this code reads")
        for key, value in LEGACY_PROBE_FIELDS.items():
            if key in d["probe"] and (got := d["probe"].pop(key)) != value and value is not None:
                raise ValueError(f"checkpoint uses probe.{key}={got!r}, which this code does not implement")
    return from_dict(VariantConfig, d, "checkpoint variant")


def expand_variants(tc: TrainVerifierSection) -> dict[str, VariantConfig]:
    if tc.variants:
        if tc.sweep:
            raise ValueError("set train.sweep or train.variants, not both")
        return {name: from_dict(VariantConfig, _deep_merge(to_dict(tc.base), over), f"variant {name}") for name, over in tc.variants.items()}
    if not tc.sweep:
        return {"base": tc.base}
    keys = list(tc.sweep)
    out = {}
    for values in itertools.product(*(tc.sweep[k] for k in keys)):
        d = to_dict(tc.base)
        for key, value in zip(keys, values):
            *parents, leaf = key.split(".")
            node = d
            for part in parents:
                node = node[part]
            if leaf not in node:
                raise ValueError(f"sweep key {key!r} is not in train.base")
            node[leaf] = value
        name = "_".join(f"{SHORT.get(k, k.split('.')[-1])}{v:g}" if isinstance(v, float) else f"{SHORT.get(k, k.split('.')[-1])}{v}" for k, v in zip(keys, values))
        out[name] = from_dict(VariantConfig, d, f"variant {name}")
    return out


def training_sequences(examples: list[Example], dedupe_prompts: bool) -> tuple[list[int], Tensor, int]:
    """-> (indices of the examples to train on, labels [n, 2] (prompt, response; NaN = no label), conflicts).

    A prompt label only at the first example with that prompt (`conflicts` counts later copies whose label
    disagrees), a response label wherever there is one.
    """
    seen: dict[str, int] = {}
    keep, labels, conflicts = [], [], 0
    for i, e in enumerate(examples):
        p = e.prompt_label
        if p is not None and dedupe_prompts:
            if e.prompt in seen:
                conflicts += int(seen[e.prompt] != p)
                p = None
            else:
                seen[e.prompt] = p
        r = e.response_label
        if p is None and r is None:
            continue
        keep.append(i)
        labels.append([math.nan if p is None else p, math.nan if r is None else r])
    return keep, torch.tensor(labels, dtype=torch.float32).view(-1, 2), conflicts


def label_positions(rendered: list[Rendered]) -> Tensor:
    """[B, 2]: last prompt token, last response token (the prompt end again when there is no response)."""
    return torch.tensor([[r.prompt_end, r.prompt_end if r.response_end is None else r.response_end] for r in rendered])


def verifier_loss(logits: Tensor, y: Tensor, pos_weight: Tensor) -> Tensor:
    """logits [B, 2 positions, 2 outputs]; BCE of prompt harm at the prompt end and response harm at the response end."""
    sel = logits.diagonal(dim1=1, dim2=2)  # [B, 2]
    valid = ~y.isnan()
    return F.binary_cross_entropy_with_logits(sel[valid], y[valid], pos_weight=pos_weight.expand_as(sel)[valid])


def sample_prefix_positions(rendered: list[Rendered], labels: Tensor, n: int, rng: np.random.Generator) -> Tensor:
    """[N, n] response positions before the response's last token (sampled without replacement), -1 = none (no response,
    no response label, or a response shorter than n + 1 tokens)."""
    out = torch.full((len(rendered), n), -1, dtype=torch.long)
    for i, r in enumerate(rendered):
        if r.response_end is None or math.isnan(float(labels[i, 1])):
            continue
        candidates = np.arange(r.response_start, r.response_end)
        pick = np.sort(rng.choice(candidates, size=min(n, len(candidates)), replace=False))
        out[i, : len(pick)] = torch.from_numpy(pick)
    return out


def response_spans(rendered: list[Rendered], labels: Tensor) -> Tensor:
    """[N, 2] (start, end): the response positions before its last token are start <= p < end; (0, 0) = none (no
    response or no response label). The candidates of sample_prefix_positions, all of them (prefix_positions = -1)."""
    out = torch.zeros((len(rendered), 2), dtype=torch.long)
    for i, r in enumerate(rendered):
        if r.response_end is not None and not math.isnan(float(labels[i, 1])):
            out[i] = torch.tensor([r.response_start, r.response_end])
    return out


def load_onsets(pattern: str) -> dict[str, int]:
    """Example id -> the teacher's unsafe onset, relative to the response start (-1 = never flagged)."""
    files = sorted(glob.glob(str(resolve(pattern))))
    if not files:
        raise FileNotFoundError(f"train.prefix_teacher={pattern!r} matches no file; run scripts/teacher_onsets.py")
    return {r["id"]: r["onset"] for f in files for r in map(json.loads, Path(f).read_text().splitlines())}


def teacher_onsets(rendered: list[Rendered], labels: Tensor, ids: list[str], onsets: dict[str, int]) -> Tensor:
    """[N] absolute onset position per training sequence: response start + the teacher's onset for unsafe responses it
    flags before their last token, else -1 (safe, unlabeled, or never flagged in time)."""
    missing = [i for i, y in zip(ids, labels[:, 1].tolist()) if y == 1 and i not in onsets]
    if missing:
        raise ValueError(f"{len(missing)} unsafe training responses have no teacher onset, e.g. {missing[:3]}")
    out = torch.full((len(rendered),), -1, dtype=torch.long)
    for k, (r, i, y) in enumerate(zip(rendered, ids, labels[:, 1].tolist())):
        if y == 1 and onsets[i] >= 0 and r.response_start + onsets[i] < r.response_end:
            out[k] = r.response_start + onsets[i]
    return out


def prefix_loss(
    logits: Tensor, y: Tensor, valid: Tensor, scheme: str, weight: float, pos_weight: Tensor, prefix: Tensor | None = None, onset: Tensor | None = None
) -> Tensor:
    """Weighted response-head BCE summed over prefix positions. logits [B, n] (response logit at the prefixes), y [B] the
    response labels, valid [B, n]; each response's prefixes share `weight` in total. Divide by the batch's label count.
    The teacher scheme also takes prefix [B, n] (the positions) and onset [B] (absolute, -1 = none: teacher_onsets)."""
    if scheme == "none":
        return logits.new_zeros(())
    target = y[:, None].expand_as(logits)
    if scheme == "safe":
        use = valid & (y[:, None] == 0)
    elif scheme == "teacher":
        if prefix is None or onset is None:
            raise ValueError(f"prefix_labels={scheme} needs the prefix positions and the teacher onsets")
        unsafe, found = y[:, None] == 1, onset[:, None] >= 0
        target = (unsafe & found & (prefix >= onset[:, None])).to(logits.dtype)
        use = valid
    else:
        raise ValueError(f"prefix_labels={scheme!r}: expected teacher, safe or none")
    if not bool(use.any()):
        return logits.new_zeros(())
    per_row = weight / use.sum(1, keepdim=True).clamp(min=1)
    bce = F.binary_cross_entropy_with_logits(logits, target.nan_to_num(0.0), pos_weight=pos_weight[1], reduction="none")
    return (bce * per_row * use).sum()


def micro_batches(idx: list[int], lengths: list[int], micro_batch_tokens: int | None, micro_batch_seqs: int | None = None) -> list[list[int]]:
    """Split a batch into contiguous parts whose padded size (len x longest) is at most micro_batch_tokens and that
    hold at most micro_batch_seqs sequences (None = no limit)."""
    if micro_batch_tokens is None and micro_batch_seqs is None:
        return [idx]
    parts, current = [], []
    for i in idx:
        if current and (
            (micro_batch_tokens is not None and (len(current) + 1) * max(lengths[j] for j in [*current, i]) > micro_batch_tokens)
            or (micro_batch_seqs is not None and len(current) >= micro_batch_seqs)
        ):
            parts.append(current)
            current = []
        current.append(i)
    return parts + [current]


def class_pos_weight(labels: Tensor) -> Tensor:
    pos = labels.nanmean(0).clamp(1e-3, 1 - 1e-3)
    return (1 - pos) / pos


def epoch_batches(lengths: list[int], batch_tokens: int, pool_size: int, rng: np.random.Generator) -> list[list[int]]:
    """Shuffle, sort by length within pools (little padding), cut by token budget, shuffle the batches."""
    order = rng.permutation(len(lengths))
    batches = []
    for start in range(0, len(order), pool_size):
        pool = order[start : start + pool_size]
        batches += [[int(pool[i]) for i in b] for b in token_budget_batches([lengths[i] for i in pool], batch_tokens)]
    rng.shuffle(batches)
    return batches


def build_trainable(cfg: Qwen3Config, vc: VariantConfig) -> Verifier | GeneratorProbe:
    if vc.probe is not None:
        return GeneratorProbe(cfg, vc.probe)
    if vc.verifier.head.norm == "standardize":
        raise NotImplementedError("verifier heads support norm none / rms (standardize needs fitted statistics)")
    return Verifier(cfg, vc.verifier)


def capture_chunks(backbone: Qwen3Min, first: int, rendered: list[Rendered], labels: Tensor, pad_id: int, batch_tokens: int) -> list[tuple]:
    """(frozen generator captures, label positions, labels) per token-budget batch, computed once. For small sets
    trained for many steps (the overfit test); the captures stay on the GPU."""
    chunks = []
    for idx in token_budget_batches([len(r.input_ids) for r in rendered], batch_tokens):
        part = [rendered[i] for i in idx]
        ids, _ = collate_right_padded(part, pad_id)
        device = backbone.device
        chunks.append((generator_captures(backbone, ids.to(device), first), label_positions(part).to(device), labels[idx].to(device)))
    return chunks


def fit_full_batch(verifier: Verifier, backbone: Qwen3Min, chunks: list[tuple], steps: int, lr: float, head_lr: float, pos_weight: Tensor) -> None:
    """Full-batch AdamW (gradients accumulated over the chunks), no schedule."""
    opt = torch.optim.AdamW([{"params": verifier.lora.parameters(), "lr": lr}, {"params": verifier.head.parameters(), "lr": head_lr}])
    n = sum(len(y) for *_, y in chunks)
    verifier.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        for captures, positions, y in chunks:
            (verifier_loss(verifier(backbone, captures, positions), y, pos_weight) * len(y) / n).backward()
        opt.step()
    verifier.eval()


@torch.no_grad()
def chunk_accuracy(verifier: Verifier, backbone: Qwen3Min, chunks: list[tuple]) -> list[float]:
    """Per-task accuracy at probability 0.5 over the labeled positions."""
    verifier.eval()
    pred = torch.cat([verifier(backbone, c, p).diagonal(dim1=1, dim2=2) > 0 for c, p, _ in chunks])
    y = torch.cat([y for *_, y in chunks])
    return [float((pred[:, t] == y[:, t].bool())[~y[:, t].isnan()].float().mean()) for t in range(len(TASKS))]


@torch.no_grad()
def score_variants(
    backbone: Qwen3Min, variants: dict[str, Verifier], rendered: list[Rendered], pad_id: int, batch_tokens: int
) -> dict[str, list[Scored]]:
    """Per variant and example: sigmoid prompt-harm score at the prompt end, response-harm score at the response end."""
    for v in variants.values():
        v.eval()
    first = min(v.first for v in variants.values())
    out: dict[str, list[Scored | None]] = {name: [None] * len(rendered) for name in variants}
    for idx in token_budget_batches([len(r.input_ids) for r in rendered], batch_tokens):
        batch = [rendered[i] for i in idx]
        ids, _ = collate_right_padded(batch, pad_id)
        captures = generator_captures(backbone, ids.to(backbone.device), first)
        positions = label_positions(batch).to(backbone.device)
        for name, v in variants.items():
            probs = v(backbone, captures, positions).diagonal(dim1=1, dim2=2).sigmoid().cpu()
            for row, i in enumerate(idx):
                has_response = rendered[i].response_end is not None
                out[name][i] = Scored(prompt_score=float(probs[row, 0]), response_score=float(probs[row, 1]) if has_response else None)
    for v in variants.values():
        v.train()
    return out  # type: ignore[return-value]


@dataclass
class Trainee:
    verifier: Verifier
    opt: torch.optim.Optimizer
    sched: torch.optim.lr_scheduler.LambdaLR
    loss_sum: Tensor
    loss_n: int = 0
    best: float = -math.inf
    best_step: int = -1
    best_state: dict | None = None
    history: list[tuple[int, float]] = field(default_factory=list)


def lr_lambda(total: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))

    return f


def main(argv: list[str] | None = None) -> dict:
    cfg = config_from_cli(TrainVerifierConfig, argv, default_config=str(CONFIGS_ROOT / "train" / "tora.yaml"))
    require_visible_devices()
    seed = cfg.run.seed
    seed_everything(seed)
    tc = cfg.train
    variant_cfgs = expand_variants(tc)
    with RunLogger(cfg.run.phase, make_run_id(cfg.run.name), to_dict(cfg), seed, cfg.run.results_dir) as log:
        backbone = Qwen3Min.from_pretrained(
            cfg.model.repo_id,
            revision=cfg.model.revision,
            dtype=getattr(torch, cfg.model.dtype),
            max_seq_len=max(cfg.model.max_seq_len, tc.max_len),
            hf_home=cfg.run.hf_home,
        )
        device = backbone.device
        fmt = ChatFormatter(load_tokenizer(backbone.source_path))

        t0 = time.perf_counter()
        data: dict[tuple[str, str], tuple[list[Example], list[Rendered]]] = {}
        needed = [(n, "train") for n in tc.train_datasets] + [(n, "val") for n in tc.train_datasets + tc.eval_datasets]
        needed += [(n, "test") for n in tc.eval_datasets]
        for name, split in dict.fromkeys(needed):
            if not processed_path(cfg.run.data_dir, name, split).exists():
                if split == "val" and name not in tc.train_datasets:
                    continue  # e.g. BeaverTails: test only
                raise FileNotFoundError(f"{name}/{split} not prepared; run `python -m tora.data.prepare`")
            examples = load_processed(cfg.run.data_dir, name, split)
            data[(name, split)] = (examples, [fmt.render_example(e, max_len=tc.max_len) for e in examples])
        print(f"rendered {sum(len(v[0]) for v in data.values())} examples in {time.perf_counter() - t0:.0f}s", flush=True)

        train_rendered: list[Rendered] = []
        train_ids: list[str] = []
        label_parts, conflicts = [], {}
        for name in tc.train_datasets:
            examples, rendered = data[(name, "train")]
            keep, labels, conflicts[name] = training_sequences(examples, tc.dedupe_prompts)
            train_rendered += [rendered[i] for i in keep]
            train_ids += [examples[i].id for i in keep]
            label_parts.append(labels)
        train_labels = torch.cat(label_parts)
        if tc.train_fraction < 1.0:
            sub = np.sort(np.random.default_rng(seed).permutation(len(train_rendered))[: int(tc.train_fraction * len(train_rendered))])
            train_rendered, train_ids, train_labels = [train_rendered[i] for i in sub], [train_ids[i] for i in sub], train_labels[sub]
        pos_weight = class_pos_weight(train_labels).to(device)
        lengths = [len(r.input_ids) for r in train_rendered]
        prefixes = tc.prefix_positions != 0
        train_prefix = None  # [N, n] sampled positions; None with prefix_positions = -1 (every position in train_spans)
        train_spans = response_spans(train_rendered, train_labels)
        if prefixes:
            if tc.prefix_positions < -1:
                raise ValueError(f"train.prefix_positions={tc.prefix_positions}: expected > 0, 0 (off) or -1 (every position)")
            if tc.prefix_positions > 0:
                train_prefix = sample_prefix_positions(train_rendered, train_labels, tc.prefix_positions, np.random.default_rng(seed + 1))
                n_prefix, n_resp = int((train_prefix >= 0).sum()), int((train_prefix >= 0).any(1).sum())
            else:
                n_prefix, n_resp = int((train_spans[:, 1] - train_spans[:, 0]).sum()), int((train_spans[:, 1] > train_spans[:, 0]).sum())
            print(f"prefix positions: {n_prefix} over {n_resp} responses", flush=True)
        train_onset = None
        if any(vc.prefix_labels == "teacher" for vc in variant_cfgs.values()) and prefixes:
            if not tc.prefix_teacher:
                raise ValueError("prefix_labels teacher needs train.prefix_teacher (scripts/teacher_onsets.py)")
            train_onset = teacher_onsets(train_rendered, train_labels, train_ids, load_onsets(tc.prefix_teacher))
            unsafe, found = train_labels[:, 1] == 1, train_onset >= 0
            if train_prefix is not None:
                after = int(((train_prefix >= train_onset[:, None]) & found[:, None] & (train_prefix >= 0)).sum())
                n_unsafe = int((train_prefix[unsafe] >= 0).sum())
            else:
                after = int((train_spans[found, 1] - train_onset[found]).sum())
                n_unsafe = int((train_spans[unsafe, 1] - train_spans[unsafe, 0]).sum())
            print(f"teacher onsets: {int(found.sum())} of {int(unsafe.sum())} unsafe responses flagged before their last "
                  f"token; {after} of {n_unsafe} unsafe-response prefix positions at or after the onset", flush=True)
        stats = {
            "train_sequences": len(train_rendered),
            "train_tokens": sum(lengths),
            "train_rows": {t: int((~train_labels[:, i].isnan()).sum()) for i, t in enumerate(TASKS)},
            "train_pos_frac": {t: float(train_labels[:, i].nanmean()) for i, t in enumerate(TASKS)},
            "prompt_label_conflicts_dropped": conflicts,
            "variants": {name: asdict(vc) for name, vc in variant_cfgs.items()},
        }
        log.log({"data": stats})
        print("training data:", {k: v for k, v in stats.items() if k != "variants"}, flush=True)

        rng = np.random.default_rng(seed)
        schedule = [epoch_batches(lengths, tc.batch_tokens, tc.pool_size, rng) for _ in range(tc.epochs)]
        total = sum(len(b) for b in schedule)
        warmup = max(1, int(tc.warmup_frac * total))
        eval_at = {sum(len(b) for b in schedule[:e]) + (k * len(schedule[e])) // tc.evals_per_epoch - 1 for e in range(tc.epochs) for k in range(1, tc.evals_per_epoch + 1)}

        trainees: dict[str, Trainee] = {}
        for name, vc in variant_cfgs.items():
            torch.manual_seed(seed)  # a variant's init depends only on the seed and its own shapes
            v = build_trainable(backbone.cfg, vc).to(device)
            groups = [{"params": list(v.lora.parameters()), "lr": vc.lr}, {"params": list(v.head.parameters()), "lr": vc.head_lr}]
            opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=vc.weight_decay)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, warmup))
            trainees[name] = Trainee(v, opt, sched, torch.zeros((), device=device))
        first = min(t.verifier.first for t in trainees.values())
        n_params = {name: sum(p.numel() for p in t.verifier.parameters()) for name, t in trainees.items()}
        print(f"{len(trainees)} variant(s), {total} steps ({tc.epochs} epochs), trainable params: {n_params}", flush=True)

        val_keys = [(n, "val") for n in tc.train_datasets]
        val_rendered = [r for k in val_keys for r in data[k][1]]

        def run_eval(step: int) -> None:
            t_eval = time.perf_counter()
            scored = score_variants(backbone, {n: t.verifier for n, t in trainees.items()}, val_rendered, fmt.pad_id, tc.eval_batch_tokens)
            for name, t in trainees.items():
                splits, start = [], 0
                for k in val_keys:
                    examples = data[k][0]
                    splits.append(list(zip(examples, scored[name][start : start + len(examples)])))
                    start += len(examples)
                score = mean_auprc(splits)
                train_loss = float(t.loss_sum) / max(1, t.loss_n)
                t.loss_sum.zero_()
                t.loss_n = 0
                t.history.append((step, score))
                log.log({"variant": name, "val_auprc": score, "train_loss": train_loss}, step=step)
                if score > t.best:
                    t.best, t.best_step = score, step
                    t.best_state = {k: v.detach().cpu().clone() for k, v in t.verifier.state_dict().items()}
            ranked = sorted(trainees.items(), key=lambda kv: -kv[1].history[-1][1])
            print(f"  [eval @ step {step + 1}/{total}, {time.perf_counter() - t_eval:.0f}s] val AUPRC: "
                  + ", ".join(f"{n} {t.history[-1][1]:.4f}" for n, t in ranked), flush=True)

        step, t_train, last_log, tokens = 0, time.perf_counter(), time.perf_counter(), 0
        for epoch, batches in enumerate(schedule):
            for idx in batches:
                for t in trainees.values():
                    t.opt.zero_grad(set_to_none=True)
                n_valid = int((~train_labels[idx].isnan()).sum())
                for part in micro_batches(idx, lengths, tc.micro_batch_tokens, tc.micro_batch_seqs):
                    batch = [train_rendered[i] for i in part]
                    ids, _ = collate_right_padded(batch, fmt.pad_id)
                    captures = generator_captures(backbone, ids.to(device), first)
                    positions = label_positions(batch).to(device)
                    y = train_labels[part].to(device)
                    n_part = int((~y.isnan()).sum())
                    if train_prefix is not None:  # the prefix positions follow the two label positions
                        prefix = train_prefix[part].to(device)
                        prefix_valid = prefix >= 0
                        read_at = torch.cat([positions, torch.where(prefix_valid, prefix, positions[:, 1:])], dim=1)
                    elif prefixes:  # every position: all rows at once, then the label positions picked out in front
                        span = train_spans[part].to(device)
                        prefix = torch.arange(ids.shape[1], device=device).expand(len(part), -1)
                        prefix_valid = (prefix >= span[:, :1]) & (prefix < span[:, 1:])
                    for name, t in trainees.items():
                        if not prefixes:
                            loss = verifier_loss(t.verifier(backbone, captures, positions), y, pos_weight) * n_part / n_valid
                        else:
                            vc = variant_cfgs[name]
                            if train_prefix is not None:
                                logits = t.verifier(backbone, captures, read_at)
                            else:
                                every = t.verifier(backbone, captures, None)
                                logits = torch.cat([gather_positions(every, positions), every], dim=1)
                            onset = train_onset[part].to(device) if train_onset is not None else None
                            prefix_sum = prefix_loss(logits[:, 2:, 1], y[:, 1], prefix_valid, vc.prefix_labels, vc.prefix_weight, pos_weight, prefix, onset)
                            loss = (verifier_loss(logits[:, :2], y, pos_weight) * n_part + prefix_sum) / n_valid
                        loss.backward()
                        t.loss_sum += loss.detach()
                    del captures
                    tokens += ids.numel()
                for t in trainees.values():
                    torch.nn.utils.clip_grad_norm_(t.verifier.parameters(), tc.grad_clip)
                    t.opt.step()
                    t.sched.step()
                    t.loss_n += 1
                if time.perf_counter() - last_log > tc.log_every_s:
                    last_log = time.perf_counter()
                    rate = tokens / (last_log - t_train)
                    eta = (total - step - 1) * (last_log - t_train) / (step + 1)
                    print(f"  epoch {epoch} step {step + 1}/{total}  {rate:,.0f} padded tok/s  ETA {eta / 60:.0f} min  "
                          f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
                if step in eval_at:
                    run_eval(step)
                step += 1
        train_s = time.perf_counter() - t_train

        for t in trainees.values():
            t.verifier.load_state_dict(t.best_state)
        eval_keys = [(n, s) for n in tc.eval_datasets for s in ("test", "val") if (n, s) in data]
        scored_all = {k: score_variants(backbone, {n: t.verifier for n, t in trainees.items()}, data[k][1], fmt.pad_id, tc.eval_batch_tokens) for k in eval_keys}
        out_dir = log.path.with_suffix("")
        summaries = {}
        for name, t in trainees.items():
            scored = {k: list(zip(data[k][0], scored_all[k][name])) for k in eval_keys}
            results = evaluate_scored(scored, tc.eval_datasets, tc.subsets)
            log.log({
                "variant": name, "config": asdict(variant_cfgs[name]), "best_step": t.best_step, "best_val_auprc": t.best,
                "val_auprc_history": t.history, "results": results, "train_s": train_s, "trainable_params": n_params[name],
            })
            summaries[name] = summarize([results])
            if tc.save_states:
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"variant": asdict(variant_cfgs[name]), "model": model_tag(cfg.model), "seed": seed, "state_dict": t.best_state}, out_dir / f"{name}.pt")
            if tc.save_scores:
                for (dataset, split), pairs in scored.items():
                    save_scores(out_dir / f"{name}__{dataset}__{split}.jsonl", pairs, with_tokens=False)
        log.log({"summary": {name: s for name, s in summaries.items()}, "seed": seed, "train_s": train_s})

        for name in sorted(trainees, key=lambda n: -trainees[n].best):
            print_summary(f"{name} (best val AUPRC {trainees[name].best:.4f} at step {trainees[name].best_step + 1})", summaries[name], 1)
        print(f"\ntrained {total} steps in {train_s / 60:.1f} min; peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    return {"summaries": summaries, "results_path": log.path, "best": {n: t.best for n, t in trainees.items()}}


if __name__ == "__main__":
    main()
