"""Dataset checks (S6a-S6d) and the split/dedupe operations they police."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np

from tora.data.formatting import ChatFormatter, Rendered
from tora.data.schema import Example


def normalize_prompt(prompt: str) -> str:
    """Key for duplicate detection: case- and whitespace-insensitive, so exact duplicates are a subset."""
    return " ".join(prompt.split()).casefold()


# --- S6a ------------------------------------------------------------------------------------------


def split_stats(examples: list[Example]) -> dict:
    """Per-split counts and label balance."""
    stats: dict = {}
    for split in sorted({e.split for e in examples}):
        exs = [e for e in examples if e.split == split]
        p = [e.prompt_label for e in exs if e.prompt_label is not None]
        r = [e.response_label for e in exs if e.response_label is not None]
        stats[split] = {
            "n": len(exs),
            "unique_prompts": len({normalize_prompt(e.prompt) for e in exs}),
            "n_prompt_labeled": len(p),
            "prompt_harmful_frac": float(np.mean(p)) if p else None,
            "n_with_response": sum(e.response is not None for e in exs),
            "n_response_labeled": len(r),
            "response_harmful_frac": float(np.mean(r)) if r else None,
        }
    return stats


def format_stats(name: str, stats: dict) -> str:
    def frac(x):
        return "   -  " if x is None else f"{x:6.1%}"

    lines = [f"{name}:", f"  {'split':6} {'n':>7} {'uniq.prompts':>12} {'prompt lab.':>11} {'harmful':>7} {'resp lab.':>9} {'harmful':>7}"]
    for split, s in stats.items():
        lines.append(
            f"  {split:6} {s['n']:7d} {s['unique_prompts']:12d} {s['n_prompt_labeled']:11d} {frac(s['prompt_harmful_frac'])} "
            f"{s['n_response_labeled']:9d} {frac(s['response_harmful_frac'])}"
        )
    return "\n".join(lines)


# --- validation holdout ---------------------------------------------------------------------------


def holdout_by_prompt(examples: list[Example], fraction: float, seed: int) -> list[Example]:
    """Move `fraction` of train to val, grouping by unique prompt and stratifying by the group's labels."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        if e.split == "train":
            groups[normalize_prompt(e.prompt)].append(i)
    strata: dict[tuple, list[str]] = defaultdict(list)
    for key, idx in groups.items():
        first = examples[idx[0]]
        strata[(first.prompt_label, first.response_label)].append(key)
    rng = random.Random(seed)
    to_val: set[int] = set()
    for stratum in sorted(strata, key=str):
        keys = sorted(strata[stratum])
        rng.shuffle(keys)
        for key in keys[: round(fraction * len(keys))]:
            to_val.update(groups[key])
    return [replace(e, split="val") if i in to_val else e for i, e in enumerate(examples)]


# --- S6b ------------------------------------------------------------------------------------------


def remove_overlap(datasets: dict[str, list[Example]], held: str, drop: tuple[str, ...]) -> tuple[dict[str, list[Example]], dict]:
    """Drop examples of the `drop` splits whose prompt appears in ANY dataset's `held` split. Returns (datasets,
    removed counts)."""
    keys = {normalize_prompt(e.prompt) for exs in datasets.values() for e in exs if e.split == held}
    cleaned, removed = {}, {}
    for name, exs in datasets.items():
        gone = [e.split in drop and normalize_prompt(e.prompt) in keys for e in exs]
        cleaned[name] = [e for e, g in zip(exs, gone) if not g]
        removed[name] = dict(Counter(e.split for e, g in zip(exs, gone) if g))
    return cleaned, removed


def remove_test_overlap(datasets: dict[str, list[Example]]) -> tuple[dict[str, list[Example]], dict]:
    """Drop train/val examples whose prompt appears in ANY dataset's test split."""
    return remove_overlap(datasets, "test", ("train", "val"))


def cross_split_duplicates(datasets: dict[str, list[Example]], a: str = "train", b: str = "test", exact: bool = True) -> int:
    key = (lambda p: p) if exact else normalize_prompt
    in_a = {key(e.prompt) for exs in datasets.values() for e in exs if e.split == a}
    in_b = {key(e.prompt) for exs in datasets.values() for e in exs if e.split == b}
    return len(in_a & in_b)


# --- S6c ------------------------------------------------------------------------------------------


def check_label_positions(fmt: ChatFormatter, examples: list[Example], n: int, seed: int) -> list[dict]:
    """Render n random examples and assert the label indices sit on the last prompt / last response token.

    Independent of render(): the expected tokens come from tokenizing the prompt part and the response on
    their own. Returns per-example records (decoded tokens around each label position) for printing.
    """
    rng = random.Random(seed)
    sample = rng.sample(examples, min(n, len(examples)))
    tok = fmt.tokenizer
    records = []
    for e in sample:
        r = fmt.render_example(e)
        prompt_text = fmt.prompt_text(e.prompt, e.meta.get("thinking", False))
        prompt_ids = fmt.encode(prompt_text)
        assert r.prompt_end == len(prompt_ids) - 1, e.id
        assert r.input_ids[r.prompt_end] == prompt_ids[-1], e.id
        assert tok.decode(r.input_ids[: r.prompt_end + 1]) == prompt_text, e.id
        rec = {"id": e.id, "prompt_end_token": tok.decode(r.input_ids[r.prompt_end]), "before_prompt_end": tok.decode(r.input_ids[max(0, r.prompt_end - 6) : r.prompt_end])}
        if e.response is not None:
            response_ids = fmt.encode(e.response)
            assert r.response_end == len(r.input_ids) - 1, e.id
            assert r.input_ids[r.response_end] == response_ids[-1], e.id
            assert r.input_ids[r.prompt_end + 1 :] == response_ids, e.id
            rec |= {
                "response_end_token": tok.decode(r.input_ids[r.response_end]),
                "response_tail": tok.decode(r.input_ids[max(r.prompt_end + 1, r.response_end - 8) : r.response_end + 1]),
                "response_text_tail": e.response[-40:],
            }
        records.append(rec)
    return records


def check_reference_tokens(fmt: ChatFormatter, examples: list[Example]) -> int:
    """Examples that carry their source's own token stream (meta["input_ids"], Qwen3GuardTest) must render to exactly
    it plus the turn-closing tokens, so token spans and Think's reasoning tokens line up with the official evaluation.
    Returns the number checked."""
    checked = 0
    for e in examples:
        if "input_ids" in e.meta:
            if fmt.render_example(e).input_ids + fmt.closing_ids() != e.meta["input_ids"]:
                raise AssertionError(f"{e.id}: our rendering differs from the dataset's input_ids")
            checked += 1
    return checked


# --- S6d ------------------------------------------------------------------------------------------


def length_stats(lengths: list[int], max_len: int) -> dict:
    arr = np.asarray(lengths)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        **{f"p{q}": float(np.percentile(arr, q)) for q in (50, 90, 99, 99.9)},
        "max": int(arr.max()),
        "max_len": max_len,
        "frac_over_max_len": float((arr > max_len).mean()),
    }


def truncation_stats(rendered: list[Rendered]) -> dict:
    n = len(rendered)
    return {
        "frac_prompt_truncated": sum(r.prompt_trimmed > 0 for r in rendered) / n,
        "frac_response_truncated": sum(r.response_trimmed > 0 for r in rendered) / n,
        "max_prompt_tokens_dropped": max((r.prompt_trimmed for r in rendered), default=0),
    }


def text_histogram(lengths: list[int], edges: list[int]) -> str:
    counts = np.histogram(lengths, bins=edges + [max(max(lengths), edges[-1]) + 1])[0]
    width = max(counts.max(), 1)
    lines = []
    for lo, c in zip(edges, counts):
        lines.append(f"  >= {lo:5d}: {c:7d} {'#' * int(40 * c / width)}")
    return "\n".join(lines)
