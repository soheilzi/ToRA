"""One loader per dataset -> unified Examples.

Each loader prints the raw schema it sees and checks it against what the mapping below expects,
raising (with the actual columns) on any mismatch. Schemas as inspected on the Hub (Oct 2026):

wildguardmix  allenai/wildguardmix (gated), {train,test}/*.parquet: prompt, adversarial, response,
              prompt_harm_label / response_harm_label in {harmful, unharmful, None},
              response_refusal_label in {refusal, compliance, None}, subcategory
aegis2        nvidia/Aegis-AI-Content-Safety-Dataset-2.0, {train,validation,test}.json: id, prompt,
              response (None or "" when absent), prompt_label / response_label in {safe, unsafe, None},
              violated_categories, *_label_source; some prompts are the literal 'REDACTED'
beavertails   PKU-Alignment/BeaverTails, round0/{30k,330k}/{train,test}.jsonl.{gz,xz}: prompt,
              response, is_safe (bool, response-level only), category (dict of 14 bools)
qwen3guardtest Qwen/Qwen3GuardTest, {response_loc,thinking,thinking_loc}.jsonl: message (user,
              assistant), input_ids (Qwen3 chat template), unsafe_start_index / unsafe_end_index
              (positions in input_ids), unsafe_type, label in {Safe, Unsafe}
singstreambench inclusionAI/SingStreamBench, SingStreamBench.jsonl: Query, Response, Query_Label /
              Response_Label in {safe, unsafe}, Unsafe_Start_Index (character offset into Response)

WildGuard's evaluation suite (allenai/safety-eval, evaluation/tasks/classification/<task>/__init__.py), which
Qwen3Guard "generally follows" for its English benchmarks; each loader reproduces that task's load():
saferlhf      PKU-Alignment/PKU-SafeRLHF@v0, round0/test.jsonl.xz: prompt, response_0/1, is_response_0/1_safe (bool)
harmbench_*   safety-eval files: harmbench_text_behaviors_val_set.json ({behavior_id: [test_case, generation,
              experiment, human_0..2 (0/1), ...]}) and harmbench_behaviors_text_test.csv (Behavior,
              FunctionalCategory, ...)
openai_mod    safety-eval file openai_moderation_evaluation_dataset.jsonl: prompt, S, H, V, HR, SH, S3, H2, V2
              (0/1, missing when not annotated)
toxicchat     lmsys/toxic-chat, data/0124/toxic-chat_annotation_{train,test}.csv: user_input, model_output,
              human_annotation (bool), toxicity (0/1), jailbreaking (0/1), openai_moderation
aegis1        nvidia/Aegis-AI-Content-Safety-Dataset-1.0, *_{train,test}_release_0418_v1.parquet: text, text_type
              (user_message | llm_response | combined | multi_turn), labels_0..labels_4 ('Safe' or categories)
simplesafetytests Bertievidgen/SimpleSafetyTests, sst_test_cases.csv: prompt, harm_area, category (all harmful)
xstest_response allenai/xstest-response (gated), data/response_harmfulness-*.parquet: prompt, response, label in
              {harmful, unharmful}, prompt_type in {prompt_harmful, prompt_safe}, prompt_harm_category

ExpGuard's fork of that suite (brightjade/ExpGuard @9ed9185, safety-eval/evaluation/tasks/classification/), which adds
the two sets SingProbe reports and the original paper's XSTest prompts:
xstest_prompt Paul/XSTest, xstest_prompts.csv: id, prompt, type, label in {safe, unsafe}, focus, note
expguard      6rightjade/expguardmix (gated), expguard{train,test}.parquet: prompt, response ("" when absent),
              prompt_label in {safe, unsafe}, response_label in {safe, unsafe, ""}, prompt_category,
              response_category, domain, scenario
"""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import random
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from tora.data.schema import Example
from tora.utils.config import DataConfig
from tora.utils.hf import access_error, hub_cache_dir, hf_token


def _download(cfg: DataConfig, filename: str, hf_home: str | None) -> Path:
    try:
        return Path(
            hf_hub_download(
                cfg.repo_id,
                filename,
                repo_type="dataset",
                revision=cfg.revision,
                cache_dir=hub_cache_dir(hf_home),
                token=hf_token(),
            )
        )
    except (GatedRepoError, RepositoryNotFoundError) as err:
        raise access_error(cfg.repo_id, err) from err


def _check_columns(source: str, columns: set[str], required: set[str]) -> None:
    print(f"[{source}] columns: {sorted(columns)}")
    missing = required - columns
    if missing:
        raise ValueError(f"[{source}] schema changed: missing {sorted(missing)}; columns are {sorted(columns)}")


def _binary(value, positive: str, negative: str, where: str) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if value == positive:
        return 1
    if value == negative:
        return 0
    raise ValueError(f"{where}: unexpected label {value!r} (expected {positive!r}, {negative!r} or None)")


def _text(value) -> str | None:
    """None / NaN / empty or whitespace-only -> None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value if value.strip() else None


# --- WildGuardMix ---------------------------------------------------------------------------------

WILDGUARD_FILES = {"wildguardtrain": "train/wildguard_train.parquet", "wildguardtest": "test/wildguard_test.parquet"}


def wildguard_record(rec: dict, idx: int, source_split: str, split: str) -> Example:
    where = f"wildguardmix/{source_split}[{idx}]"
    response = _text(rec.get("response"))
    response_label = _binary(rec.get("response_harm_label"), "harmful", "unharmful", where)
    return Example(
        id=f"wildguard-{source_split}-{idx}",
        prompt=rec["prompt"],
        response=response,
        prompt_label=_binary(rec.get("prompt_harm_label"), "harmful", "unharmful", where),
        response_label=response_label if response is not None else None,
        source="wildguardmix",
        split=split,
        meta={
            "adversarial": bool(rec["adversarial"]) if rec.get("adversarial") is not None else None,
            "subcategory": rec.get("subcategory"),
            "response_refusal_label": rec.get("response_refusal_label"),
        },
    )


def load_wildguardmix(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    out = []
    for source_split, split in cfg.splits.items():
        df = pd.read_parquet(_download(cfg, WILDGUARD_FILES[source_split], hf_home))
        _check_columns(f"wildguardmix/{source_split}", set(df.columns), {"prompt", "response", "prompt_harm_label", "response_harm_label", "adversarial"})
        out += [wildguard_record(rec, i, source_split, split) for i, rec in enumerate(df.to_dict("records"))]
    return out


# --- Aegis 2.0 ------------------------------------------------------------------------------------


def aegis_record(rec: dict, source_split: str, split: str) -> Example:
    where = f"aegis2/{source_split}/{rec['id']}"
    response = _text(rec.get("response"))
    return Example(
        id=f"aegis2-{rec['id']}",
        prompt=rec["prompt"],
        response=response,
        prompt_label=_binary(rec["prompt_label"], "unsafe", "safe", where),
        # Aegis labels some empty-string responses 'safe'; with no response text there is nothing to classify.
        response_label=_binary(rec["response_label"], "unsafe", "safe", where) if response is not None else None,
        source="aegis2",
        split=split,
        meta={
            "violated_categories": rec.get("violated_categories"),
            "prompt_label_source": rec.get("prompt_label_source"),
            "response_label_source": rec.get("response_label_source"),
        },
    )


def load_aegis2(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    out = []
    for source_split, split in cfg.splits.items():
        records = json.loads(_download(cfg, f"{source_split}.json", hf_home).read_text())
        _check_columns(f"aegis2/{source_split}", set(records[0]), {"id", "prompt", "response", "prompt_label", "response_label"})
        redacted = [r for r in records if r["prompt"] == "REDACTED"]
        if not cfg.keep_redacted:
            records = [r for r in records if r["prompt"] != "REDACTED"]
        print(f"[aegis2/{source_split}] {'kept' if cfg.keep_redacted else 'dropped'} {len(redacted)} REDACTED prompts")
        out += [aegis_record(r, source_split, split) for r in records]
    return out


# --- BeaverTails ----------------------------------------------------------------------------------

BEAVERTAILS_FILES = {
    "30k_train": ("round0/30k/train.jsonl.gz", gzip.open),
    "30k_test": ("round0/30k/test.jsonl.gz", gzip.open),
    "330k_train": ("round0/330k/train.jsonl.xz", lzma.open),
    "330k_test": ("round0/330k/test.jsonl.xz", lzma.open),
}


def beavertails_record(rec: dict, idx: int, source_split: str, split: str) -> Example:
    if not isinstance(rec["is_safe"], bool):
        raise ValueError(f"beavertails/{source_split}[{idx}]: is_safe={rec['is_safe']!r}")
    response = _text(rec["response"])
    return Example(
        id=f"beavertails-{source_split}-{idx}",
        prompt=rec["prompt"],
        response=response,
        prompt_label=None,  # BeaverTails only labels the QA pair
        response_label=(0 if rec["is_safe"] else 1) if response is not None else None,
        source="beavertails",
        split=split,
        meta={"categories": sorted(k for k, v in rec["category"].items() if v)},
    )


def load_beavertails(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    out = []
    for source_split, split in cfg.splits.items():
        filename, opener = BEAVERTAILS_FILES[source_split]
        with opener(_download(cfg, filename, hf_home), "rt") as fh:
            records = [json.loads(line) for line in fh]
        _check_columns(f"beavertails/{source_split}", set(records[0]), {"prompt", "response", "is_safe", "category"})
        out += [beavertails_record(r, i, source_split, split) for i, r in enumerate(records)]
    return out


# --- Qwen3GuardTest -------------------------------------------------------------------------------


def qwen3guardtest_record(rec: dict, source_split: str, split: str) -> Example:
    where = f"qwen3guardtest/{source_split}/{rec['unique_id']}"
    roles = [m["role"] for m in rec["message"]]
    if roles != ["user", "assistant"]:
        raise ValueError(f"{where}: expected a user/assistant pair, got roles {roles}")
    meta = {"unique_id": rec["unique_id"], "unsafe_type": rec.get("unsafe_type"), "generator": rec.get("source")}
    if "unsafe_start_index" in rec:
        # Positions in the dataset's input_ids (= our rendering + '<|im_end|>\n'); Qwen's eval treats the span as
        # inclusive, and unsafe_end_index can point one past the last token.
        meta |= {k: rec[k] for k in ("input_ids", "unsafe_start_index", "unsafe_end_index")}
    return Example(
        id=f"qwen3guardtest-{source_split}-{rec['unique_id']}",
        prompt=rec["message"][0]["content"],
        response=rec["message"][1]["content"],
        prompt_label=None,
        response_label=_binary(rec["label"], "Unsafe", "Safe", where),
        source="qwen3guardtest",
        split=split,
        meta=meta,
    )


def think_record(rec: dict, split: str) -> Example:
    """Qwen3GuardTest `thinking` (Qwen3Guard's "Think" benchmark): the assistant turn is '<think>...</think>' + answer.
    The response is that turn as Qwen3's chat template renders it (reasoning stripped of surrounding newlines),
    minus the '<think>\\n' the thinking-mode prompt part ends with; with empty reasoning the template gives the
    ordinary non-thinking turn. The dataset's input_ids (unsafe items only) are kept to check the rendering."""
    where = f"qwen3guardtest/thinking/{rec['unique_id']}"
    if [m["role"] for m in rec["message"]] != ["user", "assistant"]:
        raise ValueError(f"{where}: expected a user/assistant pair")
    content = rec["message"][1]["content"]
    if not content.startswith("<think>") or "</think>" not in content:
        raise ValueError(f"{where}: expected '<think>...</think>' at the start of the assistant turn")
    reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].strip("\n")
    answer = content.split("</think>")[-1].lstrip("\n")
    meta = {"unique_id": rec["unique_id"], "unsafe_type": rec.get("unsafe_type"), "generator": rec.get("source"), "thinking": bool(reasoning)}
    if rec.get("input_ids") is not None:
        meta["input_ids"] = rec["input_ids"]
    return Example(
        id=f"think-{rec['unique_id']}",
        prompt=rec["message"][0]["content"],
        response=reasoning + "\n</think>\n\n" + answer if reasoning else answer,
        prompt_label=None,
        response_label=_binary(rec["label"], "Unsafe", "Safe", where),
        source="think",
        split=split,
        meta=meta,
    )


def load_think(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    out = []
    for source_split, split in cfg.splits.items():
        with _download(cfg, f"{source_split}.jsonl", hf_home).open() as fh:
            records = [json.loads(line) for line in fh]
        _check_columns(f"qwen3guardtest/{source_split}", set(records[0]), {"message", "input_ids", "label", "unique_id"})
        out += [think_record(r, split) for r in records]
    return out


def load_qwen3guardtest(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    out = []
    for source_split, split in cfg.splits.items():
        with _download(cfg, f"{source_split}.jsonl", hf_home).open() as fh:
            records = [json.loads(line) for line in fh]
        _check_columns(f"qwen3guardtest/{source_split}", set(records[0]), {"message", "input_ids", "label", "unique_id"})
        out += [qwen3guardtest_record(r, source_split, split) for r in records]
    return out


# --- SingStreamBench ------------------------------------------------------------------------------


def singstreambench_record(rec: dict, idx: int, split: str) -> Example:
    where = f"singstreambench[{idx}]"
    response_label = _binary(rec["Response_Label"], "unsafe", "safe", where)
    onset = rec["Unsafe_Start_Index"]
    if not 0 <= onset <= len(rec["Response"]) or (response_label == 0 and onset != len(rec["Response"])):
        raise ValueError(f"{where}: Unsafe_Start_Index={onset} inconsistent with label / response length")
    return Example(
        id=f"singstreambench-{idx}",
        prompt=rec["Query"],
        response=rec["Response"],
        prompt_label=_binary(rec["Query_Label"], "unsafe", "safe", where),
        response_label=response_label,
        source="singstreambench",
        split=split,
        meta={"unsafe_char_start": onset},  # character offset into the response; len(response) when safe
    )


def load_singstreambench(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    out = []
    for source_split, split in cfg.splits.items():
        with _download(cfg, f"{source_split}.jsonl", hf_home).open() as fh:
            records = [json.loads(line) for line in fh]
        _check_columns(f"singstreambench/{source_split}", set(records[0]), {"Query", "Response", "Query_Label", "Response_Label", "Unsafe_Start_Index"})
        out += [singstreambench_record(r, i, split) for i, r in enumerate(records)]
    return out


# --- WildGuard's evaluation suite -------------------------------------------------------------------

SAFETY_EVAL_SHA256 = {  # files under evaluation/tasks/classification/ of allenai/safety-eval
    "harmbench/harmbench_text_behaviors_val_set.json": "e740727b6874e492063e097122a9916833d57f9bd2ce3e86713ec188741a41c8",
    "harmbench/harmbench_behaviors_text_test.csv": "75d257b3e7428c52eb7b0154318f455af3e01b09a3794b5e2f3d36054f3c0e29",
    "openai_mod/openai_moderation_evaluation_dataset.jsonl": "f27c25f726fd0aa1b431118a68a4a6b9b5f0bb1ffd415f9767e9af59c4c422f4",
}


def _safety_eval_file(cfg: DataConfig, name: str, hf_home: str | None) -> Path:
    """A data file of allenai/safety-eval at the commit `cfg.revision`, cached next to the hub cache and checked
    against its pinned sha256."""
    if cfg.repo_id != "allenai/safety-eval" or cfg.revision is None or len(cfg.revision) != 40:
        raise ValueError(f"{cfg.name}: repo_id must be allenai/safety-eval with revision = a full commit hash")
    path = hub_cache_dir(hf_home).parent / "github" / cfg.repo_id / cfg.revision / name
    if not path.exists():
        url = f"https://raw.githubusercontent.com/{cfg.repo_id}/{cfg.revision}/evaluation/tasks/classification/{name}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SAFETY_EVAL_SHA256[name]:
        raise ValueError(f"{path}: sha256 {digest} != pinned {SAFETY_EVAL_SHA256[name]}")
    return path


def _only_test(cfg: DataConfig) -> str:
    if set(cfg.splits) != {"test"}:
        raise ValueError(f"{cfg.name}: only a 'test' source split exists, got {sorted(cfg.splits)}")
    return cfg.splits["test"]


SAFERLHF_FILES = {"train": "round0/train.jsonl.xz", "test": "round0/test.jsonl.xz"}


def load_saferlhf(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    """test (the WildGuard suite's set): pairs whose two responses differ in safety give one safe and one harmful
    response each; then 1,000 of each, at the same indices, drawn by random.seed(42); random.sample(range(n_safe), 1000).
    train: v0 train repeats each prompt ~21 times (297,394 pairs, 14,016 prompts), so one pair per unique prompt,
    drawn by random.Random(0), with both of its responses."""
    out = []
    for source_split, split in cfg.splits.items():
        if source_split not in SAFERLHF_FILES:
            raise ValueError(f"saferlhf: unknown source split {source_split!r}; known: {sorted(SAFERLHF_FILES)}")
        with lzma.open(_download(cfg, SAFERLHF_FILES[source_split], hf_home), "rt") as fh:
            records = [json.loads(line) for line in fh]
        _check_columns(f"saferlhf/{source_split}", set(records[0]), {"prompt", "response_0", "response_1", "is_response_0_safe", "is_response_1_safe"})
        for i, r in enumerate(records):
            flags = (r["is_response_0_safe"], r["is_response_1_safe"])
            if not all(isinstance(f, bool) for f in flags):
                raise ValueError(f"saferlhf/{source_split}[{i}]: is_response_*_safe = {flags}")
        out += _saferlhf_test(records, split) if source_split == "test" else saferlhf_one_pair_per_prompt(records, source_split, split)
    return out


def saferlhf_one_pair_per_prompt(records: list[dict], source_split: str, split: str) -> list[Example]:
    by_prompt: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_prompt.setdefault(r["prompt"], []).append(i)
    rng = random.Random(0)
    picked = [(i, k) for i in (rng.choice(idxs) for idxs in by_prompt.values()) for k in (0, 1)]
    empty = [(i, k) for i, k in picked if _text(records[i][f"response_{k}"]) is None]  # v0 train has 75 of 594,788
    if empty:
        print(f"[saferlhf/{source_split}] skipped {len(empty)} empty responses of the picked pairs")
    return [
        Example(id=f"saferlhf-{source_split}-{i}-{k}", prompt=records[i]["prompt"], response=records[i][f"response_{k}"],
                prompt_label=None, response_label=int(not records[i][f"is_response_{k}_safe"]), source="saferlhf", split=split)
        for i, k in picked if (i, k) not in empty
    ]


def _saferlhf_test(records: list[dict], split: str) -> list[Example]:
    safe, harmful = [], []
    for i, r in enumerate(records):
        flags = (r["is_response_0_safe"], r["is_response_1_safe"])
        if flags[0] == flags[1]:
            continue
        for k in (0, 1):
            (safe if flags[k] else harmful).append((i, k, r["prompt"], r[f"response_{k}"]))
    if len(safe) != len(harmful):
        raise AssertionError("each mixed pair gives one safe and one harmful response")
    picked = random.Random(42).sample(range(len(safe)), 1000)
    return [
        Example(id=f"saferlhf-test-{i}-{k}", prompt=prompt, response=response, prompt_label=None, response_label=label,
                source="saferlhf", split=split)
        for pool, label in ((safe, 0), (harmful, 1)) for (i, k, prompt, response) in (pool[j] for j in picked)
    ]


def load_harmbench_response(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    """HarmBench's human-labeled classifier val set: harmful when at least 2 of the 3 annotators say so."""
    split = _only_test(cfg)
    raw = json.loads(_safety_eval_file(cfg, "harmbench/harmbench_text_behaviors_val_set.json", hf_home).read_text())
    out = []
    for behavior, items in raw.items():
        for j, x in enumerate(items):
            if j == 0 and not out:
                _check_columns("harmbench_response", set(x), {"test_case", "generation", "experiment", "human_0", "human_1", "human_2"})
            votes = [int(x[f"human_{a}"]) for a in range(3)]
            if any(v not in (0, 1) for v in votes):
                raise ValueError(f"harmbench_response/{behavior}[{j}]: votes {votes}")
            out.append(Example(id=f"harmbench_response-{behavior}-{j}", prompt=x["test_case"], response=x["generation"],
                               prompt_label=None, response_label=int(sum(votes) > 1), source="harmbench", split=split,
                               meta={"adversarial": x["experiment"] != "default"}))
    return out


def load_harmbench_prompt(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    """HarmBench text test behaviors with FunctionalCategory standard or copyright, all harmful."""
    split = _only_test(cfg)
    df = pd.read_csv(_safety_eval_file(cfg, "harmbench/harmbench_behaviors_text_test.csv", hf_home))
    _check_columns("harmbench_prompt", set(df.columns), {"Behavior", "FunctionalCategory", "BehaviorID"})
    df = df[df["FunctionalCategory"].isin(["standard", "copyright"])]
    return [Example(id=f"harmbench_prompt-{r.BehaviorID}", prompt=r.Behavior, response=None, prompt_label=1, response_label=None,
                    source="harmbench", split=split, meta={"functional_category": r.FunctionalCategory}) for r in df.itertuples()]


OPENAI_MOD_CLASSES = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]


def load_openai_mod(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    """OpenAI's moderation evaluation set: harmful when any class is 1 (a missing class counts as 0)."""
    split = _only_test(cfg)
    df = pd.read_json(_safety_eval_file(cfg, "openai_mod/openai_moderation_evaluation_dataset.jsonl", hf_home), orient="records", lines=True)
    _check_columns("openai_mod", set(df.columns), {"prompt", *OPENAI_MOD_CLASSES})
    return [Example(id=f"openai_mod-{i}", prompt=row["prompt"], response=None,
                    prompt_label=int(any(row[c] == 1 for c in OPENAI_MOD_CLASSES)), response_label=None, source="openai_mod", split=split)
            for i, row in df.iterrows()]


TOXICCHAT_FILES = {"train": "data/0124/toxic-chat_annotation_train.csv", "test": "data/0124/toxic-chat_annotation_test.csv"}


def load_toxicchat(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    """toxicchat0124, human-annotated rows only (the WildGuard suite's test set); prompt label = toxicity."""
    out = []
    for source_split, split in cfg.splits.items():
        df = pd.read_csv(_download(cfg, TOXICCHAT_FILES[source_split], hf_home))
        _check_columns(f"toxicchat/{source_split}", set(df.columns), {"conv_id", "user_input", "human_annotation", "toxicity", "jailbreaking"})
        if df["human_annotation"].dtype != bool or not df["toxicity"].isin([0, 1]).all():
            raise ValueError(f"toxicchat/{source_split}: human_annotation / toxicity have unexpected values")
        df = df[df["human_annotation"]]
        out += [Example(id=f"toxicchat-{source_split}-{r.conv_id}", prompt=r.user_input, response=None, prompt_label=int(r.toxicity),
                        response_label=None, source="toxicchat", split=split, meta={"adversarial": bool(r.jailbreaking)})
                for r in df.itertuples()]
    return out


AEGIS1_FILES = {
    "train": "Content Moderation Extracted Annotations 02.08.24_train_release_0418_v1.parquet",
    "test": "Content Moderation Extracted Annotations 02.08.24_test_release_0418_v1.parquet",
}


def load_aegis1(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    """Aegis 1.0 user messages; harmful when at most one of labels_0..labels_2 is 'Safe' (the WildGuard suite's rule,
    also on rows with a 4th annotation). The text is stripped, as there."""
    out = []
    for source_split, split in cfg.splits.items():
        df = pd.read_parquet(_download(cfg, AEGIS1_FILES[source_split], hf_home))
        _check_columns(f"aegis1/{source_split}", set(df.columns), {"id", "text", "text_type", "labels_0", "labels_1", "labels_2"})
        df = df[df["text_type"] == "user_message"]
        lacking = df[["labels_0", "labels_1", "labels_2"]].isna().any(axis=1)
        if lacking.any() and split != "train":
            raise ValueError(f"aegis1/{source_split}: a user message lacks one of its first three labels")
        if lacking.any():  # train: 3 of 3,099 user messages have only 2 labels (num_annotations says 3)
            print(f"[aegis1/{source_split}] dropped {int(lacking.sum())} user messages with fewer than 3 labels")
            df = df[~lacking]
        out += [Example(id=f"aegis1-{source_split}-{r.id}", prompt=r.text.strip(), response=None,
                        prompt_label=int(sum(v == "Safe" for v in (r.labels_0, r.labels_1, r.labels_2)) <= 1),
                        response_label=None, source="aegis1", split=split) for r in df.itertuples()]
    return out


def load_simplesafetytests(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    split = _only_test(cfg)
    df = pd.read_csv(_download(cfg, "sst_test_cases.csv", hf_home))
    _check_columns("simplesafetytests", set(df.columns), {"id", "prompt", "harm_area", "category"})
    return [Example(id=f"simplesafetytests-{r.id}", prompt=r.prompt, response=None, prompt_label=1, response_label=None,
                    source="simplesafetytests", split=split, meta={"harm_area": r.harm_area}) for r in df.itertuples()]


def load_xstest_response(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    """XSTest-Response, split response_harmfulness (safety-eval xstest_response_harm): response label only."""
    split = _only_test(cfg)
    df = pd.read_parquet(_download(cfg, "data/response_harmfulness-00000-of-00001.parquet", hf_home))
    _check_columns("xstest_response", set(df.columns), {"prompt", "response", "label", "prompt_type", "prompt_harm_category"})
    out = []
    for i, r in enumerate(df.to_dict("records")):
        if _text(r["response"]) is None:
            raise ValueError(f"xstest_response[{i}]: empty response")
        out.append(Example(id=f"xstest_response-{i}", prompt=r["prompt"], response=r["response"], prompt_label=None,
                           response_label=_binary(r["label"], "harmful", "unharmful", f"xstest_response[{i}]"),
                           source="xstest", split=split,
                           meta={"prompt_type": r["prompt_type"], "prompt_harm_category": r["prompt_harm_category"]}))
    return out


def load_xstest_prompt(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    """The original XSTest prompts (ExpGuard's safety-eval xstest_prompt_harm): harmful when label is 'unsafe'."""
    split = _only_test(cfg)
    df = pd.read_csv(_download(cfg, "xstest_prompts.csv", hf_home))
    _check_columns("xstest_prompt", set(df.columns), {"id", "prompt", "type", "label"})
    return [Example(id=f"xstest_prompt-{r.id}", prompt=r.prompt, response=None,
                    prompt_label=_binary(r.label, "unsafe", "safe", f"xstest_prompt/{r.id}"), response_label=None,
                    source="xstest", split=split, meta={"type": r.type}) for r in df.itertuples()]


EXPGUARD_FILES = {"train": "expguardtrain.parquet", "test": "expguardtest.parquet"}


def expguard_record(rec: dict, idx: int, source_split: str, split: str) -> Example:
    """As in ExpGuard's expguardtest task: every row has a prompt label; a row with a non-empty response also has a
    response label. A response that is only whitespace, or a label without a response (or the reverse), raises."""
    where = f"expguard/{source_split}[{idx}]"
    response = _text(rec["response"])
    if (response is None) != (rec["response"] == "") or (response is None) != (rec["response_label"] == ""):
        raise ValueError(f"{where}: response {rec['response']!r:.40} vs response_label {rec['response_label']!r}")
    return Example(id=f"expguard-{source_split}-{idx}", prompt=rec["prompt"], response=response,
                   prompt_label=_binary(rec["prompt_label"], "unsafe", "safe", where),
                   response_label=None if response is None else _binary(rec["response_label"], "unsafe", "safe", where),
                   source="expguard", split=split, meta={"domain": rec["domain"]})


def load_expguard(cfg: DataConfig, hf_home: str | None) -> list[Example]:
    """ExpGuardMix: 1,188 of the 2,275 test rows have a response. Train gets the same mapping."""
    out = []
    for source_split, split in cfg.splits.items():
        df = pd.read_parquet(_download(cfg, EXPGUARD_FILES[source_split], hf_home))
        _check_columns(f"expguard/{source_split}", set(df.columns), {"prompt", "response", "prompt_label", "response_label", "domain"})
        out += [expguard_record(r, i, source_split, split) for i, r in enumerate(df.to_dict("records"))]
    return out


LOADERS: dict[str, Callable[[DataConfig, str | None], list[Example]]] = {
    "wildguardmix": load_wildguardmix,
    "aegis2": load_aegis2,
    "beavertails": load_beavertails,
    "qwen3guardtest": load_qwen3guardtest,
    "singstreambench": load_singstreambench,
    "think": load_think,
    "saferlhf": load_saferlhf,
    "harmbench_response": load_harmbench_response,
    "harmbench_prompt": load_harmbench_prompt,
    "openai_mod": load_openai_mod,
    "toxicchat": load_toxicchat,
    "aegis1": load_aegis1,
    "simplesafetytests": load_simplesafetytests,
    "xstest_response": load_xstest_response,
    "xstest_prompt": load_xstest_prompt,
    "expguard": load_expguard,
}


def load_examples(cfg: DataConfig, hf_home: str | None = None) -> list[Example]:
    if cfg.loader not in LOADERS:
        raise ValueError(f"unknown loader {cfg.loader!r}; known: {sorted(LOADERS)}")
    bad = set(cfg.splits.values()) - {"train", "val", "test"}
    if bad:
        raise ValueError(f"{cfg.name}: unified splits must be train/val/test, got {sorted(bad)}")
    return LOADERS[cfg.loader](cfg, hf_home)
