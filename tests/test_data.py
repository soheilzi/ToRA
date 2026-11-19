"""Loader record mappings (on records shaped like the inspected Hub schemas), splits, dedupe, truncation."""

import pytest

from tora.data import checks
from tora.data.formatting import ChatFormatter, load_tokenizer, token_budget_batches
from tora.data.loaders import (
    aegis_record,
    beavertails_record,
    expguard_record,
    qwen3guardtest_record,
    saferlhf_one_pair_per_prompt,
    singstreambench_record,
    think_record,
    wildguard_record,
)
from tora.data.schema import Example
from tora.utils.hf import download_model


@pytest.fixture(scope="module")
def fmt() -> ChatFormatter:
    return ChatFormatter(load_tokenizer(download_model("Qwen/Qwen3-0.6B")))


def ex(i, prompt, split="train", pl=0, response=None, rl=None):
    return Example(f"x{i}", prompt, response, pl, rl, "unit", split)


# --- loaders --------------------------------------------------------------------------------------


def test_aegis_record():
    rec = {"id": "a1", "prompt": "p", "response": "r", "prompt_label": "unsafe", "response_label": "safe"}
    e = aegis_record(rec, "test", "test")
    assert (e.prompt_label, e.response_label, e.split) == (1, 0, "test")
    empty = aegis_record(rec | {"response": "", "response_label": "safe"}, "test", "test")
    assert empty.response is None and empty.response_label is None
    with pytest.raises(ValueError, match="unexpected label"):
        aegis_record(rec | {"prompt_label": "needs caution"}, "test", "test")


def test_wildguard_record():
    rec = {"prompt": "p", "adversarial": True, "response": None, "prompt_harm_label": "harmful", "response_harm_label": None, "response_refusal_label": None, "subcategory": "x"}
    e = wildguard_record(rec, 3, "wildguardtest", "test")
    assert (e.prompt_label, e.response, e.response_label, e.meta["adversarial"]) == (1, None, None, True)
    e = wildguard_record(rec | {"response": "ok", "response_harm_label": "unharmful"}, 3, "wildguardtest", "test")
    assert e.response_label == 0


def test_beavertails_record():
    e = beavertails_record({"prompt": "p", "response": "r", "is_safe": False, "category": {"a": True, "b": False}}, 0, "30k_test", "test")
    assert (e.prompt_label, e.response_label, e.meta["categories"]) == (None, 1, ["a"])
    with pytest.raises(ValueError):
        beavertails_record({"prompt": "p", "response": "r", "is_safe": "no", "category": {}}, 0, "30k_test", "test")


def test_qwen3guardtest_record():
    rec = {"unique_id": 7, "message": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "r"}], "input_ids": [1, 2], "unsafe_start_index": 1, "unsafe_end_index": 1, "unsafe_type": "Violent", "label": "Unsafe", "source": "BeaverTails"}
    e = qwen3guardtest_record(rec, "response_loc", "test")
    assert (e.response_label, e.meta["unsafe_start_index"]) == (1, 1)
    with pytest.raises(ValueError, match="user/assistant"):
        qwen3guardtest_record(rec | {"message": rec["message"][:1]}, "response_loc", "test")


def test_think_record(fmt):
    user = {"role": "user", "content": "p"}
    rec = {"unique_id": 3, "message": [user, {"role": "assistant", "content": "<think>\n\nLet me see.\n\n</think>\nNo."}], "input_ids": None, "label": "Safe"}
    e = think_record(rec, "test")
    # the assistant turn as Qwen3's template renders it, minus the '<think>\n' that ends the thinking-mode prompt part
    official = fmt.tokenizer.apply_chat_template(rec["message"], tokenize=False, add_generation_prompt=False, enable_thinking=False)
    assert (e.meta["thinking"], e.response_label, "input_ids" in e.meta) == (True, 0, False)
    assert fmt.prompt_text("p", thinking=True) + e.response + "<|im_end|>\n" == official
    empty = think_record(rec | {"message": [user, {"role": "assistant", "content": "<think>\n\n</think>\n\nNo."}]}, "test")
    assert (empty.meta["thinking"], empty.response) == (False, "No.")  # rendered as an ordinary non-thinking turn
    with pytest.raises(ValueError, match="<think>"):
        think_record(rec | {"message": [user, {"role": "assistant", "content": "No."}]}, "test")


def test_saferlhf_one_pair_per_prompt():
    pair = lambda p, s0, s1: {"prompt": p, "response_0": f"{p}0", "response_1": f"{p}1", "is_response_0_safe": s0, "is_response_1_safe": s1}
    records = [pair("a", True, False), pair("a", True, True), pair("b", False, False), pair("a", False, True)]
    exs = saferlhf_one_pair_per_prompt(records, "train", "train")
    assert [e.prompt for e in exs] == ["a", "a", "b", "b"]
    assert [e.response_label for e in exs[2:]] == [1, 1] and exs[0].id[:-2] == exs[1].id[:-2]
    assert exs == saferlhf_one_pair_per_prompt(records, "train", "train")  # seeded
    records[2]["response_1"] = " "
    assert [e.response for e in saferlhf_one_pair_per_prompt(records, "train", "train")][2:] == ["b0"]  # empty: skipped


def test_expguard_record():
    rec = {"prompt": "p", "response": "r", "prompt_label": "unsafe", "response_label": "safe", "domain": "law"}
    e = expguard_record(rec, 0, "test", "test")
    assert (e.prompt_label, e.response, e.response_label) == (1, "r", 0)
    e = expguard_record(rec | {"response": "", "response_label": ""}, 0, "test", "test")
    assert (e.prompt_label, e.response, e.response_label) == (1, None, None)
    for bad in ({"response": ""}, {"response_label": ""}, {"response": " ", "response_label": ""}):
        with pytest.raises(ValueError, match="response"):
            expguard_record(rec | bad, 0, "test", "test")


def test_singstreambench_record():
    base = {"Query": "q", "Response": "abcdef", "Query_Label": "unsafe", "Response_Label": "unsafe", "Unsafe_Start_Index": 2}
    assert singstreambench_record(base, 0, "test").meta["unsafe_char_start"] == 2
    with pytest.raises(ValueError):  # a safe response must have its onset at len(response)
        singstreambench_record(base | {"Response_Label": "safe"}, 0, "test")


# --- splits and leakage ---------------------------------------------------------------------------


def test_holdout_by_prompt_keeps_groups_together():
    exs = [ex(i, f"prompt {i // 2}", pl=(i // 2) % 2) for i in range(400)]  # 200 prompts, 2 rows each
    out = checks.holdout_by_prompt(exs, 0.05, seed=0)
    by_prompt: dict = {}
    for e in out:
        by_prompt.setdefault(e.prompt, set()).add(e.split)
    assert all(len(s) == 1 for s in by_prompt.values())
    val_prompts = [p for p, s in by_prompt.items() if s == {"val"}]
    assert len(val_prompts) == 10
    assert sum(int(p.split()[1]) % 2 for p in val_prompts) == 5  # stratified by label
    assert checks.holdout_by_prompt(exs, 0.05, seed=0) == out  # deterministic


def test_remove_test_overlap_across_datasets():
    a = [ex(0, "Hello  World", "train"), ex(1, "unique", "train"), ex(2, "x", "val")]
    b = [ex(3, "hello world", "test"), ex(4, "X", "test")]
    cleaned, removed = checks.remove_test_overlap({"a": a, "b": b})
    assert [e.id for e in cleaned["a"]] == ["x1"]
    assert removed["a"] == {"train": 1, "val": 1}
    assert len(cleaned["b"]) == 2
    assert checks.cross_split_duplicates(cleaned, "train", "test", exact=False) == 0


def test_remove_train_in_val_across_datasets():
    a = [ex(0, "Hello  World", "train"), ex(1, "unique", "train"), ex(2, "x", "val")]
    b = [ex(3, "hello world", "val"), ex(4, "X", "train"), ex(5, "x", "test")]
    cleaned, removed = checks.remove_overlap({"a": a, "b": b}, "val", ("train",))
    assert [e.id for e in cleaned["a"]] == ["x1", "x2"] and [e.id for e in cleaned["b"]] == ["x3", "x5"]
    assert removed == {"a": {"train": 1}, "b": {"train": 1}}


def test_split_stats():
    stats = checks.split_stats([ex(0, "a", "test", 1, "r", 0), ex(1, "b", "test", 0)])
    assert stats["test"]["n"] == 2 and stats["test"]["prompt_harmful_frac"] == 0.5 and stats["test"]["n_response_labeled"] == 1


# --- formatting / truncation ----------------------------------------------------------------------


def test_truncation_drops_prompt_from_left(fmt):
    prompt = " ".join(f"word{i}" for i in range(400))
    full = fmt.render(prompt, "The answer is short.")
    r = fmt.render(prompt, "The answer is short.", max_len=128)
    assert len(r.input_ids) == 128 and r.prompt_trimmed == len(full.input_ids) - 128 and r.response_trimmed == 0
    assert r.input_ids[-len(fmt.encode("The answer is short.")) :] == fmt.encode("The answer is short.")
    assert r.input_ids[r.prompt_end] == full.input_ids[full.prompt_end]  # template tail kept
    text = fmt.tokenizer.decode(r.input_ids)
    assert text.startswith("<|im_start|>user\n") and "word399" in text and "word0 " not in text


def test_truncation_cuts_response_only_when_needed(fmt):
    response = " ".join(f"tok{i}" for i in range(300))
    r = fmt.render("short?", response, max_len=64)
    assert len(r.input_ids) == 64 and r.response_trimmed > 0
    assert r.input_ids[r.response_end] == fmt.encode(response)[-1]  # response end never cut
    with pytest.raises(ValueError):
        fmt.render(" ".join(["w"] * 50), None, max_len=8)  # the template alone does not fit


def test_label_position_check_runs(fmt):
    exs = [ex(i, f"Question {i}?", "test", 0, f"Answer {i}.", 0) for i in range(5)] + [ex(9, "Only a prompt", "test")]
    assert len(checks.check_label_positions(fmt, exs, n=6, seed=0)) == 6


def test_token_budget_batches():
    lengths = [5, 100, 7, 50, 60, 3]
    batches = token_budget_batches(lengths, max_tokens=120)
    assert sorted(i for b in batches for i in b) == list(range(6))
    assert all(max(lengths[i] for i in b) * len(b) <= 120 or len(b) == 1 for b in batches)
