import pytest

from tora.data.formatting import ChatFormatter, collate_right_padded, load_tokenizer
from tora.utils.hf import download_model


@pytest.fixture(scope="module")
def fmt() -> ChatFormatter:
    return ChatFormatter(load_tokenizer(download_model("Qwen/Qwen3-0.6B")))


def test_thinking_disabled(fmt):
    text = fmt.prompt_text("What is 2+2?")
    print("\nrendered:", repr(text))
    assert text == "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_thinking_mode_prompt_part(fmt):
    """Thinking traces: the prompt part ends with '<think>\\n', so the response starts at the first reasoning token."""
    text = fmt.prompt_text("Hi", thinking=True)
    assert text == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    r = fmt.render("Hi", "Hmm.\n</think>\n\nHello.", thinking=True)
    assert fmt.tokenizer.decode(r.input_ids[: r.prompt_end + 1]) == text
    assert fmt.tokenizer.decode(r.input_ids[r.prompt_end + 1 :]) == "Hmm.\n</think>\n\nHello."


def test_prompt_tokens_match_chat_template(fmt):
    prompt = "Tell me about owls."
    expected = fmt.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    if isinstance(expected, dict) or hasattr(expected, "input_ids"):
        expected = expected["input_ids"]
    assert fmt.render(prompt, None).input_ids == list(expected)


def test_label_positions(fmt):
    r = fmt.render("Tell me about owls.", "Owls hunt at night.")
    tok = fmt.tokenizer
    assert tok.decode(r.input_ids[: r.prompt_end + 1]) == fmt.prompt_text("Tell me about owls.")
    assert tok.decode(r.input_ids[r.prompt_end + 1 : r.response_end + 1]) == "Owls hunt at night."
    assert r.response_end == len(r.input_ids) - 1


def test_prompt_only_and_empty_response(fmt):
    r = fmt.render("Hi", None)
    assert r.response_end is None and r.prompt_end == len(r.input_ids) - 1
    with pytest.raises(ValueError):
        fmt.render("Hi", "")


def test_collate_right_padded(fmt):
    a, b = fmt.render("Hi", "Hello there."), fmt.render("A much longer question about owls?", "Yes.")
    ids, lengths = collate_right_padded([a, b], fmt.pad_id)
    assert ids.shape == (2, max(len(a.input_ids), len(b.input_ids)))
    assert ids[0, : lengths[0]].tolist() == a.input_ids
    assert (ids[0, lengths[0] :] == fmt.pad_id).all()
