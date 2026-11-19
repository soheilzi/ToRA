"""Chat-template rendering, label positions, truncation.

Rendering (Qwen3 chat template, thinking disabled):
  prompt part    template([user], add_generation_prompt=True, enable_thinking=False), i.e.
                 '<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'
  response part  the response text tokenized on its own, the way the generator would emit it
  input_ids      tokens(prompt part) + tokens(response)
Thinking traces (Example.meta["thinking"]: the response is a reasoning trace, then '</think>\n\n' and the answer):
  prompt part    template([user], add_generation_prompt=True, enable_thinking=True) + '<think>\n', i.e. the
                 response starts at the first reasoning token, as Qwen3Guard's eval_stream.py --thinking

Label positions (checked by S6c):
  prompt_end     last token of the prompt part: the last position the backbone sees before the
                 first response token is generated
  response_end   last response token

Truncation to max_len: drop tokens from the left of the user's prompt content (the template
around it is kept). Only if the response alone does not fit is it cut, also from the left, so the
response end -- the label position -- is never cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tora.data.schema import Example

USER_HEADER = "<|im_start|>user\n"
THINK_OPEN = "<think>\n"
TURN_END = "<|im_end|>\n"


@dataclass(frozen=True)
class Rendered:
    input_ids: list[int]
    prompt_end: int
    response_end: int | None  # None for prompt-only examples
    prompt_trimmed: int = 0  # prompt-content tokens dropped from the left
    response_trimmed: int = 0  # response tokens dropped from the left

    @property
    def response_start(self) -> int | None:
        return None if self.response_end is None else self.prompt_end + 1


def load_tokenizer(model_path: str | Path) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_path)


class ChatFormatter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer has no pad token")
        self.pad_id: int = tokenizer.pad_token_id
        self.im_end_id: int = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def prompt_text(self, prompt: str, thinking: bool = False) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        return text + THINK_OPEN if thinking else text

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _encode_prompt(self, prompt: str, thinking: bool) -> tuple[list[int], int, int]:
        """-> (prompt-part ids, [c0, c1) token span of the user content)."""
        text = self.prompt_text(prompt, thinking)
        if not text.startswith(USER_HEADER) or text[len(USER_HEADER) : len(USER_HEADER) + len(prompt)] != prompt:
            raise ValueError(f"unexpected chat template rendering: {text[:80]!r}")
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        lo, hi = len(USER_HEADER), len(USER_HEADER) + len(prompt)
        inside = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s >= lo and e <= hi and e > s]
        c0, c1 = (inside[0], inside[-1] + 1) if inside else (0, 0)
        return enc["input_ids"], c0, c1

    def render(self, prompt: str, response: str | None, max_len: int | None = None, thinking: bool = False) -> Rendered:
        prompt_ids, c0, c1 = self._encode_prompt(prompt, thinking)
        response_ids: list[int] = []
        if response is not None:
            response_ids = self.encode(response)
            if not response_ids:
                raise ValueError("empty response; pass response=None for prompt-only examples")
        prompt_cut = response_cut = 0
        overflow = len(prompt_ids) + len(response_ids) - (max_len or 10**12)
        if overflow > 0:
            prompt_cut = min(overflow, c1 - c0)
            prompt_ids = prompt_ids[:c0] + prompt_ids[c0 + prompt_cut :]
            response_cut = overflow - prompt_cut
            if response_cut > 0 and (response is None or response_cut >= len(response_ids)):
                raise ValueError(f"cannot fit into max_len={max_len} without cutting the label position")
            response_ids = response_ids[response_cut:]
        n = len(prompt_ids)
        response_end = n + len(response_ids) - 1 if response is not None else None
        return Rendered(prompt_ids + response_ids, n - 1, response_end, prompt_cut, response_cut)

    def render_example(self, e: Example, max_len: int | None = None) -> Rendered:
        return self.render(e.prompt, e.response, max_len, thinking=e.meta.get("thinking", False))

    def closing_ids(self) -> list[int]:
        """Tokens that end an assistant turn ('<|im_end|>\\n'), appended by guards evaluated on complete turns."""
        return self.encode(TURN_END)


def collate_right_padded(batch: list[Rendered] | list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """-> input_ids [B, T_max] right-padded with pad_id, lengths [B]."""
    seqs = [r.input_ids if isinstance(r, Rendered) else r for r in batch]
    lengths = torch.tensor([len(s) for s in seqs])
    ids = torch.full((len(seqs), int(lengths.max())), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, lengths


def token_budget_batches(lengths: list[int], max_tokens: int, max_batch: int = 256) -> list[list[int]]:
    """Group indices (sorted by length) so each padded batch has <= max_tokens tokens."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        width = lengths[i]  # sorted ascending, so the newest item sets the padded width
        if current and (width * (len(current) + 1) > max_tokens or len(current) >= max_batch):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches
