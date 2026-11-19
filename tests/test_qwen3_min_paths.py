"""Internal consistency of qwen3_min's cache paths, in fp32 so that logic bugs aren't hidden by rounding.

Incremental decode, chunked prefill and right-padded batches must reproduce a single full prefill.
In fp32 they agree to ~5e-6 relative L2 (measured), so REL_TOL = 1e-4 leaves margin while any
indexing / masking / position bug shows up as O(1) error. In bf16 the same paths differ by bf16
rounding (up to ~4% relative L2 on logits / final hidden, a few ulps per element; HF's own
decode-vs-prefill shows the same drift), which is why these checks are not run in bf16.
"""

import pytest
import torch

from conftest import rel_err
from tora.data.formatting import ChatFormatter, collate_right_padded, load_tokenizer
from tora.models.qwen3_min import Qwen3Min, greedy_generate

pytestmark = pytest.mark.gpu

REL_TOL = 1e-4
PROMPTS = [
    "Hi",
    "Explain why the sky is blue in two paragraphs.",
    "List five tips for learning a new programming language, with one sentence each.",
]


@pytest.fixture(
    scope="module", params=["Qwen/Qwen3-0.6B", pytest.param("Qwen/Qwen3-1.7B", marks=pytest.mark.slow)]
)
def model32(request, device):
    model = Qwen3Min.from_pretrained(request.param, device=device, dtype=torch.float32, max_seq_len=1024, verbose=False)
    yield model, ChatFormatter(load_tokenizer(model.source_path))
    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def test_incremental_decode_matches_full_prefill(model32, device):
    model, fmt = model32
    ids = torch.tensor([fmt.render(PROMPTS[1], None).input_ids], device=device)
    new = greedy_generate(model, ids, 48)
    full = torch.cat([ids, new[None, :-1]], dim=1)
    ref = model(full).logits[0, ids.shape[1] - 1 :]
    cache = model.new_cache(1, full.shape[1])
    steps = [model(ids, cache=cache).logits[0, -1]]
    steps += [model(tok, cache=cache).logits[0, -1] for tok in new[:-1].view(-1, 1, 1)]
    err = rel_err(torch.stack(steps), ref).max().item()
    print(f"\nmax rel err {err:.2e}")
    assert err <= REL_TOL
    assert cache.length == full.shape[1]


@torch.no_grad()
def test_chunked_prefill_matches_full(model32, device):
    model, fmt = model32
    ids = torch.tensor([fmt.render(PROMPTS[2], "Here are five tips. First, practice every day.").input_ids], device=device)
    ref = model(ids).logits[0]
    cache = model.new_cache(1, ids.shape[1])
    cut = ids.shape[1] // 3
    parts = [model(ids[:, :cut], cache=cache).logits[0], model(ids[:, cut:], cache=cache).logits[0]]
    err = rel_err(torch.cat(parts), ref).max().item()
    print(f"\nmax rel err {err:.2e}")
    assert err <= REL_TOL


@torch.no_grad()
def test_right_padded_batch_matches_unpadded(model32, device):
    model, fmt = model32
    rendered = [fmt.render(p, "Sure, here you go.") for p in PROMPTS]
    ids, lengths = collate_right_padded(rendered, fmt.pad_id)
    assert len(set(lengths.tolist())) > 1, "prompts should differ in length to exercise padding"
    batch = model(ids.to(device), logits="none").final_hidden
    worst = 0.0
    for i, r in enumerate(rendered):
        single = model(torch.tensor([r.input_ids], device=device), logits="none").final_hidden[0]
        worst = max(worst, rel_err(batch[i, : lengths[i]], single).max().item())
    print(f"\nmax rel err {worst:.2e}")
    assert worst <= REL_TOL
