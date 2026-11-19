"""qwen3_min vs transformers (S1, S1b) and internal consistency of the cache paths.

Tolerances:
- vs HF (S1, S1b, K/V, hidden states): EXACT (atol = rtol = 0). qwen3_min reproduces HF's bf16 op
  sequence and SDPA calls, so results are bitwise identical (torch 2.9.1+cu126, transformers 5.21,
  RTX A6000). If a library upgrade breaks this, investigate before relaxing.
- Our own cache paths against each other are checked in fp32 in test_qwen3_min_paths.py.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from tora.data.formatting import ChatFormatter, load_tokenizer
from tora.models.qwen3_min import Qwen3Min, eos_ids, greedy_generate

pytestmark = pytest.mark.gpu

PROMPTS = [
    "Explain why the sky is blue in two paragraphs.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List five tips for learning a new programming language, with one sentence each.",
    "Describe in detail how a bicycle gear system works.",
    "What are the main differences between TCP and UDP? Answer in detail.",
]


@pytest.fixture(
    scope="module", params=["Qwen/Qwen3-0.6B", pytest.param("Qwen/Qwen3-1.7B", marks=pytest.mark.slow)]
)
def models(request, device):
    ours = Qwen3Min.from_pretrained(request.param, device=device, max_seq_len=1024)
    hf = AutoModelForCausalLM.from_pretrained(
        ours.source_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    fmt = ChatFormatter(load_tokenizer(ours.source_path))
    yield ours, hf, fmt
    del ours, hf
    torch.cuda.empty_cache()


def prompt_ids(fmt: ChatFormatter, prompt: str, device) -> torch.Tensor:
    return torch.tensor([fmt.render(prompt, None).input_ids], device=device)


def test_config_matches_hf(models):
    ours, hf, _ = models
    c, h = ours.cfg, hf.config
    rope = getattr(h, "rope_parameters", None) or {}
    assert (c.num_layers, c.hidden_size, c.intermediate_size) == (
        h.num_hidden_layers, h.hidden_size, h.intermediate_size
    )
    assert (c.num_heads, c.num_kv_heads, c.head_dim) == (h.num_attention_heads, h.num_key_value_heads, h.head_dim)
    assert (c.vocab_size, c.rms_norm_eps, c.tie_word_embeddings) == (h.vocab_size, h.rms_norm_eps, h.tie_word_embeddings)
    assert c.rope_theta == rope.get("rope_theta", getattr(h, "rope_theta", None))


@torch.no_grad()
def test_s1_prefill_logits_match_hf(models, device):
    ours, hf, fmt = models
    print("\nrendered prompt:", repr(fmt.prompt_text(PROMPTS[0])))
    for prompt in PROMPTS:
        ids = prompt_ids(fmt, prompt, device)
        got, ref = ours(ids).logits, hf(ids).logits
        print(f"T={ids.shape[1]:3d}  max|dlogit|={(got.float() - ref.float()).abs().max().item():.3g}")
        torch.testing.assert_close(got, ref, atol=0, rtol=0)


@torch.no_grad()
def test_s1_hidden_states_and_kv_match_hf(models, device):
    ours, hf, fmt = models
    ids = prompt_ids(fmt, PROMPTS[1], device)
    n = ours.cfg.num_layers
    got = ours(ids, output_hidden_states=True, capture_layers=range(n))
    ref = hf(ids, output_hidden_states=True, use_cache=True)
    # HF's hidden_states[i] for i < L is the residual entering layer i (same as ours); its last
    # entry is post-final-norm, which is our final_hidden (our hidden_states[L] is pre-norm).
    assert len(ref.hidden_states) == n + 1
    for i in range(n):
        torch.testing.assert_close(got.hidden_states[i], ref.hidden_states[i], atol=0, rtol=0)
    torch.testing.assert_close(got.final_hidden, ref.hidden_states[n], atol=0, rtol=0)
    for i in range(n):
        layer = ref.past_key_values.layers[i]
        torch.testing.assert_close(got.captures[i].k, layer.keys, atol=0, rtol=0)
        torch.testing.assert_close(got.captures[i].v, layer.values, atol=0, rtol=0)


@torch.no_grad()
def test_s1b_greedy_decode_matches_hf_generate(models, device):
    ours, hf, fmt = models
    eos = eos_ids(ours.cfg, ours.source_path / "generation_config.json")
    lengths = []
    for prompt in PROMPTS:
        ids = prompt_ids(fmt, prompt, device)
        got = greedy_generate(ours, ids, 64, eos)
        ref = hf.generate(ids, max_new_tokens=64, do_sample=False, temperature=None, top_p=None, top_k=None)
        ref = ref[0, ids.shape[1] :]
        assert got.tolist() == ref.tolist(), f"{prompt!r}\nours: {fmt.tokenizer.decode(got)!r}\nhf:   {fmt.tokenizer.decode(ref)!r}"
        lengths.append(len(got))
    print("\ngenerated lengths:", lengths)
    assert max(lengths) == 64, "no prompt produced the full 64 tokens; the check would be too weak"


@torch.no_grad()
def test_decode_step_logits_match_hf_incremental(models, device):
    """Static-cache decode equals HF's DynamicCache decode at every step, exactly."""
    ours, hf, fmt = models
    ids = prompt_ids(fmt, PROMPTS[3], device)
    new = greedy_generate(ours, ids, 32)
    cache = ours.new_cache(1, ids.shape[1] + len(new))
    got = [ours(ids, cache=cache, logits="last").logits[:, -1]]
    out = hf(ids, use_cache=True, logits_to_keep=1)  # same as generate; M=1 lm_head GEMM like ours
    ref = [out.logits[:, -1]]
    for tok in new[:-1].view(-1, 1, 1):
        got.append(ours(tok, cache=cache, logits="last").logits[:, -1])
        out = hf(tok, past_key_values=out.past_key_values, use_cache=True)
        ref.append(out.logits[:, -1])
    torch.testing.assert_close(torch.stack(got), torch.stack(ref), atol=0, rtol=0)


def test_cache_overflow_raises(models):
    ours, _, _ = models
    cache = ours.new_cache(1, 4)
    with pytest.raises(ValueError, match="overflow"):
        ours(torch.zeros(1, 5, dtype=torch.long, device=ours.device), cache=cache)
