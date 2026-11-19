"""Our trained models as run_eval scorers (offline, teacher-forced): results/train/<run>/<variant>.pt -> per-token scores.

    eval.methods=[ckpt:tora_s0=results/train/tora_seed0-<...>/tora.pt, ...]

Verifiers (ToRA and the own-KV ablation) and the frozen-feature probe score every response token in one
teacher-forced pass: sigmoid of the response-harm logit at each response position. For ToRA that equals the reference
decoder's per-step score, so it is what the streaming verifier emits while the response is generated. The prompt
score is read at the prompt's last token and `response_score` at the response's last token, both as in training;
rendering and max_len are train_verifier's (no closing tokens). There is no official rule: run_eval sets streaming
thresholds on calibration data from the token scores.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tora.data.formatting import ChatFormatter, collate_right_padded, token_budget_batches
from tora.data.schema import Example
from tora.eval.metrics import response_score
from tora.eval.scorers import Scored, Scorer
from tora.models.qwen3_min import Qwen3Min
from tora.models.verifier import generator_captures
from tora.train.evaluation import model_tag
from tora.train.train_verifier import VariantConfig, build_trainable, variant_from_checkpoint
from tora.utils.config import ModelConfig


def load_checkpoint(ckpt_path: Path, repo_id: str, max_seq_len: int, device="cuda", hf_home=None) -> tuple[Qwen3Min, torch.nn.Module, VariantConfig]:
    """A saved <variant>.pt -> (its bf16 backbone, the trained model in eval mode, its variant config)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt["model"] != model_tag(ModelConfig(repo_id=repo_id)):
        raise ValueError(f"{ckpt_path}: trained on {ckpt['model']}, not {repo_id}")
    backbone = Qwen3Min.from_pretrained(repo_id, device=device, dtype=torch.bfloat16, max_seq_len=max_seq_len, hf_home=hf_home, verbose=False)
    vc = variant_from_checkpoint(ckpt["variant"])
    model = build_trainable(backbone.cfg, vc).to(backbone.device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return backbone, model, vc


class CheckpointScorer(Scorer):
    consecutive = 1

    def __init__(self, name: str, ckpt_path: Path, fmt: ChatFormatter, max_len: int, batch_tokens: int, repo_id: str = "Qwen/Qwen3-0.6B",
                 device="cuda", hf_home=None):
        super().__init__(fmt, max_len, batch_tokens, append_turn_end=False)
        self.name = name
        self.backbone, self.model, _ = load_checkpoint(ckpt_path, repo_id, max(4096, max_len + 64), device, hf_home)

    @torch.no_grad()
    def score(self, examples: list[Example]) -> list[Scored]:
        prepared = [self.prepare(e).rendered for e in examples]
        device = self.backbone.device
        results: list[Scored | None] = [None] * len(examples)
        for idx in token_budget_batches([len(r.input_ids) for r in prepared], self.batch_tokens):
            batch = [prepared[i] for i in idx]
            ids, _ = collate_right_padded(batch, self.fmt.pad_id)
            captures = generator_captures(self.backbone, ids.to(device), self.model.first)
            probs = self.model(self.backbone, captures).float().sigmoid().cpu()  # [B, T, C] at every position
            for row, i in enumerate(idx):
                r, s = prepared[i], Scored()
                s.prompt_score = float(probs[row, r.prompt_end, 0])
                if r.response_end is not None:
                    tok = probs[row, r.response_start : r.response_end + 1, 1].numpy()
                    s.response_token_scores = tok
                    s.response_score = float(tok[-1])
                    s.alt["response_max"] = response_score(tok)
                results[i] = s
        return results  # type: ignore[return-value]
