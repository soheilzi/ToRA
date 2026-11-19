"""ToRA scored through the deployed decode step: per example, the prompt minus its last token is prefilled in
one generator call; then every token from the prompt's last one to the response's last is decoded one step at a time
by the fused dual-row step (inference/decode_static.py, mode fused), by default compiled and replayed as a CUDA
graph, the execution the latency tables time.

Same rendering, max_len and score positions as CheckpointScorer (prompt score at the prompt's last token, token
scores at every response token), so its teacher-forced scores are the reference these must match (S4 at benchmark
scale). The response is pre-generated: by S2 the step's generator row is what generating those tokens would run, so
the verifier sees what it would see while they are generated.

    eval.methods=[fused:tora_s0=results/train/tora_seed0-<...>/tora.pt]
"""

from __future__ import annotations

from pathlib import Path

import torch

from tora.data.formatting import ChatFormatter
from tora.data.schema import Example
from tora.eval.checkpoint_scorer import load_checkpoint
from tora.eval.metrics import response_score
from tora.eval.scorers import Scored, Scorer
from tora.inference.decode_static import Execution, StaticDecoder, runner
from tora.models.verifier import Verifier


class FusedScorer(Scorer):
    consecutive = 1

    def __init__(self, name: str, ckpt_path: Path, fmt: ChatFormatter, max_len: int, repo_id: str = "Qwen/Qwen3-0.6B",
                 execution: Execution = "compile+graph", device="cuda", hf_home=None):
        super().__init__(fmt, max_len, batch_tokens=0, append_turn_end=False)
        self.name, self.execution = name, execution
        backbone, model, _ = load_checkpoint(ckpt_path, repo_id, max_len, device, hf_home)
        if not isinstance(model, Verifier):
            raise ValueError(f"{ckpt_path}: the fused decode step runs a shared-KV verifier (ToRA), not {type(model).__name__}")
        self.decoder = StaticDecoder(backbone, model, "fused", max_len)  # positions 0 .. max_len - 1
        self.step = runner(self.decoder.step, execution)
        vlogits = self.step()[1]
        self.out = torch.empty((max_len, vlogits.shape[-1]), dtype=vlogits.dtype, device=vlogits.device)

    @torch.no_grad()
    def score(self, examples: list[Example]) -> list[Scored]:
        d, results = self.decoder, []
        for e in examples:
            r = self.prepare(e).rendered
            ids = torch.tensor(r.input_ids, device=d.pos.device).view(1, -1)
            last = r.prompt_end if r.response_end is None else r.response_end
            d.prefill(ids[:, : r.prompt_end])
            for t in range(r.prompt_end, last + 1):  # no host sync inside: the scores stay on the device until the end
                d.pos.fill_(t)
                d.token.copy_(ids[:, t : t + 1])
                self.out[t - r.prompt_end].copy_(self.step()[1][0])
            probs = self.out[: last - r.prompt_end + 1].float().sigmoid().cpu()
            s = Scored(prompt_score=float(probs[0, 0]))
            if r.response_end is not None:
                tok = probs[1:, 1].numpy()
                s.response_token_scores, s.response_score = tok, float(tok[-1])
                s.alt["response_max"] = response_score(tok)
            results.append(s)
        return results
