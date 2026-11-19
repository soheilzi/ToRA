"""Greedy answers of the generator to every prompt-labeled example of the given datasets. SingProbe's query protocol
needs them: its prompt score is averaged over the positions of a generated answer (scorers.SingProbeQueryScorer).

    CUDA_VISIBLE_DEVICES=0 python -m tora.eval.generate "gen.datasets=[aegis2,openai_mod]" "gen.splits=[test,val]"

Writes <data_dir>/generated/<tag>/<dataset>__<split>.jsonl: {id, response, n_tokens, finished}. Prompts are rendered as
everywhere else (Qwen3 chat template, thinking disabled) and cut from the left to max_len - max_new_tokens; decoding
stops at <|im_end|> (finished) or after max_new_tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from tora.data.formatting import ChatFormatter, load_tokenizer
from tora.data.prepare import load_processed, processed_path
from tora.utils.config import RunConfig, config_from_cli, to_dict
from tora.utils.gpu_info import require_visible_devices
from tora.utils.hf import download_model
from tora.utils.paths import CONFIGS_ROOT, resolve
from tora.utils.runlog import RunLogger, make_run_id
from tora.utils.seeding import seed_everything


@dataclass
class GenSection:
    datasets: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=lambda: ["test"])
    model_repo: str = "Qwen/Qwen3-0.6B"
    tag: str = "qwen3-0.6b_greedy_512"  # <data_dir>/generated/<tag>/
    max_new_tokens: int = 512  # SingProbe report §4.2: "a maximum length of 512"
    max_len: int = 2048
    batch_size: int = 64


@dataclass
class GenConfig:
    run: RunConfig
    gen: GenSection = field(default_factory=GenSection)


def generated_path(data_dir: str, tag: str, name: str, split: str) -> Path:
    return resolve(data_dir) / "generated" / tag / f"{name}__{split}.jsonl"


@torch.inference_mode()
def generate(model, fmt: ChatFormatter, prompts: list[list[int]], max_new_tokens: int, batch_size: int) -> list[tuple[list[int], bool]]:
    """Left-padded greedy batches, longest first; -> (answer token ids without <|im_end|>, finished) per prompt."""
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    out: list[tuple[list[int], bool] | None] = [None] * len(prompts)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        width = max(len(prompts[i]) for i in idx)
        ids = torch.tensor([[fmt.pad_id] * (width - len(prompts[i])) + prompts[i] for i in idx], device=model.device)
        mask = torch.tensor([[0] * (width - len(prompts[i])) + [1] * len(prompts[i]) for i in idx], device=model.device)
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=fmt.im_end_id, pad_token_id=fmt.pad_id)[:, width:].tolist()
        for i, row in zip(idx, gen):
            finished = fmt.im_end_id in row
            out[i] = (row[: row.index(fmt.im_end_id)] if finished else row, finished)
        print(f"  {min(b + batch_size, len(order))}/{len(order)}", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    cfg = config_from_cli(GenConfig, argv, default_config=str(CONFIGS_ROOT / "eval" / "generate.yaml"))
    require_visible_devices()
    seed_everything(cfg.run.seed)
    g = cfg.gen
    path = download_model(g.model_repo, hf_home=cfg.run.hf_home)
    fmt = ChatFormatter(load_tokenizer(path))
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    with RunLogger(cfg.run.phase, make_run_id(cfg.run.name), to_dict(cfg), cfg.run.seed, cfg.run.results_dir) as log:
        for name in g.datasets:
            for split in g.splits:
                if not processed_path(cfg.run.data_dir, name, split).exists():
                    continue
                exs = [e for e in load_processed(cfg.run.data_dir, name, split) if e.prompt_label is not None]
                if not exs:
                    continue
                print(f"{name}/{split}: {len(exs)} prompts")
                prompts = [fmt.render(e.prompt, None, max_len=g.max_len - g.max_new_tokens).input_ids for e in exs]
                answers = generate(model, fmt, prompts, g.max_new_tokens, g.batch_size)
                dest = generated_path(cfg.run.data_dir, g.tag, name, split)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("w") as fh:
                    for e, (ids, finished) in zip(exs, answers):
                        fh.write(json.dumps({"id": e.id, "response": fmt.tokenizer.decode(ids), "n_tokens": len(ids), "finished": finished}) + "\n")
                stats = {"dataset": name, "split": split, "n": len(exs), "unfinished": sum(not f for _, f in answers),
                         "empty": sum(not ids for ids, _ in answers), "mean_tokens": sum(len(ids) for ids, _ in answers) / len(answers)}
                print(f"  -> {dest}: {stats}")
                log.log(stats)


if __name__ == "__main__":
    main()
