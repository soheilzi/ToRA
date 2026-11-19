"""Unsafe-onset labels for ToRA's per-token training (the paper's "Per-Token Labels").

Qwen3Guard-Stream-0.6B reads every unsafe training response exactly as the verifier trainer renders it (same tokenizer,
template and max_len truncation, so indices line up with the trainer's), and the onset is the first token of the
earliest two-in-a-row Unsafe / Controversial pair: its released rule, strict, in token order
(`first_unsafe_token`). Safe responses are not run: their labels never depend on Qwen. The trainer
(`train.prefix_teacher`, `prefix_labels: teacher`) labels an unsafe response's earlier positions 1 from the onset on
and 0 before it; with no onset, all of them 0, the last token keeping its ground-truth label.

    CUDA_VISIBLE_DEVICES=0 python scripts/teacher_onsets.py --shard 0 --num-shards 4    # one shard per GPU

-> data_cache/teacher/qwen3guard_stream_0p6b.shard{i}of{n}.jsonl, one record per unsafe response: id, onset
(relative to the response start; -1 = never flagged), level (unsafe / controversial / none), n_response (response tokens
as rendered). Summary: results/teacher/<run_id>.jsonl.
"""

import argparse
import json

import numpy as np
import torch

from tora.data.formatting import ChatFormatter, collate_right_padded, load_tokenizer, token_budget_batches
from tora.data.prepare import load_processed
from tora.models.baselines.qwen3guard_stream import CONTROVERSIAL, UNSAFE, Qwen3GuardStream, first_unsafe_token
from tora.utils.gpu_info import require_visible_devices
from tora.utils.hf import download_model
from tora.utils.paths import resolve
from tora.utils.runlog import RunLogger, make_run_id

TRAIN_DATASETS = ["wildguard", "aegis2", "beavertails", "saferlhf", "toxicchat", "aegis1", "expguard"]  # configs/train/tora.yaml
LEVELS = {UNSAFE: "unsafe", CONTROVERSIAL: "controversial"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--datasets", nargs="+", default=TRAIN_DATASETS)
    ap.add_argument("--max-len", type=int, default=2048, help="the trainer's train.max_len")
    ap.add_argument("--batch-tokens", type=int, default=32768)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--teacher", default="Qwen/Qwen3Guard-Stream-0.6B")
    args = ap.parse_args()
    require_visible_devices()
    out = resolve(args.data_dir) / "teacher" / f"{args.teacher.split('/')[-1].lower().replace('-', '_').replace('.', 'p')}.shard{args.shard}of{args.num_shards}.jsonl"
    log = RunLogger("teacher", make_run_id(f"teacher_onsets_shard{args.shard}of{args.num_shards}"), vars(args) | {"out": out}, seed=0)

    fmt = ChatFormatter(load_tokenizer(download_model("Qwen/Qwen3-0.6B")))
    unsafe = [e for name in args.datasets for e in load_processed(args.data_dir, name, "train") if e.response is not None and e.response_label == 1]
    unsafe = unsafe[args.shard :: args.num_shards]
    rendered = [fmt.render_example(e, max_len=args.max_len) for e in unsafe]
    model = Qwen3GuardStream.from_pretrained(args.teacher, device="cuda", max_seq_len=args.max_len)
    print(f"{len(unsafe)} unsafe responses -> {out}", flush=True)

    records: list[dict | None] = [None] * len(unsafe)
    with torch.inference_mode():
        for batch in token_budget_batches([len(r.input_ids) for r in rendered], args.batch_tokens):
            ids, lengths = collate_right_padded([rendered[i] for i in batch], fmt.pad_id)
            risk = model(ids.cuda()).response_risk_logits.argmax(-1).cpu()
            for row, i in enumerate(batch):
                r = rendered[i]
                if r.response_end is None:  # the response was truncated away; the trainer has no response label here either
                    records[i] = {"id": unsafe[i].id, "onset": -1, "level": "none", "n_response": 0}
                    continue
                onset, level = first_unsafe_token(risk[row, r.response_start : r.response_end + 1].tolist())
                records[i] = {"id": unsafe[i].id, "onset": onset, "level": LEVELS.get(level, "none"), "n_response": r.response_end - r.response_start + 1}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records))

    found = [r for r in records if r["onset"] >= 0]
    rel = np.array([r["onset"] / r["n_response"] for r in found])
    summary = {
        "n_unsafe": len(records),
        "n_no_response_tokens": sum(r["n_response"] == 0 for r in records),
        "detected_rate": len(found) / len(records),
        "level_counts": {lv: sum(r["level"] == lv for r in records) for lv in ("unsafe", "controversial", "none")},
        "onset_median": float(np.median([r["onset"] for r in found])) if found else None,
        "onset_relative_quartiles": np.quantile(rel, [0.25, 0.5, 0.75]).round(3).tolist() if found else None,
        "detected_rate_by_dataset": {
            name: float(np.mean([r["onset"] >= 0 for r in records if r["id"].startswith(f"{name}-")])) for name in args.datasets if any(r["id"].startswith(f"{name}-") for r in records)
        },
    }
    log.log(summary)
    log.close()
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
