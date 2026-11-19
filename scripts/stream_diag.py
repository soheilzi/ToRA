"""Why a verifier streams the way it does: per-token response scores of one checkpoint on the
calibration safe responses (WildGuard + Aegis 2.0 val), the safe BeaverTails test responses and Qwen3GuardTest
response_loc, then
  - the after-the-fact readout (last response token only) at the same calibration FPRs, vs the streaming rule;
  - k-in-a-row streaming rules (flag when k consecutive tokens are >= the threshold), k = 1 .. 32;
  - where safe responses peak (the max over tokens sets the streaming threshold), and, on the unsafe Qwen3GuardTest
    responses, the median peak score before the unsafe span vs from its start on (the paper's Streaming Detection).

    CUDA_VISIBLE_DEVICES=0 python scripts/stream_diag.py results/train/tora_seed0-<...>/tora.pt > results/logs/stream_diag_tora_seed0.txt
"""

import argparse
from pathlib import Path

import numpy as np

from tora.data.formatting import ChatFormatter, load_tokenizer
from tora.data.prepare import load_processed
from tora.eval import metrics as M
from tora.eval.checkpoint_scorer import CheckpointScorer
from tora.utils.gpu_info import require_visible_devices
from tora.utils.hf import download_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data-dir", default="data_cache")
    args = ap.parse_args()
    require_visible_devices()
    fmt = ChatFormatter(load_tokenizer(download_model("Qwen/Qwen3-0.6B")))
    scorer = CheckpointScorer("diag", args.checkpoint, fmt, 2048, 32768)

    def scored(name: str, split: str, label: int | None = None):
        exs = [e for e in load_processed(args.data_dir, name, split) if e.response is not None and e.response_label is not None]
        exs = [e for e in exs if label is None or e.response_label == label]
        return exs, [s.response_token_scores for s in scorer.score(exs)]

    cal = [s for name in ("wildguard", "aegis2") for s in scored(name, "val", 0)[1]]
    _, bt = scored("beavertails", "test", 0)
    q_exs, q = scored("qwen3guardtest", "test")
    spans = [(e.meta["unsafe_start_index"] - scorer.prepare(e).rendered.response_start, e.meta["unsafe_end_index"] - scorer.prepare(e).rendered.response_start) for e in q_exs]
    print(f"{args.checkpoint}\ncalibration safe responses {len(cal)}, BeaverTails safe {len(bt)}, Qwen3GuardTest unsafe {len(q)}")
    print(f"Qwen3GuardTest: median response length {np.median([len(s) for s in q]):.0f} tokens, median span start {np.median([a for a, _ in spans]):.0f}")

    print("\nAfter the fact (last response token only):")
    last_cal = np.array([s[-1] for s in cal])
    for fpr in (0.05, 0.01):
        thr = M.threshold_at_fpr(last_cal, fpr)
        print(f"  calibration FPR {fpr:.0%}: threshold {thr:.3f}, Qwen3GuardTest detected {np.mean([s[-1] >= thr for s in q]):.1%}, "
              f"BeaverTails safe flagged {np.mean([s[-1] >= thr for s in bt]):.1%}")

    print("\nStreaming, k tokens in a row >= threshold:")
    for fpr in (0.05, 0.01):
        for k in (1, 2, 4, 8, 16, 32):
            thr = M.threshold_at_fpr(np.array([M.response_score(s, k) for s in cal]), fpr)
            m = M.streaming_metrics(q, spans, thr, k, safe_scores=bt)
            print(f"  FPR {fpr:.0%} k={k:2d}: threshold {thr:.3f}, detected {m['detection_rate']:.1%}, exact hit {m['exact_hit_rate']:.1%}, "
                  f"within 128 {m['within_128_rate']:.1%}, median delay {m['delay_from_span_start_median']}, BeaverTails safe flagged {m['fpr_safe']:.1%}")

    peaks = np.array([s.max() for s in cal])
    where = np.array([int(np.argmax(s)) for s in cal])
    top = peaks >= np.quantile(peaks, 0.95)
    print(f"\nSafe calibration responses: median of the max over tokens {np.median(peaks):.3f}, median last-token score {np.median(last_cal):.3f}")
    print(f"  token index of the max, all: quartiles {np.quantile(where, [0.25, 0.5, 0.75])}; top 5% by max: {np.quantile(where[top], [0.25, 0.5, 0.75])}")
    print(f"  95th percentile of the score at token 0 / 1 / 4 / 16: " + " / ".join(f"{np.quantile([s[i] for s in cal if len(s) > i], 0.95):.3f}" for i in (0, 1, 4, 16)))
    pre = [s[: max(1, a)].max() for s, (a, _) in zip(q, spans)]
    print(f"Qwen3GuardTest unsafe: median max before the span {np.median(pre):.3f}, median max from the span on "
          f"{np.median([s[a:].max() for s, (a, _) in zip(q, spans)]):.3f}, median last-token score {np.median([s[-1] for s in q]):.3f}")


if __name__ == "__main__":
    main()
