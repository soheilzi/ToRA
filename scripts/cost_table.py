"""The analytic costs the paper cites (src/tora/bench/cost_model.py): extra FLOPs and bytes moved per decoded token
of the verifier row, relative to the generator alone, at batch 1. Rank 8 and the MLP head (hidden 1,024) throughout.

  - Adaptation depth: extra FLOPs of ToRA at K = 14 and K = 28 (the paper quotes T = 4,096).
  - Cache sharing: extra bytes of the shared-KV verifier and of the own-KV verifier at K = 28 (T = 1,024 / 4,096 /
    16,384), assuming the fused dual-row kernels (each weight and KV tile is read once for both rows).

    python scripts/cost_table.py [--model Qwen/Qwen3-0.6B]

Only the model's config.json is fetched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from tora.bench import cost_model as C
from tora.models.qwen3_min import Qwen3Config
from tora.utils.hf import hf_token

CONTEXTS = (1024, 4096, 16384)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()
    c = Qwen3Config.from_hf_json(Path(hf_hub_download(args.model, "config.json", token=hf_token())))

    def extra(t: int, k: int, own_kv: bool = False) -> tuple[float, float]:
        g, v = C.generator(c, t), C.verifier_extra(c, t, k, args.rank, "mlp", 1024, own_kv=own_kv)
        return 100 * v.flops / g.flops, 100 * v.bytes / g.bytes

    print(f"{args.model}: extra cost of the verifier row over the generator per decoded token (rank {args.rank}, MLP head)\n")
    print(f"{'verifier':28}" + "".join(f"{'T = ' + format(t, ','):>22}" for t in CONTEXTS))
    print(f"{'':28}" + "".join(f"{'FLOPs':>11}{'bytes':>11}" for _ in CONTEXTS))
    rows = [(f"ToRA, K = {k} (shared KV)", k, False) for k in (14, 28)] + [("own KV cache, K = 28", 28, True)]
    for name, k, own in rows:
        cells = [extra(t, k, own) for t in CONTEXTS]
        print(f"{name:28}" + "".join(f"{f:10.1f}%{b:10.2f}%" for f, b in cells))


if __name__ == "__main__":
    main()
