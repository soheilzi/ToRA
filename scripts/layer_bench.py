"""The dual-row FFN on a dummy model (the paper's Fig. 4b), a stack of L FFN layers
(h <- h + down(silu(gate(h)) * up(h))) at a real Qwen3 shape with distinct random weights per layer, so that, as in
a decode, no layer's weights are still in the 6 MB L2 when it runs. The verifier row (row 1) carries a rank-r LoRA
on gate / up / down. Every variant runs the whole stack under one CUDA graph; reported per layer.

Variants:
  gen            the generator row alone, cuBLAS (M = 1): the floor
  separate       gen, then the verifier row by its own M = 1 calls + LoRA matmuls (weights read twice)
  merged         gen, then the verifier row through a merged copy W + sBA (weights read twice, stored twice)
  batched        both rows as one M = 2 cuBLAS GEMM per projection, LoRA as separate small matmuls (naive)
  fused          ours: K2 (gate + up + SiLU * mul) and K1 (down), both rows in one pass, LoRA in the kernel
  fused_gen      our kernels on the generator row alone (M = 1), with the config tuned for fused (one config per
                 shape keeps the generator row bitwise the same): the floor for our kernels
  *+compile      the stack compiled with torch.compile, then graphed

    CUDA_VISIBLE_DEVICES=0 python scripts/layer_bench.py --name layer_ffn \\
        --models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B

Shapes come from each model's config.json (only that file is fetched; no weights). LoRA: r = 8, scale 2 (alpha = 2r),
fp32 A and B as trained.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from tora.bench.timing import cuda_graph, time_ms
from tora.kernels.dual_lora import dual_lora_ffn
from tora.models.qwen3_min import Qwen3Config
from tora.utils import gpu_info
from tora.utils.hf import hf_token
from tora.utils.runlog import RunLogger, make_run_id

VARIANTS = ("gen", "separate", "merged", "batched", "fused", "fused_gen", "gen+compile", "batched+compile")
PEAK_GBPS = 768.0  # RTX A6000


class Stack:
    """L FFN layers: base weights in the model dtype, LoRA (A, B) fp32, and the merged copies W + sBA."""

    def __init__(self, hidden: int, ffn: int, layers: int, rank: int, merged: bool, seed: int = 0):
        g = torch.Generator(device="cuda").manual_seed(seed)

        def rand(*shape, std, dtype=torch.bfloat16):
            return (torch.randn(*shape, generator=g, device="cuda") * std).to(dtype)

        self.scale = 2.0  # alpha = 2r, as trained
        self.w = [{"gate": rand(ffn, hidden, std=hidden**-0.5), "up": rand(ffn, hidden, std=hidden**-0.5), "down": rand(hidden, ffn, std=ffn**-0.5 / 4)}
                  for _ in range(layers)]
        self.lora = [{p: (rand(rank, w.shape[1], std=w.shape[1] ** -0.5, dtype=torch.float32), rand(w.shape[0], rank, std=0.02, dtype=torch.float32))
                      for p, w in ws.items()} for ws in self.w]
        self.merged = None
        if merged:
            self.merged = [{p: (w.float() + self.scale * lo[p][1] @ lo[p][0]).to(w.dtype) for p, w in ws.items()} for ws, lo in zip(self.w, self.lora)]
        self.x = torch.randn(2, hidden, generator=g, device="cuda").to(torch.bfloat16)
        self.weight_bytes = sum(w.numel() * w.element_size() for ws in self.w for w in ws.values()) / layers
        self.lora_bytes = sum(t.numel() * 4 for lo in self.lora for ab in lo.values() for t in ab) / layers

    def lora_delta(self, lo, x):
        a, b = lo
        return self.scale * F.linear(F.linear(x.float(), a), b)

    def run(self, variant: str):
        h = self.x
        if variant in ("gen", "gen+compile"):
            h0 = h[:1]
            for ws in self.w:
                h0 = h0 + F.linear(F.silu(F.linear(h0, ws["gate"])) * F.linear(h0, ws["up"]), ws["down"])
            return h0
        if variant == "fused_gen":
            h0 = h[:1]
            for ws in self.w:
                h0 = h0 + dual_lora_ffn(h0, ws["gate"], ws["up"], ws["down"])
            return h0
        if variant == "fused":
            for ws, lo in zip(self.w, self.lora):
                h = h + dual_lora_ffn(h, ws["gate"], ws["up"], ws["down"], lo["gate"], lo["up"], lo["down"], self.scale)
            return h
        if variant in ("batched", "batched+compile"):
            for ws, lo in zip(self.w, self.lora):
                def proj(p, x):
                    y = F.linear(x, ws[p])
                    return torch.cat([y[:1], (y[1:].float() + self.lora_delta(lo[p], x[1:])).to(y.dtype)])
                h = h + proj("down", F.silu(proj("gate", h)) * proj("up", h))
            return h
        h0, h1 = h[:1], h[1:]
        for i, ws in enumerate(self.w):
            h0 = h0 + F.linear(F.silu(F.linear(h0, ws["gate"])) * F.linear(h0, ws["up"]), ws["down"])
            if variant == "merged":
                wm = self.merged[i]
                h1 = h1 + F.linear(F.silu(F.linear(h1, wm["gate"])) * F.linear(h1, wm["up"]), wm["down"])
            elif variant == "separate":
                lo = self.lora[i]

                def proj(p, x):
                    y = F.linear(x, ws[p])
                    return (y.float() + self.lora_delta(lo[p], x)).to(y.dtype)
                h1 = h1 + proj("down", F.silu(proj("gate", h1)) * proj("up", h1))
            else:
                raise ValueError(variant)
        return torch.cat([h0, h1])


def model_shape(repo_id: str) -> Qwen3Config:
    return Qwen3Config.from_hf_json(Path(hf_hub_download(repo_id, "config.json", token=hf_token())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"])
    ap.add_argument("--layers", type=int, default=14, help="stack depth (the method's K)")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    gpu_info.require_visible_devices()
    rows = []
    with RunLogger("bench", make_run_id(args.name), vars(args), seed=0) as log:
        for repo in args.models:
            c = model_shape(repo)
            stack = Stack(c.hidden_size, c.intermediate_size, args.layers, args.rank, merged="merged" in args.variants)
            ref = None
            for variant in args.variants:
                fn = lambda v=variant: stack.run(v)  # noqa: E731
                if variant.endswith("+compile"):
                    torch._dynamo.reset()
                    fn = torch.compile(fn, fullgraph=True)
                    fn()
                graphed = cuda_graph(fn)
                out = graphed().clone()
                if variant not in ("gen", "gen+compile", "fused_gen"):  # every dual-row variant computes the same rows
                    ref = out if ref is None else ref
                    err = ((out.float() - ref.float()).norm(dim=-1) / ref.float().norm(dim=-1)).max().item()
                else:
                    err = None
                stats = time_ms(graphed, args.warmup, args.reps)
                per_layer_us = 1000 * stats["median_ms"] / args.layers
                row = {"model": repo, "hidden": c.hidden_size, "ffn": c.intermediate_size, "variant": variant, "per_layer_us": per_layer_us,
                       "weight_mb_per_layer": stack.weight_bytes / 1e6, "lora_kb_per_layer": stack.lora_bytes / 1e3,
                       "gbps_base_weights": stack.weight_bytes / (per_layer_us * 1e-6) / 1e9, "max_rel_err_vs_first_dual": err,
                       "gpu_shared": gpu_info.snapshot()["visible_gpus_shared"], **stats}
                log.log(row)
                rows.append(row)
                log.console.info("%s %-16s %8.1f us/layer  %5.0f GB/s%s", repo, variant, per_layer_us, row["gbps_base_weights"], "" if err is None else f"  rel err {err:.1e}")
            del stack
            torch.cuda.empty_cache()
        print(f"\nDummy FFN stack, {args.layers} layers, batch 1, r = {args.rank}: us per layer (median of {args.reps} graph replays); "
              "overhead over gen; GB/s = base weight bytes per layer / time")
        print("| Model (hidden x ffn, MB/layer) | " + " | ".join(args.variants) + " |")
        print("|---|" + "---|" * len(args.variants))
        for repo in args.models:
            cell = {r["variant"]: r for r in rows if r["model"] == repo}
            r0 = next(iter(cell.values()))
            base = {"gen": cell.get("gen"), "compile": cell.get("gen+compile")}
            parts = []
            for v in args.variants:
                b = base["compile"] if v.endswith("+compile") else base["gen"]
                over = "" if b is None or v in ("gen", "gen+compile") else f" ({100 * (cell[v]['per_layer_us'] / b['per_layer_us'] - 1):+.0f}%)"
                parts.append(f"{cell[v]['per_layer_us']:.1f}{over}, {100 * cell[v]['gbps_base_weights'] / PEAK_GBPS:.0f}% BW")
            print(f"| {repo} ({r0['hidden']} x {r0['ffn']}, {r0['weight_mb_per_layer']:.1f}) | " + " | ".join(parts) + " |")
        if any(r["gpu_shared"] for r in rows):
            log.console.warning("another process shared the GPU during some rows: re-run them")


if __name__ == "__main__":
    main()
