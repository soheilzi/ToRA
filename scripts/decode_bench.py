"""Per-token latency of batch-1 decode (the paper's Table I and Fig. 4a), the generator alone (gen) vs with
the ToRA verifier row by separate calls (separate), batched with the generator (batched), or on the fused dual-row
kernels (fused; fused_gen is the generator alone on the same kernels), each eager, under a CUDA graph, compiled, and
compiled under a CUDA graph (inference/decode_static.py). One step at context T: T cached positions (random K/V;
latency does not depend on their values), the token at position T. A global warm-up (eager steps for --warm-s
seconds) runs first, so the first timed row does not pay for clocks and caches warming up. Each row also logs whether
its generator logits equal, bitwise, those of the generator-only mode on the same kernels (separate vs gen, fused vs
fused_gen: S2 under that execution; same K/V and token).

    CUDA_VISIBLE_DEVICES=0 python scripts/decode_bench.py --name decode_0p6b \\
        --ckpt "results/train/tora_seed0-*/tora.pt"
    CUDA_VISIBLE_DEVICES=0 python scripts/decode_bench.py --name decode_1p7b --model Qwen/Qwen3-1.7B \\
        --ckpt "results/train/tora_seed0-*/tora.pt" --untrained

--untrained builds the checkpoint's verifier shape (K, rank, head) on --model with random LoRA weights instead of
loading them (the 0.6B checkpoint does not fit the 1.7B); latency does not depend on the values.

--baselines adds the generator with a baseline guard instead of ToRA (inference/decode_baselines.py): singprobe (its
probe on the generator's hidden states, with its own K/V) and qwen3guard_stream (a second 0.6B model reading every token, with its own KV cache), both released checkpoints:

    CUDA_VISIBLE_DEVICES=0 python scripts/decode_bench.py --name decode_guards_0p6b \
        --ckpt "results/train/tora_seed0-*/tora.pt" --modes gen fused fused_gen \
        --baselines singprobe qwen3guard_stream --execs graph compile+graph --contexts 512 2048 8191

(8,191, not 8,192: Qwen3Guard-Stream has 8,192 positions, and a step at context T is at position T; the cache shapes,
roundup(T + 2, 128) slots, are the same for both.)

A baseline name with the suffix _kernels runs that baseline on our kernels: its generator steps as fused_gen does
(cuBLAS below the checkpoint's L - K, our kernels from there on), and so does Qwen3Guard-Stream's model; their
generator rows must then equal fused_gen's. Against fused_gen, every method then adds its check to the same generator:

    CUDA_VISIBLE_DEVICES=0 python scripts/decode_bench.py --name decode_guards_kernels_0p6b \
        --ckpt "results/train/tora_seed0-*/tora.pt" --modes gen fused fused_gen \
        --baselines singprobe qwen3guard_stream singprobe_kernels qwen3guard_stream_kernels \
        --execs graph compile+graph --contexts 512 2048 8191

Each row also logs the bytes the method adds to the generator: parameters, and caches at that context (ToRA's own K/V
slot lies inside the generator cache's rounding to 128 slots, so it adds none).
"""

import argparse
import dataclasses
import glob
import time

import torch

from tora.bench.timing import time_ms
from tora.inference.decode_baselines import GuardStreamDecoder, SingProbeDecoder
from tora.inference.decode_static import EXECS, MODES, StaticDecoder, runner
from tora.models.baselines.qwen3guard_stream import Qwen3GuardStream
from tora.models.baselines.singprobe import SingProbe
from tora.models.qwen3_min import Qwen3Min
from tora.train.evaluation import model_tag
from tora.train.train_verifier import build_trainable, variant_from_checkpoint
from tora.utils import gpu_info
from tora.utils.config import ModelConfig
from tora.utils.runlog import RunLogger, make_run_id

BASELINES = {"singprobe": "inclusionAI/Qwen3-0.6B-singprobe", "qwen3guard_stream": "Qwen/Qwen3Guard-Stream-0.6B"}
KERNELS = "_kernels"  # suffix: the baseline on our kernels (module docstring)
GEN_ONLY = {"separate": "gen", "fused": "fused_gen", "fused_gen": "fused", "singprobe": "gen", "qwen3guard_stream": "gen",
            "singprobe_kernels": "fused_gen", "qwen3guard_stream_kernels": "fused_gen"}  # the mode whose generator row must match


def nbytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def build(mode: str, backbone: Qwen3Min, verifier, guards: dict, max_len: int):
    """-> (decoder, bytes of parameters it adds to the generator, bytes of caches it adds)."""
    base, on_kernels = mode.removesuffix(KERNELS), verifier if mode.endswith(KERNELS) else None
    if base == "singprobe":
        d = SingProbeDecoder(guards[base], max_len, on_kernels)
        return d, nbytes(guards[base].probe.parameters()), nbytes((d.k, d.v))
    if base == "qwen3guard_stream":
        d = GuardStreamDecoder(backbone, guards[base], max_len, on_kernels)
        return d, nbytes(guards[base].parameters()), nbytes((*d.guard_dec.k, *d.guard_dec.v))
    d = StaticDecoder(backbone, None if mode == "gen" else verifier, mode, max_len)
    return d, (0 if mode in ("gen", "fused_gen") else nbytes(verifier.parameters())), 0


def load_verifier(backbone: Qwen3Min, ckpt_glob: str, untrained: bool, repo_id: str):
    paths = sorted(glob.glob(ckpt_glob))
    if len(paths) != 1:
        raise FileNotFoundError(f"--ckpt {ckpt_glob!r} matches {len(paths)} files, expected 1")
    ckpt = torch.load(paths[0], map_location="cpu", weights_only=False)
    vc = variant_from_checkpoint(ckpt["variant"])
    v = build_trainable(backbone.cfg, vc).to(backbone.device).eval()
    if untrained:
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for name, p in v.lora.named_parameters():
                if name.endswith(".B"):
                    p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    else:
        if ckpt["model"] != model_tag(ModelConfig(repo_id=repo_id)):
            raise ValueError(f"{paths[0]} was trained on {ckpt['model']}, not {repo_id} (use --untrained)")
        v.load_state_dict(ckpt["state_dict"])
    return v, paths[0], vc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--ckpt", required=True, help="glob of one ToRA checkpoint (tora.pt)")
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--execs", nargs="+", default=list(EXECS), choices=EXECS)
    ap.add_argument("--baselines", nargs="*", default=[], choices=[*BASELINES, *(b + KERNELS for b in BASELINES)])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--warm-s", type=float, default=10.0)
    args = ap.parse_args()
    gpu_info.require_visible_devices()

    backbone = Qwen3Min.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16, max_seq_len=max(args.contexts) + 1, verbose=False)
    verifier, ckpt_path, vc = load_verifier(backbone, args.ckpt, args.untrained, args.model)
    guards, wanted = {}, {b.removesuffix(KERNELS) for b in args.baselines}
    if "singprobe" in wanted:  # its own copy of the generator (it checks the probe's base model)
        guards["singprobe"] = SingProbe.from_pretrained(BASELINES["singprobe"], base_repo=args.model, max_seq_len=backbone.max_seq_len)
    if "qwen3guard_stream" in wanted:
        guards["qwen3guard_stream"] = Qwen3GuardStream.from_pretrained(BASELINES["qwen3guard_stream"], max_seq_len=backbone.max_seq_len)
    names = args.modes + args.baselines
    config = {**vars(args), "ckpt_path": ckpt_path, "variant": dataclasses.asdict(vc), "baseline_repos": {b: BASELINES[b] for b in wanted}}
    rows = []
    with RunLogger("bench", make_run_id(args.name), config, seed=0) as log:
        warm = StaticDecoder(backbone, None, "gen", max_len=min(args.contexts) + 1)
        warm.fill_random(min(args.contexts))
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.warm_s:
            warm.step()
        torch.cuda.synchronize()
        del warm
        for ctx in args.contexts:
            for exe in args.execs:
                logits = {}
                for mode in names:
                    decoder, param_bytes, cache_bytes = build(mode, backbone, verifier, guards, ctx + 1)
                    decoder.fill_random(ctx)
                    fn = runner(decoder.step, exe)
                    logits[mode] = fn()[0].clone()
                    gen_only = GEN_ONLY.get(mode)
                    s2 = torch.equal(logits[mode], logits[gen_only]) if gen_only in logits else None
                    stats = time_ms(fn, args.warmup, args.reps)
                    shared = gpu_info.snapshot()["visible_gpus_shared"]
                    row = {"model": args.model, "context": ctx, "exec": exe, "mode": mode, **stats, "s2_generator_bitwise": s2,
                           "extra_param_bytes": param_bytes, "extra_cache_bytes": cache_bytes, "gpu_shared": shared}
                    log.log(row)
                    rows.append(row)
                    log.console.info("T=%d %-13s %-9s %.3f ms (p90 %.3f)%s%s", ctx, exe, mode, stats["median_ms"], stats["p90_ms"],
                                     "" if s2 is None else f"  S2 {s2}", "  GPU SHARED" if shared else "")
                    del decoder
                    torch.cuda.empty_cache()
        print(f"\n{args.model}, batch 1, ms per token (median of {args.reps}); overhead over gen at the same execution")
        print("| Context | Execution | " + " | ".join(names) + " |")
        print("|---|---|" + "---|" * len(names))
        for ctx in args.contexts:
            for exe in args.execs:
                cell = {r["mode"]: r["median_ms"] for r in rows if r["context"] == ctx and r["exec"] == exe}
                base = cell.get("gen")
                parts = [f"{cell[m]:.3f}" + ("" if m == "gen" or base is None else f" ({100 * (cell[m] / base - 1):+.1f}%)") for m in names]
                print(f"| {ctx} | {exe} | " + " | ".join(parts) + " |")
        broken = [(r["context"], r["exec"], r["mode"]) for r in rows if r["s2_generator_bitwise"] is False]
        print(f"S2 (generator logits bitwise equal to the generator-only mode): {'held in every row' if not broken else f'BROKEN in {broken}'}")
        if any(r["gpu_shared"] for r in rows):
            log.console.warning("another process shared the GPU during some rows: re-run them")


if __name__ == "__main__":
    main()
