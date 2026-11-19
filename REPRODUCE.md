# Reproducing the paper

This guide regenerates every number in the paper from scratch: data preparation, the baselines, ToRA's training and
evaluation, the latency benchmarks, the analytic costs and each ablation. Commands run from the repository root and
write to `data_cache/` (datasets) and `results/` (logs, checkpoints and score files), both of which git ignores. Each
job prints a summary and also logs its config, environment and metrics to `results/<phase>/<run_id>.jsonl`, where the
run id is the run name plus a timestamp.

All runs in the paper used one NVIDIA RTX A6000 (48 GB) per job, and the times quoted below are the wall-clock times
of those runs. Scoring through the deployed decode step and computing the teacher onsets are split into four shards
that run in parallel on four GPUs. Every command takes `CUDA_VISIBLE_DEVICES` as given and refuses to run without it.
The latency benchmarks need a GPU with nothing else running on it.

## 1. Environment and data access

```bash
conda env create -f environment.yml && conda activate tora && pip install -e ".[dev]"
export HF_TOKEN=...      # read by huggingface_hub; never written to the logs
```

Several datasets are gated on the Hugging Face Hub: at least `allenai/wildguardmix`, `allenai/xstest-response` and
`6rightjade/expguardmix`. Accept their terms on the Hub with the account behind `HF_TOKEN`, and do the same for any
other dataset whose download fails with an authorization error. The HarmBench and OpenAI Moderation test files come
from WildGuard's evaluation suite (`allenai/safety-eval` on GitHub) at a pinned commit, and each file is checked
against its SHA-256 before use. The models are `Qwen/Qwen3-0.6B` (the generator), `Qwen/Qwen3Guard-Stream-0.6B` (the
guard baseline and the token-level teacher) and `inclusionAI/Qwen3-0.6B-singprobe` (the probe baseline).

| Name | Source | Use |
|---|---|---|
| wildguard | allenai/wildguardmix | train, val, prompt and response test |
| aegis2 | nvidia/Aegis-AI-Content-Safety-Dataset-2.0 | train, val, prompt and response test |
| beavertails | PKU-Alignment/BeaverTails (30k) | train, response test, safe responses for the streaming FPR |
| saferlhf | PKU-Alignment/PKU-SafeRLHF | train, response test |
| toxicchat | lmsys/toxic-chat | train, prompt test |
| aegis1 | nvidia/Aegis-AI-Content-Safety-Dataset-1.0 | train, prompt test |
| expguard | 6rightjade/expguardmix | train, prompt and response test |
| think | Qwen/Qwen3GuardTest (thinking) | response test |
| qwen3guardtest | Qwen/Qwen3GuardTest (response_loc) | streaming detection |
| singstreambench | inclusionAI/SingStreamBench | streaming AUC |
| harmbench_prompt, harmbench_response, openai_mod | allenai/safety-eval files | prompt / response test |
| simplesafetytests | Bertievidgen/SimpleSafetyTests | prompt test |
| xstest_prompt | Paul/XSTest | prompt test |
| xstest_response | allenai/xstest-response | response test |

## 2. Data preparation

```bash
python -m tora.data.prepare      # CPU only; 5.5 minutes in our run
```

This downloads the 16 datasets, holds out 5% of each training set by prompt as its validation split (Aegis 2.0 ships
its own), removes every training prompt that also occurs in any test or validation split, and writes
`data_cache/processed/<dataset>/{train,val,test}.jsonl`. It then runs the data checks S6a to S6d (split counts and
label balance, leakage, label positions, lengths and truncation), stops if any prompt still leaks or a label sits at
the wrong token, and logs the report to `results/data/`.

## 3. Baselines

SingProbe's prompt protocol averages its query score over a greedy answer of the generator, so the answers are
generated first. Then the released Qwen3Guard-Stream-0.6B and SingProbe score every benchmark with their own decision
rules:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tora.eval.generate     # -> data_cache/generated/qwen3-0.6b_greedy_512/
CUDA_VISIBLE_DEVICES=0 python -m tora.eval.run_eval --config configs/eval/baselines.yaml     # 8.3 minutes
```

The second command writes per-example scores to `results/eval/baselines-<id>/` and metrics to
`results/eval/baselines-<id>.jsonl`. The baselines' streaming scores come from the streaming evaluation in Section 6.

## 4. Teacher onsets

Qwen3Guard-Stream reads every unsafe training response exactly as the trainer renders it, and the first token of its
earliest pair of consecutive unsafe (or controversial) tokens becomes that response's onset:

```bash
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/teacher_onsets.py --shard $i --num-shards 4 & done; wait
```

Each shard took 1.2 minutes and writes `data_cache/teacher/qwen3guard_stream_0p6b.shard{i}of4.jsonl`, the files that
`train.prefix_teacher` in `configs/train/tora.yaml` reads.

## 5. Training

`scripts/train.sh` runs every training job of the paper as an override of `configs/train/tora.yaml`; its header
lists the jobs and their role in the paper. ToRA itself is K = 14, rank 8, alpha 16, an MLP head and per-token labels
at N = 128 positions, trained for two epochs, with four validation passes and the checkpoint with the best mean
validation AUPRC kept.

```bash
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$s bash scripts/train.sh tora $s & done; wait     # 72 minutes each
```

Each run writes `results/train/tora_seed<s>-<id>/tora.pt` and the checkpoint's teacher-forced scores on every
validation and test split, `tora__<dataset>__<split>.jsonl`, next to it, and logs its test metrics to
`results/train/tora_seed<s>-<id>.jsonl`.

## 6. Evaluation through the deployed decode step

The paper's accuracy numbers for ToRA are measured as it would run in deployment: the prompt is prefilled, then
every later token goes through one compiled, CUDA-graph-captured decode step on the fused dual-row kernels. For each
seed, the benchmark items are split into four shards:

```bash
S=0; RUN=deployed_seed$S-$(date +%Y%m%d-%H%M%S)
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/deployed_eval.py score --run-id $RUN --shard $i --shards 4 \
    --method tora_s$S --ckpt "results/train/tora_seed$S-*/tora.pt" & done; wait     # about 50 minutes per shard
```

The teacher-forced streaming evaluation scores the baselines' streams, and also ToRA's teacher-forced streams, which
the agreement check below compares against:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/stream_eval.sh stream qwen3guard_stream_0p6b singprobe_qwen3_0p6b \
    tora_tf='results/train/tora_seed*/tora.pt'
```

Once a seed's four shards finish, the metrics step turns its score files into a regular `run_eval` run,
`results/eval/<RUN>.jsonl`. It also reports how far the deployed scores are from the teacher-forced ones:

```bash
python scripts/deployed_eval.py metrics --run-id $RUN --method tora_s$S \
    --reference "results/train/tora_seed$S-*/tora" "results/eval/stream-*/tora_tf_s$S"
```

The two paths run different kernels and summation orders in bf16, so their scores differ slightly, and the
autotuned kernel configurations can change between processes. Repeating the deployed evaluation therefore gives
scores within about 1e-2 of each other, which can move single threshold decisions.

## 7. Tables

**Per-benchmark F1 and AUPRC** (Table III):

```bash
python scripts/f1_table.py --baselines results/eval/baselines-<id> \
    --method "ToRA=results/eval/deployed_seed*-*/tora_s*" --method "ToRA, teacher-forced=results/train/tora_seed*/tora"
```

ToRA's decision threshold is the F1-maximizing threshold on the pooled WildGuard and Aegis 2.0 validation splits,
chosen separately for prompts and responses, while the baselines use their released rules. Expected: response Avg.
F1 81.4 ± 0.2 and prompt Avg. F1 86.3 ± 0.4 for ToRA, 80.4 / 84.6 for Qwen3Guard-Stream and 77.6 / 83.0 for SingProbe.

**Streaming detection** (Table II):

```bash
python scripts/stream_table.py results/eval/deployed_seed*-*.jsonl results/eval/stream-*.jsonl \
    --methods tora tora_tf qwen3guard_stream_0p6b singprobe_qwen3_0p6b
```

Expected at the 5% budget: detection 83.6 ± 0.6% (Qwen3Guard-Stream 84.7%, SingProbe 71.8%), exact hit
75.6 ± 0.1%, median delay 5 tokens (7 for both baselines). At the 1% budget: detection 59.5 ± 1.6%, exact hit
50.2 ± 2.3%, median delay 8.3 tokens. The same table gives the SingStreamBench response-level AUCs (ToRA 92.8,
Qwen3Guard-Stream 94.5, SingProbe 98.6).

**Headline table** (Table I) combines the averages above with the latency run of Section 8:

```bash
python scripts/headline_table.py --baselines results/eval/baselines-<id> \
    --stream results/eval/deployed_seed*-*.jsonl results/eval/stream-*.jsonl \
    --latency "results/bench/decode_guards_kernels_0p6b-*.jsonl" \
    --row "ToRA" "results/eval/deployed_seed*-*/tora_s*" tora fused \
    --row "SingProbe" baseline:SingProbe singprobe_qwen3_0p6b singprobe_kernels \
    --row "Qwen3Guard-Stream" baseline:Qwen3Guard-Stream qwen3guard_stream_0p6b qwen3guard_stream_kernels \
    --row "Generator alone, stock (cuBLAS)" - - gen --row "Generator alone on our kernels" - - fused_gen
```

## 8. Latency

Each benchmark times batch-1 decode steps at fixed contexts after a global warm-up, using CUDA events around every
repeat. The trained checkpoint only sets the verifier's shape, since latency does not depend on the weights' values.

```bash
CK="results/train/tora_seed0-*/tora.pt"
# Table I: ToRA and both baselines, each also on our kernels (8 minutes)
CUDA_VISIBLE_DEVICES=0 python scripts/decode_bench.py --name decode_guards_kernels_0p6b --ckpt "$CK" --modes gen fused fused_gen \
    --baselines singprobe qwen3guard_stream singprobe_kernels qwen3guard_stream_kernels --execs graph compile+graph --contexts 512 2048 8191
# Fig. 4a: the verifier row run separately, batched, and on the fused kernels, under each execution (6.4 minutes)
CUDA_VISIBLE_DEVICES=0 python scripts/decode_bench.py --name decode_0p6b --ckpt "$CK"
# The same at 1.7B, with an untrained adapter of ToRA's shape (6.4 minutes)
CUDA_VISIBLE_DEVICES=0 python scripts/decode_bench.py --name decode_1p7b --model Qwen/Qwen3-1.7B --ckpt "$CK" --untrained
# Fig. 4b: one dual-row FFN layer in a stack of 14 at each Qwen3 shape; fetches only config.json (5.6 minutes)
CUDA_VISIBLE_DEVICES=0 python scripts/layer_bench.py --name layer_ffn --models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B
```

Expected, relative to the stock generator: at 0.6B, separate calls add about 51%, one batched PyTorch forward adds
15 to 21%, and the dual-row kernels add less than 1%. In Table I, measured against the generator on the same kernels,
ToRA adds 4.0 / 3.5 / 2.8% at contexts 512 / 2k / 8k, SingProbe 1.7 / 1.6 / 1.2% and Qwen3Guard-Stream 84.3 / 85.9 /
89.3%. Per FFN layer, a merged copy of the weights costs about 100% at every size, and our kernels run both rows 3 to
14% faster than cuBLAS runs the generator row alone, at 69 to 88% of the A6000's 768 GB/s peak bandwidth (cuBLAS: 59
to 84%). At 1.7B, the kernels add 2.8 to 3.3% over our generator-only kernels, while naive batching adds 14 to 17%.
Every row also checks that the generator's logits are bitwise unchanged by the verifier row.

## 9. Analytic costs

```bash
python scripts/cost_table.py
```

This prints the verifier row's extra FLOPs and bytes per decoded token relative to the generator, from the model's
config alone:

```
verifier                                 T = 1,024             T = 4,096            T = 16,384
                                  FLOPs      bytes      FLOPs      bytes      FLOPs      bytes
ToRA, K = 14 (shared KV)          38.0%      0.37%      40.8%      0.29%      44.0%      0.16%
ToRA, K = 28 (shared KV)          77.3%      0.58%      83.6%      0.46%      90.9%      0.25%
own KV cache, K = 28              77.3%      9.23%      83.6%     27.72%      90.9%     59.25%
```

## 10. Ablations

The ablations are scored teacher-forced, from the score files each training run writes and from `stream_eval.sh`, and
use 3 seeds unless noted. `f1_table.py` takes training runs as `--method "NAME=results/train/<run>_seed*/<variant>"`.

**Shared versus own KV cache.** Both verifiers adapt all 28 layers and use last-token labels:

```bash
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$s bash scripts/train.sh lasttoken_k28 $s; done     # 56 minutes each
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$s bash scripts/train.sh ownkv $s; done             # 138 minutes each
python scripts/f1_table.py --baselines results/eval/baselines-<id> \
    --method "Shared KV=results/train/lasttoken_k28_seed*/base" --method "Own KV=results/train/ownkv_k28_seed*/base"
```

Expected: 81.4 response and 86.7 prompt F1 with the shared cache against 81.2 and 86.5 with its own, with comparable
AUPRC. The memory traffic each adds comes from Section 9: 9 to 59% for the own cache, under 0.6% when shared.

**Adapter rank** (seed 0). The rank sweep adapts all 28 layers with last-token labels and four attention-sink
positions (`verifier.sink = 4`: the first four positions read the verifier's own keys and values):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh rank 0                                     # r = 4 and 16 (63 minutes)
CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh rank 0 'train.sweep={verifier.rank: [32]}'
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh rank 0 'train.sweep={verifier.rank: [8]}'
python scripts/f1_table.py --baselines results/eval/baselines-<id> --method "r 4=results/train/rank_seed0-*/r4" \
    --method "r 8=results/train/rank_seed0-*/r8" --method "r 16=results/train/rank_seed0-*/r16" --method "r 32=results/train/rank_seed0-*/r32"
```

Expected response Avg. F1: 82.3 (r = 4), 81.4 (r = 8), 82.0 (r = 16) and 81.9 (r = 32), all within 0.9 of each other.

**Frozen features.** An MLP probe on the residual streams entering SingProbe's three tap layers (8, 17, 26), trained
on the same data with last-token labels, against ToRA with the same labels:

```bash
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$s bash scripts/train.sh mlp_probe $s; done   # 55 minutes each; three learning rates
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$s bash scripts/train.sh lasttoken $s; done   # 36.5 minutes each
python scripts/f1_table.py --baselines results/eval/baselines-<id> \
    --method "MLP probe=results/train/mlp_probe_seed*/mlp_probe_lr1e-3" --method "ToRA, last-token labels=results/train/lasttoken_k14_seed*/base"
```

The learning rate 1e-3 has the best mean validation AUPRC. Expected: 72.5 response and 81.3 prompt F1 for the probe,
against 81.5 and 86.7 for ToRA.

**Adaptation depth.** ToRA with all 28 layers adapted, scored for streaming detection next to K = 14:

```bash
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$s bash scripts/train.sh tora_k28 $s; done   # 119 minutes each
CUDA_VISIBLE_DEVICES=0 bash scripts/stream_eval.sh stream_depth tora_k14='results/train/tora_seed*/tora.pt' \
    tora_k28='results/train/tora_k28_seed*/tora.pt'
python scripts/stream_table.py results/eval/stream_depth-*.jsonl --methods tora_k14 tora_k28
```

Expected detection at the 5% budget: 83.5% for K = 14 and 84.5% for K = 28, at 41% and 84% extra FLOPs at a 4k
context (Section 9).

**Token-level supervision.** The `lasttoken` runs above are ToRA with N = 0:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/stream_eval.sh stream_n0 n0='results/train/lasttoken_k14_seed*/base.pt'
python scripts/stream_table.py results/eval/stream_n0-*.jsonl results/eval/stream_depth-*.jsonl --methods n0 tora_k14
```

Expected detection at the 5% budget: 19.9% with N = 0 against 83.5% with N = 128, with response F1 near 81.5 for both.

**Without the teacher** (seed 0). With N = 8 positions labelled only on safe responses:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh noteacher 0     # 51 minutes
CUDA_VISIBLE_DEVICES=0 bash scripts/stream_eval.sh stream_noteacher noteacher='results/train/noteacher_n8_seed0-*/safe.pt'
python scripts/stream_table.py results/eval/stream_noteacher-*.jsonl --methods noteacher
```

Expected at the 5% budget: 85.0% detected, with a median delay of 25 tokens.

**Peak scores** (seed 0). `scripts/stream_diag.py` reads one checkpoint's per-token scores and prints, among other
diagnostics, the median peak score of the unsafe Qwen3GuardTest responses before their unsafe span and from its start
on. The paper's values, 0.002 and 0.984, come from the N = 8 teacher model:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh teacher_n8 0
CUDA_VISIBLE_DEVICES=0 python scripts/stream_diag.py results/train/teacher_n8_seed0-*/teacher.pt
```

Three jobs, `lasttoken_k28`, `noteacher` and `teacher_n8`, trained a further variant alongside in the paper's runs.
That variant did not make the paper and is not in this code. Their configs match the paper's runs exactly, but the
MLP head's dropout shares one random stream across the variants of a job, so these reruns agree with the paper up to
seed-level noise rather than bit for bit.

## 11. Tests

```bash
python -m pytest -m "not gpu"                          # CPU only, seconds
CUDA_VISIBLE_DEVICES=0 python -m pytest -m "not slow"  # + GPU tests on Qwen3-0.6B and the baselines (about 10 minutes)
CUDA_VISIBLE_DEVICES=0 python -m pytest -m slow        # the kernel grid, the static and fused decode checks, Qwen3-1.7B
```

The tests check `qwen3_min` against Hugging Face Transformers bit for bit, the verifier's training paths against the
reference decoder, the dual-row kernels against their PyTorch references, and that the generator's output never
changes when the verifier row runs beside it.
