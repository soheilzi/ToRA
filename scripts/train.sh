#!/usr/bin/env bash
# Every training run in the paper: ToRA (configs/train/tora.yaml) and its ablations, as overrides of that config.
# Data: python -m tora.data.prepare; the per-token labels also need scripts/teacher_onsets.py first.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh JOB SEED [overrides ...]
#
# JOB             run name                variant(s)                     paper
# tora            tora_seedS              tora                           the method: K = 14, N = 128 teacher labels
# tora_k28        tora_k28_seedS          tora                           depth ablation: K = 28 (streaming detection)
# lasttoken       lasttoken_k14_seedS     base                           N = 0: labels at the last prompt / response token only;
#                                                                         also the frozen-feature comparison's reference
# lasttoken_k28   lasttoken_k28_seedS     base                           cache ablation, shared KV: K = 28, last-token labels
# ownkv           ownkv_k28_seedS         base                           cache ablation, own KV cache: K = 28, last-token labels
# noteacher       noteacher_n8_seedS      safe                           no-teacher pilot (seed 0): N = 8, labels on safe responses only
# teacher_n8      teacher_n8_seedS        teacher                        N = 8 teacher labels (seed 0): the model scripts/stream_diag.py
#                                                                         read for the median peak scores (Streaming Detection)
# mlp_probe       mlp_probe_seedS         mlp_probe_lr{3e-4,1e-3,3e-3}   frozen features: MLP on SingProbe's taps 8/17/26 (lr 1e-3
#                                                                         reported, the val pick)
# rank            rank_seedS              r4, r16                        rank ablation (seed 0): K = 28, last-token labels, 4
#                                                                         attention-sink positions; r = 32 and the r = 8 reference:
#                                                                         add 'train.sweep={verifier.rank: [32]}' (or [8])
#
# The paper trains 3 seeds (0, 1, 2) of tora, tora_k28, lasttoken, lasttoken_k28, ownkv and mlp_probe. Extra arguments
# are appended as config overrides (dotted keys, YAML values). Uses CUDA_VISIBLE_DEVICES as given.
#
# Every job's effective config equals that of the paper's run. Some of those runs also trained a variant that did not
# make the paper (lasttoken_k28, noteacher, teacher_n8) in the same job; variants share the generator pass and their
# initialization is seeded per variant, but the MLP head's dropout draws from one RNG stream, so these jobs agree with
# the paper's runs up to seed-level noise rather than bit for bit.
set -euo pipefail
cd "$(dirname "$0")/.."
job=$1 seed=$2; shift 2
train=(python -m tora.train.train_verifier --config configs/train/tora.yaml run.seed=$seed)
# last-token jobs: no earlier-position labels, no micro-batches, one unnamed variant
lasttoken=(train.prefix_positions=0 train.micro_batch_tokens=null 'train.variants={}')
case $job in
  tora) exec "${train[@]}" run.name=tora_seed$seed "$@" ;;
  # memory grows with K x sequences per micro-batch (N + 2 verifier rows each), so K = 28 needs smaller micro-batches
  tora_k28) exec "${train[@]}" run.name=tora_k28_seed$seed train.base.verifier.k_layers=28 \
    train.micro_batch_tokens=8192 train.micro_batch_seqs=96 "$@" ;;
  lasttoken) exec "${train[@]}" run.name=lasttoken_k14_seed$seed "${lasttoken[@]}" "$@" ;;
  lasttoken_k28) exec "${train[@]}" run.name=lasttoken_k28_seed$seed "${lasttoken[@]}" train.base.verifier.k_layers=28 "$@" ;;
  # every position of the adapted layers carries gradients: 8k-token micro-batches fit K = 28 on a 48 GB GPU
  ownkv) exec "${train[@]}" run.name=ownkv_k28_seed$seed "${lasttoken[@]}" train.base.verifier.k_layers=28 \
    train.base.verifier.kv=own train.micro_batch_tokens=8192 "$@" ;;
  noteacher) exec "${train[@]}" run.name=noteacher_n8_seed$seed train.prefix_positions=8 train.micro_batch_tokens=null \
    'train.variants={safe: {prefix_labels: safe}}' "$@" ;;
  teacher_n8) exec "${train[@]}" run.name=teacher_n8_seed$seed train.prefix_positions=8 train.micro_batch_tokens=null \
    'train.variants={teacher: {}}' "$@" ;;
  mlp_probe) exec "${train[@]}" run.name=mlp_probe_seed$seed "${lasttoken[@]}" \
    'train.variants={mlp_probe_lr3e-4: {probe: {}, head_lr: 3.0e-4}, mlp_probe_lr1e-3: {probe: {}, head_lr: 1.0e-3}, mlp_probe_lr3e-3: {probe: {}, head_lr: 3.0e-3}}' "$@" ;;
  rank) exec "${train[@]}" run.name=rank_seed$seed "${lasttoken[@]}" train.base.verifier.k_layers=28 train.base.verifier.sink=4 \
    'train.sweep={verifier.rank: [4, 16]}' "$@" ;;
  *) echo "unknown job $job (see the table at the top of $0)" >&2; exit 2 ;;
esac
