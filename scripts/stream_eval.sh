#!/usr/bin/env bash
# Streaming eval, teacher-forced (configs/eval/stream.yaml), of the released baselines and our checkpoints, in one run.
#   bash scripts/stream_eval.sh RUN_NAME METHOD... [-- overrides]
# METHOD is a registered scorer (qwen3guard_stream_0p6b, singprobe_qwen3_0p6b) or NAME='GLOB' over <variant>.pt
# checkpoints, which expands to ckpt:NAME_s0=..., NAME_s1=... in glob order (one per seed), e.g.
#   bash scripts/stream_eval.sh stream qwen3guard_stream_0p6b singprobe_qwen3_0p6b tora_tf='results/train/tora_seed*/tora.pt'
# Uses CUDA_VISIBLE_DEVICES as given.
set -euo pipefail
cd "$(dirname "$0")/.."
run=$1; shift
methods=()
while (($#)) && [[ $1 != -- ]]; do
  if [[ $1 == *=* ]]; then
    name=${1%%=*} i=0
    for f in ${1#*=}; do
      [[ -f $f ]] || { echo "no checkpoint matches ${1#*=}" >&2; exit 2; }
      methods+=("'ckpt:${name}_s$i=$f'"); i=$((i + 1))
    done
  else
    methods+=("'$1'")
  fi
  shift
done
(($#)) && shift
list=$(IFS=,; echo "${methods[*]}")
exec python -m tora.eval.run_eval --config configs/eval/stream.yaml run.name=$run "eval.methods=[$list]" "$@"
