#!/usr/bin/env bash
# One command for the whole experiment:
#   bash run_all.sh                       # configs/small.yaml on all visible GPUs
#   bash run_all.sh configs/base.yaml     # bigger model
#   NGPU=2 bash run_all.sh configs/tiny.yaml train.plain_steps=100   # extra overrides
#
# Stages (each is skipped if already finished, so the script can be re-run to resume):
#   1. train the plain model (variant 1)        -> runs/<name>/plain/final.pt
#   2. train the revision model, same FLOPs      -> runs/<name>/revise/final.pt
#   3. evaluate all variants on ID + OOD iGSM    -> runs/<name>/results/
#   4. write runs/<name>/results/results.md (+ acc_vs_compute.png)
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-configs/small.yaml}"
shift || true
OVERRIDES=("$@")

if [[ -z "${NGPU:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NGPU=$(nvidia-smi -L | wc -l)
  else
    NGPU=1
  fi
fi
[[ "$NGPU" -lt 1 ]] && NGPU=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

NAME=$(python -c "import sys; from revise.utils import load_config; print(load_config(sys.argv[1], sys.argv[2:])['out_dir'])" "$CONFIG" "${OVERRIDES[@]}")
mkdir -p "$NAME"
LOG="$NAME/run_all.log"
echo "config=$CONFIG  gpus=$NGPU  out=$NAME  log=$LOG"


launch() {
  if [[ "$NGPU" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NGPU" -m "$@"
  else
    python -m "$@"
  fi
}

{
  echo "=== [1/4] plain model ($(date)) ==="
  launch revise.train --config "$CONFIG" --stage plain "${OVERRIDES[@]}"
  echo "=== [2/4] revision model ($(date)) ==="
  launch revise.train --config "$CONFIG" --stage revise "${OVERRIDES[@]}"
  echo "=== [3/4] evaluation ($(date)) ==="
  launch revise.evaluate --config "$CONFIG" "${OVERRIDES[@]}"
  echo "=== [4/4] report ($(date)) ==="
  python -m revise.report --config "$CONFIG" "${OVERRIDES[@]}"
  echo "done ($(date)). Results: $NAME/results/results.md"
} 2>&1 | tee -a "$LOG"
