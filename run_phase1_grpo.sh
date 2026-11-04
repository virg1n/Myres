#!/usr/bin/env bash
# Stage 2 by GRPO instead of RFT with one command: plain GRPO from sftmulti and GRPO with the 4 FiLM modes of wta1,
# seeds 0 and 1 (~12-13 h on 4 GPUs). Needs the finished phase1_self run (its data and M_0 generations).
#   bash run_phase1_grpo.sh                 # smoke test in the foreground (~25-35 min), then the full run in background
#   bash run_phase1_grpo.sh --train_hours 7 # shorter training (the steps follow the calibration)
# Re-running resumes: finished steps are skipped, training continues from its last checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
P1="${P1:-runs/phase1}"
SEEDS_RUN="${SEEDS_RUN:-runs/phase1_seeds}"
SELF_RUN="${SELF_RUN:-runs/phase1_self}"
OUT="${OUT:-runs/phase1_grpo}"
HOURS="${HOURS:-14}"
MAX_HOURS="${MAX_HOURS:-16}"
export VLLM_USE_FLASHINFER_SAMPLER=0
for x in "$@"; do          # the paths belong to this script: the lock, the smoke directory and the PID file use them
  case "$x" in --run|--run=*|--seeds_run*|--phase1*|--self_run*|--smoke|--plan_only)
    echo "$x is set by this script: use OUT=, SEEDS_RUN=, P1=, SELF_RUN= instead"; exit 1;; esac
done
for pf in runs/*.pid; do   # phase1_self and every other run: the GPUs must be free
  [ -f "$pf" ] || continue
  if kill -0 "$(cat "$pf")" 2> /dev/null; then
    echo "${pf%.pid} is still running (PID $(cat "$pf")): the GPUs are busy"; exit 1
  fi
done
test -f "$SELF_RUN/data/hv.jsonl" || { echo "phase1_self data not found in $SELF_RUN/data"; exit 1; }

# one instance per run directory: the background run inherits the lock and holds it until it ends
mkdir -p "$(dirname "$OUT")"
command -v flock > /dev/null || { echo "flock not found (package util-linux)"; exit 1; }
exec 9>"$OUT.lock"
flock -n 9 || { echo "$OUT is already running (PID file: $OUT.pid); stop it first"; exit 1; }

python3 - <<'PY'
import shutil, torch, vllm, transformers
print("torch", torch.__version__, "| vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| GPUs", torch.cuda.device_count())
print(f"free disk as the OS reports it: {shutil.disk_usage('.').free / 1e9:.0f} GB (peak use ~18 GB)")
PY

ARGS=(--run "$OUT" --self_run "$SELF_RUN" --seeds_run "$SEEDS_RUN" --phase1 "$P1")
python3 -m phase1.grun "${ARGS[@]}" --hours "$HOURS" --max_hours "$MAX_HOURS" --plan_only "$@"

if [ ! -f "$OUT/STATUS.md" ]; then
  echo "== smoke run: $OUT.smoke"
  rm -rf "$OUT.smoke"
  python3 -m phase1.grun --run "$OUT.smoke" --self_run "$SELF_RUN" --seeds_run "$SEEDS_RUN" --phase1 "$P1" --smoke "$@" \
    2>&1 | tee "$OUT.smoke.log"
  rm -rf "$OUT.smoke"          # the disk is small: the real run needs this space
  echo "== smoke OK"
fi

echo "== starting the full run in the background"
nohup python3 -m phase1.grun "${ARGS[@]}" --hours "$HOURS" --max_hours "$MAX_HOURS" "$@" >> "$OUT.log" 2>&1 &
echo $! > "$OUT.pid"
echo "PID $(cat "$OUT.pid")"
echo "status:   cat $OUT/STATUS.md"
echo "log:      tail -f $OUT.log"
echo "speed:    grep min/step $OUT/logs/calib_*.log $OUT/logs/train_*.log | tail"
echo "stop:     kill \$(cat $OUT.pid)"
echo "report:   $OUT/report_grpo.md"
