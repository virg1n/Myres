#!/usr/bin/env bash
# Continuous random z with one command: wta1 with a fresh z per example / per attempt, d = 8, 16, 128, 1024, seed 0,
# evaluated at k = 32 with wta1, loramcl and sftmulti of seed 0 (~5 h on 4 GPUs). Needs phase1 and phase1_seeds.
#   bash run_phase1_z.sh            # smoke test in the foreground (~15 min), then the full run in background
# Re-running resumes: finished steps are skipped, training continues from its last checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
P1="${P1:-runs/phase1}"
SEEDS_RUN="${SEEDS_RUN:-runs/phase1_seeds}"
OUT="${OUT:-runs/phase1_z}"
HOURS="${HOURS:-5.5}"
MAX_HOURS="${MAX_HOURS:-7}"
export VLLM_USE_FLASHINFER_SAMPLER=0
for x in "$@"; do          # the paths belong to this script: the lock, the smoke directory and the PID file use them
  case "$x" in --run|--run=*|--seeds_run*|--phase1*|--smoke|--plan_only)
    echo "$x is set by this script: use OUT=, SEEDS_RUN=, P1= instead"; exit 1;; esac
done
for other in runs/phase1_seeds runs/phase1_rank; do
  if [ -f "$other.pid" ] && kill -0 "$(cat "$other.pid")" 2> /dev/null; then
    echo "$other is still running (PID $(cat "$other.pid")): the GPUs are busy"; exit 1
  fi
done

# one instance per run directory: the background run inherits the lock and holds it until it ends
mkdir -p "$(dirname "$OUT")"
command -v flock > /dev/null || { echo "flock not found (package util-linux)"; exit 1; }
exec 9>"$OUT.lock"
flock -n 9 || { echo "$OUT is already running (PID file: $OUT.pid); stop it first"; exit 1; }

python3 - <<'PY'
import shutil, torch, vllm, transformers
print("torch", torch.__version__, "| vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| GPUs", torch.cuda.device_count())
print(f"free disk as the OS reports it: {shutil.disk_usage('.').free / 1e9:.0f} GB (peak use ~15 GB)")
PY

ARGS=(--run "$OUT" --seeds_run "$SEEDS_RUN" --phase1 "$P1")
python3 -m phase1.zrun "${ARGS[@]}" --hours "$HOURS" --max_hours "$MAX_HOURS" --plan_only "$@"

if [ ! -f "$OUT/STATUS.md" ]; then
  echo "== smoke run: $OUT.smoke"
  rm -rf "$OUT.smoke"
  python3 -m phase1.zrun --run "$OUT.smoke" --seeds_run "$SEEDS_RUN" --phase1 "$P1" --smoke "$@" \
    2>&1 | tee "$OUT.smoke.log"
  rm -rf "$OUT.smoke"          # the disk is small: the real run needs this space
  echo "== smoke OK"
fi

echo "== starting the full run in the background"
nohup python3 -m phase1.zrun "${ARGS[@]}" --hours "$HOURS" --max_hours "$MAX_HOURS" "$@" >> "$OUT.log" 2>&1 &
echo $! > "$OUT.pid"
echo "PID $(cat "$OUT.pid")"
echo "status:  cat $OUT/STATUS.md"
echo "log:     tail -f $OUT.log"
echo "stop:    kill \$(cat $OUT.pid)"
echo "report:  $OUT/report_z.md"
