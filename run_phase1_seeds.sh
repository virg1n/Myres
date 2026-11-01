#!/usr/bin/env bash
# Seeds, SFT with equal data, LoRA-MCL and ModC with one command. Needs finished phases 0 and 1.
#   bash run_phase1_seeds.sh            # smoke test in the foreground (~20-30 min), then the full run in background
# Re-running resumes: finished steps are skipped, training continues from its last checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
P0="${P0:-runs/phase0}"
P1="${P1:-runs/phase1}"
OUT="${OUT:-runs/phase1_seeds}"
HOURS="${HOURS:-14}"
MAX_HOURS="${MAX_HOURS:-16}"
export VLLM_USE_FLASHINFER_SAMPLER=0
test -f "$P1/report_phase1.md" || { echo "phase 1 results not found in $P1"; exit 1; }

# one instance per run directory: the background run inherits the lock and holds it until it ends
mkdir -p "$(dirname "$OUT")"
command -v flock > /dev/null || { echo "flock not found (package util-linux)"; exit 1; }
exec 9>"$OUT.lock"
flock -n 9 || { echo "$OUT is already running (PID file: $OUT.pid); stop it first"; exit 1; }

python3 - <<'PY'
import shutil, torch, vllm, transformers
print("torch", torch.__version__, "| vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| GPUs", torch.cuda.device_count())
print(f"free disk: {shutil.disk_usage('.').free / 1e9:.0f} GB (peak use ~19-21 GB: 4 evaluations x 3.5 GB + adapters ~5 GB)")
PY

python3 -m phase1.seeds_run --run "$OUT" --phase1 "$P1" --phase0 "$P0" --hours "$HOURS" --max_hours "$MAX_HOURS" --plan_only "$@"

if [ ! -f "$OUT/STATUS.md" ]; then
  echo "== smoke run: $OUT.smoke"
  rm -rf "$OUT.smoke"
  python3 -m phase1.seeds_run --run "$OUT.smoke" --phase1 "$P1" --phase0 "$P0" --smoke "$@" 2>&1 | tee "$OUT.smoke.log"
  rm -rf "$OUT.smoke"          # the disk is small: the real run needs this space
  echo "== smoke OK"
fi

echo "== starting the full run in the background"
nohup python3 -m phase1.seeds_run --run "$OUT" --phase1 "$P1" --phase0 "$P0" --hours "$HOURS" --max_hours "$MAX_HOURS" "$@" >> "$OUT.log" 2>&1 &
echo $! > "$OUT.pid"
echo "PID $(cat "$OUT.pid")"
echo "status:  cat $OUT/STATUS.md"
echo "log:     tail -f $OUT.log"
echo "stop:    kill \$(cat $OUT.pid)"
echo "report:  $OUT/report_seeds.md"
