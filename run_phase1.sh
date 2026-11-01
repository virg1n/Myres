#!/usr/bin/env bash
# Phase 1 (trained modes on Qwen3) with one command. Needs a finished phase 0 in runs/phase0.
#   bash run_phase1.sh            # smoke test in the foreground (~25-40 min), then the full run (~8-10 h) in background
#   HOURS=13 bash run_phase1.sh   # time budget for the full run (default 12)
# Re-running resumes: finished steps are skipped, training continues from its last checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
P0="${P0:-runs/phase0}"
OUT="${OUT:-runs/phase1}"
HOURS="${HOURS:-12}"
export VLLM_USE_FLASHINFER_SAMPLER=0
test -f "$P0/report_phase0.md" || { echo "phase 0 results not found in $P0"; exit 1; }

python3 - <<'PY'
import shutil, torch, vllm, transformers
print("torch", torch.__version__, "| vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| GPUs", torch.cuda.device_count())
print(f"free disk: {shutil.disk_usage('.').free / 1e9:.0f} GB "
      "(peak use ~60 GB: up to 4 jobs x 4 exported modes x 3.5 GB, deleted after each evaluation)")
PY

echo "== smoke run: $OUT/smoke"
python3 -m phase1.run --run "$OUT/smoke" --phase0 "$P0" --smoke "$@" 2>&1 | tee "$OUT.smoke.log"

echo "== smoke OK, starting the full phase 1 in the background (budget ${HOURS} h)"
nohup python3 -m phase1.run --run "$OUT" --phase0 "$P0" --hours "$HOURS" "$@" > "$OUT.log" 2>&1 &
echo $! > "$OUT.pid"
echo "PID $(cat "$OUT.pid")"
echo "status:  cat $OUT/STATUS.md"
echo "log:     tail -f $OUT.log"
echo "stop:    kill \$(cat $OUT.pid)"
echo "report:  $OUT/report_phase1.md"
