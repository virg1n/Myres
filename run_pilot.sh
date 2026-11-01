#!/usr/bin/env bash
# One command for the whole pilot.
#   bash run_pilot.sh            # 11-hour budget
#   bash run_pilot.sh 9          # 9-hour budget
#   bash run_pilot.sh 11 --op_max 12   # extra args go to pilot.run_all
# Runs the preflight in the foreground (5-10 min), then the main run in the background (nohup),
# so you can close the terminal. Re-running the same command resumes.
set -euo pipefail
cd "$(dirname "$0")"
HOURS="${1:-11}"
if [ $# -gt 0 ]; then shift; fi
OUT="${OUT:-runs/pilot}"
mkdir -p "$OUT"

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count())"

echo "== preflight: tiny run of every job type (5-10 min)"
python3 -m pilot.run_all --out "$OUT" --hours "$HOURS" --phase preflight "$@" 2>&1 | tee -a "$OUT/run_all.log"

echo "== preflight OK, starting the main run in the background (budget ${HOURS}h from the first start)"
nohup python3 -m pilot.run_all --out "$OUT" --hours "$HOURS" --phase all "$@" >> "$OUT/run_all.log" 2>&1 &
echo $! > "$OUT/run_all.pid"
echo "PID $(cat "$OUT/run_all.pid")"
echo "status:  cat $OUT/STATUS.md"
echo "log:     tail -f $OUT/run_all.log"
echo "stop:    kill \$(cat $OUT/run_all.pid)"
echo "report:  $OUT/report.md (in the morning)"
