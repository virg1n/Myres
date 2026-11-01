#!/usr/bin/env bash
# Phase 0 (Qwen3 without training) with one command.
#   bash run_phase0.sh                 # smoke test in the foreground (~10-15 min), then the full run in background
#   bash run_phase0.sh --gpus 0,1      # extra args go to phase0.run (both smoke and full)
#   MODEL=/path/to/local/qwen bash run_phase0.sh
# Re-running resumes: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-runs/phase0}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
mkdir -p "$(dirname "$OUT")"

python3 - <<'PY'
import importlib, shutil
for m in ["torch", "transformers", "datasets", "huggingface_hub", "numpy"]:
    importlib.import_module(m)
import torch
print("torch", torch.__version__, "| GPUs:", torch.cuda.device_count())
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as e:
    print("WARNING: vllm is not importable (%s): the HF backend will be used, several times slower" % e)
free = shutil.disk_usage(".").free / 1e9
print(f"free disk: {free:.1f} GB (need ~9 GB: model ~4 GB + one noisy copy ~4 GB + outputs < 0.5 GB)")
if free < 9:
    print("WARNING: little free space -- the noise lane will stop if fewer than 6 GB are free")
PY

echo "== smoke run (tiny sizes): $OUT/smoke"
python3 -m phase0.run --run "$OUT/smoke" --model "$MODEL" --smoke "$@" 2>&1 | tee "$OUT.smoke.log"

echo "== smoke OK, starting the full phase 0 in the background"
nohup python3 -m phase0.run --run "$OUT" --model "$MODEL" "$@" > "$OUT.log" 2>&1 &
echo $! > "$OUT.pid"
echo "PID $(cat "$OUT.pid")"
echo "log:     tail -f $OUT.log"
echo "lanes:   ls $OUT/logs/ ; tail -n 3 $OUT/logs/*.log"
echo "stop:    kill \$(cat $OUT.pid)"
echo "report:  $OUT/report_phase0.md (expected in ~1.5-2 h with vLLM)"
