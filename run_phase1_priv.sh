#!/usr/bin/env bash
# Private-share sweep (shared LoRA + a private LoRA per mode, FiLM in every point with modes) with one command.
#   bash run_phase1_priv.sh math          # hv: sft, wta1 (0%), 6%, 41%, 100% private, LoRA-MCL; 3 trainings (~3-4 h)
#   bash run_phase1_priv.sh mt            # en->de of LoRA-MCL: sft, LoRA-MCL, 6%, 38%, 100% private (~5-6 h);
#                                         # first downloads ALMA-7B-Pretrain and converts it to bf16 (13.5 GB on disk)
#   bash run_phase1_priv.sh math --points 0,4,8,16,24,32 --seeds 0,1     # the full curve later, same directory
# A smoke run in the foreground first (~10-20 min), then the full run in the background. Re-running resumes.
# Disk: run the two benchmarks one after the other (math peaks at ~18 GB, mt holds 13.5 GB for its base).
set -euo pipefail
cd "$(dirname "$0")"
BENCH="${1:-}"
case "$BENCH" in math|mt) shift;; *) echo "usage: bash run_phase1_priv.sh math|mt [--points ...] [--seeds ...]"; exit 1;; esac
P0="${P0:-runs/phase0}"
P1="${P1:-runs/phase1}"
SEEDS_RUN="${SEEDS_RUN:-runs/phase1_seeds}"
SELF_RUN="${SELF_RUN:-runs/phase1_self}"
if [ "$BENCH" = math ]; then OUT="${OUT:-runs/phase1_priv}"; else OUT="${OUT:-runs/phase1_priv_mt}"; fi
HOURS="${HOURS:-6}"
MAX_HOURS="${MAX_HOURS:-10}"
export VLLM_USE_FLASHINFER_SAMPLER=0
for x in "$@"; do          # the paths belong to this script: the lock, the smoke directory and the PID file use them
  case "$x" in --run|--run=*|--bench*|--seeds_run*|--self_run*|--phase0*|--phase1*|--mt_base*|--smoke|--plan_only)
    echo "$x is set by this script: use OUT=, SEEDS_RUN=, SELF_RUN=, P0=, P1= instead"; exit 1;; esac
done
for other in runs/phase1_seeds runs/phase1_rank runs/phase1_z runs/phase1_self runs/phase1_grpo runs/phase1_priv \
             runs/phase1_priv_mt; do
  if [ "$other" != "$OUT" ] && [ -f "$other.pid" ] && kill -0 "$(cat "$other.pid")" 2> /dev/null; then
    echo "$other is still running (PID $(cat "$other.pid")): the GPUs and the disk are busy"; exit 1
  fi
done

mkdir -p "$(dirname "$OUT")"
command -v flock > /dev/null || { echo "flock not found (package util-linux)"; exit 1; }
exec 9>"$OUT.lock"
flock -n 9 || { echo "$OUT is already running (PID file: $OUT.pid); stop it first"; exit 1; }

python3 - "$BENCH" <<'PY'
import shutil, sys, torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| GPUs", torch.cuda.device_count())
print(f"free disk as the OS reports it: {shutil.disk_usage('.').free / 1e9:.0f} GB")
if sys.argv[1] == "mt":
    import sacrebleu, sacremoses, huggingface_hub   # pip install sacrebleu sacremoses
    import os
    if os.path.isdir("/dev/shm"):
        print(f"/dev/shm free: {shutil.disk_usage('/dev/shm').free / 1e9:.0f} GB (one 9.9 GB shard is converted there)")
else:
    import vllm
    print("vllm", vllm.__version__)
PY

ARGS=(--bench "$BENCH" --run "$OUT" --seeds_run "$SEEDS_RUN" --self_run "$SELF_RUN" --phase0 "$P0" --phase1 "$P1")
if [ "$BENCH" = mt ]; then
  echo "== base model: ALMA-7B-Pretrain -> $OUT/base (bf16 safetensors, once)"
  export HF_XET_CHUNK_CACHE_SIZE_BYTES=0      # no Xet chunk cache in ~/.cache: the shards would also land under the quota
  mkdir -p "$(readlink -m "$OUT/base")"      # $OUT/base may link to /dev/shm, which a pod restart empties
  python3 -m phase1.pmt base --out "$OUT/base" ${MT_TMP:+--tmp "$MT_TMP"}
  ARGS+=(--mt_base "$OUT/base")
fi
python3 -m phase1.prun "${ARGS[@]}" --hours "$HOURS" --max_hours "$MAX_HOURS" --plan_only "$@"

if [ ! -f "$OUT/STATUS.md" ] || [ "${SMOKE:-0}" != 0 ]; then     # SMOKE=1: a smoke run also for a directory in use
  echo "== smoke run: $OUT.smoke"
  rm -rf "$OUT.smoke"
  SM=("${ARGS[@]}")
  SM[3]="$OUT.smoke"                      # --run
  python3 -m phase1.prun "${SM[@]}" --smoke "$@" 2>&1 | tee "$OUT.smoke.log"
  rm -rf "$OUT.smoke"          # the disk is small: the real run needs this space
  echo "== smoke OK"
fi

echo "== starting the full run in the background"
nohup python3 -m phase1.prun "${ARGS[@]}" --hours "$HOURS" --max_hours "$MAX_HOURS" "$@" >> "$OUT.log" 2>&1 &
echo $! > "$OUT.pid"
echo "PID $(cat "$OUT.pid")"
echo "status:  cat $OUT/STATUS.md"
echo "log:     tail -f $OUT.log"
echo "stop:    kill \$(cat $OUT.pid)"
echo "report:  $OUT/report_priv.md, curves $OUT/priv_curves.png"
