#!/usr/bin/env bash
# Train HodgeMoE-vef from scratch with config.toml (400 epochs, OneCycle, fp32, batch_size 1).
#
#   bash scripts/run_train.sh                      # -> runs/train/  (train.log, ckpts, metrics.csv)
#   OUT=runs/my_run bash scripts/run_train.sh      # other output folder
#   RESUME=runs/train/<ckpt>.pth bash scripts/run_train.sh
#        resume a crashed run: restores optimizer AND scheduler state and runs only the
#        remaining epochs (--continue_schedule), reusing the run's own norm stats.
#   CPUS=0-7,12-31 bash scripts/run_train.sh      # optional: pin to a CPU set (taskset)
#   FOREGROUND=1 bash scripts/run_train.sh         # run in this shell instead of detached
#
# Plain one-shot launch: no auto-restart, no watchdog. Uses `python` from the active env
# (conda activate hodgemoe-vef), or set PYTHON=/path/to/python.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
CFG="${CFG:-config.toml}"
OUT="${OUT:-runs/train}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARGS=(--cfg_path "$CFG" --out "$OUT/")
if [ -n "${RESUME:-}" ]; then
    ARGS+=(--resume_ckpt "$RESUME" --stats_path "$OUT/airfrans_norm_stats.npz" --continue_schedule)
fi
PIN=()
[ -n "${CPUS:-}" ] && PIN=(taskset -c "$CPUS")

if [ -n "${FOREGROUND:-}" ]; then
    "${PIN[@]}" "$PYTHON" -u train.py "${ARGS[@]}" 2>&1 | tee -a "$OUT/train.log"
else
    setsid nohup "${PIN[@]}" "$PYTHON" -u train.py "${ARGS[@]}" >> "$OUT/train.log" 2>&1 < /dev/null &
    echo "launched (PID $!) -> $OUT/train.log"
fi
