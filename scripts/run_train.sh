#!/usr/bin/env bash
# Train HodgeMoE from scratch (400 epochs, OneCycle, batch_size 1). The variant decides both the
# config and the trainer: vef -> config.toml + train.py (fp32); v -> config_v.toml + train_v.py
# (bf16 autocast). Precision is a property of the training loop, not of arch/.
#
#   bash scripts/run_train.sh                      # -> runs/train/  (train.log, ckpts, metrics.csv)
#   VARIANT=v bash scripts/run_train.sh            # -> runs/train_v/  (vertex-only config)
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
V_NAME=""; V_CFG=""; V_CKPT=""; V_STATS=""; V_TRAINER=""
if [ -n "${VARIANT:-}" ]; then
    # Assign first: `eval "$(cmd)"` reports the status of eval, not of cmd, so an unknown
    # VARIANT would print its error and then silently fall through to the defaults.
    _V_ASSIGN="$("$PYTHON" source/shared/variants.py --variant "$VARIANT" --shell)" || exit 1
    eval "$_V_ASSIGN"; unset _V_ASSIGN
fi
# VARIANT=v (or CFG=config_v.toml) trains the vertex-only configuration instead of vef.
CFG="${CFG:-${V_CFG:-config.toml}}"
# The two variants were trained at different precisions, and precision lives in the training loop,
# not in arch/. Each variant names its entry point in variants.toml: train.py is the fp32 loop,
# train_v.py the bf16 one. TRAINER= overrides it.
TRAINER="${TRAINER:-${V_TRAINER:-train.py}}"
OUT="${OUT:-runs/train${VARIANT:+_$VARIANT}}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "trainer $TRAINER  cfg $CFG  out $OUT"
ARGS=(--cfg_path "$CFG" --out "$OUT/")
if [ -n "${RESUME:-}" ]; then
    ARGS+=(--resume_ckpt "$RESUME" --stats_path "$OUT/airfrans_norm_stats.npz" --continue_schedule)
fi
PIN=()
[ -n "${CPUS:-}" ] && PIN=(taskset -c "$CPUS")

if [ -n "${FOREGROUND:-}" ]; then
    "${PIN[@]}" "$PYTHON" -u "$TRAINER" "${ARGS[@]}" 2>&1 | tee -a "$OUT/train.log"
else
    setsid nohup "${PIN[@]}" "$PYTHON" -u "$TRAINER" "${ARGS[@]}" >> "$OUT/train.log" 2>&1 < /dev/null &
    echo "launched (PID $!) -> $OUT/train.log"
fi
