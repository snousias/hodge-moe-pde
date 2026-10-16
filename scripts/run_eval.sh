#!/usr/bin/env bash
# Evaluate one checkpoint: field metrics (rel-L2, R², Volume, Surf) + qualitative figures, then
# mesh-native lift (C_L rel-L2, Spearman rho_L). Defaults to the released checkpoint.
#
#   bash scripts/run_eval.sh                                  # -> runs/eval/
#   CKPT=runs/train/<ckpt>.pth STATS=runs/train/airfrans_norm_stats.npz OUT=runs/eval_mine \
#       bash scripts/run_eval.sh
#   N_QUAL=8 bash scripts/run_eval.sh                          # number of qualitative figures
#
# Check the printed "HodgeMoE-vef test samples: N" — 200 means the full official test set.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
CFG="${CFG:-config.toml}"
CKPT="${CKPT:-$(ls checkpoints/hodgemoe_vef_best_ep*.pth | head -1)}"
STATS="${STATS:-checkpoints/norm_stats.npz}"
OUT="${OUT:-runs/eval}"
N_QUAL="${N_QUAL:-6}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[1/2] field metrics  ($CKPT)"
"$PYTHON" -u eval_fields.py --cfg_path "$CFG" --ckpt "$CKPT" --stats "$STATS" \
    --out "$OUT/fields" --n_qual "$N_QUAL" 2>&1 | tee "$OUT/fields.log"
echo "[2/2] mesh-native lift"
"$PYTHON" -u eval_lift.py --cfg_path "$CFG" --ckpt "$CKPT" --stats "$STATS" \
    --out "$OUT/lift" 2>&1 | tee "$OUT/lift.log"
echo "done -> $OUT/"
