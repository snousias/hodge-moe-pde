#!/usr/bin/env bash
# Evaluate one checkpoint: field metrics (rel-L2, R², Volume, Surf) + qualitative figures, then
# mesh-native lift (C_L rel-L2, Spearman rho_L). Defaults to the released checkpoint.
#
#   bash scripts/run_eval.sh                                  # -> runs/eval/   (vef)
#   VARIANT=v   bash scripts/run_eval.sh                      # -> runs/eval_v/
#   VARIANT=vef bash scripts/run_eval.sh                      # -> runs/eval_vef/
#   CKPT=runs/train/<ckpt>.pth STATS=runs/train/airfrans_norm_stats.npz OUT=runs/eval_mine \
#       bash scripts/run_eval.sh
#   N_QUAL=8 bash scripts/run_eval.sh                          # number of qualitative figures
#
# VARIANT names a row of variants.toml and supplies its config AND its checkpoint together, so
# the two cannot drift apart (`python source/shared/variants.py --list` prints the table). An explicit
# CFG=, CKPT= or STATS= still wins over the variant, and with VARIANT unset this script behaves
# exactly as before: config.toml with the vef checkpoint.
#
# Check the printed "HodgeMoE-<modes> test samples: N" — 200 means the full official test set.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
V_NAME=""; V_CFG=""; V_CKPT=""; V_STATS=""
if [ -n "${VARIANT:-}" ]; then
    # Assign first: `eval "$(cmd)"` reports the status of eval, not of cmd, so an unknown
    # VARIANT would print its error and then silently fall through to the defaults.
    _V_ASSIGN="$("$PYTHON" source/shared/variants.py --variant "$VARIANT" --shell)" || exit 1
    eval "$_V_ASSIGN"; unset _V_ASSIGN
fi
CFG="${CFG:-${V_CFG:-config.toml}}"
CKPT="${CKPT:-${V_CKPT:-$(ls checkpoints/hodgemoe_vef_best_ep*.pth | head -1)}}"
STATS="${STATS:-${V_STATS:-checkpoints/norm_stats.npz}}"
OUT="${OUT:-runs/eval${VARIANT:+_$VARIANT}}"
N_QUAL="${N_QUAL:-6}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "variant ${V_NAME:-<none>}  cfg $CFG  ckpt $CKPT"
echo "[1/2] field metrics  ($CKPT)"
"$PYTHON" -u eval_fields.py --cfg_path "$CFG" --ckpt "$CKPT" --stats "$STATS" \
    --out "$OUT/fields" --n_qual "$N_QUAL" 2>&1 | tee "$OUT/fields.log"
echo "[2/2] mesh-native lift"
"$PYTHON" -u eval_lift.py --cfg_path "$CFG" --ckpt "$CKPT" --stats "$STATS" \
    --out "$OUT/lift" 2>&1 | tee "$OUT/lift.log"
echo "done -> $OUT/"
