#!/usr/bin/env bash
# Evaluate every Nth checkpoint of a training run and report the best by mean relative L2.
# This is how the released checkpoint was chosen (20-checkpoint sweep: epochs 20, 40, ..., 400).
#
#   RUN=runs/train bash scripts/sweep_eval.sh          # -> runs/train/eval_sweep/summary.txt
#   VARIANT=v RUN=runs/train_v bash scripts/sweep_eval.sh
#   RUN=runs/train EVERY=10 bash scripts/sweep_eval.sh
#   RUN=runs/train START=360 bash scripts/sweep_eval.sh   # only ckpts >= 360, appending to summary.txt
#
# NOTE: the sweep evaluates on the official TEST set, so picking the best checkpoint this way is
# selection on test data (no validation split exists in this protocol) — see results/README.md.
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
# CFG must match the configuration the swept checkpoints were trained with. VARIANT=v selects it
# (or CFG=config_v.toml directly); a mismatch is reported before the model is built.
CFG="${CFG:-${V_CFG:-config.toml}}"
RUN="${RUN:-runs/train${VARIANT:+_$VARIANT}}"
EVERY="${EVERY:-20}"
START="${START:-$EVERY}"
EPOCHS="${EPOCHS:-400}"
STATS="${STATS:-$RUN/airfrans_norm_stats.npz}"
OUT="${SWEEP_OUT:-$RUN/eval_sweep}"; SUM="$OUT/summary.txt"
mkdir -p "$OUT"; [ "$START" -le "$EVERY" ] && : > "$SUM"   # START > EVERY appends
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for ((ep=START; ep<=EPOCHS; ep+=EVERY)); do
    ck=$(ls "$RUN"/*_ckpt_${ep}.pth 2>/dev/null | head -1) || true
    [ -n "$ck" ] || continue
    echo "ckpt_${ep}"
    "$PYTHON" -u eval_fields.py --cfg_path "$CFG" --ckpt "$ck" --stats "$STATS" \
        --out "$OUT/ckpt_${ep}" --n_qual 0 > "$OUT/ckpt_${ep}.log" 2>&1
    echo "---- ckpt_${ep} ----" >> "$SUM"
    grep -E "^(ux|uy|p|nut|MEAN|VOLUME|SURF) " "$OUT/ckpt_${ep}.log" >> "$SUM"
done

"$PYTHON" - "$SUM" <<'PY'
import re, sys
rows, ep = {}, None
for line in open(sys.argv[1]):
    m = re.match(r"---- ckpt_(\d+) ----", line)
    if m: ep = int(m.group(1)); rows[ep] = {}; continue
    t = line.split()
    if ep is not None and t and t[0] in ("MEAN", "VOLUME", "SURF"):
        rows[ep][t[0]] = float(t[1])
rows = {e: r for e, r in rows.items() if "MEAN" in r}
if not rows: sys.exit("no checkpoints evaluated")
print(f"{'ckpt':>6} {'relL2':>8} {'Volume':>9} {'Surf':>9}")
for e in sorted(rows):
    r = rows[e]; print(f"{e:>6} {r['MEAN']:8.4f} {r.get('VOLUME', float('nan')):9.5f} {r.get('SURF', float('nan')):9.5f}")
best = min(rows, key=lambda e: rows[e]["MEAN"])
print(f"BEST ckpt_{best}  relL2 {rows[best]['MEAN']:.4f}")
PY
