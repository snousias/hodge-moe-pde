#!/usr/bin/env bash
# =====================================================================================
# END-TO-END VERIFICATION — train, sweep, evaluate, and check against the paper metrics
# =====================================================================================
# Runs one or both released variants through the pipeline and compares the result with the
# reference numbers in `variants.toml`. Exits 0 only if every checked metric passes, so this
# is usable as a CI step.
#
#   bash scripts/run_verify.sh                     # evaluate + check the RELEASED checkpoints
#   VARIANTS=v bash scripts/run_verify.sh          # one variant only
#   MODE=full bash scripts/run_verify.sh           # train from scratch, then sweep, eval, check
#   STAGES=eval,check bash scripts/run_verify.sh   # the same as the default, written out
#
# STAGES, in the order they run. The default is `eval,check`.
#   train   train the variant from scratch with its own config. Writes checkpoints,
#           metrics.csv and train.log. ~44 h for vef and ~15 h for v on one RTX 4090,
#           so this is not the default.
#   sweep   evaluate every EVERY-th checkpoint of that training run and select the best by
#           mean rel-L2 — the protocol that chose the released checkpoints.
#   eval    field metrics (rel-L2, R2, Volume, Surf) + mesh-native lift (C_L, rho_L) for the
#           checkpoint in hand: the one `sweep` selected, or the released one.
#   check   compare those metrics with `[<variant>.expect]` in variants.toml and print a
#           PASS/FAIL table.
#
# Other knobs:
#   VARIANTS=v,vef     which variants to run (default both, in that order)
#   OUT_ROOT=runs/verify   where everything is written
#   CKPT_v=... CKPT_vef=...  evaluate a specific checkpoint instead of the released one
#   EVERY=20 EPOCHS=400      checkpoint sweep spacing and range
#   N_QUAL=6           qualitative figures per variant
#   PYTHON=/path/to/python
#
# WHAT "VERIFY THE PAPER METRICS" MEANS HERE. For `v` the reference is the paper's headline
# vertex-only row. Its published numbers were measured under bf16 autocast and this code
# evaluates in fp32, so its tolerance is deliberately looser; the table prints the observed
# difference for every metric. For `vef` the reference is this repository's own measurement of
# its checkpoint, NOT the paper's supplementary VEF row — that row is a different run (0.0359 at
# epoch 560) and this checkpoint is not expected to reproduce it. `variants.toml` states the
# source of every reference number next to it.
#
# Each variant's result folder ends up as:
#   $OUT_ROOT/<variant>/final/   the checkpoint that was evaluated, its norm stats, the metric
#                               arrays, the per-epoch curves if it was trained here, the check
#                               table, and a MANIFEST.txt recording what produced them.
# =====================================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "${1:-}" in
    -h|--help|help) sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

PYTHON="${PYTHON:-python}"
VARIANTS="${VARIANTS:-v,vef}"
OUT_ROOT="${OUT_ROOT:-runs/verify}"
EVERY="${EVERY:-20}"
EPOCHS="${EPOCHS:-400}"
N_QUAL="${N_QUAL:-6}"
if [ "${MODE:-}" = "full" ]; then
    STAGES="${STAGES:-train,sweep,eval,check}"
else
    STAGES="${STAGES:-eval,check}"
fi

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/verify.log"
SUMMARY="$OUT_ROOT/summary.txt"
: > "$LOG"; : > "$SUMMARY"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

has_stage() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
say() { echo "$*" | tee -a "$LOG"; }
rule() { say "---------------------------------------------------------------------------"; }

# Fail before spending hours if a stage name is misspelt.
for s in ${STAGES//,/ }; do
    case "$s" in
        train|sweep|eval|check) ;;
        *) echo "unknown stage '$s'. Valid stages: train, sweep, eval, check" >&2; exit 2 ;;
    esac
done
for v in ${VARIANTS//,/ }; do
    "$PYTHON" source/shared/variants.py --variant "$v" --print modes > /dev/null || exit 1
done

say "HodgeMoE verification"
say "  variants : $VARIANTS"
say "  stages   : $STAGES"
say "  output   : $OUT_ROOT"
say "  started  : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
say ""
"$PYTHON" source/shared/variants.py --list | tee -a "$LOG"

OVERALL=0
RESULT_LINES=()

for VARIANT in ${VARIANTS//,/ }; do
    rule
    say "VARIANT $VARIANT"
    rule
    VDIR="$OUT_ROOT/$VARIANT"
    TRAIN_DIR="$VDIR/train"; SWEEP_DIR="$VDIR/sweep"; EVAL_DIR="$VDIR/eval"; FINAL="$VDIR/final"
    mkdir -p "$VDIR"

    CFG="$("$PYTHON" source/shared/variants.py --variant "$VARIANT" --print config)"
    STATS="$("$PYTHON" source/shared/variants.py --variant "$VARIANT" --print stats)"
    # CKPT_v / CKPT_vef override the released checkpoint for this variant.
    eval "CKPT=\${CKPT_$VARIANT:-}"
    [ -n "$CKPT" ] || CKPT="$("$PYTHON" source/shared/variants.py --variant "$VARIANT" --print checkpoint)"
    PROVENANCE="released checkpoint"

    # ── train ───────────────────────────────────────────────────────────────────────────
    if has_stage train; then
        say "[train] $CFG -> $TRAIN_DIR  (this is the long one)"
        mkdir -p "$TRAIN_DIR"
        FOREGROUND=1 VARIANT="$VARIANT" OUT="$TRAIN_DIR" PYTHON="$PYTHON" \
            bash scripts/run_train.sh 2>&1 | tee -a "$LOG"
        STATS="$TRAIN_DIR/airfrans_norm_stats.npz"
        CKPT="$(ls -1 "$TRAIN_DIR"/*_ckpt_"$EPOCHS".pth 2>/dev/null | head -1 || true)"
        [ -n "$CKPT" ] || { say "[train] FAILED: no checkpoint at epoch $EPOCHS in $TRAIN_DIR"; exit 1; }
        PROVENANCE="trained here, final epoch $EPOCHS"
        say "[train] final checkpoint: $CKPT"
    fi

    # ── sweep ───────────────────────────────────────────────────────────────────────────
    if has_stage sweep; then
        if [ ! -d "$TRAIN_DIR" ]; then
            say "[sweep] skipped: no training run at $TRAIN_DIR (add the train stage, or point"
            say "        RUN at an existing run and use scripts/sweep_eval.sh directly)"
        else
            say "[sweep] every ${EVERY} epochs up to ${EPOCHS} -> $SWEEP_DIR"
            VARIANT="$VARIANT" RUN="$TRAIN_DIR" SWEEP_OUT="$SWEEP_DIR" EVERY="$EVERY" \
                EPOCHS="$EPOCHS" STATS="$STATS" PYTHON="$PYTHON" \
                bash scripts/sweep_eval.sh 2>&1 | tee -a "$LOG" | tee "$VDIR/sweep.out"
            BEST_EP="$(grep -oE '^BEST ckpt_[0-9]+' "$VDIR/sweep.out" | tail -1 | grep -oE '[0-9]+$' || true)"
            if [ -n "$BEST_EP" ]; then
                BEST_CKPT="$(ls -1 "$TRAIN_DIR"/*_ckpt_"$BEST_EP".pth 2>/dev/null | head -1 || true)"
                if [ -n "$BEST_CKPT" ]; then
                    CKPT="$BEST_CKPT"
                    PROVENANCE="trained here, epoch $BEST_EP selected by the sweep"
                    say "[sweep] selected epoch $BEST_EP -> $CKPT"
                fi
            else
                say "[sweep] no BEST line found; keeping $CKPT"
            fi
        fi
    fi

    # ── eval ────────────────────────────────────────────────────────────────────────────
    if has_stage eval; then
        say "[eval]  $CKPT"
        say "        cfg $CFG"
        say "        stats $STATS"
        VARIANT="$VARIANT" CFG="$CFG" CKPT="$CKPT" STATS="$STATS" OUT="$EVAL_DIR" \
            N_QUAL="$N_QUAL" PYTHON="$PYTHON" bash scripts/run_eval.sh 2>&1 | tee -a "$LOG"
    fi

    # ── check ───────────────────────────────────────────────────────────────────────────
    STATUS="not checked"
    if has_stage check; then
        set +e
        "$PYTHON" source/shared/check_metrics.py --variant "$VARIANT" --eval-dir "$EVAL_DIR" \
            2>&1 | tee -a "$LOG" | tee "$VDIR/check.txt"
        RC=${PIPESTATUS[0]}
        set -e
        case "$RC" in
            0) STATUS="PASS" ;;
            1) STATUS="FAIL"; OVERALL=1 ;;
            *) STATUS="ERROR"; OVERALL=1 ;;
        esac
        say "[check] $VARIANT: $STATUS"
    fi

    # ── collect the final artifacts ─────────────────────────────────────────────────────
    mkdir -p "$FINAL"
    [ -f "$CKPT" ] && cp -f "$CKPT" "$FINAL/"
    [ -f "$STATS" ] && cp -f "$STATS" "$FINAL/norm_stats.npz"
    cp -f "$CFG" "$FINAL/" 2>/dev/null || true
    for f in "$EVAL_DIR/fields/test_metrics.npz" "$EVAL_DIR/lift/cl_native.npz" \
             "$EVAL_DIR/fields.log" "$EVAL_DIR/lift.log" "$TRAIN_DIR/metrics.csv" \
             "$TRAIN_DIR/train.log" "$SWEEP_DIR/summary.txt" "$VDIR/check.txt"; do
        [ -f "$f" ] && cp -f "$f" "$FINAL/" || true
    done
    [ -f "$SWEEP_DIR/summary.txt" ] && cp -f "$SWEEP_DIR/summary.txt" "$FINAL/sweep_summary.txt"

    {
        echo "variant      $VARIANT"
        echo "config       $CFG"
        echo "checkpoint   $CKPT"
        echo "provenance   $PROVENANCE"
        echo "norm stats   $STATS"
        echo "stages       $STAGES"
        echo "check        $STATUS"
        echo "produced     $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        if [ -f "$CKPT" ]; then
            echo "sha256       $("$PYTHON" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$CKPT")"
        fi
    } > "$FINAL/MANIFEST.txt"
    say "[final] $FINAL/"
    sed 's/^/        /' "$FINAL/MANIFEST.txt" | tee -a "$LOG"

    RESULT_LINES+=("$(printf '%-6s %-6s %s' "$VARIANT" "$STATUS" "$FINAL")")
done

rule
{
    echo "HodgeMoE verification summary"
    echo "  stages  : $STAGES"
    echo "  finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo ""
    printf '%-6s %-6s %s\n' "VARIANT" "CHECK" "ARTIFACTS"
    for l in ${RESULT_LINES[@]+"${RESULT_LINES[@]}"}; do echo "$l"; done
    echo ""
    if [ "$OVERALL" -eq 0 ]; then
        echo "OVERALL: PASS"
    else
        echo "OVERALL: FAIL  (see $LOG and each variant's check.txt)"
    fi
} | tee -a "$LOG" | tee "$SUMMARY"

exit "$OVERALL"
