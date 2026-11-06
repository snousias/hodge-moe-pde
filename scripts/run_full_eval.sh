#!/usr/bin/env bash
# Full test-set evaluation: every checkpoint, every one of the 200 official test cases.
# Per checkpoint: per_case.csv, a qualitative figure per case, SUMMARY.md / summary.json with the
# overall metrics, and a check against variants.toml for the released checkpoints.
# Checkpoints run ONE AFTER THE OTHER (one GPU job at a time; ~0.6 GiB fp32 / ~2.7 GiB bf16).
#
#   bash scripts/run_full_eval.sh                           # both released checkpoints -> runs/full_eval/
#   VARIANTS=v bash scripts/run_full_eval.sh                # one released checkpoint
#   VARIANTS=vef RUN=runs/train bash scripts/run_full_eval.sh      # every *.pth of a training run
#   CKPTS="a.pth b.pth" VARIANTS=vef bash scripts/run_full_eval.sh # chosen checkpoints, vef config
#
# Knobs: OUT (default runs/full_eval), QUAL (-1 = every case, N = first N, 0 = none),
#        CACHE_DIR (test geometry cache; default from the config, built on first use, ~1 min/mesh),
#        PRECISION (auto|fp32|bf16; auto = bf16 for v, fp32 for vef), SAVE_PRED=1 keeps predictions,
#        WORKERS (figure processes, default 8), PYTHON.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
OUT="${OUT:-runs/full_eval}"
ARGS=(--out "$OUT" --qual "${QUAL:--1}" --precision "${PRECISION:-auto}" --workers "${WORKERS:-8}")
for v in ${VARIANTS:-}; do ARGS+=(--variant "$v"); done
[ -n "${RUN:-}" ]       && ARGS+=(--ckpt "$RUN")
[ -n "${CKPTS:-}" ]     && ARGS+=(--ckpt $CKPTS)
[ -n "${CFG:-}" ]       && ARGS+=(--cfg_path "$CFG")
[ -n "${STATS:-}" ]     && ARGS+=(--stats "$STATS")
[ -n "${CACHE_DIR:-}" ] && ARGS+=(--cache_dir "$CACHE_DIR")
[ -n "${SAVE_PRED:-}" ] && ARGS+=(--save_pred)
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PYTHON" -u eval_full.py "${ARGS[@]}" 2>&1 | tee -a "$OUT/full_eval.log"
