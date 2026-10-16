# ========================================================================================
# HELPERS for training/loop.py — metrics and per-epoch persistence.
# ========================================================================================
#!/usr/bin/env python
"""Metrics and per-epoch metric persistence.

Part of the v57_gated_hodge_vef case, split out of the former 699-line
`airfrans_regression_v57_gated_hodge_vef.py` so that ARCHITECTURE / DATA / TRAINING / EVALUATION are
separable. See README.md for the map.
"""

import os
import json
from typing import Dict

import torch


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute MSE, MAE, RMSE, R², and per-channel L2 relative error.
    Tensors are (B, N, C).
    """
    pred = pred.float().detach().cpu()
    target = target.float().detach().cpu()

    diff2 = (pred - target) ** 2
    mse = torch.mean(diff2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = mse ** 0.5

    ss_res = torch.sum(diff2).item()
    ss_tot = torch.sum((target - target.mean()) ** 2).item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    out = {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}

    C = target.shape[-1]
    for c in range(C):
        t = target[..., c]
        p = pred[..., c]
        sr = torch.sum((p - t) ** 2).item()
        st = torch.sum((t - t.mean()) ** 2).item()
        out[f"r2_c{c}"] = 1.0 - sr / (st + 1e-12)
        # L2 relative error: ‖pred − target‖₂ / ‖target‖₂
        l2_rel = (torch.norm(p - t) / (torch.norm(t) + 1e-8)).item()
        out[f"l2rel_c{c}"] = l2_rel

    return out


def _wall_col_index(feature_cols) -> int:
    """Index of the is_wall column in the normalised vertex feature tensor."""
    n_feat = len(feature_cols) if feature_cols is not None else 5
    return 3 + n_feat   # v57_gated_hodge_vef: pos(2) + area(1) + x_feat(n_feat) + is_wall  (fixed: pre-v57_gated_hodge_vef omitted the area col → wrong index)


def append_metrics(out_dir, row):
    """Persist one epoch's metrics to the RUN FOLDER, not just W&B.

    Writes `<out>/metrics.jsonl` (append-only, one JSON object per epoch — survives resumes and
    tolerates keys appearing mid-run) and rewrites `<out>/metrics.csv` as the flat table, whose
    columns are the union of every key seen so far. v57_gated_hodge_vef's `band` / `bal` terms land here, so the
    expert-specialisation and gate-balance history is recoverable without W&B.
    """
    os.makedirs(out_dir, exist_ok=True)
    jl = os.path.join(out_dir, "metrics.jsonl")
    with open(jl, "a") as f:
        f.write(json.dumps(row) + "\n")
    try:
        rows = [json.loads(l) for l in open(jl) if l.strip()]
        cols, seen = [], set()
        for r in rows:                                  # stable union of keys, first-seen order
            for k in r:
                if k not in seen:
                    seen.add(k); cols.append(k)
        import csv as _csv
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    except Exception as e:                              # never let bookkeeping kill training
        print(f"[metrics] csv rewrite skipped ({type(e).__name__}: {e})", flush=True)


