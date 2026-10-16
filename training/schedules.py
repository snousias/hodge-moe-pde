# ========================================================================================
# HELPER for training/loop.py — the warmup/flat/cosine schedule.
# ========================================================================================
#!/usr/bin/env python
"""Learning-rate schedules.

Part of the v57_gated_hodge_vef case, split out of the former 699-line
`airfrans_regression_v57_gated_hodge_vef.py` so that ARCHITECTURE / DATA / TRAINING / EVALUATION are
separable. See README.md for the map.
"""

import math

import torch


def make_warmup_flat_cosine(optimizer, warmup_epochs, flat_epochs, total_epochs,
                            eta_min_ratio: float = 2e-3):
    """
    Three-phase LR schedule:
      1. Linear warmup  : epochs [0, warmup_epochs)
      2. Flat            : epochs [warmup_epochs, warmup_epochs+flat_epochs)
      3. Cosine decay    : epochs [warmup_epochs+flat_epochs, total_epochs)
    eta_min_ratio: floor as a fraction of the base LR.
    """
    cosine_start = warmup_epochs + flat_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-6, (epoch + 1) / warmup_epochs)
        if epoch < cosine_start:
            return 1.0
        decay_ep = epoch - cosine_start
        cosine_ep = total_epochs - cosine_start
        if cosine_ep <= 0:
            return eta_min_ratio
        cos_val = 0.5 * (1.0 + math.cos(math.pi * decay_ep / cosine_ep))
        return max(eta_min_ratio, cos_val)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── loss (v11: robust + per-case normalised) ──────────────────────────────────

