# ========================================================================================
# HELPERS for training/loop.py — loss terms and target-column splitters.
# ========================================================================================
#!/usr/bin/env python
"""Loss terms for v57_gated_hodge_vef.

Part of the v57_gated_hodge_vef case, split out of the former 699-line
`airfrans_regression_v57_gated_hodge_vef.py` so that ARCHITECTURE / DATA / TRAINING / EVALUATION are
separable. See README.md for the map.
"""

import torch


def _elementwise_robust(resid, kind, delta):
    """Per-element loss on a residual tensor. 'mse' reproduces the v3–v10 loss."""
    if kind == "mse":
        return resid ** 2
    a = resid.abs()
    if kind in ("huber", "smooth_l1"):
        # Huber: quadratic for |r|<=delta, linear beyond → outlier residuals
        # contribute linearly instead of quadratically.
        return torch.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))
    if kind == "logcosh":
        # log(cosh r), numerically stable: |r| + softplus(-2|r|) - log2.
        return a + torch.nn.functional.softplus(-2.0 * a) - math.log(2.0)
    raise ValueError(f"unknown loss_type: {kind}")


def split_bands(target, cfg):
    """v57_gated_hodge_vef target layout: [C fields][cl_w][cl_true][cl_b][bands C*K].
    Returns the band targets as (B,N,C,K), or None when absent."""
    n_out = int(cfg["train"]["model"]["out_channels"])
    K = int(cfg["train"]["model"].get("moe_kw", {}).get("n_experts", 4))
    need = n_out + 3 + n_out * K
    if target.shape[-1] < need:
        return None
    return target[..., n_out + 3:need].reshape(*target.shape[:-1], n_out, K)


def band_loss(band_pred, band_true, gate, cfg):
    """Auxiliary loss pinning expert k to frequency band k, plus a load-balance term on the gate.

    Without this the experts have no reason to specialise — a softmax mixture of identical experts
    is a perfectly good optimum of the field loss alone. The balance term keeps the gate from
    collapsing onto one expert (which would silently make v57_gated_hodge_vef a narrower v51).
    """
    tcfg = cfg["train"]
    l_band = ((band_pred - band_true) ** 2).mean()
    lam_bal = float(tcfg.get("balance_lambda", 0.0))
    if lam_bal <= 0.0:
        return l_band, l_band.detach() * 0.0
    share = gate.mean(dim=(0, 1))                       # (K,) mean gate mass per expert
    K = share.numel()
    kk = share.new_tensor(float(K))                     # vef_v2: no Python scalar on the (K,) share (torch 2.14)
    l_bal = ((share - 1.0 / kk) ** 2).sum() * kk        # 0 iff perfectly balanced
    return l_band, l_bal


def split_cl(target, cfg):
    """v57_gated_hodge_vef ships [C fields][cl_w per vertex][cl_true][cl_b] in the target tensor.
    Returns (fields (B,N,C), cl_w (B,N), cl_true (B,), cl_b (B,))."""
    n_out = int(cfg["train"]["model"]["out_channels"])
    if target.shape[-1] <= n_out:
        return target, None, None, None
    return (target[..., :n_out], target[..., n_out],
            target[:, 0, n_out + 1], target[:, 0, n_out + 2])


def cl_loss(pred, cl_w, cl_true, cl_b, cfg):
    """Squared error of the LIFT INTEGRAL, in C_L units divided by a fixed scale.

    C_L is linear in the predicted wall pressure (channel 2): C_L_pred = cl_w . p_z + cl_b, with the
    per-mesh weights already folded through the target normalisation by the dataset. Scaling by a
    FIXED constant (the RMS of C_L over the data, ~1.87) rather than per-sample |C_L| keeps the term
    finite for the ~8% of cases whose lift is near zero.
    """
    tcfg = cfg["train"]
    scale = float(tcfg.get("cl_loss_scale", 1.87))
    p_z = pred[..., 2].float()                                   # (B,N) pressure channel
    cl_pred = (cl_w.float() * p_z).sum(dim=1) + cl_b.float()     # (B,)
    return (((cl_pred - cl_true.float()) / scale) ** 2).mean(), cl_pred


def compute_field_loss(pred, target, cfg, wall_mask=None):
    """
    v11 field loss. Backward-compatible: with loss_type='mse' and
    per_case_norm=false this equals the v3–v10 MSE (+ surface) loss.

      • robust elementwise penalty (huber / smooth_l1 / logcosh / mse)
      • optional PER-CASE, per-field normalisation: divide each sample's
        per-field residual by that sample's field std (floored), so a thick /
        near-stall airfoil's huge pressure swing stops dominating the gradient.
        This makes training optimise a relative (Transolver-style) error.
    """
    tcfg  = cfg["train"]
    kind  = tcfg.get("loss_type", "mse")
    delta = float(tcfg.get("huber_delta", 1.0))

    pred   = pred.float()
    target = target.float()

    if tcfg.get("per_case_norm", False):
        floor = float(tcfg.get("per_case_norm_floor", 0.1))
        # std over nodes, per sample, per field → (B, 1, C)
        scale = target.std(dim=1, keepdim=True).clamp_min(floor)
        resid = (pred - target) / scale
    else:
        resid = pred - target

    elt = _elementwise_robust(resid, kind, delta)        # (B, N, C)

    # ── v57_gated_hodge_vef: DISTINCT volume/surface MSE (disjoint node sets) ────────────────────────────────
    # Mirrors the eval metric exactly (airfrans_native.vol_surf_mse: vol = se[~wall], surf = se[wall]).
    # Each node contributes to EXACTLY ONE term. Contrast v45, whose "volume" was elt.mean() over ALL
    # nodes (wall included) with the surface term merely ADDED on top → wall nodes were double-counted
    # and the two terms were not distinct. Total = volume_loss_lambda·VOL + surface_loss_lambda·SURF.
    surf_lambda = float(tcfg.get("surface_loss_lambda", 0.0))
    if tcfg.get("distinct_vol_surf", False) and wall_mask is not None:
        vol_lambda = float(tcfg.get("volume_loss_lambda", 1.0))
        vmask = ~wall_mask
        vol_loss  = elt[vmask].mean()     if vmask.any()     else elt.sum() * 0.0   # interior nodes only
        surf_loss = elt[wall_mask].mean() if wall_mask.any() else elt.sum() * 0.0   # wall nodes only
        return vol_lambda * vol_loss + surf_lambda * surf_loss

    # ── legacy v45 path: volume over ALL nodes + additive surface term ─────────────────────────
    vol_loss = elt.mean()
    if surf_lambda > 0.0 and wall_mask is not None and wall_mask.any():
        surf_loss = elt[wall_mask].mean()                 # (n_wall, C) → scalar
        return vol_loss + surf_lambda * surf_loss
    return vol_loss


# ── training / validation loops ───────────────────────────────────────────────

