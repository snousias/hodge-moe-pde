# ========================================================================================
# TRAINING LOOP, bf16 VARIANT   —   the recipe that produced the vertex-only checkpoint
# ========================================================================================
"""bf16-autocast trainer, for `config_v.toml` / `VARIANT=v`.

`training/loop.py` is the fp32 trainer that produced `hodgemoe_vef_best_ep400.pth`, and it is
left exactly as it was. This module is the second trainer: the same loop with the forward passes
under `torch.autocast(bfloat16)`, which is how `hodgemoe_v_best_ep380.pth` was trained in the
origin case `v57_gated_hodge`. Training the vertex-only architecture with the fp32 trainer gives
a different recipe from the one behind the published 0.0300.

WHAT DIFFERS, and nothing else:

  train_epoch     the device transfer, the C_L split and the model forward run under
                  `autocast(bfloat16)`. The loss, the metrics and the optimizer step stay fp32.
  train_epoch     the analyzer's band targets are computed under `autocast(enabled=False)`:
                  sparse bmm has no bfloat16 CUDA kernel, and a supervision target should not
                  carry autocast rounding.
  validate_epoch  the same autocast scope around the validation forward.

Everything else — argument parsing, dataset construction, the loss, the schedule, checkpointing,
resume, W&B — is IMPORTED from `training.loop`, so the two trainers cannot drift apart.

`main()` rebinds the two epoch functions inside `training.loop` for the duration of the run and
restores them afterwards, because `training.loop.main` calls them by name. That rebinding is the
price of leaving `training/loop.py` untouched.

Precision is not an architectural property: `autocast` casts the inputs of individual ops at run
time and never changes `param.dtype`. Both released checkpoints are fp32 on disk and both load
into the same `arch/` package. See checkpoints/PROVENANCE.md.

    python train_v.py --variant v --out runs/train_v/
    VARIANT=v bash scripts/run_train.sh
"""
# __deployment_path_bootstrap__
import os as _o, sys as _s
_ROOT = _o.path.dirname(_o.path.abspath(__file__))
while _ROOT != _o.path.dirname(_ROOT) and not _o.path.isfile(
        _o.path.join(_ROOT, "variants.toml")):
    _ROOT = _o.path.dirname(_ROOT)                      # deployment root (holds variants.toml)
for _d in (_o.path.join(_ROOT, "source", "shared"), _o.path.join(_ROOT, "source"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)

import contextlib

import numpy as np
import torch

from training import loop as _loop
# The helpers the two lifted functions call. They are the released fp32 module's own objects:
# importing them here is what keeps the shared code single-sourced.
from training.loop import (split_cl, compute_field_loss, cl_loss, band_loss,
                           regression_metrics, _wall_col_index)

__all__ = ["train_epoch", "validate_epoch", "main"]


def train_epoch(data_loader, model, loss_fn, optimizer, scheduler, epoch, device, cfg,
                step_per_batch=False):
    model.train()
    scaler = torch.amp.GradScaler()
    epochs = cfg["train"]["epochs"]
    surf_lambda = cfg["train"].get("surface_loss_lambda", 0.0)
    # v57_gated_hodge: the distinct vol/surf loss needs the wall mask too (independent of surf_lambda)
    need_wall   = surf_lambda > 0.0 or cfg["train"].get("distinct_vol_surf", False)
    wall_col    = _wall_col_index(cfg["dataset"].get("feature_cols"))
    agg = {"loss": 0.0, "mae": 0.0}
    n_batches = 0

    for idx, (tensors, target) in enumerate(data_loader):
        optimizer.zero_grad()

        # bf16 autocast covers the transfer, the C_L split and the model forward, and ends
        # there: the loss below is computed in fp32, as in the origin case.
        with torch.autocast(device.type, dtype=torch.bfloat16):
            tensors_gpu = [t.to(device.type) for t in tensors]
            target_gpu, cl_w, cl_true, cl_b = split_cl(target.to(device.type), cfg)

            pred = model(*tensors_gpu, mask=None)   # (B, N, C)

        # is_wall column is >> 0 in normalised space for wall nodes
        wall_mask = (tensors_gpu[0][:, :, wall_col] > 0.5) if need_wall else None
        # Loss in fp32 (robust + optional per-case normalisation; see compute_field_loss)
        loss = compute_field_loss(pred, target_gpu, cfg, wall_mask=wall_mask)

        # ── v57_gated_hodge: the lift-integral term ─────────────────────────────────────────────────────
        # The disjoint vol/surf MSE terms cannot see C_L: v51 improved wall-pressure MSE by 27%
        # while its lift got 42% worse, because MSE is blind to error that is CORRELATED along the
        # contour and the integral is not. This term constrains the same pressure prediction through
        # the integral, so the two can no longer be traded off against each other.
        lam_cl = float(cfg["train"].get("cl_loss_lambda", 0.0))
        if lam_cl > 0.0 and cl_w is not None:
            l_cl, _ = cl_loss(pred, cl_w, cl_true, cl_b, cfg)
            agg["cl"] = agg.get("cl", 0.0) + float(l_cl.item())
            loss = loss + lam_cl * l_cl

        # ── v57_gated_hodge: LEARNED-analyzer band supervision + gate balance + anti-collapse ──────────
        # The bands are produced by the model's own Hodge analyzer from the TARGET, so what each
        # expert is responsible for is itself learned. Completeness (sum_k b_k == y) holds for any
        # star values, so the analyzer can only redistribute content between bands, never change the
        # total. `entropy_lambda` opposes the degenerate solution of shovelling all energy into one
        # band, which would hand the other experts trivial zero targets.
        lam_b = float(cfg["train"].get("band_loss_lambda", 0.0))
        if lam_b > 0.0 and getattr(model, "aux", None) and model.analyzer is not None:
            # fp32, explicitly: sparse bmm has no bfloat16 CUDA kernel, and a supervision
            # target should not carry autocast rounding anyway. `enabled=False` is how the
            # origin case writes it; the surrounding autocast block has already closed, so
            # this is belt and braces rather than a correction.
            with torch.autocast(device.type, enabled=False):
                bands_t = model.analyzer.bands(target_gpu.float(), tensors_gpu[0].float(),
                                               tensors_gpu[1].float(), tensors_gpu[3])
            # v57_gated_hodge: STOP-GRADIENT on the band targets. The analyzer no longer receives gradient
            # from "make the bands easy to predict" — only from the entropy term below. v54a could
            # not distinguish two explanations for its regression: (i) the analyzer gamed its own
            # auxiliary loss, or (ii) the balanced decomposition it found is simply worse for fields
            # that are 90-98% smooth. Detaching removes (i), so v54a vs v57_gated_hodge_vef decides between them.
            # v57_gated_hodge: atoms -> per-expert bands through the LEARNED assignment. Column-stochastic,
            # so sum_k target_k = sum_m atom_m = y exactly, whatever it learns.
            atoms = bands_t.detach() if cfg["train"].get("detach_band_targets", False) else bands_t
            tgt_bands = model.assign.targets(atoms)
            l_band, l_bal = band_loss(model.aux["bands"].float(), tgt_bands.float(),
                                      model.aux["gate"].float(), cfg)
            # ONE-SIDED collapse guard on the per-expert band energies (NOT entropy: v54a/b showed
            # that maximising entropy forces the balance that lost to the fixed lopsided ladder)
            l_floor, shares = model.analyzer.band_floor_penalty(
                tgt_bands.float(), min_share=cfg["train"].get("band_min_share"))
            h_ent, _ = model.analyzer.band_energy_entropy(tgt_bands.float())
            lam_e = float(cfg["train"].get("floor_lambda", 0.0))
            agg["band"] = agg.get("band", 0.0) + float(l_band.item())
            agg["bal"] = agg.get("bal", 0.0) + float(l_bal.item())
            agg["band_entropy"] = agg.get("band_entropy", 0.0) + float(h_ent.item())
            agg["band_floor"] = agg.get("band_floor", 0.0) + float(l_floor.item())
            agg["assign_entropy"] = agg.get("assign_entropy", 0.0) + float(model.assign.entropy())
            for k, v in enumerate(shares.detach().tolist()):
                agg[f"band{k}_energy"] = agg.get(f"band{k}_energy", 0.0) + float(v)
            loss = (loss + lam_b * l_band
                    + float(cfg["train"].get("balance_lambda", 0.0)) * l_bal
                    + lam_e * l_floor)                    # PENALISE starved bands (one-sided)

        if device.type == "cpu":
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

        if step_per_batch:                 # OneCycle steps every batch
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        m = regression_metrics(pred, target_gpu)      # v57_gated_hodge: fields only (C_L columns stripped)
        agg["loss"] += loss.item()
        agg["mae"] += m["mae"]
        n_batches += 1

        print(
            "[TRAIN] [EPOCH:{}/{}] [BATCH:{}/{}] Loss={:.5f}  MAE={:.5f}  lr={:.2e}".format(
                epoch, epochs, idx + 1, len(data_loader), loss.item(), m["mae"], cur_lr
            )
        )

    for k in agg:
        agg[k] /= max(n_batches, 1)

    try:
        wandb.log({"train_" + k: v for k, v in agg.items()},
                  step=epoch, commit=False)
        wandb.log({"train_lr": optimizer.param_groups[0]["lr"]}, step=epoch)
    except Exception:
        pass

    if not step_per_batch:                 # per-epoch schedulers step here
        scheduler.step()
    return agg


def validate_epoch(data_loader, model, loss_fn, epoch, device, cfg):
    model.eval()
    epochs = cfg["train"]["epochs"]
    field_names = ["ux", "uy", "p", "nut"]
    surf_lambda = cfg["train"].get("surface_loss_lambda", 0.0)
    need_wall   = surf_lambda > 0.0 or cfg["train"].get("distinct_vol_surf", False)  # v57_gated_hodge_vef
    wall_col    = _wall_col_index(cfg["dataset"].get("feature_cols"))
    agg = None
    n_batches = 0

    with torch.no_grad():
        for idx, (tensors, target) in enumerate(data_loader):

            with torch.autocast(device.type, dtype=torch.bfloat16):
                tensors = [t.to(device.type) for t in tensors]
                target, cl_w, cl_true, cl_b = split_cl(target.to(device.type), cfg)
                pred    = model(*tensors, mask=None)

            wall_mask = (tensors[0][:, :, wall_col] > 0.5) if need_wall else None
            loss = compute_field_loss(pred, target, cfg, wall_mask=wall_mask)

            m = regression_metrics(pred, target)
            m["loss"] = loss.item()
            # v57_gated_hodge: track the lift term through training (logged to W&B, not printed per batch)
            if cl_w is not None:
                l_cl_v, cl_pred_v = cl_loss(pred, cl_w, cl_true, cl_b, cfg)
                m["cl_sq"] = l_cl_v.item()
                m["cl_abs_err"] = float((cl_pred_v - cl_true.float()).abs().mean().item())
            if agg is None:
                agg = {k: 0.0 for k in m}
            for k, v in m.items():
                agg[k] += v
            n_batches += 1

            C = target.shape[-1]
            l2_str = "  ".join(
                f"L2rel({field_names[c]})={m[f'l2rel_c{c}']:.4f}"
                for c in range(C)
            )
            print(
                "[VALID] [EPOCH:{}/{}] [BATCH:{}/{}] Loss={:.5f}  R²={:.4f}  {}".format(
                    epoch, epochs, idx + 1, len(data_loader),
                    loss.item(), m["r2"], l2_str
                )
            )

    for k in agg:
        agg[k] /= max(n_batches, 1)

    # Epoch-level summary line
    C = min(4, len(field_names))
    l2_summary = "  ".join(
        f"L2rel({field_names[c]})={agg[f'l2rel_c{c}']:.4f}" for c in range(C)
    )
    print(
        "[VALID] [EPOCH:{}/{}] [AGG] MSE={:.5f}  R²={:.4f}  {}".format(
            epoch, epochs, agg["mse"], agg["r2"], l2_summary
        )
    )

    try:
        wandb.log({"val_" + k: v for k, v in agg.items()}, step=epoch)
    except Exception:
        pass

    return agg


# ── main ──────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _epochs_patched():
    """Point `training.loop.main` at the bf16 epoch functions, then put it back.

    `training/loop.py` resolves `train_epoch` / `validate_epoch` as module globals, and that file
    is deliberately not edited, so this is how the bf16 pair is substituted. The swap is undone on
    the way out even if training raises.
    """
    saved = (_loop.train_epoch, _loop.validate_epoch)
    _loop.train_epoch, _loop.validate_epoch = train_epoch, validate_epoch
    try:
        yield
    finally:
        _loop.train_epoch, _loop.validate_epoch = saved


def main():
    print("[cases.v] bf16 autocast trainer (forward passes only; loss and optimizer in fp32)",
          flush=True)
    with _epochs_patched():
        return _loop.main()


if __name__ == "__main__":
    main()
