# ========================================================================================
# TRAINING ENTRY POINT
#   main()  — called by ../train.py
# ========================================================================================
#!/usr/bin/env python
"""Training and validation loops, and the CLI entry point.

Part of the v57_gated_hodge_vef case, split out of the former 699-line
`airfrans_regression_v57_gated_hodge_vef.py` so that ARCHITECTURE / DATA / TRAINING / EVALUATION are
separable. See README.md for the map.
"""


import os
import glob
import json
import argparse
from typing import List, Tuple, Dict

import toml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    print("`wandb` was not found. `wandb` functionalities will be disabled.")

from mesh_opformer.dataset import collate_batch_pde_fn
from mesh_opformer.training.utils import validate_device, model_save

from arch.model import build_v57_gated_hodge_vef_model
from data.dataset import AirfransV39Dataset
from training.losses import (split_bands, band_loss, split_cl, cl_loss, compute_field_loss)
from training.metrics import regression_metrics, append_metrics, _wall_col_index
from training.schedules import make_warmup_flat_cosine
from paths import resolve_data_paths

_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

build_v30_model = build_v57_gated_hodge_vef_model     # legacy alias used inside the loops
AirfransV13Dataset = AirfransV39Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="./runs/")
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Checkpoint .pth to resume from (loads model + optimizer weights)")
    parser.add_argument("--continue_schedule", action="store_true",
                        help="vef_v2: with --resume_ckpt, CONTINUE the checkpoint's own LR schedule (restore "
                             "the scheduler state, run only the remaining epochs to cfg epochs) instead of a "
                             "fresh schedule. Used by the crash auto-resume launcher.")
    parser.add_argument("--stats_path", type=str, default=None,
                        help="Path to existing airfrans_norm_stats.npz (skips recomputation)")
    return parser.parse_args()


# ── data helpers ──────────────────────────────────────────────────────────────


def _collect_graph_paths(data_root: str) -> List[str]:
    """Return sorted list of AirFRANS graph .pth files under `data_root`."""
    all_pth = glob.glob(os.path.join(data_root, "**", "*.pth"), recursive=True)
    # Keep only files inside directories named graph_airfrans_data_* to avoid
    # accidentally picking up checkpoint files in runs/ or cache/.
    paths = sorted(
        p for p in all_pth
        if os.path.basename(os.path.dirname(p)).startswith("graph_airfrans_data")
    )
    if not paths:
        raise FileNotFoundError(f"No AirFRANS graph .pth files found under {data_root}")
    return paths


def initialize_airfrans_datasets(cfg, preloaded_stats=None) -> Tuple[Dict, Dict, Dict]:
    """
    Build train / test AirfransGraphDataset instances and their DataLoaders.
    Standardisation stats for inputs and targets are computed from the training
    split only.
    """
    dcfg = cfg["dataset"]
    pool_dir = dcfg["data_root"]     # dir of the v18_3 mesh POOL (<name>__s{k}.npz)
    K = dcfg.get("pool_K", 8)

    feature_cols = dcfg.get("feature_cols", None)
    target_cols = dcfg.get("target_cols", None)

    # Official AirFRANS split -> sample base NAMES that have a precomputed pool.
    official_json = dcfg["official_split_json"]
    with open(official_json) as f:
        split = json.load(f)

    def has_pool(b):
        n = os.path.splitext(b)[0]
        return os.path.isfile(os.path.join(pool_dir, n + "__s0.npz"))

    train_names = [os.path.splitext(b)[0] for b in split["train"] if has_pool(b)]
    test_names  = [os.path.splitext(b)[0] for b in split["test"]  if has_pool(b)]
    print(f"v18_3 official split: {len(train_names)} train / {len(test_names)} test samples "
          f"(pool K={K}, train mesh-augmented, test fixed s0)")

    use_bc_injection = dcfg.get("use_bc_injection", True)

    if preloaded_stats is not None:
        stats = preloaded_stats
        print("Using preloaded normalisation stats.")
    else:
        print(f"Computing input/target stats over {len(train_names)} samples (s0 meshes) …")
        stats = AirfransV13Dataset.compute_stats(
            train_names, pool_dir,
            feature_cols=feature_cols,
            target_cols=target_cols,
            max_samples=dcfg.get("stats_max_samples", None),
            use_bc_injection=use_bc_injection,
        )
    print("  feat_mean  :", np.round(stats["feat_mean"], 4))
    print("  feat_std   :", np.round(stats["feat_std"], 4))
    print("  target_mean:", np.round(stats["target_mean"], 4))
    print("  target_std :", np.round(stats["target_std"], 4))

    train_kw = dcfg["train"]["kw"]
    test_kw = dcfg["test"]["kw"]

    common = dict(
        feature_cols=feature_cols,
        target_cols=target_cols,
        feat_mean=stats["feat_mean"],
        feat_std=stats["feat_std"],
        target_mean=stats["target_mean"],
        target_std=stats["target_std"],
    )

    data = {
        "train": AirfransV13Dataset(train_names, pool_dir, K=K, train_augment=True, **common, **train_kw),
        "test": AirfransV13Dataset(test_names, pool_dir, K=K, train_augment=False, **common, **test_kw),
    }

    loader = {
        name: DataLoader(
            dataset,
            **dcfg["dataloader"][name],
            collate_fn=collate_batch_pde_fn,
        )
        for name, dataset in data.items()
    }

    return data, loader, stats


# ── metrics ───────────────────────────────────────────────────────────────────


def train_epoch(data_loader, model, loss_fn, optimizer, scheduler, epoch, device, cfg,
                step_per_batch=False):
    model.train()
    scaler = torch.amp.GradScaler()
    epochs = cfg["train"]["epochs"]
    surf_lambda = cfg["train"].get("surface_loss_lambda", 0.0)
    # v57_gated_hodge_vef: the distinct vol/surf loss needs the wall mask too (independent of surf_lambda)
    need_wall   = surf_lambda > 0.0 or cfg["train"].get("distinct_vol_surf", False)
    wall_col    = _wall_col_index(cfg["dataset"].get("feature_cols"))
    agg = {"loss": 0.0, "mae": 0.0}
    n_batches = 0

    for idx, (tensors, target) in enumerate(data_loader):
        optimizer.zero_grad()

        tensors_gpu = [t.to(device.type) for t in tensors]
        target_gpu, cl_w, cl_true, cl_b = split_cl(target.to(device.type), cfg)

        pred = model(*tensors_gpu, mask=None)   # (B, N, C)

        # is_wall column is >> 0 in normalised space for wall nodes
        wall_mask = (tensors_gpu[0][:, :, wall_col] > 0.5) if need_wall else None
        # Loss in fp32 (robust + optional per-case normalisation; see compute_field_loss)
        loss = compute_field_loss(pred, target_gpu, cfg, wall_mask=wall_mask)

        # ── v57_gated_hodge_vef: the lift-integral term ─────────────────────────────────────────────────────
        # The disjoint vol/surf MSE terms cannot see C_L: v51 improved wall-pressure MSE by 27%
        # while its lift got 42% worse, because MSE is blind to error that is CORRELATED along the
        # contour and the integral is not. This term constrains the same pressure prediction through
        # the integral, so the two can no longer be traded off against each other.
        lam_cl = float(cfg["train"].get("cl_loss_lambda", 0.0))
        if lam_cl > 0.0 and cl_w is not None:
            l_cl, _ = cl_loss(pred, cl_w, cl_true, cl_b, cfg)
            agg["cl"] = agg.get("cl", 0.0) + float(l_cl.item())
            loss = loss + lam_cl * l_cl

        # ── v57_gated_hodge_vef: LEARNED-analyzer band supervision + gate balance + anti-collapse ──────────
        # The bands are produced by the model's own Hodge analyzer from the TARGET, so what each
        # expert is responsible for is itself learned. Completeness (sum_k b_k == y) holds for any
        # star values, so the analyzer can only redistribute content between bands, never change the
        # total. `entropy_lambda` opposes the degenerate solution of shovelling all energy into one
        # band, which would hand the other experts trivial zero targets.
        lam_b = float(cfg["train"].get("band_loss_lambda", 0.0))
        if lam_b > 0.0 and getattr(model, "aux", None) and model.analyzer is not None:
            # fp32 (vef_v2: no autocast anywhere): sparse bmm has no bfloat16 CUDA kernel, and a supervision
            # target should not carry autocast rounding anyway.
            bands_t = model.analyzer.bands(target_gpu.float(), tensors_gpu[0].float(),
                                           tensors_gpu[1].float(), tensors_gpu[3])
            # v57_gated_hodge_vef: STOP-GRADIENT on the band targets. The analyzer no longer receives gradient
            # from "make the bands easy to predict" — only from the entropy term below. v54a could
            # not distinguish two explanations for its regression: (i) the analyzer gamed its own
            # auxiliary loss, or (ii) the balanced decomposition it found is simply worse for fields
            # that are 90-98% smooth. Detaching removes (i), so v54a vs v57_gated_hodge_vef decides between them.
            # v57_gated_hodge_vef: atoms -> per-expert bands through the LEARNED assignment. Column-stochastic,
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
        m = regression_metrics(pred, target_gpu)      # v57_gated_hodge_vef: fields only (C_L columns stripped)
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

            tensors = [t.to(device.type) for t in tensors]
            target, cl_w, cl_true, cl_b = split_cl(target.to(device.type), cfg)
            pred    = model(*tensors, mask=None)

            wall_mask = (tensors[0][:, :, wall_col] > 0.5) if need_wall else None
            loss = compute_field_loss(pred, target, cfg, wall_mask=wall_mask)

            m = regression_metrics(pred, target)
            m["loss"] = loss.item()
            # v57_gated_hodge_vef: track the lift term through training (logged to W&B, not printed per batch)
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


def main():
    args = parse_args()

    with open(args.cfg_path) as f:
        cfg = toml.load(f)
    cfg = resolve_data_paths(cfg, _DEPLOY_ROOT)

    _seed = int(cfg.get("seed", 0))
    torch.manual_seed(_seed); np.random.seed(_seed)

    device = validate_device(cfg["device"])

    if device.type == "cpu":
        torch.set_num_interop_threads(4)
        torch.set_num_threads(4)

    print("Device:", device)

    # Load preloaded stats if provided (skips expensive recomputation on resume).
    preloaded_stats = None
    if args.stats_path:
        raw = np.load(args.stats_path)
        preloaded_stats = {k: raw[k] for k in raw.files}
        print(f"Loaded norm stats from {args.stats_path}")

    data, loader, stats = initialize_airfrans_datasets(cfg, preloaded_stats=preloaded_stats)

    model = build_v30_model(**cfg["train"]["model"]).to(device.type)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    loss_fn  = torch.nn.MSELoss()
    _opt = cfg["train"].get("optimizer", "adam").lower()
    _wd = cfg["train"].get("weight_decay", 0.0)
    if _opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=_wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    print(f"Optimizer: {_opt} (lr={cfg['train']['lr']:g}, weight_decay={_wd:g})")

    # Build a fresh scheduler sized for cfg["train"]["epochs"] new epochs.
    sched_cfg = cfg["train"].get("schedule", {})
    step_per_batch = False
    if sched_cfg.get("type") == "onecycle":
        # LR ramps to max_lr over pct_start then cosine-anneals to ~0 (per BATCH, no
        # flat plateau). Plain Adam (no weight decay). steps = epochs * batches/epoch.
        steps_per_epoch = len(loader["train"])
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg["train"]["lr"], epochs=cfg["train"]["epochs"],
            steps_per_epoch=steps_per_epoch,
            pct_start=sched_cfg.get("pct_start", 0.15),
            div_factor=sched_cfg.get("div_factor", 25.0),
            final_div_factor=sched_cfg.get("final_div_factor", 1e4),
            anneal_strategy=sched_cfg.get("anneal_strategy", "cos"),
        )
        step_per_batch = True
        print(f"Scheduler: OneCycleLR (max_lr={cfg['train']['lr']:g}, "
              f"pct_start={sched_cfg.get('pct_start',0.15)}, per-batch over "
              f"{steps_per_epoch*cfg['train']['epochs']} steps)")
    elif sched_cfg.get("type") == "warmup_flat_cosine":
        scheduler = make_warmup_flat_cosine(
            optimizer,
            warmup_epochs=sched_cfg.get("warmup_epochs", 20),
            flat_epochs=sched_cfg.get("flat_epochs", 280),
            total_epochs=cfg["train"]["epochs"],
            eta_min_ratio=sched_cfg.get("eta_min_ratio", 2e-3),
        )
        print(f"Scheduler: warmup({sched_cfg.get('warmup_epochs',20)}) + "
              f"flat({sched_cfg.get('flat_epochs',280)}) + cosine")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["train"]["epochs"], eta_min=1e-6
        )
        print("Scheduler: CosineAnnealingLR")

    # Resume from checkpoint: restore model + optimizer weights; epoch counter.
    # Scheduler is intentionally rebuilt fresh so the new schedule applies.
    start_epoch = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device.type, weights_only=False)
        model.load_state_dict(ckpt["state"])
        # OneCycleLR does NOT hold its own max_lr — `get_lr()` reads `initial_lr` / `max_lr` /
        # `min_lr` back out of `optimizer.param_groups` every step. `load_state_dict` REPLACES
        # param_groups wholesale with the checkpoint's, which still carry the PREVIOUS run's
        # annotations, so the "fresh scheduler" above silently reverts to the old LR magnitudes:
        # a continuation configured for max_lr=2e-4 was measured ramping to 5.2e-4 by epoch 11,
        # on its way to the original 1e-3 — a full-strength warm restart of a converged model.
        # Snapshot what the new scheduler wrote, load the state, then put the annotations back.
        if args.continue_schedule:
            # vef_v2 crash-resume: same run, same schedule -> restore optimizer AND scheduler verbatim,
            # so the LR picks up exactly where the crashed run left it (no warm restart).
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            for _g, _lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
                _g["lr"] = _lr
        else:
            _sched_keys = ("initial_lr", "max_lr", "min_lr", "max_momentum", "base_momentum")
            _keep = [{k: g[k] for k in _sched_keys if k in g} for g in optimizer.param_groups]
            optimizer.load_state_dict(ckpt["optimizer"])
            for _g, _k in zip(optimizer.param_groups, _keep):
                _g.update(_k)
                if "initial_lr" in _k:
                    _g["lr"] = _k["initial_lr"]
        start_epoch = ckpt["epoch"]
        print(f"Resumed from {args.resume_ckpt}  (epoch {start_epoch} → continuing from {start_epoch + 1})")

    if cfg.get("use_wandb", False):
        os.environ["WANDB_MODE"] = cfg["wandb"]["WANDB_MODE"]
        _init_kw = dict(cfg["wandb"]["init"])
        # drop placeholder/empty entity -> default account from the ~/.netrc key
        if not _init_kw.get("entity") or _init_kw.get("entity") in ("user-wandb-account", ""):
            _init_kw.pop("entity", None)
        # pass the proper run name INTO init so it shows correctly from the start
        wandb.init(name=cfg["wandb"]["name"], config=cfg, **_init_kw)
        wandb.watch(model, log="all", log_freq=50)

    os.makedirs(args.out, exist_ok=True)

    # Persist normalisation stats next to the checkpoints for evaluation.
    np.savez(
        os.path.join(args.out, "airfrans_norm_stats.npz"),
        **{k: v for k, v in stats.items()},
    )

    ckpt_every = cfg["train"].get("ckpt_every", 10)
    run_name = cfg["wandb"]["name"]
    new_epochs = cfg["train"]["epochs"]
    if args.resume_ckpt and args.continue_schedule:          # vef_v2: only the epochs the schedule has left
        new_epochs = max(0, cfg["train"]["epochs"] - start_epoch)

    # rel_epoch: position within this training run (1..new_epochs), used for scheduler.
    # abs_epoch: absolute epoch number (start_epoch+1 .. start_epoch+new_epochs), used for checkpoints.
    for rel_epoch in range(1, new_epochs + 1):
        abs_epoch = start_epoch + rel_epoch
        tr_agg = train_epoch(loader["train"], model, loss_fn, optimizer, scheduler, rel_epoch, device, cfg,
                    step_per_batch=step_per_batch)
        va_agg = validate_epoch(loader["test"], model, loss_fn, rel_epoch, device, cfg)

        # v57_gated_hodge_vef: persist per-epoch metrics (incl. the band / balance terms) to the run folder
        row = {"epoch": int(abs_epoch), "lr": float(optimizer.param_groups[0]["lr"])}
        row.update({f"train_{k}": float(v) for k, v in (tr_agg or {}).items()})
        row.update({f"val_{k}": float(v) for k, v in (va_agg or {}).items()})
        append_metrics(args.out, row)

        _ck_epoch = abs_epoch if args.continue_schedule else rel_epoch   # vef_v2: keep the ckpt grid on resume
        if _ck_epoch % ckpt_every == 0 or rel_epoch == new_epochs:
            save_path = os.path.join(
                args.out, f"airfrans_{run_name}_ckpt_{abs_epoch}.pth"
            )
            model_save(model, optimizer, scheduler, abs_epoch, save_path)
            print("Saved:", save_path)


if __name__ == "__main__":
    main()
