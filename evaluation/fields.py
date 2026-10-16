# ========================================================================================
# EVALUATION ENTRY POINT
#   main()  — field metrics + qualitative figures for one checkpoint
# ========================================================================================
# __deployment_path_bootstrap__
# The original case-based layout globbed every sibling case dir onto sys.path (a monorepo
# convenience). This deployment is self-contained: only its own root and shared/ are needed.
import os as _o, sys as _s
_ROOT = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))    # deployment root
for _d in (_o.path.join(_ROOT, "shared"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)
# Eval for AirFRANS v14 (freestream-non-dimensionalised targets). Inverts:
# physical = redim(standardised_pred * std + mean, U_inf).
#!/usr/bin/env python
"""
Eval for AirFRANS v18 (remeshed dataset + interpolated vertex targets).

Loads the v18 test meshes (dataset_airfrans_v18.AirfransV13Dataset via the official
split), runs a checkpoint, computes PHYSICAL-space per-field metrics — R², relative
L2 (‖p-t‖/‖t‖, Transolver-style), mean-centred L2, MSE/MAE/RMSE — averaged over the
200 test samples, and (optionally) renders jet GT/prediction/abs-error figures on the
v18 mesh.

NOTE: v18 targets are interpolated onto the v18 mesh, so metrics are on v18's OWN test
set and are NOT directly comparable to v10/v11/v12 (raw BL-subsampled targets).

Usage: python eval_airfrans_v18.py --cfg_path C --ckpt K --stats S --out O [--n_qual N]
"""
import os, sys, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import toml
from torch.utils.data import DataLoader

from data.dataset import AirfransV39Dataset as AirfransV13Dataset
from data.dataset import redim_targets, uinf
from airfrans_native import vol_surf_mse, global_target_std
from mesh_opformer.dataset import collate_batch_pde_fn
from arch.model import build_v57_gated_hodge_vef_model as build_v30_model
from mesh_opformer.training.utils import validate_device
from paths import resolve_data_paths

_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELD_KEYS = ["ux", "uy", "p", "nut"]
FIELD_NAMES = ["u_x", "u_y", "p", "nu_t"]


def per_channel_metrics(pred, target):
    out = {}
    for c, k in enumerate(FIELD_KEYS):
        p, t = pred[:, c], target[:, c]
        mse = float(np.mean((p - t) ** 2)); mae = float(np.mean(np.abs(p - t))); rmse = mse ** 0.5
        ss_res = float(np.sum((p - t) ** 2)); ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        l2rel = float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8))
        l2rel_c = float(np.sqrt(max(0.0, ss_res) / (ss_tot + 1e-12)))
        out[k] = dict(mse=mse, mae=mae, rmse=rmse, r2=r2, l2rel=l2rel, l2rel_c=l2rel_c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="./eval_v18/")
    ap.add_argument("--n_qual", type=int, default=0)
    args = ap.parse_args()

    cfg = toml.load(args.cfg_path)
    cfg = resolve_data_paths(cfg, _DEPLOY_ROOT)
    device = validate_device(cfg["device"])
    os.makedirs(args.out, exist_ok=True)
    dcfg = cfg["dataset"]

    split = json.load(open(dcfg["official_split_json"]))
    pool_dir = dcfg["data_root"]; K = dcfg.get("pool_K", 8)
    has_pool = lambda b: os.path.isfile(os.path.join(pool_dir, os.path.splitext(b)[0] + "__s0.npz"))
    test_names  = [os.path.splitext(b)[0] for b in split["test"]  if has_pool(b)]
    train_names = [os.path.splitext(b)[0] for b in split["train"] if has_pool(b)]
    # Evaluates on the FIXED s0 mesh of each sample (deterministic; train uses the K-pool at random)
    test_paths = [os.path.join(pool_dir, n + "__s0.npz") for n in test_names]
    train_paths = [os.path.join(pool_dir, n + "__s0.npz") for n in train_names]
    print(f"HodgeMoE-vef test samples: {len(test_names)} (eval on fixed s0 meshes)")
    std_glob = global_target_std(train_paths)   # AirFRANS coef_norm std_out (global per-field physical std)
    print("  global target std (AirFRANS norm):", np.round(std_glob, 5))

    st = np.load(args.stats)
    fm, fs = st["feat_mean"], st["feat_std"]
    tm, ts = st["target_mean"], st["target_std"]

    ds = AirfransV13Dataset(
        test_names, pool_dir, K=K, train_augment=False,
        feature_cols=dcfg.get("feature_cols"), target_cols=dcfg.get("target_cols"),
        feat_mean=fm, feat_std=fs, target_mean=tm, target_std=ts,
        extract_kw=dcfg["test"]["kw"]["extract_kw"],
        cache_dir=dcfg["test"]["kw"].get("cache_dir"), use_bc_injection=True,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_batch_pde_fn)

    model = build_v30_model(**cfg["train"]["model"]).to(device.type)
    ck = torch.load(args.ckpt, map_location=device.type, weights_only=False)
    model.load_state_dict(ck["state"] if "state" in ck else ck["model_state_dict"])
    model.eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,} | ckpt {args.ckpt}")

    all_metrics = []
    vs_acc = []; ss_acc = []          # AirFRANS-native volume/surface normalised MSE per field
    for i, (tensors, target) in enumerate(loader):
        with torch.no_grad():
            pred = model(*[t.to(device.type) for t in tensors], mask=None)
        pred = pred[0].float().cpu().numpy()
        # v57_gated_hodge_vef's dataset appends [cl_w][cl_true][cl_b] to the target — keep only the fields
        targ = target[0].cpu().numpy()[:, :int(cfg["train"]["model"]["out_channels"])]
        # PLAIN de-standardise -> physical (v18_6 uses global per-field z-score, no U_inf)
        zz = np.load(test_paths[i])
        U = uinf(zz["x_feat"])                    # freestream non-dim: de-z-score → de-log-ν_t → re-dim by U∞
        pred_phys = redim_targets(pred * ts + tm, U)
        targ_phys = redim_targets(targ * ts + tm, U)
        # AirFRANS-native metric: MSE on globally-normalised PHYSICAL fields, vol/surf split
        vfld, sfld = vol_surf_mse(pred_phys, targ_phys, zz["wall"], std_glob)
        vs_acc.append(vfld); ss_acc.append(sfld)
        m = per_channel_metrics(pred_phys, targ_phys)
        all_metrics.append(m)
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(test_paths)}] R2mean={np.mean([m[k]['r2'] for k in FIELD_KEYS]):.3f}")

        if i < args.n_qual:
            z = np.load(test_paths[i]); pos = z["pos"]; faces = z["faces"]
            tri = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
            fig, axes = plt.subplots(4, 3, figsize=(20, 18))
            for c in range(4):
                gt, pr = targ_phys[:, c], pred_phys[:, c]
                vmin, vmax = np.percentile(gt, [1, 99])
                for j, (val, ttl) in enumerate([(gt, "GT"), (pr, "Pred"), (np.abs(pr-gt), "|err|")]):
                    ax = axes[c, j]
                    vm, vx = (0, np.percentile(np.abs(pr-gt), 99)) if j == 2 else (vmin, vmax)
                    tpc = ax.tripcolor(tri, val, cmap="jet", shading="gouraud", vmin=vm, vmax=vx)
                    ax.set_aspect("equal"); ax.set_title(f"{FIELD_NAMES[c]} {ttl}")
                    fig.colorbar(tpc, ax=ax, shrink=0.7)
            fig.suptitle(f"HodgeMoE-vef \u2014 test sample {i}", fontsize=13); fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"qual_{i:03d}.png"), dpi=90, bbox_inches="tight")
            plt.close(fig)

    # ── aggregate (mean over samples) ─────────────────────────────────────────
    print(f"\n── HodgeMoE-vef aggregate test metrics (mean over {len(test_paths)} sample{'s' if len(test_paths) != 1 else ''}, physical) ──")
    hdr = f"{'Field':<6}" + "".join(f"  {h:>9}" for h in ["L2rel", "L2rel_c", "R²", "MSE", "MAE", "RMSE"])
    print(hdr)
    l2s = []
    for k in FIELD_KEYS:
        row = {q: np.mean([m[k][q] for m in all_metrics]) for q in ["l2rel", "l2rel_c", "r2", "mse", "mae", "rmse"]}
        l2s.append(row["l2rel"])
        print(f"{k:<6}  {row['l2rel']:>9.4f}  {row['l2rel_c']:>9.4f}  {row['r2']:>9.4f}  "
              f"{row['mse']:>9.4f}  {row['mae']:>9.4f}  {row['rmse']:>9.4f}")
    print(f"{'MEAN':<6}  {np.mean(l2s):>9.4f}   (mean per-field relative L2 — headline)")

    # ── AirFRANS-native metrics (Transolver benchmark table) ──────────────────
    vol = np.mean(np.array(vs_acc), axis=0)   # per-field volume normalised MSE
    surf = np.mean(np.array(ss_acc), axis=0)  # per-field surface normalised MSE
    print("\n── AirFRANS-native normalised-MSE (Transolver table columns) ──")
    print(f"{'Field':<6}  {'VolMSE':>10}  {'SurfMSE':>10}")
    for c, k in enumerate(FIELD_KEYS):
        print(f"{k:<6}  {vol[c]:>10.5f}  {surf[c]:>10.5f}")
    print(f"{'VOLUME':<6}  {np.mean(vol):>10.5f}   (mean over fields — benchmark 'Volume' column)")
    print(f"{'SURF':<6}  {np.mean(surf):>10.5f}   (mean over fields — benchmark 'Surf' column)")

    np.savez(os.path.join(args.out, "test_metrics.npz"),
             vol_mse=vol, surf_mse=surf,
             metrics=np.array([[[m[k][q] for q in ["mse", "mae", "rmse", "r2", "l2rel", "l2rel_c"]]
                                for k in FIELD_KEYS] for m in all_metrics]),
             field_keys=FIELD_KEYS)
    print(f"Saved metrics -> {os.path.join(args.out, 'test_metrics.npz')}")


if __name__ == "__main__":
    main()
