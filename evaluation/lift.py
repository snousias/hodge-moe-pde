# ========================================================================================
# EVALUATION ENTRY POINT
#   main()  — mesh-native C_L / rho_L for one checkpoint
# ========================================================================================
#!/usr/bin/env python
# __deployment_path_bootstrap__
# The original case-based layout globbed every sibling case dir onto sys.path (a monorepo
# convenience). This deployment is self-contained: only its own root and shared/ are needed.
import os as _o, sys as _s
_ROOT = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))    # deployment root
for _d in (_o.path.join(_ROOT, "shared"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)
"""
MESH-NATIVE lift coefficient C_L / rho_L — NO interpolation to the raw ~177k CFD nodes.

The lift coefficient is integrated over the MODEL'S OWN airfoil wall nodes:
  C_L = ( basis(aoa) @ sum_edges p_edge * n_edge * L_edge )[1] / (0.5 * v^2)
(pressure integral over the airfoil surface, rotated to the lift direction; the viscous / wall-shear
term is omitted — it is <1% of lift, as the AirFRANS paper notes). Computed identically for the
predicted and the (interpolated-onto-the-mesh) true pressure, so it is a self-consistent own-grid
metric. Reports C_L relative L2 and Spearman rho_L over the test set.

Validation: also computes the library's TRUE C_L (af.Simulation.force_coefficient, on the raw mesh)
and reports its Spearman correlation with our mesh-native TRUE C_L — a high value confirms the
mesh-native pressure integral captures lift correctly.

Usage: python eval_airfrans_cl_native.py --cfg_path <v18_3 cfg> --ckpt K --stats S --out O
"""
import os, json, argparse
import numpy as np
import torch
import toml
from torch.utils.data import DataLoader

from data.dataset import AirfransV39Dataset as AirfransV18_3Dataset, redim_targets, uinf
from mesh_opformer.dataset import collate_batch_pde_fn
from arch.model import build_v57_gated_hodge_vef_model as build_mesh_opformer_model
from mesh_opformer.training.utils import validate_device
from airfrans_native import spearman
from paths import resolve_data_paths

_DEPLOY_ROOT = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))

# Was a hardcoded absolute path; now read from cfg["data"]["data_root"] in main() below
# (--cfg_path is required, so this is always overwritten before use).
RAW_ROOT = None


def order_wall_loop(P):
    """Permutation of wall-node indices forming a simple airfoil loop (camber split)."""
    i_le = int(np.argmin(P[:, 0])); i_te = int(np.argmax(P[:, 0]))
    le = P[i_le]; te = P[i_te]; d = te - le; d = d / (np.linalg.norm(d) + 1e-12)
    nrm = np.array([-d[1], d[0]]); t = (P - le) @ d; s = (P - le) @ nrm
    nbin = 60; edges = np.linspace(t.min(), t.max(), nbin + 1); mid = 0.5 * (edges[:-1] + edges[1:])
    camb = np.full(nbin, np.nan)
    for b in range(nbin):
        m = (t >= edges[b]) & (t <= edges[b + 1])
        if m.any():
            camb[b] = 0.5 * (s[m].min() + s[m].max())
    ok = ~np.isnan(camb); camb_t = np.interp(t, mid[ok], camb[ok])
    up = np.where(s >= camb_t)[0]; lo = np.where(s < camb_t)[0]
    return np.concatenate([up[np.argsort(t[up])], lo[np.argsort(-t[lo])]])


def mesh_cl(pos, wall, p, normals, inlet):
    """Pressure-integral lift coefficient over the mesh's own airfoil surface."""
    widx = np.where(wall)[0]
    W = pos[widx].astype("float64"); pw = p[widx].astype("float64")
    Nw = normals[widx].astype("float64")
    order = order_wall_loop(W)
    W = W[order]; pw = pw[order]; Nw = Nw[order]
    Nw = Nw / (np.linalg.norm(Nw, axis=1, keepdims=True) + 1e-12)   # unit outward normals
    P2 = np.roll(W, -1, axis=0)
    L = np.linalg.norm(P2 - W, axis=1)
    p_mid = 0.5 * (pw + np.roll(pw, -1))
    N_mid = 0.5 * (Nw + np.roll(Nw, -1, axis=0))
    N_mid = N_mid / (np.linalg.norm(N_mid, axis=1, keepdims=True) + 1e-12)
    fp = (p_mid[:, None] * N_mid * L[:, None]).sum(axis=0)          # sum p n L
    v = float(np.linalg.norm(inlet)); aoa = float(np.arctan2(inlet[1], inlet[0]))
    basis = np.array([[np.cos(aoa), np.sin(aoa)], [-np.sin(aoa), np.cos(aoa)]])
    return float((basis @ fp)[1] / (0.5 * v ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_path", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True); ap.add_argument("--out", default="./eval_cl_native/")
    ap.add_argument("--validate", type=int, default=30, help="N cases to cross-check vs library true C_L")
    args = ap.parse_args()

    global RAW_ROOT
    cfg = toml.load(args.cfg_path)
    cfg = resolve_data_paths(cfg, _DEPLOY_ROOT)
    RAW_ROOT = cfg["data"]["_raw_dataset_root"]
    device = validate_device(cfg["device"]); os.makedirs(args.out, exist_ok=True)
    dcfg = cfg["dataset"]; split = json.load(open(dcfg["official_split_json"]))
    pool_dir = dcfg["data_root"]; K = dcfg.get("pool_K", 8)
    has_pool = lambda b: os.path.isfile(os.path.join(pool_dir, os.path.splitext(b)[0] + "__s0.npz"))
    test_names = [os.path.splitext(b)[0] for b in split["test"] if has_pool(b)]
    test_paths = [os.path.join(pool_dir, n + "__s0.npz") for n in test_names]
    print(f"mesh-native C_L: {len(test_names)} test samples (own-grid, no 177k interp)")

    st = np.load(args.stats); fm, fs = st["feat_mean"], st["feat_std"]; tm, ts = st["target_mean"], st["target_std"]
    ds = AirfransV18_3Dataset(
        test_names, pool_dir, K=K, train_augment=False,
        feature_cols=dcfg.get("feature_cols"), target_cols=dcfg.get("target_cols"),
        feat_mean=fm, feat_std=fs, target_mean=tm, target_std=ts,
        extract_kw=dcfg["test"]["kw"]["extract_kw"],
        cache_dir=dcfg["test"]["kw"].get("cache_dir"), use_bc_injection=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_batch_pde_fn)

    model = build_mesh_opformer_model(**cfg["train"]["model"]).to(device.type)
    ck = torch.load(args.ckpt, map_location=device.type, weights_only=False)
    model.load_state_dict(ck["state"] if "state" in ck else ck["model_state_dict"]); model.eval()
    print(f"ckpt {args.ckpt}")

    cl_true, cl_pred, cl_lib = [], [], []
    for i, (tensors, target) in enumerate(loader):
        with torch.no_grad():
            pred = model(*[t.to(device.type) for t in tensors], mask=None)
        pred = pred[0].float().cpu().numpy()
        z = np.load(test_paths[i]); pos = z["pos"]; wall = z["wall"]; xf = z["x_feat"]
        inlet = np.array([xf[0, 0], xf[0, 1]]); normals = xf[:, 3:5]
        U = uinf(xf)
        p_pred = redim_targets(pred * ts + tm, U)[:, 2]         # physical predicted pressure
        p_true = z["y"][:, 2]                                    # physical true pressure (on the mesh)
        cl_pred.append(mesh_cl(pos, wall, p_pred, normals, inlet))
        cl_true.append(mesh_cl(pos, wall, p_true, normals, inlet))
        if i < args.validate:
            try:
                import airfrans as af
                sim = af.Simulation(root=RAW_ROOT, name=test_names[i])
                cl_lib.append(float(sim.force_coefficient()[1][0]))
            except Exception:
                cl_lib.append(np.nan)
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(test_names)}]")

    cl_true = np.array(cl_true); cl_pred = np.array(cl_pred)
    relL2 = float(np.linalg.norm(cl_true - cl_pred) / (np.linalg.norm(cl_true) + 1e-12))
    rho_L = spearman(cl_true, cl_pred)
    print("\n── MESH-NATIVE force metrics (own grid, no 177k interpolation) ──")
    print(f"  C_L relative L2 : {relL2:.4f}   [lower=better]")
    print(f"  rho_L (Spearman): {rho_L:.4f}   [higher=better]")
    if cl_lib:
        cl_lib = np.array(cl_lib); ok = ~np.isnan(cl_lib)
        if ok.sum() > 3:
            val = spearman(cl_true[:len(cl_lib)][ok], cl_lib[ok])
            print(f"  [validation] mesh-native TRUE vs library TRUE C_L Spearman = {val:.4f} "
                  f"(n={int(ok.sum())}; ~1.0 confirms the pressure integral captures lift)")
    np.savez(os.path.join(args.out, "cl_native.npz"), cl_true=cl_true, cl_pred=cl_pred, relL2=relL2, rho_L=rho_L)
    print(f"Saved -> {os.path.join(args.out, 'cl_native.npz')}")


if __name__ == "__main__":
    main()
