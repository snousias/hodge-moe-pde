# __deployment_path_bootstrap__  (self-contained: only the deployment root and shared/)
import os as _o, sys as _s, glob as _g
_ROOT = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))    # deployment root
for _d in (_o.path.join(_ROOT, "shared"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)
from paths import resolve_data_paths as _resolve_data_paths
#!/usr/bin/env python
"""Empirically check the SPD claim of the v34_2 Hodge stars on a real test mesh.

Captures the actual inputs of each PSD-Hodge layer with a forward hook, re-assembles the
sparse star as a DENSE matrix, and reports: symmetry, extremal eigenvalues of the star,
and the spectrum of the assembled L_v = *0^-1 d0^T *1 d0.
"""
import json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import toml
from torch.utils.data import DataLoader

from data.dataset import AirfransV39Dataset
from mesh_opformer.dataset import collate_batch_pde_fn
from arch.model import build_v57_gated_hodge_vef_model
from mesh_opformer.training.utils import validate_device


def dense_star(phi, idx, log_sigma, theta_beta, head):
    """Re-assemble psd_star() as a dense (N,N) matrix for one head."""
    p = phi[0, head].double()                      # (N,pdim)
    N, W = idx.shape
    sigma2 = (F.softplus(log_sigma[head].double()) + 1e-8) ** 2
    Wm = torch.zeros(N, N, dtype=torch.float64)
    for i in range(N):
        js = idx[i]
        js = js[js >= 0]
        d2 = ((p[i].unsqueeze(0) - p[js]) ** 2).sum(-1)
        Wm[i, js] = torch.exp(-d2 / (2.0 * sigma2))
    D = Wm.sum(-1)
    Shat = Wm / torch.sqrt(D.unsqueeze(1) * D.unsqueeze(0) + 1e-8)
    beta = 0.5 * torch.sigmoid(theta_beta[head].double())
    return (1 - beta) * torch.eye(N, dtype=torch.float64) + beta * Shat, Wm, float(beta)


def report(name, S):
    asym = float((S - S.T).abs().max())
    ev = np.linalg.eigvalsh(S.numpy())
    print(f"  {name:24s} sym_err={asym:.3e}  eig in [{ev.min():.6f}, {ev.max():.6f}]  "
          f"SPD={'YES' if ev.min() > 0 else 'NO'}")
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    args = ap.parse_args()

    cfg = toml.load(args.cfg_path); cfg = _resolve_data_paths(cfg, _ROOT)
    device = validate_device(cfg["device"])
    dcfg = cfg["dataset"]
    split = json.load(open(dcfg["official_split_json"]))
    pool_dir = dcfg["data_root"]
    import os
    has = lambda b: os.path.isfile(os.path.join(pool_dir, os.path.splitext(b)[0] + "__s0.npz"))
    test_names = [os.path.splitext(b)[0] for b in split["test"] if has(b)][:1]
    st = np.load(args.stats)
    ds = AirfransV39Dataset(
        test_names, pool_dir, K=dcfg.get("pool_K", 8), train_augment=False,
        feature_cols=dcfg.get("feature_cols"), target_cols=dcfg.get("target_cols"),
        feat_mean=st["feat_mean"], feat_std=st["feat_std"],
        target_mean=st["target_mean"], target_std=st["target_std"],
        extract_kw=dcfg["test"]["kw"]["extract_kw"],
        cache_dir=dcfg["test"]["kw"].get("cache_dir"), use_bc_injection=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_batch_pde_fn)

    model = build_v57_gated_hodge_vef_model(**cfg["train"]["model"]).to(device.type)
    ck = torch.load(args.ckpt, map_location=device.type, weights_only=False)
    model.load_state_dict(ck["state"] if "state" in ck else ck["model_state_dict"])
    model.eval()

    cap = {}
    def mk(i):
        def hook(mod, inp, out):
            cap[i] = [t.detach().cpu() if torch.is_tensor(t) else t for t in inp]
        return hook
    for i, blk in enumerate(model.hodge):
        blk.attn.register_forward_hook(mk(i))

    tensors, _ = next(iter(loader))
    with torch.no_grad():
        model(*[t.to(device.type) for t in tensors], mask=None)

    for li, blk in enumerate(model.hodge):
        attn = blk.attn
        x_v, x_e, d_0, v_idx, e_idx = cap[li]
        if v_idx.dim() == 3: v_idx = v_idx[0]
        if e_idx.dim() == 3: e_idx = e_idx[0]
        h, p = attn.h, attn.p
        b = x_v.shape[0]
        dev = next(attn.parameters()).device
        v_phi_inv = attn.v_phi_inv(x_v.to(dev)).view(b, -1, h, p).transpose(1, 2).detach().cpu()
        e_phi = attn.e_phi(x_e.to(dev)).view(b, -1, h, p).transpose(1, 2).detach().cpu()
        print(f"\nlayer {li}:  N_v={x_v.shape[1]}  N_e={x_e.shape[1]}  "
              f"window_v={v_idx.shape[1]} window_e={e_idx.shape[1]}")
        for head in range(1):                       # head 0 is representative; all share the form
            S1, W1, b1 = dense_star(e_phi, e_idx, attn.e_log_sigma.detach().cpu(),
                                    attn.e_theta_beta.detach().cpu(), head)
            S0, W0, b0 = dense_star(v_phi_inv, v_idx, attn.v_log_sigma.detach().cpu(),
                                    attn.v_theta_beta.detach().cpu(), head)
            print(f"  head {head}:  beta_edge={b1:.4f}  beta_vert={b0:.4f}")
            report("star_1 (edge)", S1)
            report("star_0inv (vertex)", S0)
            # support symmetry actually used by the kernel
            print(f"  {'W_edge support sym':24s} err={float((W1 - W1.T).abs().max()):.3e}")
            print(f"  {'W_vert support sym':24s} err={float((W0 - W0.T).abs().max()):.3e}")
            # assembled L_v = *0^-1 d0^T *1 d0
            D0 = d_0[0].to_dense().double() if d_0[0].is_sparse else d_0[0].double()
            L = S0 @ D0.T @ S1 @ D0
            ev = np.linalg.eigvals(L.numpy())
            print(f"  {'L_v spectrum':24s} max|Im|={np.abs(ev.imag).max():.3e}  "
                  f"Re in [{ev.real.min():.6e}, {ev.real.max():.4f}]  "
                  f"neg_eigs={int((ev.real < -1e-8).sum())}")
            # congruence-symmetrised form, the object that is provably PSD
            w, V = np.linalg.eigh(S0.numpy())
            S0_mhalf = V @ np.diag(w ** -0.5) @ V.T
            Lsym = S0_mhalf @ (D0.T @ S1 @ D0).numpy() @ S0_mhalf
            evs = np.linalg.eigvalsh((Lsym + Lsym.T) / 2)
            print(f"  {'L_v (congruence-sym)':24s} eig in [{evs.min():.6e}, {evs.max():.4f}]")


if __name__ == "__main__":
    main()
