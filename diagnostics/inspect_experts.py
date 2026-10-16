#!/usr/bin/env python
"""v57_gated_hodge_vef — did the experts actually specialise, and did the gate stay balanced?

A frequency MoE can reach a good field loss while quietly degenerating: the gate collapses onto one
expert (v57_gated_hodge_vef becomes a narrower v51), or the experts ignore their bands and duplicate each other. The
training log's `band` / `bal` terms answer that going forward, but for a run that predates the
metrics writer this recovers the same information from a checkpoint.

Reports, over `--n` test meshes:
  * gate mass per expert, and gate entropy (log K = perfectly mixed, 0 = collapsed)
  * gate mass on WALL vs interior vertices — does the high-frequency expert take the wall?
  * per-expert band error: how well expert k predicts band k, vs the OTHER bands (a specialised
    expert is much better on its own band than on the others)

  python diagnostics/inspect_experts.py --ckpt runs/v57_gated_hodge_vef/..._ckpt_400.pth --stats runs/v57_gated_hodge_vef/airfrans_norm_stats.npz
"""
# __deployment_path_bootstrap__  (self-contained: only the deployment root and shared/)
import os as _o, sys as _s, glob as _g
_ROOT = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))    # deployment root
for _d in (_o.path.join(_ROOT, "shared"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)
from paths import resolve_data_paths as _resolve_data_paths
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))

import json, argparse, numpy as np, torch, toml
from data.dataset import AirfransV39Dataset
from arch.model import build_v57_gated_hodge_vef_model
from training.losses import split_bands, split_cl
from mesh_opformer.dataset import collate_batch_pde_fn
from mesh_opformer.training.utils import validate_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_path", default=_o.path.join(_ROOT, "config.toml"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None, help="write the summary JSON here (default: ckpt's dir)")
    args = ap.parse_args()

    cfg = toml.load(args.cfg_path); cfg = _resolve_data_paths(cfg, _ROOT)
    dev = validate_device(cfg["device"])
    C = int(cfg["train"]["model"]["out_channels"])
    K = int(cfg["train"]["model"]["moe_kw"]["n_experts"])
    dcfg = cfg["dataset"]

    split = json.load(open(dcfg["official_split_json"]))
    pool = dcfg["data_root"]
    has = lambda b: _o.path.isfile(_o.path.join(pool, _o.path.splitext(b)[0] + "__s0.npz"))
    test_names = [_o.path.splitext(b)[0] for b in split["test"] if has(b)][:args.n]

    st = np.load(args.stats)
    ds = AirfransV39Dataset(test_names, pool, K=dcfg.get("pool_K", 8), train_augment=False,
                            feature_cols=dcfg.get("feature_cols"), target_cols=dcfg.get("target_cols"),
                            feat_mean=st["feat_mean"], feat_std=st["feat_std"],
                            target_mean=st["target_mean"], target_std=st["target_std"],
                            extract_kw=dcfg["test"]["kw"]["extract_kw"],
                            cache_dir=dcfg["test"]["kw"].get("cache_dir"),
                            band_kw=dcfg["test"]["kw"].get("band_kw"))

    model = build_v57_gated_hodge_vef_model(**cfg["train"]["model"]).to(dev.type)
    ck = torch.load(args.ckpt, map_location=dev.type, weights_only=False)
    model.load_state_dict(ck["state"] if "state" in ck else ck["model_state_dict"])
    model.eval()

    mass = np.zeros(K); mass_wall = np.zeros(K); mass_int = np.zeros(K)
    ent = 0.0; err = np.zeros((K, K)); n_seen = 0
    for i in range(len(test_names)):
        tensors, tgt = ds[i]
        bt, by = collate_batch_pde_fn([(tensors, tgt)])
        bt = [t.to(dev.type) for t in bt]; by = by.to(dev.type)
        with torch.no_grad():
            model(*bt, mask=None)
        g = model.aux["gate"].float()[0]                       # (N,K)
        bp = model.aux["bands"].float()[0]                     # (N,C,K)
        bt_true = split_bands(by, cfg).float()[0]              # (N,C,K)
        wall = (bt[0][0, :, 8] > 0.5)

        mass += g.mean(0).cpu().numpy()
        mass_wall += g[wall].mean(0).cpu().numpy()
        mass_int += g[~wall].mean(0).cpu().numpy()
        ent += float((-(g * (g + 1e-12).log()).sum(-1)).mean().cpu())
        for k in range(K):                                     # expert k vs every band j
            for j in range(K):
                err[k, j] += float(((bp[..., k] - bt_true[..., j]) ** 2).mean().cpu())
        n_seen += 1

    mass /= n_seen; mass_wall /= n_seen; mass_int /= n_seen; ent /= n_seen; err /= n_seen
    print(f"v57_gated_hodge_vef expert inspection over {n_seen} test meshes — {_o.path.basename(args.ckpt)}\n")
    print(f"gate entropy {ent:.3f} nats (max {np.log(K):.3f} = perfectly mixed, 0 = collapsed)")
    print(f"{'expert':8s} {'gate mass':>10s} {'on wall':>9s} {'interior':>9s}  {'own band err':>13s} "
          f"{'best band':>10s}")
    for k in range(K):
        print(f"  {k:<6d} {mass[k]:10.3f} {mass_wall[k]:9.3f} {mass_int[k]:9.3f}  "
              f"{err[k, k]:13.5f} {int(np.argmin(err[k])):10d}")
    spec = int(sum(np.argmin(err[k]) == k for k in range(K)))
    print(f"\nexperts whose OWN band is their best-fit band: {spec}/{K}")
    print("band-error matrix (row = expert, col = band; diagonal should be the row minimum):")
    for k in range(K):
        print("   " + "  ".join(f"{err[k, j]:9.5f}" for j in range(K)))

    out = args.out or _o.path.join(_o.path.dirname(args.ckpt), "expert_stats.json")
    json.dump(dict(ckpt=_o.path.basename(args.ckpt), n_meshes=n_seen, gate_entropy=ent,
                   gate_entropy_max=float(np.log(K)), gate_mass=mass.tolist(),
                   gate_mass_wall=mass_wall.tolist(), gate_mass_interior=mass_int.tolist(),
                   band_err=err.tolist(), n_specialised=spec, n_experts=K),
              open(out, "w"), indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
