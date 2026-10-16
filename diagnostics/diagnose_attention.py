#!/usr/bin/env python
"""Is v57_gated_hodge_vef actually using its attention, or is it a point-wise model with extra steps?

Only ONE component of v57_gated_hodge_vef mixes information across vertices: the softmax MHA inside each expert
(8 experts x 2 blocks). Embedding, Fourier features, gate and decoder are all per-vertex — the
decoder's cross-field attention runs over the 4 field tokens WITHIN a vertex. So if the MHA does
nothing, v57_gated_hodge_vef is a pointwise MLP.

Two direct tests on a trained checkpoint:

  1. SUBSET CONSISTENCY — predict a vertex with the whole mesh, then with a random subset that
     contains it, down to that vertex ALONE. A pointwise model gives the same answer every time; a
     model that uses context does not. (v57_gated_hodge_vef is a point transformer, so it accepts any vertex subset —
     no mesh tensors are needed for a forward pass.)

  2. ATTENTION ABLATION — skip the attention sub-layer at inference (keep the residual, so the block
     becomes LN->FFN only) and measure the damage on the real metric.

Verdict: if predictions survive a single-vertex forward and attention ablation barely hurts, the
model is effectively pointwise and its SDF/normal inputs are doing the geometric work.

  python diagnostics/diagnose_attention.py --ckpt runs/v57_gated_hodge_vef/..._ckpt_380.pth --stats runs/v57_gated_hodge_vef/airfrans_norm_stats.npz
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
from mesh_opformer.training.utils import validate_device


def main():
    ap = argparse.ArgumentParser()
    HERE = _ROOT
    ap.add_argument("--cfg_path", default=_o.path.join(_ROOT, "config.toml"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--n", type=int, default=12, help="test meshes")
    ap.add_argument("--probes", type=int, default=256, help="probe vertices per mesh")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = toml.load(args.cfg_path); cfg = _resolve_data_paths(cfg, _ROOT); dev = validate_device(cfg["device"])
    C = int(cfg["train"]["model"]["out_channels"]); dcfg = cfg["dataset"]
    split = json.load(open(dcfg["official_split_json"])); pool = dcfg["data_root"]
    has = lambda b: _o.path.isfile(_o.path.join(pool, _o.path.splitext(b)[0] + "__s0.npz"))
    names = [_o.path.splitext(b)[0] for b in split["test"] if has(b)][:args.n]

    st = np.load(args.stats)
    ds = AirfransV39Dataset(names, pool, K=dcfg.get("pool_K", 8), train_augment=False,
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
    rng = np.random.default_rng(0)

    # ── 1. subset consistency ─────────────────────────────────────────────────────────────
    fracs = [1.0, 0.5, 0.25, 0.1, 0.02]
    dev_rel = {f: [] for f in fracs}; dev_single = []
    for i in range(len(names)):
        tensors, _ = ds[i]
        x = tensors[0].unsqueeze(0).to(dev.type)
        N = x.shape[1]
        with torch.no_grad():
            full = model(x).float()[0]                       # (N,C)
        scale = full.std(dim=0).clamp_min(1e-6)              # per-field scale for a relative measure
        probes = rng.choice(N, size=min(args.probes, N), replace=False)
        for f in fracs:
            if f >= 1.0:
                dev_rel[f].append(0.0); continue
            keep = rng.choice(np.setdiff1d(np.arange(N), probes),
                              size=max(1, int(f * N) - probes.size), replace=False)
            idx = np.concatenate([probes, keep])
            with torch.no_grad():
                sub = model(x[:, idx]).float()[0][:probes.size]
            dev_rel[f].append(float(((sub - full[probes]).abs() / scale).mean()))
        # the extreme: each probe vertex entirely alone
        with torch.no_grad():
            alone = torch.cat([model(x[:, [p]]).float()[0] for p in probes[:32]], 0)
        dev_single.append(float(((alone - full[probes[:32]]).abs() / scale).mean()))

    print(f"v57_gated_hodge_vef attention diagnostics — {len(names)} test meshes, {args.probes} probes each")
    print(f"   checkpoint {_o.path.basename(args.ckpt)}\n")
    print("[1] SUBSET CONSISTENCY  (deviation at the SAME vertex when context is removed,")
    print("    in units of that field's own std across the mesh; 0 = perfectly point-wise)")
    for f in fracs:
        print(f"      context = {f*100:5.0f}% of vertices : {np.mean(dev_rel[f]):.4f}")
    print(f"      context = ONE vertex        : {np.mean(dev_single):.4f}")

    # ── 2. attention ablation ─────────────────────────────────────────────────────────────
    import arch.experts as moe_mod
    orig = moe_mod.VanillaTransformerBlock.forward

    def no_attn(self, x):                                    # keep the residual, drop the mixing
        return x + self.drop(self.ff(self.n2(x)))

    def field_err(fn):
        moe_mod.VanillaTransformerBlock.forward = fn
        errs = []
        for i in range(len(names)):
            tensors, tgt = ds[i]
            x = tensors[0].unsqueeze(0).to(dev.type)
            y = tgt[:, :C].unsqueeze(0).to(dev.type).float()
            with torch.no_grad():
                p = model(x).float()
            errs.append(float((p - y).norm() / y.norm()))
        return float(np.mean(errs))

    e_on = field_err(orig)
    e_off = field_err(no_attn)
    moe_mod.VanillaTransformerBlock.forward = orig
    print("\n[2] ATTENTION ABLATION  (skip the MHA sub-layer at inference; z-scored rel-L2)")
    print(f"      attention ON  : {e_on:.4f}")
    print(f"      attention OFF : {e_off:.4f}   ({e_off/max(e_on,1e-9):.1f}x worse)")

    verdict = ("EFFECTIVELY POINT-WISE — context removal barely changes the prediction"
               if np.mean(dev_single) < 0.05 and e_off < 1.5 * e_on else
               "USES CROSS-VERTEX CONTEXT — predictions depend on which other vertices are present")
    print(f"\nVERDICT: {verdict}")

    out = args.out or _o.path.join(_o.path.dirname(args.ckpt), "attention_diagnostics.json")
    json.dump(dict(ckpt=_o.path.basename(args.ckpt), n_meshes=len(names),
                   subset_deviation={str(f): float(np.mean(dev_rel[f])) for f in fracs},
                   single_vertex_deviation=float(np.mean(dev_single)),
                   relL2_attention_on=e_on, relL2_attention_off=e_off, verdict=verdict),
              open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
