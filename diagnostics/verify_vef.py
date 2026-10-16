#!/usr/bin/env python
"""v57_gated_hodge_vef — is `modes = "vef"` actually doing anything?

In v57 the flag was inert: `build_..._model` accepted `modes` and never passed it on, and
PSDHodgeLayer only ever ran the vertex operator. This case makes it real (L₀ on vertices, L₁ on
edges, L₂ on faces) so the flag has to be checked, not assumed. Four tests on one real mesh:

  1. BUILD      — parameter count, and modes="v" vs "vef" differ by exactly the e/f operators.
  2. FORWARD    — finite output; gate rows are a simplex (>=0, sum 1).
  3. LIVE WIRING— perturb the FACE features x_f and watch the gate move. Under modes="v" the face
                  stream only ever hits an FFN whose output nothing reads, so the gate cannot
                  change; under "vef" it reaches the gate through  x_f → L₁(edges) → L₀(vertices)
                  across two layers. A zero response here means the mode is not wired.
  4. FOOTPRINT  — real training-mode memory (forward + backward, fp32).

    python diagnostics/verify_vef.py
"""
# __deployment_path_bootstrap__  (self-contained: only the deployment root and shared/)
import os as _o, sys as _s, glob as _g
_ROOT = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))    # deployment root
for _d in (_o.path.join(_ROOT, "shared"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)
from paths import resolve_data_paths as _resolve_data_paths

import json, copy, numpy as np, torch, toml
from arch.model import build_v57_gated_hodge_vef_model as build
from data.dataset import AirfransV39Dataset
from mesh_opformer.dataset import collate_batch_pde_fn

HERE = _ROOT
cfg = toml.load(_o.path.join(_ROOT, "config.toml")); cfg = _resolve_data_paths(cfg, _ROOT)
mcfg = cfg["train"]["model"]; dcfg = cfg["dataset"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. build ─────────────────────────────────────────────────────────────────────────────────
m_vef = build(**mcfg).to(dev)
mcfg_v = copy.deepcopy(mcfg); mcfg_v["modes"] = "v"; mcfg_v["hodge_gate_kw"]["modes"] = "v"
m_v = build(**mcfg_v).to(dev)
n_vef = sum(p.numel() for p in m_vef.parameters())
n_v = sum(p.numel() for p in m_v.parameters())
print(f"[1] BUILD    modes='vef' : {n_vef:,} params")
print(f"             modes='v'   : {n_v:,} params   (delta = {n_vef - n_v:,} = the e/f operators)")
gate = m_vef.hodge_gate
print(f"             per layer: v_attn={gate.hodge[0].attn is not None}  "
      f"e_attn={gate.hodge[0].e_attn is not None}  f_attn={gate.hodge[0].f_attn is not None}")
assert n_vef > n_v, "modes='vef' added no parameters — the flag is still inert"

# ── one real mesh ────────────────────────────────────────────────────────────────────────────
st = np.load(_o.path.join(_ROOT, "checkpoints", "norm_stats.npz"))
split = json.load(open(dcfg["official_split_json"])); pool = dcfg["data_root"]
has = lambda b: _o.path.isfile(_o.path.join(pool, _o.path.splitext(b)[0] + "__s0.npz"))
names = [_o.path.splitext(b)[0] for b in split["test"] if has(b)][:1]
ds = AirfransV39Dataset(names, pool, K=dcfg.get("pool_K", 8), train_augment=False,
                        feature_cols=dcfg.get("feature_cols"), target_cols=dcfg.get("target_cols"),
                        feat_mean=st["feat_mean"], feat_std=st["feat_std"],
                        target_mean=st["target_mean"], target_std=st["target_std"],
                        extract_kw=dcfg["test"]["kw"]["extract_kw"],
                        cache_dir=dcfg["test"]["kw"].get("cache_dir"))
tensors, tgt = ds[0]
bt, _ = collate_batch_pde_fn([(tensors, tgt)])
T = [t.to(dev) if torch.is_tensor(t) else t for t in bt]
shapes = {n: tuple(t.shape) for n, t in zip(("x_v", "x_e", "x_f", "d_0", "d_1", "v_idx", "e_idx", "f_idx"), T)}

# ── 2. forward ───────────────────────────────────────────────────────────────────────────────
m_vef.eval()
with torch.no_grad():
    y = m_vef(*T, mask=None).float()
g = m_vef.aux["gate"].float()
print(f"\n[2] FORWARD  {shapes}")
print(f"             out {tuple(y.shape)}  finite={bool(torch.isfinite(y).all())}  "
      f"|y| mean {y.abs().mean():.4f}")
print(f"             gate {tuple(g.shape)}  min {g.min():.2e}  rowsum in "
      f"[{g.sum(-1).min():.6f}, {g.sum(-1).max():.6f}]  entropy {-(g*g.clamp_min(1e-9).log()).sum(-1).mean():.3f}"
      f" / {np.log(g.shape[-1]):.3f}")
assert torch.isfinite(y).all() and g.min() >= 0 and abs(float(g.sum(-1).mean()) - 1) < 1e-3

# ── 3. is the face stream live? ──────────────────────────────────────────────────────────────
def gate_of(model, x_f):
    args = list(T); args[2] = x_f
    with torch.no_grad():
        model(*args, mask=None)
    return model.aux["gate"].float()

torch.manual_seed(0)
x_f2 = T[2] + 0.5 * torch.randn_like(T[2])
print("\n[3] LIVE WIRING  (perturb x_f by N(0,0.5); mean |dgate| per vertex)")
for tag, mdl in (("modes='v'  ", m_v), ("modes='vef'", m_vef)):
    mdl.eval()
    d = (gate_of(mdl, x_f2) - gate_of(mdl, T[2])).abs().mean()
    print(f"      {tag} : {float(d):.3e}")
d_vef = float((gate_of(m_vef, x_f2) - gate_of(m_vef, T[2])).abs().mean())
d_v_ = float((gate_of(m_v, x_f2) - gate_of(m_v, T[2])).abs().mean())
assert d_vef > 1e-6 and d_v_ < 1e-9, f"face stream not wired into the gate (vef {d_vef}, v {d_v_})"
print("      -> faces reach the gate only under 'vef', as intended")

# ── 4. training footprint ────────────────────────────────────────────────────────────────────
if dev == "cuda":
    del m_v; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m_vef.train()
    loss = m_vef(*T, mask=None).float().pow(2).mean()
    loss.backward()
    print(f"\n[4] FOOTPRINT  peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
          f"(fwd+bwd, fp32, N_v={shapes['x_v'][1]}, N_e={shapes['x_e'][1]}, N_f={shapes['x_f'][1]})")

print("\nALL CHECKS PASSED")
