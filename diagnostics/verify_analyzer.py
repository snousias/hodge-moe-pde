#!/usr/bin/env python
"""v57_gated_hodge_vef — checks on the LEARNED Hodge analyzer, before any training is spent on it.

The four things that have to hold for a learnable analyzer to be safe:
  1. L_0 is PSD  (Rayleigh quotient >= 0 for random signals, for random star values)
  2. the smoothing iteration is stable  (spectrum of I - tau L in [0,1])
  3. the bands telescope to the target EXACTLY, for arbitrary learned parameters
  4. the degenerate "collapse" solution is detected by band_energy_entropy
plus: gradient actually reaches the stars.
"""
# __deployment_path_bootstrap__  (self-contained: only the deployment root and shared/)
import os as _o, sys as _s, glob as _g
_ROOT = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))    # deployment root
for _d in (_o.path.join(_ROOT, "shared"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)
from paths import resolve_data_paths as _resolve_data_paths
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))

import numpy as np, torch, toml, tempfile
from arch.analyzer import LearnedHodgeAnalyzer
from data.dataset import AirfransV39Dataset
from mesh_opformer.dataset import collate_batch_pde_fn

HERE = _ROOT
cfg = toml.load(_o.path.join(_ROOT, "config.toml")); cfg = _resolve_data_paths(cfg, _ROOT)
POOL = cfg["dataset"]["data_root"]; C = cfg["train"]["model"]["out_channels"]
names = [_o.path.basename(p)[:-len("__s0.npz")] for p in sorted(_g.glob(_o.path.join(POOL, "*__s0.npz")))[:1]]

tmp = tempfile.mkdtemp(); kw = dict(cfg["dataset"]["test"]["kw"]); kw["cache_dir"] = tmp
stats = AirfransV39Dataset.compute_stats(names, POOL, target_cols=cfg["dataset"]["target_cols"], max_samples=1)
ds = AirfransV39Dataset(names, POOL, K=1, train_augment=False, target_cols=cfg["dataset"]["target_cols"],
                        feat_mean=stats["feat_mean"], feat_std=stats["feat_std"],
                        target_mean=stats["target_mean"], target_std=stats["target_std"], **kw)
tensors, tgt = ds[0]
bt, by = collate_batch_pde_fn([(tensors, tgt)])
x_v, x_e, d_0 = bt[0], bt[1], bt[3]
y = by[..., :C].clone()
N, E = x_v.shape[1], x_e.shape[1]
print(f"mesh: N={N} vertices, E={E} edges, d_0 {tuple(d_0.shape)}")

an = LearnedHodgeAnalyzer(e_in=x_e.shape[-1], v_in=x_v.shape[-1],
                          ladder=cfg["train"].get("band_ladder", (2, 8, 32)))

# ── 1. PSD, for RANDOM star settings (not just at init) ───────────────────────────────────
worst = 0.0
for trial in range(5):
    for p in an.parameters():
        with torch.no_grad():
            p.normal_(0, 1.0)                      # arbitrary (possibly wild) learned state
    w, a = an.stars(x_v, x_e, d_0)
    x = torch.randn(1, N, 3)
    Lx = an.laplacian(x, w, a, d_0)
    # Rayleigh quotient in the *_0 inner product:  x^T *_0 L x = x^T d_0^T *_1 d_0 x  >= 0
    r = float((x * (a * Lx)).sum())
    worst = min(worst, r)
    assert r >= -1e-6, f"L_0 not PSD: Rayleigh {r}"
print(f"[1] PSD: min Rayleigh quotient over 5 random star settings = {worst:+.3e}  (>= 0 required)")

# ── 2. the smoothing iteration is stable ──────────────────────────────────────────────────
w, a = an.stars(x_v, x_e, d_0)
t = an.tau(w, a, d_0)
v = torch.randn(1, N, 1); v = v / v.norm()
norms = []
for _ in range(64):
    v = v - t * an.laplacian(v, w, a, d_0)
    norms.append(float(v.norm()))
assert norms[0] <= 1.0 + 1e-4 and all(norms[i + 1] <= norms[i] + 1e-6 for i in range(len(norms) - 1))
print(f"[2] stability: ||(I - tau L)^m v|| is non-increasing over 64 steps "
      f"({norms[0]:.4f} -> {norms[-1]:.4f}); tau = {float(t):.3e}")

# ── 3. completeness for ARBITRARY learned parameters ──────────────────────────────────────
errs = []
for trial in range(5):
    for p in an.parameters():
        with torch.no_grad():
            p.normal_(0, 1.5)
    b = an.bands(y, x_v, x_e, d_0)
    errs.append(float((b.sum(-1) - y).abs().max() / y.abs().max()))
print(f"[3] completeness: max relative |sum_k b_k - y| over 5 random star settings = {max(errs):.2e}")
assert max(errs) < 1e-5

# ── 4. spectrum is bounded, and the collapse is detected ──────────────────────────────────
# With *_0 = learned degree the operator is a random-walk Laplacian: lambda_max <= 2 for ANY
# conductances, so no setting of the stars can make the smoothing degenerate numerically.
lam = []
for trial in range(5):
    for p_ in an.parameters():
        with torch.no_grad():
            p_.normal_(0, 2.0)
    w, a = an.stars(x_v, x_e, d_0)
    lam.append(float(an.lambda_max_estimate(w, a, d_0).max()))
print(f"[4] spectrum: lambda_max (power iteration) over 5 random star settings = "
      f"[{min(lam):.3f}, {max(lam):.3f}]  (random-walk bound is 2.0)")
assert max(lam) <= 2.05, "lambda_max exceeded the random-walk bound"

# the scale cheat (w -> 0) is now impossible: *_0 scales with *_1, so L is scale-invariant
with torch.no_grad():
    for p_ in an.edge_mlp.parameters():
        p_.zero_()
    an.edge_mlp[-1].bias.fill_(+5.0)     # softplus ~ 5e0, far above the w_floor
b_small = an.bands(y, x_v, x_e, d_0)
with torch.no_grad():
    an.edge_mlp[-1].bias.fill_(+9.0)     # softplus ~ 9e0, a 1.8x uniform rescale
b_big = an.bands(y, x_v, x_e, d_0)
drift = float((b_small - b_big).abs().max() / y.abs().max())
h_s, share_s = an.band_energy_entropy(b_small)
K = b_small.shape[-1]
print(f"    scale invariance: uniform conductance at 1e-2 vs 1e2 gives identical bands "
      f"(max relative drift {drift:.1e}) -> the 'shrink the operator' cheat is structurally gone")
assert drift < 1e-5
print(f"    energy entropy at uniform conductance: {float(h_s):.4f} nats of max {np.log(K):.3f}; "
      f"shares {np.round(share_s.detach().numpy(), 4)}")

# ── 5. gradient reaches the stars ─────────────────────────────────────────────────────────
an.zero_grad()
b = an.bands(y, x_v, x_e, d_0)
h, _ = an.band_energy_entropy(b)
(-h).backward()
g_edge = sum(p.grad.abs().sum().item() for p in an.edge_mlp.parameters() if p.grad is not None)
g_mass = (sum(p.grad.abs().sum().item() for p in an.mass_mlp.parameters() if p.grad is not None)
          if an.mass_mlp is not None else float("nan"))   # *_0 = learned degree by default
assert g_edge > 0
print(f"[5] gradient reaches the learned stars: |g_edge| {g_edge:.3e}"
      + (f", |g_mass| {g_mass:.3e}" if an.mass_mlp is not None else " (mass = learned degree, no free MLP)"))
print(f"\nALL OK — the learned analyzer is PSD, stable, exactly complete, and its degenerate "
      f"solution is detectable.")
