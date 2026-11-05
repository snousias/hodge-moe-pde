# ========================================================================================
# HELPER for data/dataset.py — the lift functional as a per-vertex weight vector.
# ========================================================================================
#!/usr/bin/env python
"""v57_gated_hodge_vef — the lift coefficient as a LINEAR functional of the predicted wall pressure.

`C_L` is a signed integral of pressure over the airfoil surface, so it is sensitive to *spatially
correlated* error in a way pointwise MSE is not: scattering +/-eps along the wall cancels in the
integral, biasing one side does not. That is exactly what v51 did — its wall-pressure MSE beat v50
by 27% while its C_L got 42% worse. No amount of re-weighting the volume/surface MSE terms can see
that, because neither term contains the integral.

The fix is to put the integral in the loss. It is cheap because **C_L is linear in p**: everything
else in the pressure integral (wall loop order, edge lengths, normals, the lift-direction rotation)
is fixed by the mesh. Writing the eval's `mesh_cl` out,

    C_L = sum_k  0.5 (p_{q_k} + p_{q_{k+1}}) * L_k * (basis[1] . N_mid_k) / (0.5 |U|^2)
        = sum_j  p_{q_j} * (a_j + a_{j-1})            with  a_k = 0.5 L_k (basis[1].N_mid_k)/(0.5|U|^2)

so a single per-vertex weight vector `w` (zero off the wall) gives `C_L = w . p_physical` exactly —
one dot product, differentiable, no ordering inside the graph.

`cl_weights` returns that vector already folded through the target normalisation, so the loss can
consume the model's z-scored output directly:

    p_phys = (p_z * std_p + mean_p) * U^2
    C_L    = (U^2 std_p) (w . p_z)  +  (U^2 mean_p) (sum w)
           =  w' . p_z + b

Verified against the eval's `mesh_cl` to ~1e-12 relative (see verify_v57_gated_hodge_vef.py).
"""
import numpy as np


def order_wall_loop(P):
    """Permutation of wall-node indices forming a simple airfoil loop (camber split).
    Copied verbatim from eval_cl_native_* so the training loss integrates the SAME contour the
    evaluation metric does."""
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


def cl_weights(pos, wall, normals, inlet, n_vert=None):
    """Per-vertex weights w with `C_L = w . p_physical` (w = 0 off the wall).

    pos (N,2), wall (N,) bool/0-1, normals (N,2) wall outward normals, inlet (2,) freestream.
    """
    n_vert = int(n_vert or pos.shape[0])
    w = np.zeros(n_vert, dtype="float64")
    widx = np.flatnonzero(np.asarray(wall).reshape(-1) > 0.5)
    if widx.size < 3:
        return w
    P = pos[widx].astype("float64"); Nw = normals[widx].astype("float64")
    order = order_wall_loop(P)
    q = widx[order]                                  # global ids, in loop order
    P = P[order]; Nw = Nw[order]
    Nw = Nw / (np.linalg.norm(Nw, axis=1, keepdims=True) + 1e-12)

    L = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    N_mid = 0.5 * (Nw + np.roll(Nw, -1, axis=0))
    N_mid = N_mid / (np.linalg.norm(N_mid, axis=1, keepdims=True) + 1e-12)

    v = float(np.linalg.norm(inlet)); aoa = float(np.arctan2(inlet[1], inlet[0]))
    lift_dir = np.array([-np.sin(aoa), np.cos(aoa)])   # basis row 1: the lift direction
    a = 0.5 * L * (N_mid @ lift_dir) / (0.5 * v ** 2)  # per-edge coefficient
    np.add.at(w, q, a + np.roll(a, 1))                 # each node gets its two incident edges
    return w


def cl_weights_zscored(pos, wall, normals, inlet, p_mean, p_std, n_vert=None):
    """Fold the target normalisation in: returns (w_prime, b) with `C_L = w' . p_zscored + b`.

    p_mean / p_std are the pressure channel's z-score statistics (on the NON-DIMENSIONAL target,
    which is how the datasets store them); U^2 re-dimensionalises.
    """
    w = cl_weights(pos, wall, normals, inlet, n_vert)
    u2 = float(np.linalg.norm(inlet)) ** 2
    return (w * u2 * float(p_std)).astype("float32"), float(u2 * float(p_mean) * w.sum())
