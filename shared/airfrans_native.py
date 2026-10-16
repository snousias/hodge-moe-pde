#!/usr/bin/env python
"""
AirFRANS-native evaluation metrics, matching the Transolver / AirFRANS benchmark
(Bonnet et al.; thuml/Transolver Airfoil-Design-AirfRANS/utils/metrics.py):

  * VOLUME / SURF: mean-squared error on the NORMALISED (z-scored) fields, split by
    the wall mask (surface = wall nodes, volume = the rest), averaged over the 4 fields
    then over samples. This is the benchmark's headline "Volume" / "Surf" columns.
  * C_L / rho_L: lift-coefficient RELATIVE error and Spearman rank-correlation across
    the test set. Computed with the airfrans library's own force integrator: overwrite
    a Simulation's velocity+pressure with our de-normalised predictions and call
    force_coefficient() (it recomputes wall-shear-stress from the velocity gradient and
    integrates pressure+viscous forces over the airfoil), vs the true fields.
"""
import numpy as np


# ── volume / surface normalised MSE ───────────────────────────────────────────

def vol_surf_mse(pred_phys, targ_phys, wall, std_glob):
    """Per-field node-MSE on AirFRANS-normalised fields, split volume (~wall) / surface.

    Matches the benchmark (Transolver metrics.py): the fields are normalised by the
    GLOBAL per-field std of the RAW physical targets over the training set (coef_norm
    std_out) — NOT our freestream+z-score. So each field's error is measured as
    ((pred_phys - targ_phys) / std_glob)^2. For u_x the global std spans the freestream
    range, which is what makes the benchmark's 'Volume' number small.

    pred_phys, targ_phys: (N, C) PHYSICAL prediction/target. wall: (N,) bool.
    std_glob: (C,) global per-field physical std. Returns (vol_per_field, surf_per_field).
    """
    d = (pred_phys.astype("float64") - targ_phys.astype("float64")) / (np.asarray(std_glob) + 1e-12)
    se = d ** 2
    w = wall.astype(bool)
    vol = se[~w].mean(axis=0) if (~w).any() else np.full(se.shape[1], np.nan)
    surf = se[w].mean(axis=0) if w.any() else np.full(se.shape[1], np.nan)
    return vol, surf


def global_target_std(train_paths):
    """AirFRANS coef_norm std_out: global per-field std of RAW physical targets over train.

    Cached to <dataset_dir>/_global_target_std.npy (keyed by the dataset folder) so a
    12-checkpoint sweep doesn't reload all 800 training .npz every time.
    """
    import os
    cache = os.path.join(os.path.dirname(train_paths[0]), "_global_target_std.npy")
    if os.path.isfile(cache):
        return np.load(cache)
    acc = [np.load(p)["y"].astype("float64") for p in train_paths]
    std = np.concatenate(acc, axis=0).std(axis=0)
    try:
        np.save(cache, std)
    except Exception:
        pass
    return std


# ── lift coefficient via the airfrans force integrator ────────────────────────

def lift_true_pred(name, our_pos, ux, uy, p, root):
    """(cl_true, cl_pred) for one simulation.

    our_pos: (M,2) our mesh vertex positions. ux, uy, p: (M,) PHYSICAL predictions on
    those vertices. Predictions are interpolated onto the raw CFD nodes, injected into a
    Simulation, and its force_coefficient() lift is compared to the true-field lift.
    """
    import airfrans as af
    from scipy.interpolate import griddata

    sim_t = af.Simulation(root=root, name=name)
    cl_true = float(sim_t.force_coefficient()[1][0])          # (cl, clp, clv) -> cl

    P = sim_t.position                                        # (Nfull, 2)
    def interp(v):
        out = griddata(our_pos, v, P, method="linear")
        nan = np.isnan(out)
        if nan.any():                                         # far-field points outside our hull
            out[nan] = griddata(our_pos, v, P[nan], method="nearest")
        return out.astype("float64")

    sim_p = af.Simulation(root=root, name=name)
    sim_p.velocity = np.column_stack([interp(ux), interp(uy)])
    sim_p.pressure = interp(p).reshape(-1, 1)
    cl_pred = float(sim_p.force_coefficient()[1][0])
    return cl_true, cl_pred


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


# ── raw-mesh back-projection (v21 evaluation) ─────────────────────────────────

def backproject(src_pos, vals, dst_pos):
    """Interpolate multi-field values from src nodes onto dst nodes (linear + nearest).

    src_pos: (n,2) source (remesh) nodes. vals: (n,C) fields on them. dst_pos: (m,2) target
    (raw CFD) nodes. Builds ONE Delaunay for all C fields (LinearNDInterpolator), then fills
    dst points outside the source hull (far field) with nearest-neighbour. Returns (m,C).
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    lin = LinearNDInterpolator(src_pos, vals)
    out = np.asarray(lin(dst_pos), dtype="float64")
    if out.ndim == 1:
        out = out[:, None]
    nan = np.isnan(out).any(axis=1)
    if nan.any():
        out[nan] = NearestNDInterpolator(src_pos, vals)(dst_pos[nan])
    return out


def build_backproject_operator(src_pos, dst_pos):
    """Sparse (m x n) linear-interpolation operator W: W @ vals(n,C) = vals interpolated
    from the n source (remesh) nodes onto the m target (raw) nodes.

    Barycentric weights from the source Delaunay for points inside the hull (3 nonzeros/row);
    nearest-neighbour (1 nonzero/row) for points outside (far field). Reproduces
    LinearNDInterpolator + nearest fallback, but as a fixed matrix — so every checkpoint's
    back-projection is one sparse matmul (the expensive find_simplex is paid once and cached).
    """
    from scipy.spatial import Delaunay, cKDTree
    from scipy.sparse import csr_matrix
    src = np.asarray(src_pos, dtype="float64"); dst = np.asarray(dst_pos, dtype="float64")
    m = len(dst); n = len(src)
    tri = Delaunay(src)
    s = tri.find_simplex(dst)
    inside = s >= 0
    rows = []; cols = []; data = []
    # inside: barycentric interpolation over the containing simplex
    si = s[inside]
    T = tri.transform[si]                                   # (k, 3, 2)
    d = dst[inside] - T[:, 2, :]                            # (k, 2)
    b01 = np.einsum("kij,kj->ki", T[:, :2, :], d)           # (k, 2)
    bary = np.column_stack([b01, 1.0 - b01.sum(axis=1)])    # (k, 3)
    verts = tri.simplices[si]                               # (k, 3)
    ridx = np.where(inside)[0]
    for j in range(3):
        rows.append(ridx); cols.append(verts[:, j]); data.append(bary[:, j])
    # outside: nearest source node
    out = ~inside
    if out.any():
        nn = cKDTree(src).query(dst[out])[1]
        rows.append(np.where(out)[0]); cols.append(nn); data.append(np.ones(out.sum()))
    W = csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(m, n))
    return W


def load_or_build_bp_operator(name, src_pos, dst_pos, cache_dir):
    """Cache the sparse back-projection operator per sample (built once, reused across
    checkpoints/versions that share the same remesh)."""
    import os
    from scipy.sparse import csr_matrix, save_npz, load_npz
    W = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        f = os.path.join(cache_dir, name + ".npz")
        if os.path.isfile(f):
            try:
                W = load_npz(f)
            except Exception:
                W = None
        if W is None:
            W = build_backproject_operator(src_pos, dst_pos)
            try:
                save_npz(f, W)
            except Exception:
                pass
            return W
        return W
    return build_backproject_operator(src_pos, dst_pos)


def raw_global_target_std(train_names, rawdir, cache_path=None):
    """AirFRANS coef_norm std_out on the RAW mesh: streaming global per-field std of the raw
    physical targets over the training set (no concatenation — sum/sumsq accumulation)."""
    import os
    from mesh_opformer.dataset.dataset_airfrans import load_airfrans_graph
    if cache_path and os.path.isfile(cache_path):
        return np.load(cache_path)
    s = np.zeros(4); ss = np.zeros(4); n = 0
    for name in train_names:
        y = load_airfrans_graph(os.path.join(rawdir, name + ".pth"))[2][:, :4].astype("float64")
        s += y.sum(0); ss += (y ** 2).sum(0); n += len(y)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
    if cache_path:
        try:
            np.save(cache_path, std)
        except Exception:
            pass
    return std
