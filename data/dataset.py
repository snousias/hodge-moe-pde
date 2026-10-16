# ========================================================================================
# DATA ENTRY POINT
#   AirfransV39Dataset(...)  — the only class training/ and evaluation/ construct
# ========================================================================================
#!/usr/bin/env python
"""
v33 dataset — SELF-CONTAINED (consolidated).

This flattens the full v13→v14→v16→v18_3→v18_6→v22_geodesic inheritance chain into
ONE file so v33 imports nothing from sibling case dirs (previously it reached across
into cases/v18_6, cases/v18_3, cases/v14, ... via the sys.path bootstrap). Behaviour is
byte-for-byte the same as `AirfransV22GeodesicDataset` was for v33; only the packaging
changed. It depends solely on the stable packages (mesh_sim, mesh_opformer,
mesh_opformer_geodesic).

What it does (the v33 data pipeline):
  * DATA: v18_3 augmented mesh POOL (K=8 gmsh anisotropic-BL meshes per airfoil). Train
    picks a RANDOM pool mesh per __getitem__; test uses the fixed s0 mesh.
  * INPUTS (v_in=8): positions(2) + per-vertex dual AREA(1) + x_feat(5)=[inlet_u,inlet_v,
    SDF,n_x,n_y].
  * TARGETS: freestream non-dim (u/U∞, p/U∞², ν_t/U∞) THEN per-field z-score. Eval decodes
    with redim_targets.
  * GEOMETRY: geodesic-ring stratified neighbourhoods (window 128, hops=8) built on the
    unit-topology mesh; optional metric (length-weighted) Laplacian via
    extract_kw["laplacian_weight"] ∈ {unit,length,inv_length,sqrt_length}. Cached per mesh.

ROBUSTNESS: the geometry-cache loader is self-healing — a corrupt/truncated .npz (e.g. a
partial write left by a disk-full crash) is caught, deleted, and rebuilt, instead of
crashing the DataLoader worker (which is what happened after the earlier out-of-space event).
"""
import os
import zlib
import glob
import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset

from mesh_sim.mesh import MeshSignedIncidenceMatrices
from mesh_sim.utils_ml import extract_mesh_features
from mesh_opformer.dataset.utils import sparse_scipy_to_torch
from data.cl_weights import cl_weights
from arch.experts import band_decompose
from mesh_opformer_geodesic import (extract_geodesic_ring_neighborhoods,
                                    extract_metric_ring_neighborhoods, length_weighted_incidence)
try:  # opt-in fused C geodesic BFS+select (mesh_bfs_c, option 6). Default OFF -> Python path unchanged.
    from mesh_bfs_c import extract_geodesic_ring_neighborhoods_fast as _extract_geodesic_fast_c
except Exception:  # package not built -> silently fall back to the reference extractor
    _extract_geodesic_fast_c = None


def _select_geodesic_extractor(gk):
    """Pick the geodesic-ring extractor. Returns the fused C fast path iff opted-in via config
    ``[dataset...geodesic] fast_c = true`` OR env ``MESH_BFS_C=1``; else the reference Python/graphblas
    extractor. Both return the identical dict format. NOTE: switching the flag only affects cache
    MISSES — clear the .geo cache dir to rebuild existing meshes with the other path."""
    use_fast = bool(gk.get("fast_c", False)) or os.environ.get("MESH_BFS_C") == "1"
    if use_fast and _extract_geodesic_fast_c is not None:
        return _extract_geodesic_fast_c
    return extract_geodesic_ring_neighborhoods


try:  # opt-in fused C Dijkstra + metric-ring select (mesh_bfs_c). Default OFF -> Python reference path.
    from mesh_bfs_c import extract_metric_ring_neighborhoods_fast as _extract_metric_fast_c
except Exception:
    _extract_metric_fast_c = None


def _select_metric_extractor(gk):
    """Fast C metric extractor iff opted-in (config `fast_c=true` or env MESH_BFS_C=1), else the
    reference Python one. Both give identical ring histograms; the far-ring sub-sample differs only in
    which members (any subset of an over-quota band is valid)."""
    use_fast = bool(gk.get("fast_c", False)) or os.environ.get("MESH_BFS_C") == "1"
    if use_fast and _extract_metric_fast_c is not None:
        return _extract_metric_fast_c
    return extract_metric_ring_neighborhoods


# ── freestream non-dimensionalisation (was dataset_airfrans_v14) ──────────────────

def uinf(xf):
    """Freestream speed from the (constant) inlet-velocity feature columns 0,1."""
    return float(np.sqrt(xf[0, 0] ** 2 + xf[0, 1] ** 2))


def nondim_targets(y, U):
    """(N,4) [u_x,u_y,p,nu_t] physical → dimensionless (copy-safe)."""
    out = np.array(y, copy=True)
    out[:, 0] /= U
    out[:, 1] /= U
    out[:, 2] /= U ** 2
    out[:, 3] /= U
    return out


def redim_targets(y_nd, U):
    """dimensionless → physical (inverse of nondim_targets)."""
    out = np.array(y_nd, copy=True)
    out[:, 0] *= U
    out[:, 1] *= U
    out[:, 2] *= U ** 2
    out[:, 3] *= U
    return out


# ── ν_t target: LINEAR (no log compression) ──────────────────────────────────────────
# v57_gated_hodge_vef sits on the NO-LOG branch: target = freestream non-dim, then per-field z-score, with ν_t
# left linear. The log1p compression (`compress_nut`/`decompress_nut`) belongs to v43 and lives in
# `cases/v43/dataset_airfrans_v43.py`; the copies inherited here were never called and have been
# removed so the case cannot be misread as a log run. Measured outcome of that fork: v43 (log)
# rel-L2 0.0867 / Volume 0.01378 / ν_t 0.1684 vs its no-log sibling v39_2 0.0822 / 0.00803 / 0.1449
# — the log lost on every metric including ν_t itself (docs/RESULTS.md §1-2).


# ── per-vertex dual area (was dataset_airfrans_v18_6.vertex_areas) ────────────────

def vertex_areas(pos, faces):
    """Per-vertex dual (barycentric) area = 1/3 * sum of incident triangle areas. (N,)"""
    p = pos[faces]                                   # (F,3,2)
    v1 = p[:, 1] - p[:, 0]; v2 = p[:, 2] - p[:, 0]
    tri = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])   # (F,)
    va = np.zeros(pos.shape[0], dtype=np.float64)
    for k in range(3):
        np.add.at(va, faces[:, k], tri / 3.0)
    return va.astype("float32")


_GEO_DEFAULTS = dict(v_max=256, e_max=256, f_max=1, max_hops=8, decay=0.55)


def symmetrize_idx(idx, cap=None):
    """Make a neighbour-index array symmetric: j∈N(i) ⇒ i∈N(j). Required for the v34 PSD Hodge
    stars to be EXACTLY symmetric (Variant A). idx:(N,W) with −1 pad. Returns (N,cap) with self
    first, −1 padded; if a symmetrized degree exceeds cap the extra (highest-id) neighbours are
    dropped — so exact symmetry needs cap ≥ max symmetric degree (else ~approximate; PSD still holds
    via the (1−β)I diagonal). Vectorised via a scipy sparse union `A + Aᵀ` (fast; runs once per mesh)."""
    from scipy.sparse import coo_matrix
    N, W = idx.shape
    cap = cap or W
    rows = np.repeat(np.arange(N, dtype=np.int64), W)
    cols = idx.reshape(-1)
    valid = cols >= 0
    A = coo_matrix((np.ones(valid.sum(), np.int8), (rows[valid], cols[valid])), shape=(N, N))
    A = (A + A.T).tocsr(); A.sort_indices()                  # symmetric union, sorted neighbours
    indptr, indices = A.indptr, A.indices
    out = np.full((N, cap), -1, dtype=np.int64)
    for i in range(N):                                        # O(nnz) numpy slices — fast
        nb = indices[indptr[i]:indptr[i + 1]]
        rest = nb[nb != i][:cap - 1]                          # self goes first, then neighbours
        out[i, 0] = i
        out[i, 1:1 + len(rest)] = rest
    return out


class AirfransV39Dataset(Dataset):
    """Consolidated v33 dataset: v18_3 pool sampling + v18_6 (pos,area,x_feat) inputs +
    freestream-non-dim z-score targets + geodesic-ring geometry. No cross-case imports."""

    # ── construction (was v18_3.__init__ over the v13 base __init__) ──────────────
    def __init__(self, names, pool_dir, K=8, train_augment=True,
                 feature_cols=None, target_cols=None,
                 feat_mean=None, feat_std=None, target_mean=None, target_std=None,
                 extract_kw=None, cache_dir=None, use_bc_injection=True,
                 band_kw=None, **ignored):
        self.paths = list(names)                     # sample base names (idx/len only)
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.extract_kw = extract_kw or {"mode": "bigbird"}
        self.cache_dir = cache_dir
        self.band_kw = dict(band_kw or {})       # v57_gated_hodge_vef: frequency-band ladder for the expert targets
        self.use_bc_injection = use_bc_injection
        # v34: symmetric neighbour support -> exactly-symmetric (PSD) Hodge stars (Variant A).
        self.symmetric_support = bool((extract_kw or {}).get("symmetric_support", True))
        self.feat_mean = None if feat_mean is None else np.asarray(feat_mean, "float32")
        self.feat_std = None if feat_std is None else np.asarray(feat_std, "float32")
        self.target_mean = None if target_mean is None else np.asarray(target_mean, "float32")
        self.target_std = None if target_std is None else np.asarray(target_std, "float32")
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        # pool of K precomputed meshes per sample
        self.pool_dir = pool_dir
        self.train_augment = train_augment
        self.rng = np.random.default_rng()
        self.pool = {}
        for n in names:
            files = sorted(glob.glob(os.path.join(pool_dir, n + "__s*.npz")))
            self.pool[n] = files[:K] if K else files

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _select(arr, cols):
        return arr if cols is None else arr[:, list(cols)]

    def _pick(self, name):
        files = self.pool[name]
        if not files:
            raise FileNotFoundError(f"no pool meshes for {name} in {self.pool_dir}")
        return self.rng.choice(files) if self.train_augment else files[0]

    def _geodesic_kw(self):
        gk = dict(_GEO_DEFAULTS)
        gk.update((self.extract_kw or {}).get("geodesic", {}))
        return gk

    # ── geometry: geodesic-ring neighbourhoods + optional metric Laplacian ────────
    # (was v22_geodesic._geometry) with a SELF-HEALING cache load.
    def _geometry(self, name, pos, faces):
        key = os.path.join(self.cache_dir, name + ".geo.npz") if self.cache_dir else None
        if key and os.path.isfile(key):
            try:
                z = np.load(key)
                d0 = coo_matrix((z["d0d"], (z["d0r"], z["d0c"])), shape=tuple(z["d0s"]))
                d1 = coo_matrix((z["d1d"], (z["d1r"], z["d1c"])), shape=tuple(z["d1s"]))
                out = dict(x_e=z["x_e"], x_f=z["x_f"], v_idx=z["v_idx"],
                           e_idx=z["e_idx"], f_idx=z["f_idx"], d0=d0, d1=d1)
                z.close()
                return out
            except Exception as e:
                # corrupt/truncated cache (e.g. a partial write from a disk-full crash):
                # drop it and rebuild rather than crashing the worker.
                print(f"[v57_gated_hodge_vef] corrupt geometry cache, rebuilding: {os.path.basename(key)} "
                      f"({type(e).__name__})", flush=True)
                try:
                    os.remove(key)
                except OSError:
                    pass

        v3 = np.zeros((pos.shape[0], 3), "float32"); v3[:, :2] = pos
        mesh = MeshSignedIncidenceMatrices(v3, faces, uid="airfrans")

        # geometry (x_e/x_f) + incidence (v2e=d_0, e2f=d_1); bigbird neighbours discarded
        ekw = {k: v for k, v in (self.extract_kw or {}).items()
               if k not in ("geodesic", "laplacian_weight", "symmetric_support")}
        _, x_e, x_f, v2e, e2f, _v_bb, _e_bb, f_idx = extract_mesh_features(mesh, **ekw)

        # geodesic-ring neighbourhoods (builder wants d_0 = edges×verts, d_1 = faces×edges)
        gk = self._geodesic_kw()
        seed = zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF
        # geodesic.mode: "hop" (BFS hop-count rings, default) | "metric" (true geodesic-distance rings,
        # Dijkstra with Euclidean edge weights, equal-width distance bands). Both -> identical dict format.
        if str(gk.get("mode", "hop")).lower() == "metric":
            nb = _select_metric_extractor(gk)(
                {"d_0": v2e.T.tocsr(), "d_1": e2f.T.tocsr()}, pos,
                v_max=gk["v_max"], e_max=gk["e_max"], f_max=gk["f_max"],
                n_rings=int(gk.get("n_rings", gk["max_hops"])),
                ring_width=gk.get("ring_width", None), radius=gk.get("radius", None),
                equal_quota=bool(gk.get("equal_quota", True)), decay=gk["decay"],
                seed=seed, modes=("v", "e"),
            )
        else:
            nb = _select_geodesic_extractor(gk)(
                {"d_0": v2e.T.tocsr(), "d_1": e2f.T.tocsr()},
                v_max=gk["v_max"], e_max=gk["e_max"], f_max=gk["f_max"],
                max_hops=gk["max_hops"], decay=gk["decay"],
                seed=seed, modes=("v", "e"),
            )
        v_idx = nb["v_idx"]; e_idx = nb["e_idx"]

        # v34: symmetrize the neighbour support so the Gaussian Hodge stars are EXACTLY symmetric
        # (j∈N(i) ⇔ i∈N(j)) → provably PSD. Done once here, then cached.
        if self.symmetric_support:
            gk_ = self._geodesic_kw()
            v_idx = symmetrize_idx(v_idx, cap=gk_["v_max"])
            e_idx = symmetrize_idx(e_idx, cap=gk_["e_max"])

        # unit ±1 (default) OR metric-weighted incidence by edge length (config-selectable)
        lw = (self.extract_kw or {}).get("laplacian_weight", "unit")
        if lw and lw != "unit":
            v2e, e2f, _ = length_weighted_incidence(v2e, e2f, pos, mode=lw)
        d0 = v2e.tocoo(); d1 = e2f.tocoo()

        geo = dict(x_e=x_e.astype("float32"), x_f=x_f.astype("float32"),
                   v_idx=v_idx.astype("int64"), e_idx=e_idx.astype("int64"),
                   f_idx=np.asarray(f_idx).astype("int64"), d0=d0, d1=d1)
        if key:
            os.makedirs(os.path.dirname(key), exist_ok=True)
            # write to a temp file then atomically rename, so an interrupted write never
            # leaves a half-written .npz behind (the failure mode that crashed the run).
            # NB np.savez appends ".npz" unless present, so the temp name must already
            # end in ".npz" or os.replace would chase the wrong path.
            tmp = key + f".tmp{os.getpid()}.npz"
            try:
                np.savez(tmp, x_e=geo["x_e"], x_f=geo["x_f"], v_idx=geo["v_idx"],
                         e_idx=geo["e_idx"], f_idx=geo["f_idx"],
                         d0d=d0.data, d0r=d0.row, d0c=d0.col, d0s=d0.shape,
                         d1d=d1.data, d1r=d1.row, d1c=d1.col, d1s=d1.shape)
                os.replace(tmp, key)
            except OSError as e:
                # out of space / IO error: skip caching this mesh, keep training
                print(f"[v57_gated_hodge_vef] could not cache geometry ({type(e).__name__}): "
                      f"{os.path.basename(key)}", flush=True)
                try:
                    if os.path.isfile(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
        return geo

    # ── per-sample loading (was v18_6.__getitem__: 8 features, nondim+z-score) ─────
    def __getitem__(self, idx):
        name = self.paths[idx]
        path = self._pick(name)
        mesh_name = os.path.splitext(os.path.basename(path))[0]
        z = np.load(path)
        pos = z["pos"].astype("float32"); faces = z["faces"].astype("int64")
        xf = z["x_feat"].astype("float32"); y = z["y"].astype("float32")

        geo = self._geometry(mesh_name, pos, faces)

        area = vertex_areas(pos, faces).reshape(-1, 1)
        wall = z["wall"].astype("float32").reshape(-1, 1)           # v57_gated_hodge_vef: is_wall (from npz) — feeds the surface loss + informs the model
        x_v = np.hstack([pos, area, xf, wall]).astype("float32")    # (N, 2+1+5+1 = 9); is_wall = col 8
        if self.feat_mean is not None and self.feat_std is not None:
            x_v = (x_v - self.feat_mean) / (self.feat_std + 1e-8)

        target = self._select(y, self.target_cols).astype("float32")
        target = nondim_targets(target, uinf(xf))                  # freestream non-dim before z-score
        if self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / (self.target_std + 1e-8)

        # ── v57_gated_hodge_vef: the C_L integral as a linear functional of the wall pressure ───────────────
        # C_L = w . p_physical exactly (verified to 1e-15 vs the eval's mesh_cl). Folded through the
        # target normalisation so the loss can consume the model's z-scored output directly:
        #     C_L = (U^2 std_p)(w . p_z) + (U^2 mean_p)(sum w) = w' . p_z + b
        w_raw = self._cl_w(mesh_name, pos, wall, xf)
        u2 = float(uinf(xf)) ** 2
        p_mean = float(self.target_mean[2]); p_std = float(self.target_std[2])
        cl_w = (w_raw * u2 * p_std).astype("float32").reshape(-1, 1)
        cl_b = float(u2 * p_mean * w_raw.sum())
        cl_true = float(w_raw @ y[:, 2].astype("float64"))          # from the RAW physical field
        n_v = x_v.shape[0]

        # ── v57_gated_hodge_vef: frequency bands of the (normalised) target, one per expert ─────────────────
        # Decomposed on the mesh graph and cached per mesh. The bands sum to the target EXACTLY,
        # so the auxiliary loss cannot bias the total. The MODEL never sees the mesh — this is a
        # supervision-side construction only.
        bands = self._bands(mesh_name, faces, target)          # (N, C, K)
        band_cols = bands.reshape(n_v, -1)                     # (N, C*K)

        extra = np.hstack([cl_w,
                           np.full((n_v, 1), cl_true, dtype="float32"),
                           np.full((n_v, 1), cl_b, dtype="float32"),
                           band_cols]).astype("float32")

        tensors = (
            torch.FloatTensor(x_v),
            torch.FloatTensor(geo["x_e"]),
            torch.FloatTensor(geo["x_f"]),
            sparse_scipy_to_torch(geo["d0"], transpose=True),
            sparse_scipy_to_torch(geo["d1"], transpose=True),
            torch.LongTensor(geo["v_idx"]),
            torch.LongTensor(geo["e_idx"]),
            torch.LongTensor(geo["f_idx"]),
        )
        # target columns: [C fields][cl_w per vertex][cl_true][cl_b]; the two constants are
        # broadcast so the stock collate needs no change.
        return tensors, torch.FloatTensor(np.hstack([target, extra]))

    def _bands(self, mesh_name, faces, target):
        """Frequency bands of the z-scored target, cached per mesh (the smoothing operator is
        static; the target is too, for a given mesh)."""
        ladder = tuple(self.band_kw.get("ladder", (2, 8, 32)))
        tag = "-".join(str(m) for m in ladder)
        cache = (os.path.join(self.cache_dir, f"{mesh_name}.bands{tag}.npz")
                 if self.cache_dir else None)
        if cache and os.path.isfile(cache):
            try:
                return np.load(cache)["b"]
            except (zlib.error, OSError, ValueError, KeyError):
                try:
                    os.remove(cache)
                except OSError:
                    pass
        b = band_decompose(faces, target, ladder=ladder)
        if cache:
            tmp = cache + f".tmp{os.getpid()}.npz"
            try:
                np.savez_compressed(tmp, b=b); os.replace(tmp, cache)
            except OSError:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return b

    def _cl_w(self, mesh_name, pos, wall, xf):
        """Per-vertex C_L weights, cached beside the geometry (static per mesh)."""
        cache = (os.path.join(self.cache_dir, f"{mesh_name}.clw.npz") if self.cache_dir else None)
        if cache and os.path.isfile(cache):
            try:
                return np.load(cache)["w"]
            except (zlib.error, OSError, ValueError, KeyError):
                try:
                    os.remove(cache)
                except OSError:
                    pass
        w = cl_weights(pos, wall, xf[:, 3:5], np.array([xf[0, 0], xf[0, 1]]), n_vert=pos.shape[0])
        if cache:
            tmp = cache + f".tmp{os.getpid()}.npz"       # np.savez appends .npz unless present
            try:
                np.savez_compressed(tmp, w=w); os.replace(tmp, cache)
            except OSError:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return w

    # ── stats (was v18_6.compute_stats: feats over [pos,area,xf]; nondim target stats) ─
    @classmethod
    def compute_stats(cls, names, pool_dir, feature_cols=None, target_cols=None,
                      max_samples=None, **ignored):
        paths = [os.path.join(pool_dir, n + "__s0.npz") for n in names]
        paths = [p for p in paths if os.path.isfile(p)]
        if max_samples:
            paths = paths[:max_samples]
        feat_acc, tgt_acc = [], []
        for p in paths:
            z = np.load(p)
            pos = z["pos"].astype("float64"); faces = z["faces"].astype("int64")
            xf = z["x_feat"].astype("float64"); y = z["y"].astype("float64")
            area = vertex_areas(pos.astype("float32"), faces).astype("float64").reshape(-1, 1)
            feat_acc.append(np.hstack([pos, area, xf, z["wall"].astype("float64").reshape(-1, 1)]))  # v57_gated_hodge_vef: +is_wall (col 8)
            yy = y if target_cols is None else y[:, list(target_cols)]
            yy = nondim_targets(yy, uinf(xf))
            tgt_acc.append(yy)
        feats = np.concatenate(feat_acc, 0); tgts = np.concatenate(tgt_acc, 0)
        return dict(feat_mean=feats.mean(0).astype("float32"),
                    feat_std=feats.std(0).astype("float32"),
                    target_mean=tgts.mean(0).astype("float32"),
                    target_std=tgts.std(0).astype("float32"))
