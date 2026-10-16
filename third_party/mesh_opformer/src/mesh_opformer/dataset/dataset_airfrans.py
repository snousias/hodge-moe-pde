"""
dataset_airfrans.py
-------------------
HodgeFormer dataset for the **AirFRANS** 2-D RANS benchmark
(Bonnet et al., NeurIPS 2022) served as PyTorch-Geometric ``Data`` graphs.

Each ``graph_airfrans_data_*.pth`` file stores a single airfoil simulation as a
graph with

    pos        : (N, 2)  node coordinates in the 2-D flow domain
    x          : (N, 5)  per-node input features (inlet condition, distance to
                         the airfoil / SDF, surface normals)
    y          : (N, 4)  per-node targets  ->  (u_x, u_y, p, nu_t)
    edge_index : (2, ~10 N)  a k-NN connectivity graph

Crucially the graphs contain **no faces / cells** — only the 1-skeleton.  The
HodgeFormer backbone needs a genuine 2-complex (vertices + edges + faces and
the signed incidence operators dâ‚€, dâ‚).  We therefore reconstruct a 2-D
triangulation from ``pos`` (Delaunay, with far-field / airfoil-spanning
triangles pruned by edge length), embed it flat into 3-D (z = 0) so the
existing ``mesh_sim`` geometry/feature machinery applies unchanged, and regress
the four physical fields per vertex.

No geometric augmentation is applied: rotating / jittering the domain would
corrupt the flow boundary conditions.  Because the geometry is therefore fixed
per sample, the heavy per-mesh artefacts (triangulation, incidence operators,
neighbourhoods, edge/face features) are cached to disk and reused every epoch.
"""

from typing import List, Dict, Optional, Tuple

import os
import hashlib

import numpy as np
import torch

from scipy.spatial import Delaunay
from torch.utils.data import Dataset

from .utils import sparse_scipy_to_torch

try:
    from mesh_sim.mesh import MeshSignedIncidenceMatrices
    from mesh_sim.utils_ml import extract_mesh_features
except ImportError:
    raise ImportError("Need to install mesh-sim library first.")


# ──────────────────────────────────────────────────────────────────────────────
# 2-D triangulation reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_2d_triangulation(
    pos: np.ndarray,
    max_edge_pct: Optional[float] = 99.0,
    max_edge_abs: Optional[float] = None,
) -> np.ndarray:
    """
    Build a triangular mesh from a 2-D point set via Delaunay triangulation.

    Delaunay fills the convex hull of the points.  For an airfoil flow domain
    that means (a) huge skinny triangles reaching the rectangular/circular
    far-field boundary and (b) triangles bridging across the (node-free) airfoil
    body.  Both are pruned by dropping triangles that contain an edge longer
    than a threshold.  The threshold is the ``max_edge_pct`` percentile of all
    Delaunay edge lengths (robust, scale-free) unless an absolute
    ``max_edge_abs`` is supplied.

    Parameters
    ----------
    pos : np.ndarray, shape (N, 2)
        Node coordinates.
    max_edge_pct : float | None
        Percentile (0-100) of triangle max-edge-length above which triangles
        are discarded.  ``None`` disables percentile pruning.
    max_edge_abs : float | None
        Absolute max-edge-length threshold.  Takes precedence over the
        percentile when given.

    Returns
    -------
    faces : np.ndarray, shape (M, 3), int32
        Triangle vertex indices (orientation as returned by Delaunay).
    """
    pos = np.ascontiguousarray(pos[:, :2], dtype="float64")

    tri = Delaunay(pos)
    faces = tri.simplices.astype("int64")  # (M, 3)

    if max_edge_pct is None and max_edge_abs is None:
        return faces.astype("int32")

    # Per-triangle longest edge.
    p = pos[faces]  # (M, 3, 2)
    e0 = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
    e1 = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
    e2 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    tri_max_edge = np.maximum.reduce([e0, e1, e2])

    if max_edge_abs is not None:
        thr = float(max_edge_abs)
    else:
        all_edges = np.concatenate([e0, e1, e2])
        thr = float(np.percentile(all_edges, max_edge_pct))

    keep = tri_max_edge <= thr
    faces = faces[keep]

    # Drop any vertices that became isolated? No — we keep the full vertex set
    # so that per-vertex targets stay index-aligned with the original graph.
    return faces.astype("int32")


# ──────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_airfrans_graph(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single AirFRANS PyG ``Data`` graph and return ``(pos, x, y)`` as
    numpy arrays.  Requires ``torch_geometric`` to be importable (needed to
    unpickle the ``Data`` object).
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    def _get(key):
        if hasattr(obj, key):
            return getattr(obj, key)
        return obj[key]

    pos = np.asarray(_get("pos"), dtype="float32")
    x = np.asarray(_get("x"), dtype="float32")
    y = np.asarray(_get("y"), dtype="float32")
    return pos, x, y


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class AirfransGraphDataset(Dataset):
    """
    Dataset wrapping AirFRANS PyG graphs for per-vertex field regression with
    HodgeFormer.

    Parameters
    ----------
    graph_paths : List[str]
        Paths to ``graph_airfrans_data_*.pth`` files.
    feature_cols : List[int] | None
        Which columns of the AirFRANS ``x`` array to use as vertex inputs.
        ``None`` uses all 5.  The reconstructed 2-D ``pos`` is always prepended,
        so ``v_in = 2 + len(feature_cols)``.
    target_cols : List[int] | None
        Which columns of ``y`` to regress.  ``None`` uses all 4
        (u_x, u_y, p, nu_t).
    feat_mean, feat_std : np.ndarray | None
        Per-channel standardisation stats for the *vertex input* features
        (length ``v_in``).  Computed from the training split.
    target_mean, target_std : np.ndarray | None
        Per-channel standardisation stats for the targets (length
        ``len(target_cols)``).
    extract_kw : Dict
        Forwarded to ``mesh_sim.utils_ml.extract_mesh_features`` (mode +
        neighbourhood kwargs) for the edge/face features and neighbourhoods.
    tri_kw : Dict
        Forwarded to :func:`reconstruct_2d_triangulation`.
    cache_dir : str | None
        Directory for caching the heavy per-mesh artefacts.  ``None`` disables
        caching (recompute every ``__getitem__``).
    max_nodes : int | None
        Optionally subsample to at most this many nodes (random, seeded) before
        triangulation.  Useful for quick CPU smoke tests; ``None`` = full mesh.
    """

    def __init__(
        self,
        graph_paths: List[str],
        feature_cols: Optional[List[int]] = None,
        target_cols: Optional[List[int]] = None,
        feat_mean: Optional[np.ndarray] = None,
        feat_std: Optional[np.ndarray] = None,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
        extract_kw: Dict = {"mode": "bigbird"},
        tri_kw: Dict = {"max_edge_pct": 99.0},
        cache_dir: Optional[str] = None,
        max_nodes: Optional[int] = None,
        seed: int = 123,
        use_bc_injection: bool = False,
        use_bl_sampling: bool = False,
        bl_sdf_threshold: float = 0.05,
        bl_oversample: int = 4,
        target_log_cols: Optional[List[int]] = None,
        target_log_scale: float = 100.0,
    ):
        super().__init__()
        self.graph_paths = graph_paths
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.extract_kw = extract_kw
        self.tri_kw = tri_kw
        self.cache_dir = cache_dir
        # Treat 0 / None alike: full mesh, no subsampling.
        self.max_nodes = max_nodes or None
        self.seed = seed
        self.use_bc_injection = use_bc_injection
        self.use_bl_sampling = use_bl_sampling
        self.bl_sdf_threshold = bl_sdf_threshold
        self.bl_oversample = bl_oversample
        # Columns (positions in the *selected* target array) to log-compress
        # before standardisation: t = log1p(scale * raw). Heavy-tailed,
        # non-negative fields such as nu_t benefit. Inverted in eval.
        self.target_log_cols = list(target_log_cols) if target_log_cols else []
        self.target_log_scale = float(target_log_scale)

        self.feat_mean = None if feat_mean is None else np.asarray(feat_mean, "float32")
        self.feat_std = None if feat_std is None else np.asarray(feat_std, "float32")
        self.target_mean = (
            None if target_mean is None else np.asarray(target_mean, "float32")
        )
        self.target_std = (
            None if target_std is None else np.asarray(target_std, "float32")
        )

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.graph_paths)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_bc_features(
        pos: np.ndarray, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build three BC-injection feature columns of shape (N, 1) each:
          bc_ux  — known u_x at BC nodes (0 at wall, u_inlet_x at inlet)
          bc_uy  — known u_y at BC nodes (0 at wall, u_inlet_y at inlet)
          is_bc  — 1.0 if this node is on a known Dirichlet BC face

        Wall nodes  : SDF column (x[:, 2]) == 0 → no-slip (u = 0).
        Inlet nodes : x < x_min + 10% of domain width (far-left boundary face).
        u_inlet is broadcast uniformly, so x[0, 0] / x[0, 1] give the value.
        """
        N = pos.shape[0]
        bc_ux = np.zeros((N, 1), dtype="float32")
        bc_uy = np.zeros((N, 1), dtype="float32")
        is_bc = np.zeros((N, 1), dtype="float32")

        wall_mask = x[:, 2] == 0.0
        is_bc[wall_mask] = 1.0
        # bc_ux / bc_uy stay 0 at wall (no-slip)

        x_min = float(pos[:, 0].min())
        x_max = float(pos[:, 0].max())
        inlet_thr = x_min + 0.10 * (x_max - x_min)
        inlet_mask = pos[:, 0] < inlet_thr
        u_inlet_x = float(x[0, 0])
        u_inlet_y = float(x[0, 1])
        bc_ux[inlet_mask] = u_inlet_x
        bc_uy[inlet_mask] = u_inlet_y
        is_bc[inlet_mask] = 1.0

        return bc_ux, bc_uy, is_bc

    def _select_cols(self, arr: np.ndarray, cols: Optional[List[int]]) -> np.ndarray:
        if cols is None:
            return arr
        return arr[:, list(cols)]

    def _subsample(self, pos, x, y):
        if self.max_nodes is None or pos.shape[0] <= self.max_nodes:
            return pos, x, y, None
        if self.use_bl_sampling:
            return self._subsample_bl(pos, x, y)
        rs = np.random.RandomState(self.seed)
        keep = np.sort(rs.choice(pos.shape[0], self.max_nodes, replace=False))
        return pos[keep], x[keep], y[keep], keep

    def _subsample_bl(self, pos, x, y):
        """
        Boundary-layer-preserving stratified subsample.

        Guarantees all wall nodes (SDF=0) are kept, over-represents near-wall
        nodes (0 < SDF < bl_sdf_threshold) by bl_oversample×, and fills the
        remaining budget uniformly from the far field.
        """
        quota = self.max_nodes
        rs = np.random.RandomState(self.seed)

        wall_mask     = x[:, 2] == 0.0
        near_mask     = (~wall_mask) & (x[:, 2] < self.bl_sdf_threshold)
        far_mask      = ~wall_mask & ~near_mask

        wall_idx      = np.where(wall_mask)[0]
        near_idx      = np.where(near_mask)[0]
        far_idx       = np.where(far_mask)[0]

        n_wall        = len(wall_idx)
        remaining     = quota - n_wall

        # Weighted budget: near-wall gets bl_oversample× relative density.
        n_near_full   = len(near_idx)
        n_far_full    = len(far_idx)
        total_nw      = n_near_full + n_far_full
        near_weight   = (n_near_full / total_nw) * self.bl_oversample
        far_weight    = (n_far_full  / total_nw) * 1.0
        near_budget   = int(remaining * near_weight / (near_weight + far_weight))
        far_budget    = remaining - near_budget

        near_sel = rs.choice(n_near_full,
                             min(near_budget, n_near_full), replace=False)
        far_sel  = rs.choice(n_far_full,
                             min(far_budget,  n_far_full),  replace=False)

        keep = np.sort(np.concatenate([
            wall_idx,
            near_idx[near_sel],
            far_idx[far_sel],
        ]))
        return pos[keep], x[keep], y[keep], keep

    def _cache_key(self, path: str, n_nodes: int) -> Optional[str]:
        if self.cache_dir is None:
            return None
        spec = {
            "tri": self.tri_kw,
            "extract": self.extract_kw,
            "max_nodes": self.max_nodes,
            "n": n_nodes,
        }
        h = hashlib.md5(repr(sorted(spec.items())).encode()).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(self.cache_dir, f"{stem}__{h}.npz")

    def _compute_geometry(self, pos: np.ndarray):
        """
        Reconstruct the triangulation, build the mesh and extract the edge/face
        features, incidence operators and neighbourhoods.  Returns numpy /
        scipy artefacts (cache-friendly).
        """
        faces = reconstruct_2d_triangulation(pos, **self.tri_kw)

        # Embed flat into 3-D so mesh_sim / open3d work unchanged.
        verts3d = np.zeros((pos.shape[0], 3), dtype="float32")
        verts3d[:, :2] = pos

        mesh = MeshSignedIncidenceMatrices(verts3d, faces, uid="airfrans")

        x_v_geo, x_e, x_f, v2e, e2f, v_idx, e_idx, f_idx = extract_mesh_features(
            mesh, **self.extract_kw
        )

        return {
            "faces": faces,
            "x_e": x_e.astype("float32"),
            "x_f": x_f.astype("float32"),
            "v_idx": v_idx.astype("int64"),
            "e_idx": e_idx.astype("int64"),
            "f_idx": f_idx.astype("int64"),
            # store dâ‚€ = v2e (csc) and dâ‚ = e2f (csc) in raw COO form
            "d0": v2e.tocoo(),
            "d1": e2f.tocoo(),
        }

    def _load_or_compute_geometry(self, path: str, pos: np.ndarray):
        key = self._cache_key(path, pos.shape[0])

        if key is not None and os.path.isfile(key):
            z = np.load(key, allow_pickle=False)
            from scipy.sparse import coo_matrix

            d0 = coo_matrix(
                (z["d0_data"], (z["d0_row"], z["d0_col"])), shape=tuple(z["d0_shape"])
            )
            d1 = coo_matrix(
                (z["d1_data"], (z["d1_row"], z["d1_col"])), shape=tuple(z["d1_shape"])
            )
            return {
                "faces": z["faces"],
                "x_e": z["x_e"],
                "x_f": z["x_f"],
                "v_idx": z["v_idx"],
                "e_idx": z["e_idx"],
                "f_idx": z["f_idx"],
                "d0": d0,
                "d1": d1,
            }

        geo = self._compute_geometry(pos)

        if key is not None:
            d0, d1 = geo["d0"], geo["d1"]
            np.savez(
                key,
                faces=geo["faces"],
                x_e=geo["x_e"],
                x_f=geo["x_f"],
                v_idx=geo["v_idx"],
                e_idx=geo["e_idx"],
                f_idx=geo["f_idx"],
                d0_data=d0.data, d0_row=d0.row, d0_col=d0.col, d0_shape=d0.shape,
                d1_data=d1.data, d1_row=d1.row, d1_col=d1.col, d1_shape=d1.shape,
            )

        return geo

    # ------------------------------------------------------------------

    def _apply_target_log(self, target: np.ndarray) -> np.ndarray:
        """Apply t = log1p(scale * raw) to configured target columns in-place."""
        if not self.target_log_cols:
            return target
        target = target.copy()
        for c in self.target_log_cols:
            target[:, c] = np.log1p(self.target_log_scale * target[:, c])
        return target

    @staticmethod
    def invert_target_log(target: np.ndarray, log_cols, scale: float) -> np.ndarray:
        """Inverse of `_apply_target_log`: raw = expm1(t) / scale."""
        if log_cols is None or len(log_cols) == 0:
            return target
        target = target.copy()
        for c in log_cols:
            target[..., c] = np.expm1(target[..., c]) / scale
        return target

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        tensors : tuple
            ``(x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx)`` — the standard
            HodgeFormer mesh-feature tuple.
        target : torch.FloatTensor, shape (N, C)
            Standardised per-vertex target fields.
        """
        path = self.graph_paths[idx]
        pos, x, y = load_airfrans_graph(path)
        pos, x, y, _ = self._subsample(pos, x, y)

        geo = self._load_or_compute_geometry(path, pos)

        # Vertex inputs: 2-D position ++ selected AirFRANS node features
        # ++ binary wall indicator ++ optional BC value injection.
        x_feat = self._select_cols(x, self.feature_cols)
        parts = [pos.astype("float32"), x_feat.astype("float32")]
        if getattr(self, "use_wall_flag", True):
            is_wall = (x[:, 2] == 0.0).astype("float32").reshape(-1, 1)
            parts.append(is_wall)
        if self.use_bc_injection:
            bc_ux, bc_uy, is_bc = self._build_bc_features(pos, x)
            parts.extend([bc_ux, bc_uy, is_bc])
        x_v = np.hstack(parts)

        if self.feat_mean is not None and self.feat_std is not None:
            x_v = (x_v - self.feat_mean) / (self.feat_std + 1e-8)

        # Targets.
        target = self._select_cols(y, self.target_cols).astype("float32")
        target = self._apply_target_log(target)
        if self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / (self.target_std + 1e-8)

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
        return tensors, torch.FloatTensor(target)

    # ------------------------------------------------------------------

    @classmethod
    def compute_stats(
        cls,
        graph_paths: List[str],
        feature_cols: Optional[List[int]] = None,
        target_cols: Optional[List[int]] = None,
        max_samples: Optional[int] = None,
        use_bc_injection: bool = False,
        target_log_cols: Optional[List[int]] = None,
        target_log_scale: float = 100.0,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-channel mean/std of the vertex input features
        (``[pos | x[:, feature_cols] | is_wall | (bc_ux, bc_uy, is_bc)]``)
        and the targets over a list of graphs.

        If ``target_log_cols`` is given, those target columns are log-compressed
        (``log1p(scale * raw)``) before stats are computed, so the standardised
        space matches what ``__getitem__`` produces.

        Returns a dict with ``feat_mean, feat_std, target_mean, target_std`` and
        ``target_log_cols, target_log_scale`` (for inversion at eval time).
        """
        log_cols = list(target_log_cols) if target_log_cols else []
        feat_acc, tgt_acc = [], []
        paths = graph_paths if max_samples is None else graph_paths[:max_samples]

        for p in paths:
            pos, x, y = load_airfrans_graph(p)
            if feature_cols is None:
                xf = x
            else:
                xf = x[:, list(feature_cols)]
            is_wall = (x[:, 2] == 0.0).astype("float64").reshape(-1, 1)
            row = [pos, xf, is_wall]
            if use_bc_injection:
                bc_ux, bc_uy, is_bc = cls._build_bc_features(pos, x)
                row.extend([bc_ux.astype("float64"),
                             bc_uy.astype("float64"),
                             is_bc.astype("float64")])
            feat_acc.append(np.hstack(row).astype("float64"))
            if target_cols is None:
                t = y.astype("float64")
            else:
                t = y[:, list(target_cols)].astype("float64")
            for c in log_cols:
                t[:, c] = np.log1p(target_log_scale * t[:, c])
            tgt_acc.append(t)

        feats = np.concatenate(feat_acc, axis=0)
        tgts = np.concatenate(tgt_acc, axis=0)

        return {
            "feat_mean": feats.mean(0).astype("float32"),
            "feat_std": feats.std(0).astype("float32"),
            "target_mean": tgts.mean(0).astype("float32"),
            "target_std": tgts.std(0).astype("float32"),
            "target_log_cols": np.asarray(log_cols, dtype="int64"),
            "target_log_scale": np.asarray(target_log_scale, dtype="float32"),
        }
