"""
geodesic_metric.py
------------------
Metric (true geodesic-distance) ring neighbourhoods — a geometry-aware alternative to the
hop-count rings of :mod:`neighborhoods`.

Why
===
:func:`neighborhoods.extract_geodesic_ring_neighborhoods` labels each neighbour by its **BFS hop
count** and its edge weight is the uniform "1 per hop". On an *anisotropic* mesh (e.g. an airfoil
boundary layer with very thin wall-normal cells) hop count is a poor proxy for physical distance:
one wall-normal hop crosses a far smaller distance than one streamwise hop, so a "3-hop ring" mixes
neighbours that are physically near (wall-normal) with ones that are far (streamwise).

This module instead walks a **weighted shortest path** — Dijkstra with Euclidean edge lengths as
weights (``d(u,v) = ‖P_u − P_v‖`` where ``P`` is the element position: vertex coordinate, edge
midpoint, or face centroid) — and bins neighbours into **equal-width geodesic-distance rings**
``ring(j) = min(⌊dist(i→j) / Δ⌋, n_rings−1)``. So a ring is a band of *physical distance*, the same
Δ metres thick regardless of how the mesh happens to be discretised there.

Selection is stratified per distance-ring exactly like the hop version (keep a ring whole if it fits
its quota, else random sub-sample; top up with the nearest unchosen; output nearest-first). With
``equal_quota=True`` (the default) **every ring gets the same slot budget** — so the attention window
samples uniformly in *distance space* rather than in *node-count space*, giving each distance band the
same representation regardless of how many mesh nodes populate it.

Output format is identical to :func:`extract_geodesic_ring_neighborhoods` — ``(n_elem, n_max)`` int32
``*_idx`` (gather indices, ``-1`` padded) and ``*_ring`` (the distance-band id per slot, for the
``RingBias`` positional term) — so it is a drop-in geometry alternative.

Cost: one bounded Dijkstra per element (scipy C kernel with a ``limit`` radius) + a per-element Python
selection pass, comparable to the hop extractor; cache the result. A fused C version (the Dijkstra
analogue of ``mesh_bfs_c``) could accelerate it further if cold-build time matters.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .neighborhoods import _select_ring_stratified

# Element source matrix + whether to transpose to (n_elem x m), matching neighborhoods._INC_TYPE:
#   v : d_0.T (verts x edges) -> vertex adjacency      P = vertex coords
#   e : d_0   (edges x verts) -> edge line graph       P = edge midpoints
#   f : d_1   (faces x edges) -> face adjacency        P = face centroids
_SOURCE = {"v": ("d_0", True), "e": ("d_0", False), "f": ("d_1", False)}


def equal_ring_quotas(n_max: int, n_rings: int) -> Dict[int, int]:
    """Uniform per-ring slot budget. Ring 0 is the element **itself** (always exactly 1 slot); the
    ``n_rings`` geodesic-distance bands are rings ``1..n_rings`` and split the remaining ``n_max - 1``
    slots **equally**. This is the "same number of neighbours per distance band regardless of how many
    mesh nodes populate it" policy."""
    quotas = {0: 1}
    remaining = max(n_max - 1, 0)
    base = remaining // n_rings
    extra = remaining - base * n_rings  # spread the remainder over the nearest bands
    for r in range(1, n_rings + 1):
        quotas[r] = max(1, base + (1 if (r - 1) < extra else 0))
    return quotas


def _element_positions(inc: Dict[str, object], pos: np.ndarray) -> Dict[str, np.ndarray]:
    """Position of each element type: vertices (coords), edges (midpoints), faces (centroids)."""
    pos = np.asarray(pos, dtype=np.float64)
    out = {"v": pos}
    d0 = inc.get("d_0")  # (n_e edges, n_v verts)
    d1 = inc.get("d_1")  # (n_f faces, n_e edges)
    if d0 is not None:
        a0 = d0.tocsr().astype(np.float64); a0.data = np.abs(a0.data)  # |v2e|^T = edges x verts, 2 ones/row
        deg = np.asarray(a0.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        out["e"] = np.asarray(a0 @ pos) / deg[:, None]                 # edge midpoint = mean of endpoints
        if d1 is not None:
            a1 = d1.tocsr().astype(np.float64); a1.data = np.abs(a1.data)  # faces x edges
            b = (a1 @ a0).tocsr()                                       # faces x verts (incidence counts)
            fdeg = np.asarray(b.sum(axis=1)).ravel()
            fdeg[fdeg == 0] = 1.0
            out["f"] = np.asarray(b @ pos) / fdeg[:, None]             # face centroid = mean of its verts
    return out


def _weighted_adjacency(mtrx, P: np.ndarray) -> csr_matrix:
    """Symmetric, positively weighted node->node adjacency for Dijkstra.

    Pattern = off-diagonal nonzeros of ``|mtrx| @ |mtrx|.T`` (elements that share a lower/upper
    simplex — identical connectivity to the BFS graph); weight on edge (u, v) = ``‖P_u − P_v‖``.
    """
    m = mtrx.tocsr().astype(np.float64)
    m.data = np.abs(m.data)
    a = (m @ m.T).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    a.sort_indices()
    n = a.shape[0]
    rows = np.repeat(np.arange(n), np.diff(a.indptr))
    cols = a.indices
    w = np.linalg.norm(P[rows] - P[cols], axis=1)
    w = np.maximum(w, 1e-12)  # strictly positive so Dijkstra distances are well defined
    return csr_matrix((w, a.indices.copy(), a.indptr.copy()), shape=(n, n))


def _rings_for_mode_metric(
    mtrx,
    P: np.ndarray,
    n_max: int,
    n_rings: int,
    ring_width: float | None,
    radius: float | None,
    quotas: Dict[int, int] | None,
    equal_quota: bool,
    decay: float,
    rng: np.random.RandomState,
    chunk: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the ``(n_elem, n_max)`` index + distance-ring arrays for one element type."""
    from .neighborhoods import default_ring_quotas

    W = _weighted_adjacency(mtrx, P)
    n_elem = W.shape[0]

    # Ring width Δ and search radius R. Auto: one ring ≈ one mean adjacency-edge length, so the
    # ball spans ~n_rings mean-edge-lengths of geodesic distance (the metric analogue of n_rings hops).
    mean_w = float(W.data.mean()) if W.nnz else 1.0
    dz = ring_width if ring_width is not None else mean_w
    dz = max(dz, 1e-12)
    R = radius if radius is not None else n_rings * dz

    if quotas is None:
        quotas = equal_ring_quotas(n_max, n_rings) if equal_quota \
            else default_ring_quotas(n_max, n_rings, decay=decay)

    idx = np.full((n_elem, n_max), -1, dtype=np.int32)
    ring = np.full((n_elem, n_max), -1, dtype=np.int32)

    for start in range(0, n_elem, chunk):
        srcs = np.arange(start, min(start + chunk, n_elem))
        # bounded single-source Dijkstra for each source in the chunk (C kernel, capped at radius R)
        D = dijkstra(W, directed=False, indices=srcs, limit=R)  # (len(srcs), n_elem), inf past R
        for row_i, i in enumerate(srcs):
            d = D[row_i]
            reach = np.where(np.isfinite(d))[0]
            if reach.size == 0:
                idx[i, 0] = i
                ring[i, 0] = 0
                continue
            dd = d[reach]
            order = np.argsort(dd, kind="stable")  # nearest-first so within-ring order is by true distance
            cols = reach[order].astype(np.int64)
            dist = dd[order]
            # ring 0 == the source itself (dist 0, uniquely — all real edges have weight > 0); the
            # n_rings equal-width distance bands are rings 1..n_rings. This guarantees slot 0 = self
            # and mirrors the hop extractor's "ring 0 = self, then rings".
            band = np.minimum((dist / dz).astype(np.int64), n_rings - 1)  # 0..n_rings-1
            levels = np.where(cols == i, 0, band + 1)                      # self->0, bands->1..n_rings
            sel_cols, sel_rings = _select_ring_stratified(cols, levels, n_max, quotas, rng)
            k = sel_cols.shape[0]
            idx[i, :k] = sel_cols.astype(np.int32)
            ring[i, :k] = sel_rings.astype(np.int32)

    return idx, ring


def extract_metric_ring_neighborhoods(
    inc: Dict[str, object],
    pos: np.ndarray,
    v_max: int = 128,
    e_max: int = 128,
    f_max: int = 1,
    n_rings: int = 8,
    ring_width: float | None = None,
    radius: float | None = None,
    decay: float = 1.0,
    equal_quota: bool = True,
    quotas: Dict[str, Dict[int, int]] | None = None,
    clip_nbors: bool = False,
    seed: int | None = None,
    modes: Tuple[str, ...] = ("v", "e"),
) -> Dict[str, np.ndarray]:
    """Metric (geodesic-distance) ring neighbourhoods — drop-in geometry alternative to
    :func:`extract_geodesic_ring_neighborhoods`.

    Parameters
    ----------
    inc : dict
        Incidence bundle with ``d_0`` (edges×verts) and ``d_1`` (faces×edges) scipy sparse matrices —
        the same bundle the dataset already passes to the hop extractor.
    pos : np.ndarray (n_v, 2 or 3)
        Vertex coordinates — needed to weight the graph by Euclidean edge length and to place edge
        midpoints / face centroids.
    v_max, e_max, f_max : int
        Window widths per element type.
    n_rings : int
        Number of equal-width geodesic-distance bands.
    ring_width : float or None
        Physical band width Δ. ``None`` → auto = mean adjacency-edge length for that stream (so one
        ring ≈ one edge-length of geodesic distance).
    radius : float or None
        Max geodesic distance searched (Dijkstra ``limit``). ``None`` → ``n_rings * ring_width``.
    decay : float
        Only used when ``equal_quota=False`` — geometric per-ring budget decay (see
        ``default_ring_quotas``).
    equal_quota : bool
        ``True`` (default): every distance ring gets the same slot budget (uniform sampling in
        distance space). ``False``: geometric decay toward the near field, like the hop extractor.
    quotas : dict or None
        Optional explicit ``{mode: {ring: budget}}`` override.
    clip_nbors : bool
        Cap each window at ``sqrt(n_elem)`` like the stock extractor.
    seed : int or None
        RNG seed for the per-ring sub-sampling (vary per epoch for augmentation).
    modes : tuple of str
        Which element neighbourhoods to build; subset of ``("v", "e", "f")``.

    Returns
    -------
    dict
        ``{"v_idx","v_ring","e_idx","e_ring","f_idx","f_ring"}`` for the requested modes — identical
        format to :func:`extract_geodesic_ring_neighborhoods`; ``*_ring`` now holds the
        distance-band id (0..n_rings-1) of each slot instead of the hop count.
    """
    rng = np.random.RandomState(seed)
    P = _element_positions(inc, pos)
    d = {"d_0": inc.get("d_0"), "d_1": inc.get("d_1")}
    windows = {"v": v_max, "e": e_max, "f": f_max}

    out: Dict[str, np.ndarray] = {}
    for mode in modes:
        key, transpose = _SOURCE[mode]
        mtrx = d.get(key)
        if mtrx is None or mode not in P:
            continue
        mtrx = mtrx.T.tocsr() if transpose else mtrx.tocsr()
        n_max = windows[mode]
        if clip_nbors:
            n_max = max(1, min(n_max, int(mtrx.shape[0] ** 0.5)))
        q = quotas.get(mode) if quotas is not None else None
        idx, ring = _rings_for_mode_metric(
            mtrx, P[mode], n_max, n_rings, ring_width, radius, q, equal_quota, decay, rng
        )
        out[f"{mode}_idx"] = idx
        out[f"{mode}_ring"] = ring

    return out
