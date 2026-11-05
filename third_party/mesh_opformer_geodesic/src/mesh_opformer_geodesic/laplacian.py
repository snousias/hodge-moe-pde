"""
laplacian.py
------------
Metric-weighted discrete boundary operators for HodgeFormer.

The stock HodgeFormer boundary operators are COMBINATORIAL: d_0 (v2e) and d_1 (e2f) hold ±1
(oriented incidence) and 0 elsewhere, so the discrete Hodge-Laplacian L₀ = d₀ᵀ ⋆₁ d₀ knows only
graph connectivity, not geometry. This module rescales the nonzero incidence entries by the
**length of the connected edge** (or a function of it), so the operator becomes metric-aware:
each edge's contribution to the boundary is weighted by |e|.

    d_0[e, v]  :  ±1                    ->   ±|e|            (edge e = (v_a, v_b), |e| = ‖x_a − x_b‖)
    d_1[f, e]  :  ±1                    ->   ±|e|

Selectable via a config flag (`laplacian_weight`): "unit" (default, ±1 = stock), "length" (±|e|),
"inv_length" (±1/|e|), "sqrt_length" (±√|e|). Sign/orientation is always preserved, so d∘d = 0
still holds and dᵀ(·)d stays a valid Laplacian.
"""
import numpy as np
from scipy.sparse import diags


def edge_lengths_from_incidence(v2e, pos):
    """Primal edge lengths from the vertex→edge incidence and vertex positions.

    Parameters
    ----------
    v2e : scipy.sparse (n_v, n_e)
        Signed vertex→edge incidence (each edge column has exactly two ±1 entries).
    pos : np.ndarray (n_v, 2 or 3)
        Vertex coordinates.

    Returns
    -------
    np.ndarray (n_e,)
        Euclidean length of each edge.
    """
    coo = v2e.tocoo()
    n_e = v2e.shape[1]
    # group the two endpoints of each edge: sort by (col=edge, row=vertex) -> contiguous pairs
    order = np.lexsort((coo.row, coo.col))
    cols = coo.col[order]; rows = coo.row[order]
    counts = np.bincount(cols, minlength=n_e)
    if not np.all(counts == 2):
        # non-manifold / dangling edge: fall back to a robust per-edge distance (0-length -> eps)
        lengths = np.zeros(n_e, dtype="float64")
        start = 0
        for e in range(n_e):
            k = counts[e]
            if k >= 2:
                v = rows[start:start + k]
                lengths[e] = float(np.linalg.norm(pos[v[0]] - pos[v[1]]))
            start += k
        return lengths
    verts = rows.reshape(n_e, 2)                      # (n_e, 2) endpoints, in edge-index order
    d = pos[verts[:, 0]].astype("float64") - pos[verts[:, 1]].astype("float64")
    return np.linalg.norm(d, axis=1)


def _weights(lengths, mode, eps=1e-8):
    if mode == "length":
        return lengths
    if mode == "inv_length":
        return 1.0 / (lengths + eps)
    if mode == "sqrt_length":
        return np.sqrt(np.maximum(lengths, 0.0))
    raise ValueError(f"unknown laplacian_weight mode: {mode!r} "
                     "(use 'unit'|'length'|'inv_length'|'sqrt_length')")


def length_weighted_incidence(v2e, e2f, pos, mode="length"):
    """Return (v2e_w, e2f_w, edge_lengths) with the incidence rescaled by the edge-length weight.

    `v2e` is (n_v, n_e) -> scale each EDGE COLUMN by w_e.
    `e2f` is (n_e, n_f) -> scale each EDGE ROW by w_e (or None to skip).  Sign preserved.
    """
    L = edge_lengths_from_incidence(v2e, pos)
    w = _weights(L, mode)
    W = diags(w)
    v2e_w = (v2e @ W).tocoo()                          # column-scale (edges are columns of v2e)
    e2f_w = (W @ e2f).tocoo() if e2f is not None else None   # row-scale (edges are rows of e2f)
    return v2e_w, e2f_w, L
