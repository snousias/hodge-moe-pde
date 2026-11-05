"""
neighborhoods.py
----------------
Geodesic-ring sparse-attention neighbourhoods for HodgeFormer.

Motivation
==========
The stock ``bigbird`` ``modes=["l"]`` neighbourhood does BFS out to ``k`` hops and
then keeps the ``n`` **smallest-node-index** members of that ball
(``build_adj_array`` -> ``compactify(how="smallest")``). Once the k-hop ball is
larger than ``n`` (always, on a ~5k-node airfoil remesh) the truncation is by
node id, which:

  * discards the ring / hop structure entirely, and
  * gives no guarantee that far rings are represented at all — the attention
    receptive field is whatever the low-id nodes happen to be.

This module keeps the same BFS substrate (``mesh_sim`` graphblas levels) but
selects the ``n`` slots **stratified by hop-ring**: near rings kept densely,
far rings sub-sampled with a geometric budget. That yields a genuine
multi-scale, surface-intrinsic neighbourhood (dense local + coarse far-field)
for the *same* ``n`` — i.e. a wider effective receptive field per byte of VRAM.

It also returns, alongside the index array, the **hop-distance (ring id) of each
selected slot**, so a downstream attention can add a learned relative-geodesic
positional bias (see ``layers/ring_attention.py``).

Output format is identical to the stock extractor: ``(n_elem, n_max)`` int32
arrays padded with ``-1``, ready for the grouped/sparse-attention gather.
"""

from typing import Dict, List, Tuple

import numpy as np

# mesh_sim.utils_ml runs graphblas.init() at import time; import it before the
# bfs module (same ordering constraint the 3-mesh package documents).
from mesh_sim.utils_ml import pad_neighbors  # noqa: F401  (re-exported for parity)
from mesh_sim.algos.bfs import get_bfs_matrix


# Which incidence / line-graph adjacency to walk for each element type, matching
# the convention used throughout mesh_opformer / mesh_opformer_3_mesh.
_INC_TYPE = {"v": "v2e", "e": "e2v", "f": "f2e"}


def default_ring_quotas(
    n_max: int, max_hops: int, decay: float = 0.55
) -> Dict[int, int]:
    """
    Geometric per-ring slot budget summing to (about) ``n_max``.

    Ring 0 (the element itself) always gets exactly one slot. The remaining
    ``n_max - 1`` slots are split across rings ``1..max_hops`` with weights
    ``decay**(r-1)`` — so ring 1 is densest and each further ring gets a
    ``decay`` fraction, but every reachable ring is guaranteed at least one slot.

    Parameters
    ----------
    n_max : int
        Total neighbour slots per element (the attention window width).
    max_hops : int
        Number of BFS rings considered.
    decay : float
        Geometric decay of the per-ring budget (0 < decay <= 1). Smaller ->
        more concentrated on the near field.

    Returns
    -------
    Dict[int, int]
        Mapping ``ring -> slot budget``.
    """
    quotas = {0: 1}
    remaining = max(n_max - 1, 0)
    weights = [decay ** (r - 1) for r in range(1, max_hops + 1)]
    total_w = sum(weights) or 1.0
    for r, w in zip(range(1, max_hops + 1), weights):
        quotas[r] = max(1, int(round(remaining * w / total_w)))
    return quotas


def _select_ring_stratified(
    cols: np.ndarray,
    levels: np.ndarray,
    n_max: int,
    quotas: Dict[int, int],
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pick <= ``n_max`` neighbours of one element, stratified by hop-ring.

    Per ring: keep everything if the ring has <= quota members, else randomly
    sub-sample ``quota`` of them (the far-field sampling — this is also what
    makes the neighbourhood a cheap on-the-fly augmentation when ``rng`` varies).
    After the quota pass, if slots remain unfilled (sparse far rings), top up
    with the nearest not-yet-selected neighbours so the window stays full.

    Returns
    -------
    sel_cols : np.ndarray
        Selected neighbour indices (nearest-first), length <= ``n_max``.
    sel_rings : np.ndarray
        Hop distance of each selected neighbour, aligned with ``sel_cols``.
    """
    order = np.argsort(levels, kind="stable")
    cols, levels = cols[order], levels[order]

    chosen_mask = np.zeros(cols.shape[0], dtype=bool)

    # ring boundaries in the sorted array
    uniq = np.unique(levels)
    for r in uniq:
        ring_pos = np.where(levels == r)[0]
        quota = quotas.get(int(r), 1)
        if ring_pos.size <= quota:
            chosen_mask[ring_pos] = True
        else:
            pick = rng.choice(ring_pos, size=quota, replace=False)
            chosen_mask[pick] = True

    # top up to n_max with the nearest unchosen neighbours (levels already sorted)
    n_chosen = int(chosen_mask.sum())
    if n_chosen < n_max:
        spare = np.where(~chosen_mask)[0]
        need = min(n_max - n_chosen, spare.size)
        if need > 0:
            chosen_mask[spare[:need]] = True

    sel = np.where(chosen_mask)[0]
    # trim if a ring overshot the total (rare); keep nearest
    if sel.size > n_max:
        sel = sel[np.argsort(levels[sel], kind="stable")[:n_max]]

    # return nearest-first for interpretability (slot 0 == self, ring 0)
    sel = sel[np.argsort(levels[sel], kind="stable")]
    return cols[sel], levels[sel]


def _rings_for_mode(
    inc_matrix,
    inc_type: str,
    n_max: int,
    max_hops: int,
    dilations: int,
    decay: float,
    quotas: Dict[int, int] | None,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the ``(n_elem, n_max)`` index + ring arrays for one element type.
    """
    # BFS level matrix D: D[i, j] == hop distance i->j (0 on the diagonal).
    # `_max` early-stops BFS once the average collected degree passes it, so we
    # ask for a few times the window to be sure every ring up to max_hops is seen.
    D = get_bfs_matrix(
        inc_matrix,
        inc_type=inc_type,
        k=max_hops,
        dilations=dilations,
        _max=max(4 * n_max, n_max),
    )

    rows, cols, levels = D.to_coo()
    rows = np.asarray(rows)
    cols = np.asarray(cols, dtype=np.int64)
    levels = np.asarray(levels, dtype=np.int64)

    n_elem = D.nrows
    if quotas is None:
        quotas = default_ring_quotas(n_max, max_hops, decay=decay)

    idx = np.full((n_elem, n_max), -1, dtype=np.int32)
    ring = np.full((n_elem, n_max), -1, dtype=np.int32)

    # rows from graphblas to_coo are sorted ascending -> contiguous per-row spans
    starts = np.searchsorted(rows, np.arange(n_elem), side="left")
    ends = np.searchsorted(rows, np.arange(n_elem), side="right")

    for i in range(n_elem):
        s, e = starts[i], ends[i]
        if s == e:
            idx[i, 0] = i  # isolated element attends to itself only
            ring[i, 0] = 0
            continue
        sel_cols, sel_rings = _select_ring_stratified(
            cols[s:e], levels[s:e], n_max, quotas, rng
        )
        k = sel_cols.shape[0]
        idx[i, :k] = sel_cols.astype(np.int32)
        ring[i, :k] = sel_rings.astype(np.int32)

    return idx, ring


def extract_geodesic_ring_neighborhoods(
    inc: Dict[str, object],
    v_max: int = 128,
    e_max: int = 128,
    f_max: int = 1,
    max_hops: int = 8,
    dilations: int = 1,
    decay: float = 0.55,
    quotas: Dict[str, Dict[int, int]] | None = None,
    clip_nbors: bool = False,
    seed: int | None = None,
    modes: Tuple[str, ...] = ("v", "e"),
) -> Dict[str, np.ndarray]:
    """
    Build geodesic-ring neighbourhoods for the requested element types.

    Parameters
    ----------
    inc : dict
        Incidence bundle with (some of) the keys ``d_0`` (v2e, shape (m, n)) and
        ``d_1`` (e2f, shape (k, m)) as scipy sparse matrices, matching what the
        dataset already builds for d_0 / d_1. ``inc`` may also carry ``n_v`` /
        ``n_e`` / ``n_f`` counts (used only for ``clip_nbors``).
    v_max, e_max, f_max : int
        Window widths per element type.
    max_hops : int
        Number of BFS rings to consider before stratified selection.
    dilations : int
        BFS hop dilation (passed through to ``get_bfs_matrix``).
    decay : float
        Geometric per-ring budget decay (see ``default_ring_quotas``).
    quotas : dict | None
        Optional explicit ``{mode: {ring: budget}}`` override; ``None`` -> default.
    clip_nbors : bool
        If True, cap each window at ``sqrt(n_elem)`` like the stock extractor.
    seed : int | None
        RNG seed for the far-field sub-sampling (vary per epoch for augmentation).
    modes : tuple of str
        Which element neighbourhoods to build; subset of ``("v", "e", "f")``.

    Returns
    -------
    dict
        ``{"v_idx", "v_ring", "e_idx", "e_ring", "f_idx", "f_ring"}`` for the
        requested modes. ``*_idx`` are the ``-1``-padded gather indices; ``*_ring``
        hold the hop distance of each slot (``-1`` on pad), for the ring bias.
    """
    rng = np.random.RandomState(seed)

    d_0 = inc.get("d_0")  # v2e : (m edges, n verts)
    d_1 = inc.get("d_1")  # e2f : (k faces, m edges)

    # adjacency source per element (matches mesh_opformer_3_mesh.extract_tet_neighborhoods):
    #   v : v2e = d_0.T  (verts x edges)   [inc_type "v2e" -> vertex adjacency]
    #   e : e2v = d_0    (edges x verts)   [inc_type "e2v" -> edge line graph]
    #   f : f2e = d_1    (faces x edges)   [inc_type "f2e" -> face line graph]
    # d_0 is the boundary operator (m edges, n verts); d_1 is (k faces, m edges).
    sources = {
        "v": (None if d_0 is None else d_0.T.tocsr(), "v2e", v_max),
        "e": (d_0, "e2v", e_max),
        "f": (d_1, "f2e", f_max),
    }

    out: Dict[str, np.ndarray] = {}
    for mode in modes:
        mtrx, inc_type, n_max = sources[mode]
        if mtrx is None:
            continue
        if clip_nbors:
            # both inc_to_adj_matrix and the line-graph adjacency are square with
            # nrows == number of elements of this type.
            n_elem = mtrx.shape[0]
            n_max = max(1, min(n_max, int(n_elem ** 0.5)))
        q = None if quotas is None else quotas.get(mode)
        idx, ring = _rings_for_mode(
            mtrx, inc_type, n_max, max_hops, dilations, decay, q, rng
        )
        out[f"{mode}_idx"] = idx
        out[f"{mode}_ring"] = ring

    return out
