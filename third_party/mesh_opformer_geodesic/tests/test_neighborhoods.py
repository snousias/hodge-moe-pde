"""
Smoke tests for geodesic-ring neighbourhoods + ring attention.

Run:  conda run -n hodgeformer python -m pytest packages/mesh_opformer_geodesic/tests -q
(requires mesh_sim's graphblas stack + mesh_opformer in the env)
"""
import numpy as np
import scipy.sparse as sp
import torch

from mesh_opformer_geodesic import (
    extract_geodesic_ring_neighborhoods,
    default_ring_quotas,
    RingBias,
    ring_sparse_attention,
)
from mesh_opformer.layers.attention.attention_sparse import sparse_attention


def _grid_d0(n=8):
    """d_0 (edges x verts, +-1) of an n x n triangulated grid — a stand-in mesh."""
    idx = lambda r, c: r * n + c
    rows, cols, vals, e = [], [], [], 0
    def edge(a, b):
        nonlocal e
        rows.extend([e, e]); cols.extend([a, b]); vals.extend([-1, 1]); e += 1
    for r in range(n):
        for c in range(n):
            if c + 1 < n: edge(idx(r, c), idx(r, c + 1))
            if r + 1 < n: edge(idx(r, c), idx(r + 1, c))
            if r + 1 < n and c + 1 < n: edge(idx(r, c), idx(r + 1, c + 1))
    return sp.csr_matrix((vals, (rows, cols)), shape=(e, n * n))


def test_quotas_sum_and_monotonic():
    q = default_ring_quotas(n_max=128, max_hops=8)
    assert q[0] == 1
    assert sum(q.values()) <= 128 + 8   # rounding slack
    # near rings get at least as many slots as far rings
    assert q[1] >= q[2] >= q[8] >= 1


def test_ring_neighbourhood_shapes_and_selfslot():
    d0 = _grid_d0(8)
    out = extract_geodesic_ring_neighborhoods(
        {"d_0": d0}, v_max=24, max_hops=6, modes=("v",), seed=0
    )
    v_idx, v_ring = out["v_idx"], out["v_ring"]
    n = 64
    assert v_idx.shape == (n, 24) and v_ring.shape == (n, 24)
    # slot 0 is the element itself, ring 0
    assert (v_idx[np.arange(n), 0] == np.arange(n)).all()
    assert (v_ring[np.arange(n), 0] == 0).all()
    # rings are non-decreasing across slots (nearest-first), ignoring -1 pad
    for i in range(n):
        r = v_ring[i]; r = r[r >= 0]
        assert (np.diff(r) >= 0).all()
    # far rings actually present (multi-scale, not a 1-ring ball)
    assert v_ring.max() >= 3


def _grid_pos(n=8):
    """(n*n, 2) coordinates of the n x n grid used by ``_grid_d0``."""
    return np.array([[r, c] for r in range(n) for c in range(n)], dtype="float64")


def test_metric_ring_shapes_selfslot_and_bands():
    from mesh_opformer_geodesic import extract_metric_ring_neighborhoods, equal_ring_quotas
    n = 8
    d0 = _grid_d0(n); pos = _grid_pos(n)
    n_rings = 5
    out = extract_metric_ring_neighborhoods(
        {"d_0": d0}, pos, v_max=24, n_rings=n_rings, modes=("v",), seed=0
    )
    v_idx, v_ring = out["v_idx"], out["v_ring"]
    N = n * n
    assert v_idx.shape == (N, 24) and v_ring.shape == (N, 24)
    # slot 0 is the element itself, ring 0 (guaranteed even though near neighbours share band 0)
    assert (v_idx[np.arange(N), 0] == np.arange(N)).all()
    assert (v_ring[np.arange(N), 0] == 0).all()
    # distance-ring labels span 0 (self) .. n_rings (the n_rings bands are rings 1..n_rings)
    assert v_ring.max() <= n_rings
    # rings non-decreasing across slots (nearest-first)
    for i in range(N):
        r = v_ring[i]; r = r[r >= 0]
        assert (np.diff(r) >= 0).all()
    # every valid gather index is in range and not a pad
    valid = v_idx[v_idx >= 0]
    assert valid.min() >= 0 and valid.max() < N


def test_equal_ring_quotas_uniform():
    from mesh_opformer_geodesic import equal_ring_quotas
    q = equal_ring_quotas(n_max=257, n_rings=8)
    assert q[0] == 1                      # self
    band = [q[r] for r in range(1, 9)]    # 8 distance bands
    assert max(band) - min(band) <= 1     # equal up to remainder
    assert sum(q.values()) == 257


def test_metric_deterministic():
    from mesh_opformer_geodesic import extract_metric_ring_neighborhoods
    d0 = _grid_d0(7); pos = _grid_pos(7)
    a = extract_metric_ring_neighborhoods({"d_0": d0}, pos, v_max=20, n_rings=6, modes=("v",), seed=3)
    b = extract_metric_ring_neighborhoods({"d_0": d0}, pos, v_max=20, n_rings=6, modes=("v",), seed=3)
    assert np.array_equal(a["v_idx"], b["v_idx"]) and np.array_equal(a["v_ring"], b["v_ring"])


def test_ring_bias_zero_is_noop():
    """ring_sparse_attention with zero-init RingBias == stock sparse_attention."""
    torch.manual_seed(0)
    b, h, n, d, j = 1, 4, 20, 16, 6
    x = torch.randn(b, h, n, d)
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    idx = torch.randint(0, n, (b, n, j))
    ring = torch.randint(0, 6, (b, n, j))

    bias = RingBias(h, max_ring=8)(ring)          # zero-init -> all zeros
    out_ring = ring_sparse_attention(x, q, k, idx, ring_bias=bias)
    out_base = sparse_attention(x, q, k, idx)
    assert torch.allclose(out_ring, out_base, atol=1e-6)


if __name__ == "__main__":
    test_quotas_sum_and_monotonic()
    test_ring_neighbourhood_shapes_and_selfslot()
    test_metric_ring_shapes_selfslot_and_bands()
    test_equal_ring_quotas_uniform()
    test_metric_deterministic()
    test_ring_bias_zero_is_noop()
    print("ok")
