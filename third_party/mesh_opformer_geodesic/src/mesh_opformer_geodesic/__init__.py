"""
mesh_opformer_geodesic
======================
Geodesic-ring sparse attention for HodgeFormer.

Two contributions over the stock ``bigbird`` local neighbourhood:

1. :func:`extract_geodesic_ring_neighborhoods` — replaces the "k-hop ball
   truncated by node id" selection with a **hop-ring-stratified** selection:
   dense near rings, geometrically sub-sampled far rings. Same window width,
   genuinely multi-scale surface-intrinsic receptive field.

2. :class:`RingBias` + :func:`ring_sparse_attention` — a learned
   relative-geodesic positional bias added to the attention logits, indexed by
   each neighbour's hop distance. Zero-initialised (starts as an exact no-op).

3. :func:`extract_metric_ring_neighborhoods` — a geometry-aware variant of (1)
   that rings by **true geodesic distance** (Dijkstra with Euclidean edge-length
   weights) instead of BFS hop count, binned into equal-width distance bands with
   an equal per-band slot quota. On anisotropic meshes (boundary layers) a
   distance band is physically meaningful where a hop count is not. Same output
   format as (1); ``*_ring`` holds the distance-band id (0 = self, 1..n_rings =
   bands) — so with ``n_rings == max_hops`` it drops straight into the same
   ``RingBias``. See :mod:`.geodesic_metric`.

Reuses ``mesh_sim`` (BFS levels) and ``mesh_opformer`` (gather kernels); it does
not modify either package.
"""

from .neighborhoods import (
    extract_geodesic_ring_neighborhoods,
    default_ring_quotas,
)
from .geodesic_metric import (
    extract_metric_ring_neighborhoods,
    equal_ring_quotas,
)
from .layers.ring_attention import RingBias, ring_sparse_attention
from .laplacian import length_weighted_incidence, edge_lengths_from_incidence

__all__ = [
    "extract_geodesic_ring_neighborhoods",
    "default_ring_quotas",
    "extract_metric_ring_neighborhoods",
    "equal_ring_quotas",
    "RingBias",
    "ring_sparse_attention",
    "length_weighted_incidence",
    "edge_lengths_from_incidence",
]
