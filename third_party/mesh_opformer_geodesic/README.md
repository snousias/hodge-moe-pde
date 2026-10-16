# mesh_opformer_geodesic

Geodesic-ring sparse attention for HodgeFormer — a focused sibling package
(like `mesh_opformer_3_mesh`). It does **not** modify `mesh_opformer` or
`mesh_sim`; it reuses their BFS-level and gather kernels and adds two things.

## Why

The stock `bigbird` local neighbourhood (`modes=["l"]`) runs BFS to `k` hops and
then keeps the `n` **smallest-node-index** members of that ball
(`build_adj_array` → `compactify(how="smallest")`). On a ~5k-node airfoil remesh
the k-hop ball is far larger than `n`, so:

- the ring / hop structure is thrown away, and
- the far field is kept or dropped essentially by node id — no guarantee the
  receptive field is multi-scale.

Widening the window (v18 128 → v18_6 256 → 512) helps but pays O(window) VRAM
for unstratified slots.

## What it adds

1. **`extract_geodesic_ring_neighborhoods(...)`** — selects the `n` slots
   **stratified by hop-ring**: near rings dense, far rings geometrically
   sub-sampled (`default_ring_quotas`). Same window width, genuinely multi-scale
   surface-intrinsic receptive field. Varying `seed` per epoch makes the
   far-field sub-sample a cheap, geometry-preserving data augmentation.
   Returns `*_idx` (the usual `-1`-padded gather indices) **and** `*_ring` (the
   hop distance of each slot).

2. **`RingBias` + `ring_sparse_attention(...)`** — a learned per-(head, ring)
   additive bias on the attention logits, indexed by each neighbour's hop
   distance (ALiBi-style, but the metric is mesh-graph hops). Zero-initialised,
   so training starts as an exact no-op vs. the stock kernel.

## Install

```bash
conda run -n hodgeformer pip install -e packages/mesh_opformer_geodesic
conda run -n hodgeformer python -m pytest packages/mesh_opformer_geodesic/tests -q
```

## Wiring into a HodgeFormer run (sketch — not yet applied)

**Data side** (in a case's `dataset_*.py` `_compute_geometry`): replace the
`bigbird`/`extract_kw` call that produces `v_idx`/`e_idx`/`f_idx` with

```python
from mesh_opformer_geodesic import extract_geodesic_ring_neighborhoods
nb = extract_geodesic_ring_neighborhoods(
    {"d_0": d_0_scipy, "d_1": d_1_scipy},
    v_max=cfg.v_max, e_max=cfg.e_max, max_hops=8, seed=epoch_seed, modes=("v", "e"),
)
geo["v_idx"], geo["v_ring"] = nb["v_idx"], nb["v_ring"]   # + e_idx / e_ring
```

carry `*_ring` through the collate (same `-1` pad path as `*_idx`).

**Model side** (only if using the learned bias): in the grouped-Hodge path,
build the bias once per layer from `ring` and pass it to
`ring_sparse_attention` instead of `sparse_attention`. The neighbourhood change
(1) is independently useful with the **unmodified** model — start there, add the
bias (2) as a second ablation.

## Suggested ablations

- `geo-a`: v18_6 recipe, **rings only** (unbiased attention), window 128 →
  should match/beat 256-Euclidean at half the VRAM.
- `geo-b`: `geo-a` + `RingBias`.
- `geo-c`: sweep `decay` (0.4 concentrated ↔ 0.7 spread) and `max_hops`.
