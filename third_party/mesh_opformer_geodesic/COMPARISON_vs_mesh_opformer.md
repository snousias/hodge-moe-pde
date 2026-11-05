# `mesh_opformer` vs `mesh_opformer_geodesic` — what changed, and exactly where

## 0. The short version

`mesh_opformer_geodesic` is **not a rewrite, a fork, or a patched copy** of `mesh_opformer`. It is a
small companion package that sits *beside* the original and offers three optional, geometry-aware
replacements for specific steps in the pipeline. The original package is left completely untouched.

The size difference makes the relationship obvious: `mesh_opformer` is 7,754 lines across roughly thirty
modules (models, attention layers, datasets, the training loop); `mesh_opformer_geodesic` is 618 lines
across four modules. It is an add-on, not an alternative.

The dependency runs in exactly one direction. The geodesic package imports helper functions *from*
`mesh_opformer` and `mesh_sim` and builds on top of them. Neither of those two packages contains a single
`import mesh_opformer_geodesic` — I checked, and there are none. This matters in practice: it means you
can add, change, or delete the geodesic package without any risk of breaking existing experiments, and it
is why cases like v18 and v23 continued to run unchanged while the geodesic work was going on.

One point of possible confusion worth clearing up immediately: the word "geodesic" *does* appear inside
`mesh_opformer`, in `training/utils.py` and `layers/task_utils.py` (`geodesic_label_errors`,
`all_pairs_geodesic_worker`). That is a completely unrelated piece of code — a segmentation *metric* that
measures label error using geodesic distance across a mesh surface. It has nothing to do with this
package, and finding it during a grep should not be mistaken for an integration point.

The package is installed in editable mode at
`packages/mesh_opformer_geodesic/src/mesh_opformer_geodesic`.

---

## 1. How the two packages are laid out

| | `mesh_opformer` (the base) | `mesh_opformer_geodesic` (the add-on) |
|---|---|---|
| Lines of code | 7,754 | 618 |
| Modules | `layers/` (attention, embedding, encoder, model, tasks, transformer, …), `dataset/`, `training/` | `neighborhoods.py`, `layers/ring_attention.py`, `laplacian.py`, `__init__.py` |
| What it is | The complete HodgeFormer implementation: the models, all layers, the data pipeline, the training loop | Three self-contained, optional replacements for individual steps |
| Depends on | `mesh_sim` | `mesh_sim` **and** `mesh_opformer` — it reuses their primitives rather than reimplementing them |

```
mesh_opformer_geodesic/
├── src/mesh_opformer_geodesic/
│   ├── __init__.py               # exports the six public symbols
│   ├── neighborhoods.py          # (A) geodesic-ring neighbour selection
│   ├── layers/ring_attention.py  # (B) RingBias + ring_sparse_attention
│   └── laplacian.py              # (C) length-weighted (metric) incidence
└── tests/test_neighborhoods.py
```

Everything the package exposes comes down to three ideas, described one at a time below. For each, I
first explain what the original does and why it is a limitation, then what the geodesic version does
instead.

---

## 2. Change (A) — how each node chooses its neighbours

This is the change that actually gets used in every geodesic-enabled experiment, so it is the one worth
understanding properly.

### The background: why a node needs to "choose" neighbours at all

HodgeFormer uses *sparse* attention. A mesh in these experiments has roughly 5,000 vertices, and letting
every vertex attend to every other vertex would be quadratic in both compute and memory — far too
expensive. So each element (vertex, edge, or face) attends only to a fixed-size window of `n` neighbours,
typically `n = 128` or `n = 256`. That window width is a hard budget set by GPU memory.

The interesting question is therefore: *given that you can only afford `n` neighbours, which `n` do you
pick?* Both the original and the geodesic version start from the same place — a breadth-first search
(BFS) outward from each element across the mesh graph, which labels every nearby element with its **hop
distance**: how many mesh edges you must traverse to reach it. All elements at hop distance `r` form
"ring `r`". The two implementations differ purely in how they cut that BFS result down to `n` entries.

### What the original does

In the stock path (`mesh_sim/algos/bfs.py::build_adj_array`, reached through
`mesh_sim/utils_ml.py::extract_mesh_features`), BFS runs out to `k` hops to collect a "ball" of nearby
elements, and the ball is then truncated to `n` entries by this line:

```python
khop_arr = mtrx.ss.compactify(how="smallest", k=k, asindex=True)
```

`how="smallest"` keeps the entries with the **smallest node indices**. Node indices are essentially
arbitrary labels assigned when the mesh was built — they carry no geometric meaning whatsoever.

As long as the k-hop ball happens to contain fewer than `n` elements, this is harmless: everything in the
ball is kept and nothing is thrown away. But on a ~5,000-vertex airfoil remesh the ball is essentially
always larger than the window, so the truncation is always active, and it has two unfortunate
consequences:

1. **The ring structure is discarded.** After truncation there is no record of which neighbours were
   close and which were far; the hop distances are simply dropped.
2. **There is no guarantee that distant rings survive at all.** Whether a far-away region is represented
   in the window depends entirely on whether its nodes happen to have low index numbers. The effective
   receptive field becomes an artifact of mesh numbering rather than a deliberate choice.

In other words, the model's view of its surroundings is shaped by an arbitrary labelling rather than by
the geometry of the surface.

### What the geodesic version does

`neighborhoods.py::extract_geodesic_ring_neighborhoods` keeps the same BFS, but allocates the `n` slots
**by ring**, deliberately. It gives each ring a budget that shrinks geometrically with distance, so near
rings are sampled densely and far rings sparsely — but far rings are always represented.

The budget is computed by `default_ring_quotas`. Ring 0 (the element itself) always receives exactly one
slot; the remaining `n_max − 1` slots are divided among rings `1 … max_hops` in proportion to
`decay^(r−1)`, with every reachable ring guaranteed at least one slot:

```python
quotas = {0: 1}
remaining = max(n_max - 1, 0)
weights = [decay ** (r - 1) for r in range(1, max_hops + 1)]
for r, w in zip(range(1, max_hops + 1), weights):
    quotas[r] = max(1, int(round(remaining * w / total_w)))
```

With the default `decay = 0.55`, ring 1 gets the largest share, ring 2 gets about 55% of that, ring 3
about 55% of *that*, and so on. The `max(1, …)` floor is what guarantees the far field is never
completely dropped.

Selection then proceeds ring by ring (`_select_ring_stratified`). If a ring contains no more elements
than its quota, all of them are kept. If it contains more, a random subset of exactly `quota` elements is
drawn:

```python
if ring_pos.size <= quota:
    chosen_mask[ring_pos] = True
else:
    pick = rng.choice(ring_pos, size=quota, replace=False)
    chosen_mask[pick] = True
```

If quotas do not fill the window (which happens when far rings are sparse), the remaining slots are
topped up with the nearest not-yet-chosen elements, so the window is always used to its full width.

The result is a genuinely **multi-scale, surface-intrinsic** neighbourhood: fine detail nearby, coarse
coverage far away, for exactly the same memory cost. The random draw within an over-full ring has a
useful side effect — it acts as a cheap data augmentation, since a different draw yields a slightly
different neighbourhood. The cases keep this deterministic and cacheable by seeding the generator per
mesh with `CRC32(mesh_name)`.

### What stays the same

Both paths walk the *same* BFS levels, using `mesh_sim.algos.bfs.get_bfs_matrix` — the geodesic package
does not reimplement BFS. Both emit results in the *same* format: an `(n_elem, n_max)` int32 array padded
with `-1`. Because the output format is identical, every downstream gather kernel and attention layer
works unchanged; the swap is invisible to the rest of the model.

The geodesic version additionally returns a second array, `*_ring`, holding the hop distance of each
selected slot. Nothing in the base package can consume this, which is precisely what change (B) was built
for.

The mapping from element type to the adjacency being walked follows the base convention: vertices use
`v2e` (`d_0.T`), edges use `e2v` (`d_0`), and faces use `f2e` (`d_1`).

---

## 3. Change (B) — telling attention *how far away* each neighbour is

### The gap this addresses

Standard sparse attention scores a neighbour purely by the content of its features. It has no idea
whether a given neighbour sits one hop away or twenty hops away — the window is just an unordered bag of
elements. For a mesh, where distance along the surface is meaningful physical information, that is a real
loss.

### What the original does

`mesh_opformer/layers/attention/attention_sparse.py::sparse_attention` gathers the neighbour keys and
values, computes scaled dot-product logits, masks the padding slots, applies softmax, and returns the
weighted values:

```python
qk = torch.einsum("bhnd,bhnjd->bhnj", _q, _k) * (d**-0.5)
mask = expand_dim(mask, 1, h)
qk = qk.masked_fill_(mask, float("-inf"))
qk = torch.nn.functional.softmax(qk, dim=-1)
```

### What the geodesic version adds

`layers/ring_attention.py::ring_sparse_attention` reproduces that path exactly, then adds a single
element: a learned bias term, added to the logits before the softmax, indexed by each neighbour's hop
distance.

```python
if ring_bias is not None:
    qk = qk + ring_bias
```

The bias itself is `RingBias`, a small embedding table with one learnable scalar per (attention head, hop
ring) pair — so each head can develop its own preferred sense of scale, one perhaps focusing tightly on
the immediate neighbourhood while another attends broadly. It is **initialised to exactly zero**:

```python
self.table = nn.Embedding(max_ring + 1, h)
nn.init.zeros_(self.table.weight)
```

Conceptually this is ALiBi-style relative positional bias, except that "position" is not an index in a
sequence but **hop distance across the mesh surface**.

Two deliberate design choices make it safe. First, it imports the gather and masking primitives
(`prepare_indices`, `batched_index_select`, `expand_dim`) directly *from* `mesh_opformer` instead of
copying them, so the two kernels cannot drift apart. Second, because of that reuse and the zero
initialisation, calling it with `ring_bias=None` is numerically identical to the stock kernel, and a
freshly-initialised `RingBias` is an exact no-op — training starts from stock behaviour and departs from
it only as the bias is learned.

### Status: implemented and tested, but never actually used

No experiment wires this in. `v22_geodesic` is explicitly the *rings-only* variant — its own config
describes the learned RingBias as "a separate follow-up", intended for a "v22b" case that was never run.
Every case that uses this package imports only the neighbour extractor and (sometimes) the Laplacian
weighting. So **none of the reported results involve `RingBias` or `ring_sparse_attention`.** They remain
available, and untested in training.

---

## 4. Change (C) — giving the Laplacian a sense of distance

### The gap this addresses

HodgeFormer's central operator is the discrete Hodge–Laplacian, `L₀ = d₀ᵀ ⋆₁ d₀`, built from the boundary
operators `d_0` (vertices→edges) and `d_1` (edges→faces). In the stock implementation those operators are
**combinatorial**: their non-zero entries are `+1` or `−1`, encoding only *which* elements are connected
and with what orientation.

The consequence is that the operator cannot distinguish a very short edge from a very long one. On the
anisotropic boundary-layer meshes used here that is a significant blind spot, because edge lengths span
roughly a factor of 70 — from around 0.0025 for the thin cells hugging the airfoil wall out to about
0.177 in the coarse far field. A purely combinatorial operator treats those identically.

### What the geodesic version does

`laplacian.py::length_weighted_incidence` rescales the non-zero incidence entries by a function of the
connected edge's length:

$$ d_0[e,v] : \pm 1 \;\longrightarrow\; \pm|e|, \qquad d_1[f,e] : \pm 1 \;\longrightarrow\; \pm|e|, \qquad |e| = \lVert x_a - x_b \rVert $$

Implementation is a pair of diagonal scalings — columns of `v2e` and rows of `e2f` both correspond to
edges:

```python
L = edge_lengths_from_incidence(v2e, pos)
w = _weights(L, mode)
W = diags(w)
v2e_w = (v2e @ W).tocoo()      # scale each edge COLUMN of v2e
e2f_w = (W @ e2f).tocoo()      # scale each edge ROW of e2f
```

Four modes are available: `"unit"` (the default, identical to the stock ±1), `"length"`, `"inv_length"`,
and `"sqrt_length"`. The choice is made per case through the config key `extract_kw.laplacian_weight`.

Two properties are preserved by construction, and both matter mathematically. **Sign and orientation are
untouched**, so the identity `d∘d = 0` still holds and `dᵀ(·)d` remains a valid Laplacian. And **the
sparsity pattern is unchanged** — only the stored values differ, never which entries are non-zero. So the
operator becomes metric-aware without becoming a different operator structurally.

---

## 5. Where each change lives, at file and function level

| # | Change | Implemented in (geodesic package) | Replaces (base packages) |
|---|---|---|---|
| A | Ring-stratified neighbour selection | `neighborhoods.py` — `extract_geodesic_ring_neighborhoods`, `_rings_for_mode`, `_select_ring_stratified`, `default_ring_quotas` | `mesh_sim/algos/bfs.py::build_adj_array` (the `compactify(how="smallest")` truncation), reached via `mesh_sim/utils_ml.py::extract_mesh_features` |
| B | Relative-geodesic attention bias | `layers/ring_attention.py` — `ring_sparse_attention`, `RingBias` | `mesh_opformer/layers/attention/attention_sparse.py::sparse_attention` |
| C | Metric boundary operators | `laplacian.py` — `length_weighted_incidence`, `edge_lengths_from_incidence`, `_weights` | The ±1 combinatorial `d_0`/`d_1` built by `mesh_sim` / the dataset |

Reused unchanged from the base packages: `mesh_sim.algos.bfs.get_bfs_matrix` (the BFS levels),
`mesh_sim.utils_ml.extract_mesh_features` and `pad_neighbors` (mesh geometry and incidence), and
`mesh_opformer.layers.attention.attention_sparse`'s `prepare_indices`, `batched_index_select` and
`expand_dim` (the gather and masking primitives).

---

## 6. How an experiment actually opts in

Perhaps the most important structural point: **no model and no layer is modified anywhere.** A case
adopts the geodesic neighbourhoods entirely inside its own dataset class, in the `_geometry` method. It
calls the stock feature extractor as usual, *discards* the bigbird neighbour arrays it returns, and
substitutes the geodesic ones:

```python
_, x_e, x_f, v2e, e2f, _v_bb, _e_bb, f_idx = extract_mesh_features(mesh, **ekw)   # bigbird output discarded
nb = extract_geodesic_ring_neighborhoods(
        {"d_0": v2e.T.tocsr(), "d_1": e2f.T.tocsr()},
        v_max=…, e_max=…, max_hops=…, decay=…, seed=CRC32(name), modes=("v", "e"))
v_idx, e_idx = nb["v_idx"], nb["e_idx"]
```

Note the transposes, which are an easy source of bugs: the neighbourhood builder expects `d_0` as
(edges × vertices) and `d_1` as (faces × edges), whereas the dataset stores `v2e` as (vertices × edges)
and `e2f` as (edges × faces).

The relevant config keys are `extract_kw.geodesic.{v_max, e_max, f_max, max_hops, decay}` and
`extract_kw.laplacian_weight`. Because changing the neighbour selection changes the cached geometry,
every variant requires its **own geometry cache directory** — this is why the cases each carry a distinct
`cache/<version>_{train,test}_geo`.

Which experiments use what:

| Case | Imports from the package | Actually uses |
|---|---|---|
| `v22_geodesic`, `v22_geodesic_24_256` | `extract_geodesic_ring_neighborhoods`, `length_weighted_incidence` | Ring neighbourhoods only |
| `v31` | `length_weighted_incidence` (through the v22 dataset) | The metric Laplacian (`laplacian_weight="length"`) |
| `v33`, `v34`, `v34_2`, `v35` | `extract_geodesic_ring_neighborhoods`, `length_weighted_incidence` | Ring neighbourhoods (weighting left at `"unit"`) |
| `viz_ring_selection.py`, `ring_ball_overlap.py` | Additionally `default_ring_quotas` | Analysis and figures |

---

## 7. The add-on is a strict superset of the original

Every one of the three changes collapses back to stock behaviour at its default setting:

- `ring_sparse_attention(..., ring_bias=None)` is numerically identical to `sparse_attention` — same
  gather, same mask, same softmax, using the very same primitives.
- `laplacian_weight="unit"` reproduces the stock ±1 incidence exactly.
- A freshly constructed `RingBias` is zero-valued, hence an exact no-op at step 0.

This is what makes the A/B comparisons between cases trustworthy: when a geodesic case differs from a
stock case, the difference comes from the knob that was deliberately turned, not from incidental drift
between two implementations.

---

## 8. An important empirical caveat: "geodesic" often behaves like "bigbird"

Ring stratification can only matter when there is something to stratify — that is, when the k-hop ball
is **larger** than the attention window. If the entire ball already fits inside the window, both methods
keep everything and produce the same neighbourhood.

That is very nearly the situation in most of these experiments. On the anisotropic-BL AirFRANS mesh an
8-hop ball contains only about 213 vertices, which is smaller than a 256-wide window. At `hops=8`,
`v_max=256` the two selections therefore overlap almost completely, with a measured Jaccard similarity of
0.96. Measured overlap across settings:

| `(hops, v_max)` | (8, 256) | (8, 96) | (24, 256) | (40, 128) | (60, 128) |
|---|---|---|---|---|---|
| Jaccard vs bigbird | 0.96 | 0.81 | 0.79 | 0.62 | 0.53 |

To make the geodesic selection meaningfully different, you must either increase `max_hops` (into the
24–60 range) or shrink the window (to 96–128), so that the ball genuinely overflows.

This has two consequences that should be stated plainly whenever these results are reported:

- **`v22_geodesic` was run at the null setting** (8 hops, 256 window). It is effectively a re-run of
  v18_6, and its result is *not* evidence that geodesic neighbourhoods help.
- **v33, v34, v34_2 and v35 also run at 8/256**, so their neighbourhoods are essentially
  bigbird-equivalent too. Their improvements — in particular v34_2's symmetric-PSD Hodge stars — come
  from the **model**, not from the neighbour selection. Attributing those gains to "geodesic attention"
  would be incorrect.

---

## 9. Summary

`mesh_opformer_geodesic` extends an entirely unmodified `mesh_opformer` with three optional,
geometry-aware components.

**(A)** It replaces neighbour selection based on arbitrary node numbering with selection stratified by
hop distance, producing a dense-near, sparse-far, multi-scale receptive field at identical memory cost,
and exporting each neighbour's hop distance alongside the indices.

**(B)** It adds `RingBias` and `ring_sparse_attention`, an ALiBi-style learned bias on mesh-hop distance
applied to the attention logits — implemented and tested, but never wired into any trained model.

**(C)** It provides length-weighted boundary operators that turn the purely combinatorial Laplacian
`L₀ = d₀ᵀ⋆₁d₀` into a metric-aware one, while preserving orientation, `d∘d = 0`, and sparsity.

All three default to exact stock behaviour, all three reuse the base packages' BFS and gather primitives
rather than duplicating them, and integration happens purely within each experiment's dataset code. The
practical caveat is that at the neighbourhood settings actually used in the experiments (8 hops, 256
window) the geodesic selection is close to indistinguishable from the original, so the accuracy gains
observed in v34/v34_2 should be credited to the model architecture rather than to this package.
