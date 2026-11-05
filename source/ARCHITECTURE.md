# HodgeMoE v57 gated-Hodge — architecture

A frequency mixture of experts for steady-state PDE regression on unstructured meshes. The
per-vertex expert gate comes from a **Hodge spectral controller** instead of a feature-space MLP
router. Validated on AirFRANS (2D RANS airfoil flow): the model predicts `u_x, u_y, p, ν_t` on
about 4.9k mesh vertices from geometry and boundary conditions.

This release contains two checkpoints of this architecture, differing only in the controller's
`modes` flag:

| checkpoint | config | `modes` | origin case | params | rel-L2 |
|---|---|---|---|--:|--:|
| `hodgemoe_v_best_ep380.pth` | `config_v.toml` | `"v"` | `v57_gated_hodge` | 3,023,962 | 0.0300 |
| `hodgemoe_vef_best_ep400.pth` | `config.toml` | `"vef"` | `v57_gated_hodge_vef_v2` | 3,106,490 | 0.0329 |

At `modes = "v"` the controller puts a Hodge-Laplacian on the vertex stream alone. At
`modes = "vef"` every cochain stream has one:

| stream | operator | composition |
|---|---|---|
| vertices | L₀ | `⋆₀⁻¹ d₀ᵀ ⋆₁ d₀` |
| edges | L₁ = L₁ᵈᵒʷⁿ + L₁ᵘᵖ | `d₀ ⋆₀⁻¹ d₀ᵀ ⋆₁` + `⋆₁⁻¹ d₁ᵀ ⋆₂ d₁` |
| faces | L₂ᵈᵒʷⁿ | `d₁ ⋆₁⁻¹ d₁ᵀ ⋆₂` |

These follow the mode structure of the original HodgeFormer layer (`MeshHodgeFormerAttention`)
term by term. The only difference is the stars: each is the symmetric-PSD Gaussian kernel of
`source/arch/psd_hodge.py` instead of a row-softmax attention.

At `modes = "v"` the edge and face streams are still embedded and refined by their own
position-wise FFNs, and the edge stream reaches the vertex operator through the edge star ⋆₁; only
the face stream is read by nothing. The `"v"` checkpoint was trained and evaluated under
bf16 autocast, the `"vef"` checkpoint in fp32; the code in this deployment is fp32 (see
`checkpoints/PROVENANCE.md`, "Precision").

## TL;DR

Two branches read the **same** per-vertex input `x_v`:

- **Expert branch**: 8 vanilla-transformer experts. Fourier positional features at one spatial
  frequency scale bias each expert towards that scale. A per-vertex softmax gate blends them. This
  branch never sees the mesh (no edges, faces or incidence matrices).
- **Hodge branch**: a 2-layer Hodge-Laplacian attention stack. It receives the simplicial complex
  (vertices, edges, faces and their incidence matrices) and outputs the gate logits. It never
  predicts a field.

The two branches train jointly, end to end. The loss is the field-prediction loss plus an
auxiliary term that pins expert *k* to frequency band *k* of the *target*. The target bands come
from a third component that exists only during training: a learned Hodge-Laplacian **analyzer**.
It decomposes each ground-truth field into telescoping frequency atoms and does not run at
inference.

```
                      ┌──────────────┐
              x_v ──▶ │ Hodge branch │──▶ gate logits (N, 8)
              x_e     │ 2 PSD-Hodge  │
              x_f     │ layers, vef  │
              d_0,d_1 └──────────────┘
                             │
              x_v ──▶ embed ─┴─▶ [ expert_0 .. expert_7 ] ──gated mix──▶ decoder ──▶ û (N, 4)

  (training only, auxiliary)
  target y ──▶ Hodge analyzer ──▶ 9 frequency atoms ──▶ assignment P (8×9) ──▶ 8 target bands
                                                                        ──▶ supervise each expert's
                                                                            own band head
```

The architecture figure is [`figures/overview_landscape.png`](../figures/overview_landscape.png) (from the paper).

## Inputs and notation

One sample is one whole mesh with `N ≈ 4.9k` vertices. `batch_size = 1` is required for
correctness, because attention treats every vertex of one mesh as one sequence. Do not increase
it without first changing the model to handle batched meshes.

| tensor | shape | meaning | used by |
|---|---|---|---|
| `x_v` | `(N, 9)` | vertex features | expert branch, Hodge branch |
| `x_e` | `(N_e, 21)` | edge (1-cell) DEC features | Hodge branch, analyzer (training only) |
| `x_f` | `(N_f, 13)` | face (2-cell) DEC features | Hodge branch |
| `d_0` | `(N_e, N)` sparse | vertex→edge incidence (coboundary) | Hodge branch, analyzer |
| `d_1` | `(N_f, N_e)` sparse | edge→face incidence | Hodge branch |
| `v_idx`, `e_idx` | `(N, 256)`, `(N_e, 256)` | neighbour supports (BFS hop rings, nearest first) | Hodge branch |
| `f_idx` | `(N_f, 1)` | face support (self only) | Hodge branch (⋆₂ is diagonal, see below) |

The 9 vertex channels are `[x, y | dual-area | inlet u_x, u_y | SDF | wall n_x, n_y | is_wall]`.
Each sample is non-dimensionalised by its own freestream `U∞` and then z-scored globally. Both
steps are inverted before any metric is computed (`data/dataset.py:redim_targets`, `uinf`).

Targets: `y = (u_x, u_y, p, ν_t) ∈ ℝ^(N×4)`.

## The Hodge branch (`source/arch/hodge_gate.py`, `source/arch/psd_hodge.py`)

The branch has two layers of **symmetric-PSD Gaussian Hodge-Laplacian attention**, which is a
diffusion operator and not softmax attention. Each star (discrete Hodge `⋆` operator) mixes a
cell with its neighbours:

```
⋆ x = (1 − β) x + β Ŝ x,     β = ½·sigmoid(θ) ∈ (0, ½)
Ŝ = D^(-1/2) W D^(-1/2),     w_ij = exp(−‖φ_i − φ_j‖² / 2σ²)     RBF kernel, tied Q=K, σ learned per head
```

All three operators read the same pre-update streams and are applied at the same time, as in
the stock layer. Each stream then passes through its own pre-norm residual FFN.

**⋆₂ is diagonal.** The cached face support `f_idx` is one column wide (self only). On a
width-1 support the Gaussian star equals the identity for every φ, because `w_ii = 1` gives
`Ŝ = 1`. A Gaussian ⋆₂ would therefore silently drop the face kernel. `psd_diag_star` uses the
classical DEC form instead: a positive per-face scaling (a learned area),
`λ = exp(−‖φ‖²/pσ²) ∈ (0,1]`, with eigenvalues in `[1−β, 1]`.

**Support windows.** The vertex operators use the inner `window = 64` of the 256-wide vertex
support. The edge supports are truncated separately to `window_e = 32`, because edges outnumber
vertices about 3:1. A read-out MLP then maps the final vertex stream to per-vertex logits over
the 8 experts.

**How faces reach the gate.** The read-out is per-vertex, so `x_f` can influence it only
through the complex. In layer 1, the L₁ᵘᵖ term `⋆₁⁻¹ d₁ᵀ ⋆₂ d₁` carries face information onto
the edges. In layer 2, the edge star ⋆₁ inside L₀ carries it onto the vertices.
`source/diagnostics/verify_vef.py` checks this path: perturbing `x_f` moves the gate under `"vef"`,
and under `"v"` the gate does not move at all. Under `"v"` the face embedding never trains. Here
it does.

**Does the PSD guarantee hold? No.** It requires a symmetric kernel `W`, which in turn requires
a symmetric neighbour support (j is in i's list if and only if i is in j's list).
`symmetric_support = true` symmetrises the 256-wide lists, but truncating them to the inner 64
(vertices) or 32 (edges) breaks the symmetry again. On a test mesh, 2.6% of directed vertex
neighbour pairs have no reverse pair at width 256, and 76% at width 64. This was measured for
the `"v"` release on the same cache and windows. Edge supports at width 32 were not measured.
`W` is therefore not symmetric, and the operators can have complex eigenvalues. The reported
metrics are measured and unaffected, but the operator should not be called PSD.

## The expert branch (`source/arch/experts.py`)

Each expert computes `Linear(x_v) + FourierFeatures(pos, σ_k) → project → 2× vanilla
self-attention block → project out → (representation, own band prediction)`. `FourierFeatures`
is a NeRF-style sin/cos positional encoding at one spatial scale `σ_k ∈ [1.0, …, 38.0]`. Expert
settings: `d_expert = 88`, 8 heads, FFN width 352. The gate blends the 8 representations
(not the band predictions), and a LayerNorm and the cross-field attention decoder follow:
4 field tokens per vertex, 2 blocks of 4 heads, and a shared 64-wide read-out.

## The analyzer (`source/arch/analyzer.py`): training only, defines the supervision

The analyzer is a separate Hodge Laplacian `L₀ = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀` with **diagonal** stars:
`⋆₁ = diag(softplus(MLP(x_e)))` is an edge conductance, and `⋆₀` is the resulting vertex degree.
It smooths the target `y` with `(I − ½L₀)^m` for `m ∈ [2, 4, 6, 8, 12, 16, 24, 32]` and
telescopes the differences into 9 atoms that sum back to `y` exactly. A learned
column-stochastic assignment `P` (8×9) groups the atoms into one band per expert, so the 8
bands also sum to `y` exactly.

A one-sided floor loss penalises only a band whose energy drops below 2% of the total
(`band_min_share`). It does not force balance across bands, because the fields are lopsided:
ν_t has about 40% of its energy in the sharpest band, while u_x, u_y and p are 90–98% smooth.

## Loss (`source/training/losses.py`, `config.toml [train]`)

```
loss = volume_loss_lambda  · MSE(non-wall nodes)       # "Volume" in the eval metric
     + surface_loss_lambda · MSE(wall nodes)            # "Surf"   in the eval metric (disjoint node sets)
     + band_loss_lambda    · Σ_k MSE(expert_k band head, target band k)   # targets detached
     + floor_lambda        · one-sided starved-band penalty
     + balance_lambda      · gate-collapse penalty
```

The loss is computed on a relative, per-sample, per-field basis (`per_case_norm`). The
optimiser is AdamW (lr 1e-3, weight decay 1e-4) with a per-batch OneCycle schedule: 4e-5, rising
to 1e-3 at 30% of training, then cosine-annealed to about 4e-9 over 400 epochs.
`cl_loss_lambda = 0`: in an earlier ablation, adding an explicit lift-integral term made the
lift metric worse.

## Parameters

| | `modes = "v"` | `modes = "vef"` |
|---|--:|--:|
| total | **3,023,962** | **3,106,490** |
| the e/f operators, present only at `vef` | – | 82,528 |
| never updated in training (see below) | 131,401 | 179,913 |

Peak GPU memory for one forward+backward pass in fp32 at `modes = "vef"` is 3.31 GiB on a
4,755-vertex mesh (`source/diagnostics/verify_vef.py`).

## Known limitations

Each item below comes from auditing the released `vef` checkpoint (epoch 400; epoch 350 gives the
same numbers): parameters with no optimizer state, the trained assignment logits, and the β values.
None of them changes the reported metrics, which are measured, but several limit what the model can
be said to *do*. The `"v"` figures are from the same audit applied to `hodgemoe_v_best_ep380.pth`.

- **63 of 442 parameter tensors (179,913 parameters) never receive an update** at `modes = "vef"`.
  They have no optimizer state. They are still counted in the 3,106,490 total.
  | module | params | why |
  |---|--:|---|
  | `moe.gate` | 12,872 | the expert branch's own MLP gate. The Hodge branch supplies the gate logits instead |
  | `analyzer.edge_mlp` | 1,473 | training-only analyzer. `detach_band_targets = true` cuts its gradient |
  | `hodge_gate.hodge` | 165,568 | the face operator L₂ᵈᵒʷⁿ (`f_attn`), its norm and FFN **in both layers**; the edge operator L₁ (`e_attn`), its norm and FFN in layer 2; `attn.v_phi` in both layers |

  So `vef` is effectively **L₀ in both layers plus L₁ in layer 1**. Layer 1's edge operator,
  including the ⋆₂-weighted face term, is the only addition over `"v"` that affects the output.
  Layer 1's face update feeds only layer 2's edge and face operators, and nothing reads their
  output: the read-out is per-vertex, and layer 2's vertex operator reads the layer-2 *input*
  edges. A 1-layer-deeper read-out, or L₂ only in layer 1, would remove this dead weight. This
  was not tried.
- **The analyzer does not learn.** Its band definitions come from the conductance MLP's initial
  (random) weights. Only the assignment `P` trains.
- **`P`'s diagonal initialisation is overwritten.** `build_model` sets logit 2.0 on the diagonal
  (expert k ← atom k) and then runs `xavier_uniform_` over every parameter with more than one
  dimension, including `assign.logits`, so `P` starts random. In the trained checkpoint the
  diagonal is the row maximum in 0 of 8 rows.
- **The PSD claim does not hold**, because of the window truncation (see "The Hodge branch").
- **β is capped at ½** (`β = ½·sigmoid(θ)`). Layer 2's vertex-operator β reaches 0.44–0.50, at
  the cap.
- **The dataset computes a second band decomposition that training never uses.** It builds 4
  fixed bands (`band_kw.ladder = [2, 8, 32]`) and appends them to the target tensor, but the
  band loss uses the analyzer's atoms. The pipeline is kept because the checkpoint was trained
  with it.
- **`sigmas` are fixed buffers.** The per-expert Fourier scales never receive a gradient.

At `modes = "v"` the same audit gives **34 of 390 tensors, 131,401 parameters**: `f_ff` and `n_f2`
in both layers, `e_ff` and `n_e2` in layer 2, `attn.v_phi` in both layers, the face-feature
embedding `hodge_gate.embed.f_embed`, `moe.gate`, and `analyzer.edge_mlp`. The per-module breakdown
is in `checkpoints/PROVENANCE.md`.

None of these is changed in this deployment, because the code must stay identical to the code that
produced the released checkpoints. The one exception is `PSDHodgeLayer.n_f1`, the face pre-norm.
The `vef` code created it at every `modes` setting and applied it unconditionally. At
`modes = "v"` nothing reads its output, so it was 2·d_f parameters that received no gradient and
reached no output, and the `"v"` checkpoint, trained by code that did not have it, could not be
loaded strictly. `source/arch/psd_hodge.py` now creates `n_f1` only when `"e"` or `"f"` is in `modes`.
At `modes = "vef"` the module graph and the forward computation are unchanged: the `vef`
checkpoint still loads strictly, still reports 3,106,490 parameters, and produces outputs bitwise
equal to those of the previous code.
