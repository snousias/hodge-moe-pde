# v57\_gated\_hodge — zero to hero

*A complete, formula-level walkthrough of the v57 architecture: a frequency mixture of experts whose
per-node gate is produced by a **Hodge spectral controller**. Companion to `architecture_v57.dot`
(→ `architecture_v57.png` / `.pdf`).*

---

## 0. TL;DR

v57 keeps the v55 predictor — a bank of eight band-specialised **vanilla transformer experts** whose
outputs are blended per vertex by a softmax gate — and changes exactly **one thing**: *who computes the
gate*. In v53–v56 the gate was an MLP on the vanilla latent, a router with **no geometric information**
deciding which *geometric* frequency band a vertex needs. In v57 the gate is produced by a small
**Hodge branch** — two PSD-Hodge layers on the full simplicial complex — whose only job is to route.

The division of labour the measurements have been pointing at all along:

> **The Hodge operator routes (it is a natural frequency analyser); the vanilla transformers fit.**

Both branches read the **same** input `x_v`. The Hodge side never predicts a field; the vanilla side
never sees the mesh. `v55 → v57` isolates **who decides the gate**.

Two paths, drawn in parallel in the figure:

* **Inference path** (runs at test time): `x_v → {Hodge controller → gate}  and  {embed → experts}
  → gated mix → LayerNorm → decoder → û`.
* **Training path** (parallel, auxiliary): a learned Hodge analyzer decomposes each *target* field into
  frequency bands that supervise the experts; it never runs at test time.

---

## 1. Notation and inputs

One sample = **one whole mesh** `M = (V, E, F)` with `N = |V| ≈ 4.9k` vertices; `batch_size = 1` is a
correctness constraint (attention treats all `N` vertices as one sequence).

| tensor | shape | meaning | used by |
|---|---|---|---|
| `x_v` | `(N, 9)` | vertex features | **predictor** and **controller** |
| `x_e` | `(N_e, 21)` | edge (1-cell) DEC features | controller, analyzer |
| `x_f` | `(N_f, 13)` | face (2-cell) DEC features | controller |
| `d_0` | `(N_e, N)` sparse | vertex→edge incidence (coboundary) | controller, analyzer |
| `d_1` | `(N_f, N_e)` sparse | edge→face incidence | controller |
| `v_idx, e_idx` | `(N, 256), (N_e, 256)` | neighbour supports (geodesic rings, nearest-first) | controller |

The **9 vertex channels**:

```
x_v = [ x, y | a | u_in,x, u_in,y | SDF | n_x, n_y | is_wall ]
        pos    dual-area  inlet vel.   signed dist.  wall normal  wall tag
```

Each sample is non-dimensionalised by its own freestream `U∞` (`u/U∞`, `p/U∞²`, `ν_t/U∞`) then a global
z-score; both are inverted before any metric is computed.

**Targets** (what we predict): the four RANS fields `y = (u_x, u_y, p, ν_t) ∈ ℝ^{N×4}`.

---

## 2. Background: frequency on a mesh

Let `A` be the vertex adjacency and `D = diag(deg)`. The symmetric normalised Laplacian is
`L̃ = I − D^{-1/2} A D^{-1/2}`, symmetric PSD, with eigenpairs `L̃ u_i = μ_i u_i`, `0 = μ_0 ≤ … ≤ 2`.
The `u_i` are the **graph Fourier basis** and `μ_i` is the **frequency** of `u_i`. Through the Rayleigh
quotient

```
xᵀ L̃ x = Σ_{(i,j)∈E} (x_i − x_j)²        (Dirichlet energy)
```

a small-`μ` signal barely changes across an edge (smooth — the far field) and a large-`μ` signal flips
sign between neighbours (sharp — the boundary layer, suction peak, wake).

The **self-looped, symmetrically normalised** operator

```
S = D^{-1/2} (A + I) D^{-1/2},     S = I − L̃_loop,     λ(S) = 1 − μ
```

is a low-pass filter: `Sᵐ` has response `g_m(λ) = λᵐ`, with half-power cutoff `μ_c ≈ ln2 / m` and a
diffusion radius `≈ √m` hops. This is the operator behind both the frequency-band decomposition
(§5.1) and, in learned form, the analyzer.

**Why the Hodge operator is a router, not a fitter.** The discrete Hodge Laplacian
`L_0 = ⋆_0^{-1} d_0ᵀ ⋆_1 d_0` *is* a Dirichlet-energy Laplacian, so as a layer it carries a
**smoothing / low-graph-frequency prior**. That mis-specifies sharp near-wall RANS structure — which is
why every HodgeFormer *fitting* variant lost to a plain transformer. But a diffusion operator is a
natural **frequency analyser**: applying it tells you how much high-frequency content sits at a vertex.
v57 gives the Hodge operator that job.

---

## 3. Inference path (the forward pass)

The predictor is

```python
h  = embed(x_v)                                        # (N,192)  token embedding
gl = hodge_gate(x_v, x_e, x_f, d_0, d_1, v_idx, e_idx) # (N,K)    per-node gate LOGITS  (K = 8)
h, bands, g = moe(h, pos, gate_logits=gl)              # experts + per-node gate
û  = decoder(h)                                        # (N,4)    the four fields
```

### 3.1 Token embedding

```
h⁰(v) = W₂ · GELU( W₁ x_v(v) ),     W₁ ∈ ℝ^{192×9},  W₂ ∈ ℝ^{192×192}
```

This is the **only** input to the fitting branch. It carries positions among its features, so the
experts are an implicitly position-aware point-set model; they never see the mesh.

### 3.2 The Hodge spectral controller (produces the gate)

The controller is `x_v (+ mesh) ↦ per-node logits over the K experts`. It sees everything the experts do
not: the vertex/edge/face streams, the incidence operators, and the neighbour supports.

**(a) Neighbour embedding.** The three streams are lifted by the standard HodgeFormer neighbour
embedding: `(x_v, x_e, x_f, d_0, d_1) ↦ (h_v, h_e, h_f)`, each of width `d = 64`.

**(b) PSD-Hodge star (the core operator).** A learned Hodge star is a **symmetric positive-definite**
Gaussian-diffusion operator applied over a neighbour support, with tied query = key so no softmax is
needed. For values `x` on the `k`-cells with kernel features `φ`:

```
φ_i        = W_φ x_i                                   (tied query = key)
w_ij       = exp( −‖φ_i − φ_j‖² / (2 σ²) ),  j ∈ N(i)  (RBF, a PD kernel, in (0,1])
Ŝ          = D^{-1/2} W D^{-1/2},  D_i = Σ_j w_ij      (symmetric normalisation)
(⋆ x)_i    = (1 − β) x_i + β (Ŝ x)_i,   β = ½·sigmoid(θ) ∈ (0, ½)
```

Per head, `σ = softplus(logσ)` and `β` are learned. Because `Ŝ` is symmetric with `λ(Ŝ) ∈ [−1, 1]`, the
star `⋆` is symmetric with `λ(⋆) ∈ [1−2β, 1] ⊂ (0,1]` — **SPD for β < ½**. (No softmax: softmax would
re-introduce a row-stochastic, asymmetric, non-PSD operator.)

**(c) The vertex Hodge-Laplacian.** One PSD-Hodge vertex layer applies the assembled operator

```
x_v  ─d₀→  (edges)  ─⋆₁→  (edges)  ─d₀ᵀ→  (vertices)  ─⋆₀⁻¹→   =   ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀ · x_v  =  L₀ x_v
```

where the two `d_0` lifts are the **exact** sparse incidence operators (run in fp32) and only the two
stars `⋆_1` (edge) and `⋆_0^{-1}` (vertex, its own kernel feature) are learned. Each `PSDHodgeLayer` is
a pre-norm residual block: `x ← x + L₀-attention(x)`, then a per-stream position-wise FFN.

**(d) Read-out to logits.** Two PSD-Hodge layers are stacked, then

```
gate_logits(v) = W_b · GELU( W_a · LayerNorm(h_v(v)) ) ∈ ℝ^K,   W_a: 64→64,  W_b: 64→K
```

**Memory note.** The cached supports are 256 wide and those gathers dominate memory; the controller only
has to decide *which band* a vertex needs, not reconstruct a field, so it runs on the inner
`window = 64` nearest neighbours (the geodesic rings are ordered nearest-first, so this is a truncation
to the inner rings). At full 256 width training needs ~19.9 GB.

### 3.3 The frequency experts

`K = 8` experts. Expert `k` is a small transformer biased to a spatial scale `σ_k`:

```
Fourier features:  φ_k(pos) = [ sin(pos · B_k),  cos(pos · B_k) ],   B_k ∼ N(0, σ_k²) ∈ ℝ^{2×8}
input mix:         z_k⁰(v)  = W_in^k [ h(v) ; φ_k(pos(v)) ],         W_in^k: (192+16) → 88
body:              z_k(v)   = LayerNorm( TransformerBlock×2 ( z_k⁰ ) ),   d_expert = 88, 8 heads
two heads:         proj_k(v) = W_out^k z_k(v) ∈ ℝ^{192}     (feature contribution → the mix)
                   b_k(v)    = W_band^k z_k(v) ∈ ℝ^{4}      (band prediction → auxiliary loss only)
```

Geometric scales `σ = (1.0, 1.7, 2.8, 4.8, 8.0, 13.5, 22.6, 38.0)`: low-`σ` features are smooth in
space and high-`σ` features oscillate, cheap for the matching expert and hard for a mismatched one. The
**prediction uses `proj_k`** (the representation), not the band head `b_k`.

### 3.4 The per-node gate and the gated mix  *(how the softmax acts on the mix)*

The gate logits from the controller are turned into a **per-vertex distribution over the experts**:

```
g(v) = softmax( gate_logits(v) ) ∈ Δ^K,      g_k(v) ≥ 0,   Σ_{k=1}^{K} g_k(v) = 1
```

so at **each vertex** `g(v)` is a point on the `K`-simplex — a set of `K` non-negative weights summing
to 1. The mixed latent is the **per-node convex combination** of the expert representations:

```
h_mix(v) = Σ_{k=1}^{K} g_k(v) · proj_k(v)        ∈ ℝ^{192}
```

Concretely (with `H(v) = [proj_1(v), …, proj_K(v)] ∈ ℝ^{192×K}`): `h_mix = (H ⊙ g).sum(experts)`.

This is exactly what the figure’s red edge means: the softmax weight `g_k(v)` **scales expert `k`’s
contribution at vertex `v`**. Because the weights are a convex combination, each vertex chooses its own
blend of the eight band experts — a wall vertex puts most of its mass on the sharp-band experts, a
far-field vertex on the smooth-band experts. Crucially, in v57 that choice is made by the **Hodge
controller**, which has actually measured the local spectral content, rather than by an MLP that has not.

### 3.5 Cross-field decoder

`h_mix` is `LayerNorm`-ed, then decoded by a cross-field attention head. Each vertex spawns four field
tokens (the latent plus a learned per-field embedding); two attention blocks (4 heads) let the four
fields attend to one another; a shared read-out MLP emits one scalar per field:

```
û(v) = readout( CrossFieldAttn×2 ( h_mix(v) + field_emb ) ) ∈ ℝ^4 = (u_x, u_y, p, ν_t)
```

At inference the story ends here. `bands` and `g` are stashed on `self.aux` only for the training loss.

---

## 4. Training path (parallel, auxiliary supervision)

The training path exists only to **shape the experts and the gate**. It decomposes each *target* field
into frequency bands and pins expert `k` to band `k`. None of it runs at test time.

### 4.1 The learned Hodge analyzer

The analyzer is itself a small learned Hodge Laplacian, used to define the bands:

```
⋆₁ = diag(w_e),   w_e = softplus( MLP(x_e) )               (learned positive edge conductance)
⋆₀ = diag(a_v),   a_v = |d₀|ᵀ w                            (learned vertex degree)
L₀ = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀                                        (PSD; random-walk form, λ(L₀) ∈ [0,2])
S  = I − ½ L₀                                              (a diffusion step, constant ½ from the [0,2] bound)
```

`L₀` is PSD by construction (`xᵀ d₀ᵀ ⋆₁ d₀ x = Σ_e w_e (x_i − x_j)² ≥ 0`), and the degree choice makes
`λ(L₀) ≤ 2`, so **no power iteration and no eigendecomposition** appear anywhere in the training path.

### 4.2 Telescoped frequency atoms

Apply `S` on an increasing, even **ladder** `m = (2, 4, 6, 8, 12, 16, 24, 32)` (nine atoms; cost is set
by `max(m) = 32`) and difference the low-passes:

```
a_low(y)  = S^{m_{M-1}} y
a_j(y)    = S^{m_j} y − S^{m_{j+1}} y,     j = M-1 … 1
a_high(y) = y − S^{m_1} y
```

giving `M = 9` atoms `a_m(y) ∈ ℝ^{N×4}`. Two properties hold for **any** stars:

* **Completeness (telescoping):** `Σ_m a_m = y` exactly (measured rel. reconstruction `9.6e-8`).
* **Partition of unity:** the spectral responses satisfy `Σ_m g_m(λ) = 1` for every `λ`.

### 4.3 The learnable assignment (learnable band boundaries)

A learned **column-stochastic** matrix groups the nine atoms into `K = 8` per-expert band targets:

```
P = softmax( logits / τ,  over EXPERTS )  ∈ ℝ^{K×M},     Σ_k P_{k,m} = 1  for every atom m
t_k(v) = Σ_{m=1}^{M} P_{k,m} · a_m(y)(v)                  (per-expert band target)
```

Because each **column** sums to 1, `Σ_k t_k = Σ_m a_m = y` **exactly for any `P`**: completeness
survives learning the band boundaries. `P` is initialised on contiguous atom slices (a v53-style
hand-picked split) and is free to move from there.

### 4.4 The objective

Four terms:

```
ℒ = ℒ_field                                   (disjoint interior/wall MSE, per-case normalised)
  + 0.5  · ℒ_band                             expert band head b_k regressed onto t_k
  + 0.01 · ℒ_bal                              gate load balance (no expert collapse)
  + 1.0  · ℒ_floor                            one-sided starved-band penalty
```

with, writing `s_k` for the energy share of band `k` and `ḡ` for the mean gate mass,

```
ℒ_field = Σ_{v∉wall} ‖û(v) − y(v)‖² / σ̂²  +  Σ_{v∈wall} ‖û(v) − y(v)‖² / σ̂²      (σ̂ = per-sample per-field std, floor 0.1)
ℒ_band  = Σ_k  ‖ b_k − t_k ‖²                                                     (the analyzer's targets are detached: detach_band_targets = true)
ℒ_bal   = ( ḡ − 1/K )² · K
ℒ_floor = Σ_k  [ relu( 0.02 − s_k ) ]²                                            (penalise only bands below 2% of the energy)
```

Three design choices worth calling out:

* **`ℒ_field` is a *disjoint* interior/wall split**, mirroring the AirFRANS Volume/Surf metric.
* **`ℒ_floor` replaces entropy maximisation.** Earlier learned analyzers *maximised* band entropy,
  which **forced** balance and lost to the fixed ladder because the fields are lopsided (`ν_t` puts 40%
  of its energy in the sharpest band, the others 90–98% in the smoothest). The one-sided floor blocks
  only the degenerate empty-band solution and leaves the decomposition free to stay lopsided.
* **`detach_band_targets = true`:** the analyzer receives gradient only from its structural terms, not
  from "make the band easy to predict".

---

## 5. Training setup

| | |
|---|---|
| optimiser | AdamW, weight decay `1e-4` |
| schedule | OneCycleLR, `max_lr 1e-3`, `pct_start 0.3`, `div 25`, `final_div 1e4` (peak 4e-5→1e-3→~4e-9) |
| epochs / batch | 400 / 1 mesh |
| precision | bf16 autocast; the Hodge incidence ops and the analyzer run in **fp32** (sparse bmm has no bf16 kernel, and a supervision target must not carry autocast rounding) |
| loss weights | `band 0.5`, `bal 0.01`, `floor 1.0`, `cl 0.0` (the lift term is off — v52 showed it costs C_L) |
| data | `graph_airfrans_data_v18_3` (~4.9k-vertex anisotropic-BL remeshes) |
| split | official AirFRANS **full 800/200** (`airfrans_official_split.json`) |
| train aug. | pool `__s0..__s7` (K=8), one random mesh per epoch; test = fixed `__s0` |
| checkpoints | every 20 epochs → `runs/v57_gated_hodge_vef/` |

---

## 6. Component / config summary

| block | role | key hyper-parameters |
|---|---|---|
| Token embed | vanilla latent | `9 → 192 → 192`, GELU |
| **Hodge controller** (`hodge_gate_kw`) | produces the gate | `d=64`, `d_hidden=256`, `heads=4`, `n_layers=2`, `readout_hidden=64`, `window=64` |
| PSD-Hodge stars | SPD Gaussian diffusion | per-head `σ` (init ≈1), `β=½σ(θ)` (init ≈0.15–0.30) |
| Experts (`moe_kw`) | fit the bands | `n_experts=8`, `d_expert=88`, `blocks=2`, `heads=8`, `d_hidden=352`, `n_freq=8`, `σ` geometric 1→38 |
| Gate | per-node softmax over experts | logits from the controller (not an MLP) |
| Decoder (`decoder_kw`) | cross-field read-out | `n_dec_layers=2`, `dec_heads=4`, `readout_hidden=64` |
| Analyzer (`analyzer_kw`) | band supervision | `hidden=64`, `ladder=(2,4,6,8,12,16,24,32)`, `learn_mass=false` |
| Assignment | learnable band boundaries | `K×M = 8×9`, column-stochastic |

The predictor (experts + gate + decoder) is ≈ v55's `2.75M` parameters; v57 **adds** the Hodge
controller (two PSD-Hodge layers at `d = 64`). Everything else is v55.

---

## 7. Why v57 — what it isolates

`v53–v56` used an MLP gate on the vanilla latent: a router with **no geometric information** deciding
which *geometric* frequency band a vertex needs. v57 changes only the gate’s source, so **`v55 → v57`
is a clean ablation of "who decides the gate"**:

* the **Hodge branch** measures local spectral content and **routes**;
* the **vanilla experts** carry no mesh and **fit**.

Reference bars (v55, same protocol): `rel-L2 0.0345 · Vol 0.00165 · Surf 0.00192 · C_L(band) 0.0265`.

---

## 8. Reproduce

```bash
# render this figure
cd cases/v57_gated_hodge_vef/docs
dot -Tpng -Gdpi=140 architecture_v57.dot -o architecture_v57.png
dot -Tpdf architecture_v57.dot -o architecture_v57.pdf

# train (detached)
bash cases/v57_gated_hodge_vef/run_train_v57_gated_hodge_vef.sh

# 20-checkpoint sweep + mesh-native lift + expert diagnostics
bash cases/v57_gated_hodge_vef/sweep_eval_v57_gated_hodge_vef.sh
python cases/v57_gated_hodge_vef/inspect_experts_v57_gated_hodge_vef.py   # gate entropy, band-error matrix
python cases/v57_gated_hodge_vef/diagnose_attention_v57_gated_hodge_vef.py
```

---

## 9. Provenance

| item | source |
|---|---|
| forward flow | `model_v57_gated_hodge_vef.py` (`MoEModel.forward`, `build_v57_gated_hodge_vef_model`) |
| gate / controller | `hodge_gate_v57_gated_hodge_vef.py` (`HodgeGate`), `psd_hodge_v57_gated_hodge_vef.py` (`psd_star`, `PSDHodgeLayer`) |
| experts, gate, assignment | `moe_v57_gated_hodge_vef.py` (`Expert`, `FrequencyMoE.forward`, `LearnableAssignment`) |
| analyzer | `learned_hodge_v57_gated_hodge_vef.py` (`LearnedHodgeAnalyzer`) |
| losses / schedule / data | `airfrans_regression_v57_gated_hodge_vef_cfg.toml`, `airfrans_regression_v57_gated_hodge_vef.py` |
| frequency-band theory | `../../docs/special_cases/v53_frequency_bands_theory.md` |
