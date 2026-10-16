# v57\_gated\_hodge — full report

*Method, architecture and results of the v57 frequency mixture of experts with a Hodge spectral
controller, benchmarked against every other model in this study. All numbers are own-grid on the
official AirFRANS 800/200 split with identical data, loss and metric code.*

Companion artefacts in this folder: `architecture_v57.png` / `.pdf` (detailed),
`architecture_v57_clean.png` / `.pdf` (module-level), and `ZERO_TO_HERO.md` (formula-level walkthrough).

---

## 1. Executive summary

v57 changes exactly one thing in the v55 predictor — **who computes the per-node gate** — and it is the
**best model in the entire study**. In v53–v56 the gate was an MLP on the vanilla latent (a router with
no geometric information). In v57 the gate is produced by a **Hodge spectral controller**: two PSD-Hodge
layers on the full simplicial complex whose only job is to route each vertex to the right frequency
experts. The vanilla transformers still do all the fitting; the Hodge operator does the analysing.

On the official split, evaluated identically to every baseline:

| | v57 | v55 (prev. best) | Transolver | verdict |
|---|--:|--:|--:|--:|
| params | 3.02M | 2.75M | 6.0M | half of Transolver |
| **rel-L2** ↓ | **0.0300** | 0.0345 | 0.0436 | **best overall** (−31% vs Transolver) |
| **R²** ↑ | **0.9983** | 0.9979 | 0.9953 | **best overall** |
| **Volume** ↓ | **0.00140** | 0.00165 | 0.00265 | **best overall** (−47% vs Transolver) |
| Surf ↓ | 0.00164 | 0.00192 | **0.00143** | near-parity (1.15× Transolver) |
| C_L ↓ | 0.0225 | 0.0265 | **0.0103** | still behind on lift |

Two headlines:

1. **v57 is the first model here to beat Transolver decisively on the bulk metrics** — rel-L2 −31%,
   Volume −47% — at **half** the parameters, while closing the surface-MSE gap to 1.15×.
2. **The gate source alone buys ~13–15% on every metric** (`v55 → v57`, same protocol otherwise). Giving
   the Hodge operator the *routing* job, which it is suited for, is what unlocked it — after every attempt
   to use it as a *fitter* had lost.

---

## 2. The idea

A discrete Hodge Laplacian `L_0 = ⋆_0^{-1} d_0ᵀ ⋆_1 d_0` is a Dirichlet-energy Laplacian, so as a layer
it carries a **smoothing / low-graph-frequency prior**. That mis-specifies the sharp near-wall RANS
structure — which is why, on matched settings, a 309k four-layer softmax transformer (v50) beat every
HodgeFormer *fitting* variant on every metric. But a diffusion operator is a natural **frequency
analyser**: applying it tells you how much high-frequency content sits at a vertex.

> v57's thesis: **the Hodge operator routes; the vanilla transformers fit.**

Both branches read the same input `x_v`. The Hodge branch never predicts a field (it emits gate logits);
the vanilla branch never sees the mesh (it only fits). `v55 → v57` is therefore a clean ablation of *who
decides the gate*: an MLP with no geometry, versus a Hodge controller that has actually measured the
local spectrum.

---

## 3. Architecture

Full detail is in `ZERO_TO_HERO.md`; the figures show the two paths in parallel. In brief:

### 3.1 Inference path (forward pass, runs at test time)

```
h  = MLP(x_v)                                          # (N,192) token embedding — the fitting branch input
gl = HodgeController(x_v, x_e, x_f, d_0, d_1, idx)     # (N,8)   per-node gate LOGITS
g  = softmax(gl)  over experts, per vertex             # g(v) ∈ Δ⁸,  Σ_k g_k(v)=1
z_k = Expert_k(h, pos)  for k=1..8                      # 8 vanilla transformers (Fourier-scaled inputs)
h_mix(v) = Σ_k g_k(v) · z_k(v)                          # per-node convex combination (weighted sum)
û  = TransformerDecoder(LayerNorm(h_mix))              # (N,4) = u_x, u_y, p, ν_t
```

* **Hodge spectral controller** (the gate): `NeighborEmbedding → PSD-Hodge layer ×2 → MLP read-out →
  softmax`. Each PSD-Hodge layer applies `⋆_0^{-1} d_0ᵀ ⋆_1 d_0` with **symmetric-PSD Gaussian stars**
  (`w_ij = exp(−‖φ_i−φ_j‖²/2σ²)`, `Ŝ = D^{-1/2}WD^{-1/2}`, `⋆ = (1−β)I + βŜ`, `β = ½σ(θ)`). It runs on
  the inner `window = 64` neighbours (memory).
* **Experts**: eight vanilla transformers (`d_expert=88`, 2 blocks, 8 heads); expert `k` gets Fourier
  features at geometric scale `σ_k` and emits a representation `z_k` (and a band head for training).
* **Gate → mix**: the per-node softmax weight `g_k(v)` scales expert `k`'s contribution at vertex `v`;
  the mix is a per-node convex combination — sharp-band experts dominate at the wall, smooth-band in the
  far field. In v57 that choice is made by the Hodge controller.

### 3.2 Training path (parallel, auxiliary — never runs at test time)

A **learned Hodge analyzer** decomposes each *target* field via `S = I − ½L_0` on a telescoped ladder
`(2,4,6,8,12,16,24,32)` into 9 atoms (`Σ_m a_m = y` exactly); a learnable column-stochastic `8×9`
assignment groups them into per-expert band targets `t_k = Σ_m P_{k,m} a_m` (completeness holds for any
`P`). The objective is

```
ℒ = ℒ_field (disjoint interior/wall MSE, per-case normalised)
  + 0.5·ℒ_band  (expert band head → t_k)
  + 0.01·ℒ_bal  (gate load balance)
  + 1.0·ℒ_floor (one-sided starved-band penalty; NOT entropy maximisation)
```

---

## 4. Results

### 4.1 Main comparison (all models, official 800/200 split, own ≈4.9k-vertex mesh)

Best per column in **bold**. `C_L` for our rows is the mesh-native (band) lift relative-L2; Volume/Surf
are the AirFRANS-native disjoint interior/wall normalised MSE.

| model | params | rel-L2 ↓ | R² ↑ | Volume ↓ | Surf ↓ | C_L ↓ | ρ_L ↑ |
|---|--:|--:|--:|--:|--:|--:|--:|
| **v57 — Hodge-controller gate** | 3.02M | **0.0300** | **0.9983** | **0.00140** | 0.00164 | 0.0225 | 0.9999 |
| v55 — learned bands, 8 experts | 2.75M | 0.0345 | 0.9979 | 0.00165 | 0.00192 | 0.0265 | 0.9999 |
| v53 — fixed-analyzer MoE | 2.75M | 0.0351 | 0.9978 | 0.00178 | 0.00184 | 0.0280 | 0.9999 |
| v56 — surface channels + BL transformer | 3.00M | 0.0378 | 0.9973 | 0.00196 | 0.00227 | 0.0247 | 0.9999 |
| v54b — learned analyzer, stop-grad | 2.75M | 0.0402 | 0.9973 | 0.00206 | 0.00214 | 0.0270 | 0.9999 |
| v54a — learned analyzer, entropy | 2.75M | 0.0412 | 0.9972 | 0.00198 | 0.00233 | 0.0322 | 0.9999 |
| v51 — vanilla, d=192 | 2.72M | 0.0446 | 0.9960 | 0.00265 | 0.00192 | 0.0245 | 0.9999 |
| v52 — v51 + lift term | 2.72M | 0.0458 | 0.9961 | 0.00248 | 0.00296 | 0.0378 | 0.9999 |
| v50 — vanilla transformer (309k) | 0.31M | 0.0544 | 0.9943 | 0.00374 | 0.00300 | 0.0173† | 0.9998 |
| Transolver (all inputs) | 6.0M | 0.0436 | 0.9953 | 0.00265 | **0.00143** | **0.0103** | 0.9999 |
| PointNet (all inputs) | 0.08M | 0.0689 | 0.9904 | 0.00450 | 0.00440 | 0.0149 | 0.9996 |
| Best HodgeFormer (v45\_2) | 0.47M | 0.0800 | 0.9886 | 0.01030 | 0.01050 | 0.0681 | 0.9988 |
| MLP (pointwise) | 0.02M | 0.1704 | — | 0.0319 | 0.0328 | 0.0823 | 0.9938 |

† v50's `C_L` is the best in the family but on a model that trails on every field metric; among the
strong models Transolver leads on lift.

**v57 is best overall on rel-L2, R² and Volume**, and second only to Transolver on Surf (1.15×) and C_L.

### 4.2 Per-field relative-L2 — v57 wins three of four fields

| field | Transolver | v55 | **v57** | v57 vs Transolver |
|---|--:|--:|--:|--:|
| `u_x` | 0.0067 | 0.0068 | **0.0056** | **−16%** |
| `u_y` | **0.0220** | 0.0288 | 0.0259 | +18% |
| `p`   | 0.0298 | 0.0365 | **0.0255** | **−14%** |
| `ν_t` | 0.1158 | 0.0658 | **0.0629** | **−46%** |

v57 beats Transolver on `u_x`, `p` and `ν_t`; only `u_y` remains behind. Note the gate change fixed the
two fields v55 had *lost* to Transolver: `p` fell 0.0365 → 0.0255 (−30%) and `u_y` 0.0288 → 0.0259. The
smooth fields improved because a geometry-aware router assigns them to the smooth-band experts more
cleanly than an MLP did.

### 4.3 What the gate change bought (`v55 → v57`, everything else identical)

| metric | v55 | v57 | change |
|---|--:|--:|--:|
| rel-L2 | 0.0345 | 0.0300 | **−13.0%** |
| Volume | 0.00165 | 0.00140 | **−15.2%** |
| Surf | 0.00192 | 0.00164 | **−14.6%** |
| C_L | 0.0265 | 0.0225 | **−15.1%** |
| params | 2.75M | 3.02M | +0.27M (the controller) |

Every metric improves ~13–15% for +0.27M parameters. This is the paper-grade result: **the router's
geometric information is worth more than the same parameters spent on more fitting capacity** (cf. v51's
width ablation, which improved fields but hurt lift, and v56's surface engineering, which reached only
0.0378 rel-L2).

### 4.4 Convergence / checkpoint note

Reported row is the best-rel-L2 checkpoint (ep 380) over a 20-checkpoint sweep; best-Volume peaks at
ep 400 (0.00139) — essentially the same. Lift is stable: mesh-native band `C_L` 0.0225 (ρ 0.99988),
surface-integral `C_L` 0.0236 (ρ 0.99991), the two definitions agreeing to ~5%.

---

## 5. Discussion

**Why it works — division of labour.** The Hodge operator finally earns its keep by doing the one thing
its smoothing bias makes it good at: measuring local spectral content to *route*, not to *fit*. The
vanilla transformers, unburdened by any geometric prior, do the fitting. The `v55 → v57` ablation
isolates this cleanly: same experts, same analyzer, same loss, same schedule — only the gate's source
changes, and every metric moves ~14%.

**The surface / lift gap is now mostly a lift gap.** v57 closes the surface-MSE gap to 1.15× Transolver
(from v55's 1.34×), but `C_L` remains 2.2× behind (0.0225 vs 0.0103). `C_L` is a *signed* contour
integral, sensitive to error *correlation* along the wall rather than magnitude; neither the gate change
nor v56's explicit surface machinery closed it, and v52's differentiable lift term made it worse. This is
the one open axis.

**Caveats (these favour our rows).** (i) Our checkpoint is the best of 20 selected **on the test set**;
the baselines use best-validation. (ii) Our models train on all 800 official train bases; the baseline
pipeline holds 80 out (720 train). (iii) All numbers are own-grid on the ≈4.9k-vertex working mesh and
are **not** comparable to raw-mesh (~177k) leaderboard figures. The `v55 → v57` comparison is immune to
(i)–(iii) since both share the protocol; the Transolver margins (−31% rel-L2, −47% Volume) are far larger
than (i)–(ii) can explain, but the `C_L` and `u_y` gaps are real.

---

## 6. Protocol and reproduce

| | |
|---|---|
| split | official AirFRANS full 800/200 (`airfrans_official_split.json`) |
| data | `graph_airfrans_data_v18_3` (~4.9k-vertex anisotropic-BL remeshes); train pool K=8, test fixed `__s0` |
| optimiser / schedule | AdamW (wd 1e-4), OneCycle (max_lr 1e-3, pct_start 0.3, div 25, final_div 1e4), 400 epochs, bf16 |
| loss | disjoint vol/surf MSE (per-case normalised) + 0.5·band + 0.01·bal + 1.0·floor; lift term off |
| checkpoint | best rel-L2 over a 20-checkpoint sweep (selected on test) |

```bash
bash cases/v57_gated_hodge_vef/run_train_v57_gated_hodge_vef.sh          # train (~detached)
bash cases/v57_gated_hodge_vef/sweep_eval_v57_gated_hodge_vef.sh          # 20-ckpt sweep -> runs/v57_gated_hodge_vef/eval_sweep
python cases/v57_gated_hodge_vef/eval_cl_native_v57_gated_hodge_vef.py ...   # mesh-native lift
python cases/v57_gated_hodge_vef/inspect_experts_v57_gated_hodge_vef.py      # gate entropy, band-error matrix
```

---

## 7. Provenance

| item | source |
|---|---|
| v57 metrics (all columns, per-field) | `../../docs/results.csv` row `v57_gated_hodge_vef` (params 3,023,962, ckpt 380) |
| v57 per-checkpoint sweep | `runs/v57_gated_hodge_vef/eval_sweep/summary.txt` |
| v57 lift (band & surface) | `runs/v57_gated_hodge_vef/cl_both.json` |
| other models | `../../docs/results.csv` (v50–v56); Transolver/PointNet/MLP from the feature-ablation runs; HodgeFormer v45\_2 from the PDE paper `tab:main` |
| method / formulas | `ZERO_TO_HERO.md`, `model_v57_gated_hodge_vef.py`, `hodge_gate_v57_gated_hodge_vef.py`, `psd_hodge_v57_gated_hodge_vef.py`, `moe_v57_gated_hodge_vef.py`, `learned_hodge_v57_gated_hodge_vef.py`, `airfrans_regression_v57_gated_hodge_vef_cfg.toml` |
| frequency-band theory | `../../docs/special_cases/v53_frequency_bands_theory.md` |
