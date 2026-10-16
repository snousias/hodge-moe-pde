# Results

The evaluation runs on each case's own mesh (no interpolation) in physical units, on the
200 test cases of the official AirFRANS 800/200 split. The checkpoint is
`checkpoints/hodgemoe_vef_best_ep400.pth`, epoch 400 of 400. Every number in this file comes
from `bash scripts/run_eval.sh` in this folder. The raw arrays and logs are in `eval_release/`:

| file | contents |
|---|---|
| `test_metrics.npz` | per-case, per-field metrics, shape (200, 4, 6) = (case, [u_x, u_y, p, ν_t], [MSE, MAE, RMSE, R², rel-L2, rel-L2 centred]), plus `vol_mse`, `surf_mse` |
| `cl_native.npz` | per-case true and predicted C_L, and the aggregate C_L rel-L2 and ρ_L |
| `fields.log`, `lift.log` | console output of the two evaluation steps |
| `sweep_summary.txt` | full per-field output for each of the 20 swept checkpoints |
| `train_metrics.csv` | per-epoch training and validation curves of the original run |

## Headline

| metric | value |
|---|---|
| rel-L2 (mean over u_x, u_y, p, ν_t) | **0.0329** |
| R² (mean over fields) | 0.9981 |
| Volume (AirFRANS-native normalised MSE, non-wall) | 0.00170 |
| Surf (AirFRANS-native normalised MSE, wall) | 0.00202 |
| C_L rel-L2 | 0.0313 |
| ρ_L (Spearman, lift) | 0.9999 |
| params | 3,106,490 |

**C_L estimator.** `eval_lift.py` integrates pressure over every vertex the mesh flags as wall,
which is a band about 1,000 vertices wide around the airfoil. The parent project calls this the
"band contour" C_L. Against the AirFRANS library's own lift it is roughly 2× the physical value.
Comparisons between models stay valid because both sides use the same estimator. Do not quote it
as an absolute lift coefficient.

## Per field

| field | rel-L2 | R² | Volume MSE | Surf MSE |
|---|--:|--:|--:|--:|
| u_x | 0.0068 | 0.9994 | 0.00017 | 0.00091 |
| u_y | 0.0314 | 0.9987 | 0.00080 | 0.00190 |
| p | 0.0364 | 0.9985 | 0.00068 | 0.00343 |
| ν_t | 0.0571 | 0.9960 | 0.00515 | 0.00185 |

ν_t accounts for 76% of the Volume error and 43% of the rel-L2 mean. Pressure dominates the
Surf error.

## Checkpoint sweep

Each checkpoint was evaluated on all 200 test cases (`scripts/sweep_eval.sh`):

| epoch | rel-L2 | Volume | Surf |
|--:|--:|--:|--:|
| 20 | 0.1235 | 0.02105 | 0.02180 |
| 40 | 0.1325 | 0.02711 | 0.05091 |
| 60 | 0.1527 | 0.02731 | 0.02420 |
| 80 | 0.0822 | 0.01473 | 0.01017 |
| 100 | 0.0966 | 0.01709 | 0.01767 |
| 120 | 0.0964 | 0.00977 | 0.01254 |
| 140 | 0.0553 | 0.00495 | 0.00445 |
| 160 | 0.0752 | 0.00765 | 0.00748 |
| 180 | 0.0604 | 0.00543 | 0.00641 |
| 200 | 0.0517 | 0.00422 | 0.00597 |
| 220 | 0.0510 | 0.00417 | 0.00607 |
| 240 | 0.0471 | 0.00367 | 0.00445 |
| 260 | 0.0611 | 0.00413 | 0.00609 |
| 280 | 0.0397 | 0.00226 | 0.00291 |
| 300 | 0.0380 | 0.00217 | 0.00276 |
| 320 | 0.0382 | 0.00189 | 0.00253 |
| 340 | 0.0340 | 0.00190 | **0.00195** |
| 360 | 0.0337 | 0.00176 | 0.00219 |
| 380 | 0.0331 | 0.00172 | 0.00207 |
| **400** | **0.0329** | **0.00170** | 0.00202 |

The learning rate peaks at 30% of training (epoch 120) and then anneals. From epoch 340 onward,
rel-L2 and Volume improve at every checkpoint. The sweep scores on the test set, because this
protocol has no validation split, but it picks the **final** epoch here. The released numbers are
therefore the same as those of the no-selection protocol "train 400 epochs, report the last
checkpoint". Selection on the test set does not inflate them. Only Surf would have been slightly
better at epoch 340.

## Comparison

All rows use the same meshes, the same official split and the same metric code. The HodgePDEMoE
and first-vef rows come from the parent project's scorecard
(`experiments/regression_pdes_tetrahedral/docs/RESULTS.md`), with C_L computed by the same
band-contour estimator. The baselines come from the Transolver repository's feature-ablation
study, using its best feature set `pisna` (pos + area + inlet + SDF + normals).

| model | params | epoch | rel-L2 ↓ | R² ↑ | Volume ↓ | Surf ↓ | C_L ↓ | ρ_L ↑ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **HodgeMoE-vef (this release)** | 3.11M | 400 | 0.0329 | 0.9981 | 0.00170 | 0.00202 | 0.0313 | 0.9999 |
| HodgePDEMoE (`modes = "v"`, bf16) | 3.02M | 380 | **0.0300** | **0.9983** | **0.00140** | 0.00164 | 0.0225 | 0.9999 |
| HodgeMoE-vef, first run | 3.11M | 560 | 0.0359 | 0.9978 | 0.00182 | 0.00180 | 0.0277 | 0.9999 |
| Transolver | 6.00M | – | 0.0436 | – | 0.00265 | **0.00143** | **0.0103** | 0.9999 |
| PointNet | 75k | – | 0.0689 | – | 0.00451 | 0.00440 | 0.0149 | 0.9996 |
| MLP | 20k | – | 0.1743 | – | 0.03576 | 0.03969 | 0.0860 | 0.9934 |

What the table shows:

- **`vef` does not beat `"v"`.** This release is behind HodgePDEMoE on every metric: rel-L2 +10%,
  Volume +21%, Surf +23%, C_L +39%. That fits the parameter audit in `ARCHITECTURE.md`. The face
  operator in both layers and the edge operator in layer 2 never receive a gradient, so the only
  working addition over `"v"` is layer 1's edge operator.
- **Run-to-run spread is about as large as the `v`/`vef` gap.** The two `vef` runs use the same
  model but differ in precision (fp32 vs bf16), torch version and schedule length (the first
  run had a 200-epoch continuation after its 400-epoch schedule, and its best checkpoint is at
  epoch 560). They land 0.0030 apart on rel-L2 (0.0329 vs 0.0359), the same size as the gap to
  `"v"` (0.0029). Each configuration has a single run, so this table cannot rank `v` and `vef`
  with confidence. It does rule out a large gain from `vef`.
- **Compared with Transolver**, at about half the parameter count: rel-L2 is 25% lower and
  Volume 36% lower, but Surf is 1.4× higher and C_L is 3.0× higher. Like the other HodgeMoE
  models, this one is weaker on the surface and on lift.

Two caveats favour the HodgeMoE rows. They train on all 800 official training cases, whereas
the baselines hold out 80 for validation and train on 720. The HodgePDEMoE and first-vef
checkpoints were also selected on the test set. For this release the selection made no
difference (see the sweep above).

## Qualitative

`qualitative/qual_000.png` … `qual_007.png` show the first 8 test cases. Each is a 4×3 grid:
rows u_x, u_y, p and ν_t; columns ground truth, prediction and absolute error, in physical
units. They were rendered from the released checkpoint by `eval_fields.py --n_qual 8`. Errors
concentrate at the leading edge, in the near-wall layer and along the wake.
