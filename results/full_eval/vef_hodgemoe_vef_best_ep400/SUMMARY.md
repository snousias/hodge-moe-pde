# Full test-set evaluation — HodgeMoE-vef · hodgemoe_vef_best_ep400.pth

- Checkpoint: `checkpoints/hodgemoe_vef_best_ep400.pth` (epoch 400, SHA-256 `f785812fe067a3c0…`)
- Config: `config.toml` · modes `vef` · precision **fp32** · 3,106,490 parameters
- Test cases: **200** (official AirFRANS `full` split, fixed `__s0` meshes)

## Overall

| rel-L2 (mean) | rel-L2 (median) | rel-L2 (90th pct) | R² | Volume | Surf | C_L rel-L2 | ρ_L | mean abs C_L error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.0329** | 0.0308 | 0.0398 | 0.9981 | 0.00170 | 0.00202 | 0.0313 | 0.99991 | 0.0491 |

rel-L2 and R² are per-case values in physical units, averaged over the 4 fields and the test cases. Volume / Surf are the AirFRANS-native MSE of fields normalised by the global training std, on non-wall / wall vertices. C_L is the mesh-native pressure-integral lift (see README: about twice the physical AirFRANS lift; compare only like with like).

## Per field

| Field | rel-L2 | R² | MSE | MAE | RMSE | Volume MSE | Surf MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| u_x | 0.0068 | 0.9994 | 0.2074 | 0.2625 | 0.4147 | 0.00017 | 0.00091 |
| u_y | 0.0314 | 0.9987 | 0.1061 | 0.1916 | 0.2964 | 0.00080 | 0.00190 |
| p | 0.0364 | 0.9985 | 1366 | 19.03 | 29.96 | 0.00068 | 0.00343 |
| ν_t | 0.0571 | 0.9960 | 1.485e-08 | 1.67e-05 | 8.325e-05 | 0.00515 | 0.00185 |

## By angle of attack

| AoA band | cases | rel-L2 | p rel-L2 | ν_t rel-L2 | C_L rel-L2 |
|---|---:|---:|---:|---:|---:|
| AoA < 0° | 55 | 0.0369 | 0.0412 | 0.0612 | 0.0382 |
| 0–5° | 46 | 0.0320 | 0.0382 | 0.0534 | 0.0343 |
| 5–10° | 61 | 0.0296 | 0.0325 | 0.0525 | 0.0310 |
| ≥ 10° | 38 | 0.0336 | 0.0335 | 0.0633 | 0.0308 |

## Worst 10 cases

| Case | U∞ (m/s) | AoA (°) | rel-L2 | abs C_L error | figure |
|---|---:|---:|---:|---:|---|
| `airFoil2D_SST_32.432_14.036_6.578_2.722_5.928` | 32.4 | 14.04 | 0.0894 | 0.1533 | [png](qual/airFoil2D_SST_32.432_14.036_6.578_2.722_5.928.png) |
| `airFoil2D_SST_75.101_-3.321_6.728_6.938_17.749` | 75.1 | -3.32 | 0.0892 | 0.0327 | [png](qual/airFoil2D_SST_75.101_-3.321_6.728_6.938_17.749.png) |
| `airFoil2D_SST_92.85_-3.031_6.083_2.008_9.267` | 92.8 | -3.03 | 0.0664 | 0.0433 | [png](qual/airFoil2D_SST_92.85_-3.031_6.083_2.008_9.267.png) |
| `airFoil2D_SST_47.017_-4.369_4.85_6.296_6.401` | 47.0 | -4.37 | 0.0588 | 0.0284 | [png](qual/airFoil2D_SST_47.017_-4.369_4.85_6.296_6.401.png) |
| `airFoil2D_SST_35.44_14.039_5.196_1.691_9.27` | 35.4 | 14.04 | 0.0524 | 0.1043 | [png](qual/airFoil2D_SST_35.44_14.039_5.196_1.691_9.27.png) |
| `airFoil2D_SST_90.415_14.684_4.356_6.209_11.299` | 90.4 | 14.68 | 0.0478 | 0.0907 | [png](qual/airFoil2D_SST_90.415_14.684_4.356_6.209_11.299.png) |
| `airFoil2D_SST_70.228_11.772_6.988_2.903_7.517` | 70.2 | 11.77 | 0.0478 | 0.1079 | [png](qual/airFoil2D_SST_70.228_11.772_6.988_2.903_7.517.png) |
| `airFoil2D_SST_90.036_-0.06_6.959_5.619_10.438` | 90.0 | -0.06 | 0.0440 | 0.0713 | [png](qual/airFoil2D_SST_90.036_-0.06_6.959_5.619_10.438.png) |
| `airFoil2D_SST_70.291_2.396_6.998_5.692_7.274` | 70.3 | 2.40 | 0.0427 | 0.0600 | [png](qual/airFoil2D_SST_70.291_2.396_6.998_5.692_7.274.png) |
| `airFoil2D_SST_52.518_12.114_1.135_6.268_0.0_5.676` | 52.5 | 12.11 | 0.0422 | 0.1019 | [png](qual/airFoil2D_SST_52.518_12.114_1.135_6.268_0.0_5.676.png) |

## Best 5 cases

| Case | U∞ (m/s) | AoA (°) | rel-L2 | abs C_L error | figure |
|---|---:|---:|---:|---:|---|
| `airFoil2D_SST_39.741_14.642_3.175_3.316_1.0_10.583` | 39.7 | 14.64 | 0.0261 | 0.0781 | [png](qual/airFoil2D_SST_39.741_14.642_3.175_3.316_1.0_10.583.png) |
| `airFoil2D_SST_69.687_7.38_3.55_0.0_14.415` | 69.7 | 7.38 | 0.0261 | 0.0488 | [png](qual/airFoil2D_SST_69.687_7.38_3.55_0.0_14.415.png) |
| `airFoil2D_SST_70.349_11.188_2.899_4.134_1.0_13.659` | 70.3 | 11.19 | 0.0263 | 0.1072 | [png](qual/airFoil2D_SST_70.349_11.188_2.899_4.134_1.0_13.659.png) |
| `airFoil2D_SST_91.602_10.45_2.763_4.44_0.0_16.132` | 91.6 | 10.45 | 0.0264 | 0.0906 | [png](qual/airFoil2D_SST_91.602_10.45_2.763_4.44_0.0_16.132.png) |
| `airFoil2D_SST_87.86_9.36_4.554_3.608_8.92` | 87.9 | 9.36 | 0.0264 | 0.0838 | [png](qual/airFoil2D_SST_87.86_9.36_4.554_3.608_8.92.png) |

## Files

- `per_case.csv`: every metric for every case · `summary.json`: this page as data
- `test_metrics.npz`, `cl_native.npz`: same layout as `scripts/run_eval.sh` (readable by `source/shared/check_metrics.py`)
- `fig/overview.png`, `fig/per_field_box.png` · `qual/<case>.png`: one figure per test case
