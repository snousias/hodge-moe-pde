# Full test-set evaluation — HodgeMoE-v · hodgemoe_v_best_ep380.pth

- Checkpoint: `checkpoints/hodgemoe_v_best_ep380.pth` (epoch 380, SHA-256 `3752dc70a101f23c…`)
- Config: `config_v.toml` · modes `v` · precision **bf16** · 3,023,962 parameters
- Test cases: **200** (official AirFRANS `full` split, fixed `__s0` meshes)

## Overall

| rel-L2 (mean) | rel-L2 (median) | rel-L2 (90th pct) | R² | Volume | Surf | C_L rel-L2 | ρ_L | mean abs C_L error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.0300** | 0.0280 | 0.0383 | 0.9983 | 0.00140 | 0.00164 | 0.0225 | 0.99988 | 0.0322 |

rel-L2 and R² are per-case values in physical units, averaged over the 4 fields and the test cases. Volume / Surf are the AirFRANS-native MSE of fields normalised by the global training std, on non-wall / wall vertices. C_L is the mesh-native pressure-integral lift (see README: about twice the physical AirFRANS lift; compare only like with like).

## Per field

| Field | rel-L2 | R² | MSE | MAE | RMSE | Volume MSE | Surf MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| u_x | 0.0056 | 0.9996 | 0.1542 | 0.1992 | 0.3426 | 0.00011 | 0.00073 |
| u_y | 0.0259 | 0.9991 | 0.08063 | 0.1503 | 0.2502 | 0.00052 | 0.00174 |
| p | 0.0255 | 0.9992 | 829.1 | 12.25 | 21.74 | 0.00031 | 0.00248 |
| ν_t | 0.0629 | 0.9952 | 1.332e-08 | 1.545e-05 | 8.678e-05 | 0.00466 | 0.00160 |

## By angle of attack

| AoA band | cases | rel-L2 | p rel-L2 | ν_t rel-L2 | C_L rel-L2 |
|---|---:|---:|---:|---:|---:|
| AoA < 0° | 55 | 0.0357 | 0.0316 | 0.0751 | 0.0326 |
| 0–5° | 46 | 0.0282 | 0.0216 | 0.0637 | 0.0147 |
| 5–10° | 61 | 0.0256 | 0.0215 | 0.0532 | 0.0203 |
| ≥ 10° | 38 | 0.0309 | 0.0280 | 0.0599 | 0.0243 |

## Worst 10 cases

| Case | U∞ (m/s) | AoA (°) | rel-L2 | abs C_L error | figure |
|---|---:|---:|---:|---:|---|
| `airFoil2D_SST_75.101_-3.321_6.728_6.938_17.749` | 75.1 | -3.32 | 0.0978 | 0.0800 | [png](qual/airFoil2D_SST_75.101_-3.321_6.728_6.938_17.749.png) |
| `airFoil2D_SST_32.432_14.036_6.578_2.722_5.928` | 32.4 | 14.04 | 0.0795 | 0.1446 | [png](qual/airFoil2D_SST_32.432_14.036_6.578_2.722_5.928.png) |
| `airFoil2D_SST_92.85_-3.031_6.083_2.008_9.267` | 92.8 | -3.03 | 0.0752 | 0.0308 | [png](qual/airFoil2D_SST_92.85_-3.031_6.083_2.008_9.267.png) |
| `airFoil2D_SST_47.017_-4.369_4.85_6.296_6.401` | 47.0 | -4.37 | 0.0614 | 0.0236 | [png](qual/airFoil2D_SST_47.017_-4.369_4.85_6.296_6.401.png) |
| `airFoil2D_SST_35.44_14.039_5.196_1.691_9.27` | 35.4 | 14.04 | 0.0475 | 0.0768 | [png](qual/airFoil2D_SST_35.44_14.039_5.196_1.691_9.27.png) |
| `airFoil2D_SST_81.115_-3.901_4.771_2.338_11.129` | 81.1 | -3.90 | 0.0461 | 0.0131 | [png](qual/airFoil2D_SST_81.115_-3.901_4.771_2.338_11.129.png) |
| `airFoil2D_SST_52.518_12.114_1.135_6.268_0.0_5.676` | 52.5 | 12.11 | 0.0450 | 0.0747 | [png](qual/airFoil2D_SST_52.518_12.114_1.135_6.268_0.0_5.676.png) |
| `airFoil2D_SST_54.799_-2.718_5.872_6.096_7.675` | 54.8 | -2.72 | 0.0434 | 0.0244 | [png](qual/airFoil2D_SST_54.799_-2.718_5.872_6.096_7.675.png) |
| `airFoil2D_SST_70.291_2.396_6.998_5.692_7.274` | 70.3 | 2.40 | 0.0433 | 0.0206 | [png](qual/airFoil2D_SST_70.291_2.396_6.998_5.692_7.274.png) |
| `airFoil2D_SST_70.228_11.772_6.988_2.903_7.517` | 70.2 | 11.77 | 0.0432 | 0.0773 | [png](qual/airFoil2D_SST_70.228_11.772_6.988_2.903_7.517.png) |

## Best 5 cases

| Case | U∞ (m/s) | AoA (°) | rel-L2 | abs C_L error | figure |
|---|---:|---:|---:|---:|---|
| `airFoil2D_SST_35.507_5.777_1.428_6.645_0.0_10.474` | 35.5 | 5.78 | 0.0198 | 0.0172 | [png](qual/airFoil2D_SST_35.507_5.777_1.428_6.645_0.0_10.474.png) |
| `airFoil2D_SST_39.31_8.883_4.707_0.0_12.552` | 39.3 | 8.88 | 0.0217 | 0.0319 | [png](qual/airFoil2D_SST_39.31_8.883_4.707_0.0_12.552.png) |
| `airFoil2D_SST_65.717_9.69_2.928_3.947_15.912` | 65.7 | 9.69 | 0.0217 | 0.0414 | [png](qual/airFoil2D_SST_65.717_9.69_2.928_3.947_15.912.png) |
| `airFoil2D_SST_54.238_7.911_2.123_3.832_1.0_9.902` | 54.2 | 7.91 | 0.0220 | 0.0378 | [png](qual/airFoil2D_SST_54.238_7.911_2.123_3.832_1.0_9.902.png) |
| `airFoil2D_SST_81.349_2.289_1.592_3.425_10.894` | 81.3 | 2.29 | 0.0220 | 0.0070 | [png](qual/airFoil2D_SST_81.349_2.289_1.592_3.425_10.894.png) |

## Files

- `per_case.csv`: every metric for every case · `summary.json`: this page as data
- `test_metrics.npz`, `cl_native.npz`: same layout as `scripts/run_eval.sh` (readable by `source/shared/check_metrics.py`)
- `fig/overview.png`, `fig/per_field_box.png` · `qual/<case>.png`: one figure per test case
