# Evaluation artifacts — vertex-only checkpoint

Arrays and logs for `checkpoints/hodgemoe_v_best_ep380.pth`. The matching directory for the
`vef` checkpoint is `../eval_release/`.

| file | contents |
|---|---|
| `test_metrics.npz` | per-case, per-field metrics at epoch 380, shape (200, 4, 6) = (case, [u_x, u_y, p, ν_t], [MSE, MAE, RMSE, R², rel-L2, rel-L2 centred]), plus `vol_mse`, `surf_mse` |
| `cl_native.npz` | per-case true and predicted C_L, and the aggregate C_L rel-L2 and ρ_L |
| `sweep_summary.txt` | full per-field output for each of the 20 swept checkpoints (epochs 20, 40, …, 400) |
| `lift.log` | console output of the lift evaluation at epoch 380 |
| `train_metrics.csv` | per-epoch training and validation curves of the run, epochs 1–400 |

## Where these came from

Unlike `../eval_release/`, these files were **not** produced by `scripts/run_eval.sh` in this
folder. They were copied from the origin case `cases/v57_gated_hodge` in the parent repository,
from `runs/v57_gated_hodge/`:

| file here | origin file |
|---|---|
| `test_metrics.npz` | `eval_sweep/ckpt_380/test_metrics.npz` |
| `cl_native.npz` | `eval_cl_ckpt380/cl_native.npz` |
| `sweep_summary.txt` | `eval_sweep/summary.txt` |
| `lift.log` | `eval_cl_ckpt380.log`, with the repeated sparse-tensor `UserWarning` lines removed |
| `train_metrics.csv` | `metrics.csv` |

The origin evaluation code runs the forward pass under `torch.autocast(bfloat16)`; the evaluation
code in this deployment is fp32. Re-running `scripts/run_eval.sh` on these weights is therefore not
expected to reproduce the numbers in these files exactly. See `checkpoints/PROVENANCE.md`,
"Precision".

There are no `fields.log` or qualitative figures for this checkpoint in this folder; the per-field
numbers are in `sweep_summary.txt` under `---- ckpt_380 ----`, and the paper's qualitative figure
for this configuration is reproduced in its supplementary material.

## Reading `sweep_summary.txt`

Each `---- ckpt_N ----` block has two parts. The first has one row per field with the columns

```
field   rel-L2   rel-L2-centred   R²   MSE   MAE   RMSE
```

followed by `MEAN`, the mean per-field rel-L2. The second has one row per field with the columns

```
field   Volume-MSE   Surf-MSE
```

followed by `VOLUME` and `SURF`, their means over the four fields. Volume and Surf use the
AirFRANS-native normalisation and split interior from wall vertices.
