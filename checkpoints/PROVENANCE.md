# Checkpoint provenance

`hodgemoe_vef_best_ep400.pth` is the released HodgeMoE-vef checkpoint.

| | |
|---|---|
| origin case | `experiments/regression_pdes_tetrahedral/cases/v57_gated_hodge_vef_v2` in the parent repository |
| origin file | `runs/v57_gated_hodge_vef_v2/airfrans_hodgeformer-airfrans-v57_gated_hodge_vef_v2_ckpt_400.pth` (byte-identical copy) |
| sha256 | `f785812fe067a3c0c48c05afa7a0e83175229172636d5aadfcb7efc10fce912c` |
| epoch | 400 / 400, the final epoch. It is also the best of the 20-checkpoint sweep (epochs 20, 40, …, 400) by rel-L2 |
| params | 3,106,490 (of which 179,913 never receive an update; see `docs/ARCHITECTURE.md`, Known limitations) |
| precision | fp32 throughout (no autocast) |
| software | torch 2.14.1+cu130, numpy 2.5.4, Python 3.12 (`environment.yml`) |
| hardware | one RTX 4090, `batch_size = 1` |
| trained | 2026-10-12 22:42 to 2026-10-14 18:35 CEST |
| repo commit at packaging time | `15876bbb60f34a40dc923980710a25afab8d447b` (the case code was uncommitted working-tree code on top of it) |
| contains | `state` (model weights), `optimizer`, `scheduler`, `epoch`, so it can be resumed with `RESUME=... bash scripts/run_train.sh` |

## Training history

The run used the exact recipe in `config.toml`. Only the data paths, `use_wandb`, the W&B names
and `ckpt_every` differ; the original run saved every epoch. The process stopped four times
during epochs 2–3. Twice it raised `RuntimeError: Event device type CUDA does not match blocking
stream's device type CPU` (a torch 2.14 issue), and twice it exited without a traceback. Each
time it was resumed from the last saved checkpoint with
`--continue_schedule`, which restores the optimizer, the OneCycle scheduler position and the
epoch, so the learning-rate schedule is the uninterrupted one. After epoch 3 it ran to 400
without interruption. The per-epoch training and validation curves are in
`docs/eval_release/train_metrics.csv`.

## Measured results (200 official AirFRANS test cases, own grid, physical space)

Produced by `bash scripts/run_eval.sh` in this folder (logs and arrays in `docs/eval_release/`):

| metric | value |
|---|---|
| rel-L2 (mean over u_x, u_y, p, ν_t) | **0.0329** |
| R² (mean over fields) | 0.9981 |
| Volume (AirFRANS-native normalised MSE, non-wall) | 0.00170 |
| Surf (AirFRANS-native normalised MSE, wall) | 0.00202 |
| C_L rel-L2 (wall-band contour, `eval_lift.py`) | 0.0313 |
| ρ_L (Spearman, lift) | 0.9999 |

## Reproducing this checkpoint from scratch

```
bash scripts/run_train.sh
```

with `config.toml` unchanged except `[data] data_root`. A bit-for-bit reproduction is not
guaranteed, because CUDA reduction kernels are not fully deterministic over 400 epochs. The
architecture, data, loss, seed and schedule are identical.

## `norm_stats.npz`

Standardisation statistics (mean and std of the vertex features `x_v` and of the
freestream-non-dimensionalised targets), computed once over the 800 official training cases
(s0 meshes). Copied from `runs/v57_gated_hodge_vef_v2/airfrans_norm_stats.npz`. Both evaluation
entry points require it (`--stats checkpoints/norm_stats.npz`). Training recomputes and saves
its own copy.
