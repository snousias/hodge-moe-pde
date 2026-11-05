# Checkpoint provenance

Two checkpoints are released. They are the same architecture at two settings of the controller's
`modes` flag, and they are **not** two runs of the same configuration.

| file | config | `modes` | params | epoch | rel-L2 |
|---|---|---|--:|--:|--:|
| `hodgemoe_v_best_ep380.pth` | `config_v.toml` | `"v"` | 3,023,962 | 380 / 400 | **0.0300** |
| `hodgemoe_vef_best_ep400.pth` | `config.toml` | `"vef"` | 3,106,490 | 400 / 400 | 0.0329 |

Both are loaded with the same `norm_stats.npz`. Each file contains `state` (model weights),
`optimizer`, `scheduler` and `epoch`, so either can be resumed with `RESUME=... bash scripts/run_train.sh`.

Each checkpoint must be used with its own config. `variants.toml` holds the pairing, so naming the
variant selects both:

```
VARIANT=v   bash scripts/run_eval.sh      # vertex-only       -> runs/eval_v/
VARIANT=vef bash scripts/run_eval.sh      # vertex+edge+face  -> runs/eval_vef/
```

A crossed pair is caught before the model is built and reported by name. To confirm either
checkpoint reproduces its published metrics:

```
VARIANTS=v bash scripts/run_verify.sh
```

---

# `hodgemoe_v_best_ep380.pth` — the vertex-only checkpoint

This is the checkpoint behind the paper's headline vertex-only ($V$) row: 3,023,962 parameters,
mean relative $L_2$ 0.0300, selected at epoch 380.

| | |
|---|---|
| origin case | `experiments/regression_pdes_tetrahedral/cases/v57_gated_hodge` in the parent repository |
| origin file | `runs/v57_gated_hodge/airfrans_hodgeformer-airfrans-v57_gated_hodge_ckpt_380.pth` (byte-identical copy) |
| sha256 | `3752dc70a101f23c3da1efaef50594c61905596cba36a539c5f82d24359f2382` |
| epoch | 380 / 400, selected by mean rel-L2 over the 20-checkpoint sweep (epochs 20, 40, …, 400) |
| params | 3,023,962 (of which 131,401 never receive an update; see "Parameters that do not update" below) |
| precision | **bf16 autocast** during training and during the evaluation that produced the numbers below. The stored weights are fp32 |
| software | Python 3.12 (conda env `hodgeformer`). The run log does not record the torch version |
| hardware | one CUDA device; the run log does not record the model |
| trained | 2026-08-18 01:03 to 15:39 CEST, 400 epochs in one uninterrupted process (W&B run `qc9jg4a1`) |

## Measured results (200 official AirFRANS test cases, own grid, physical space)

These come from the origin repository's sweep and lift evaluation, copied into
`results/eval_release_v/`. They were **not** produced by `scripts/run_eval.sh` in this folder — see
"Precision" below.

| metric | value |
|---|---|
| rel-L2 (mean over u_x, u_y, p, ν_t) | **0.0300** |
| R² (mean over fields) | 0.9983 |
| Volume (AirFRANS-native normalised MSE, non-wall) | 0.00140 |
| Surf (AirFRANS-native normalised MSE, wall) | 0.00164 |
| C_L rel-L2 (wall-band contour, `eval_lift.py`) | 0.0225 |
| ρ_L (Spearman, lift) | 0.9999 |

Per field at epoch 380:

| field | rel-L2 | R² | Volume MSE | Surf MSE |
|---|--:|--:|--:|--:|
| u_x | 0.0056 | 0.9996 | 0.00011 | 0.00073 |
| u_y | 0.0259 | 0.9991 | 0.00052 | 0.00174 |
| p | 0.0255 | 0.9992 | 0.00031 | 0.00248 |
| ν_t | 0.0629 | 0.9952 | 0.00466 | 0.00160 |

Neighbouring checkpoints: 0.0307 at epoch 360, 0.0301 at epoch 400. The full 20-checkpoint sweep is
in `results/eval_release_v/sweep_summary.txt`.

## Precision

The origin case runs both training and evaluation under `torch.autocast(bfloat16)`. The code in this
deployment is fp32 throughout, because it is the code that produced the `vef` checkpoint. Two
consequences:

- `scripts/run_eval.sh` on this checkpoint computes the forward pass in fp32, not bf16. The metrics
  it prints are therefore not guaranteed to match the table above to the last digit. The fp32
  evaluation of these weights has not been run, so the size of the difference is not reported here.
- `bash scripts/run_train.sh` with `config_v.toml` trains the same architecture, data, loss, seed and
  schedule in fp32. It does not reproduce this checkpoint's arithmetic.

## Parameters that do not update

34 of 390 parameter tensors (131,401 parameters) have no entry in the saved optimizer state, so they
hold their initial values. They are counted in the 3,023,962 total.

| module | params | why |
|---|--:|---|
| `hodge_gate.hodge` | 107,840 | `f_ff` and `n_f2` in both layers, `e_ff` and `n_e2` in layer 2, `attn.v_phi` in both layers |
| `moe.gate` | 12,872 | the expert branch's own MLP gate. The Hodge branch supplies the gate logits instead |
| `hodge_gate.embed` | 9,216 | the face-feature embedding `f_embed`; at `modes = "v"` nothing reads the face stream |
| `analyzer.edge_mlp` | 1,473 | training-only analyzer. `detach_band_targets = true` cuts its gradient |

The corresponding figure for `vef` is 179,913 parameters in 63 of 442 tensors; see
`source/ARCHITECTURE.md`.

---

# `hodgemoe_vef_best_ep400.pth` — the vertex, edge and face checkpoint

| | |
|---|---|
| origin case | `experiments/regression_pdes_tetrahedral/cases/v57_gated_hodge_vef_v2` in the parent repository |
| origin file | `runs/v57_gated_hodge_vef_v2/airfrans_hodgeformer-airfrans-v57_gated_hodge_vef_v2_ckpt_400.pth` (byte-identical copy) |
| sha256 | `f785812fe067a3c0c48c05afa7a0e83175229172636d5aadfcb7efc10fce912c` |
| epoch | 400 / 400, the final epoch. It is also the best of the 20-checkpoint sweep (epochs 20, 40, …, 400) by rel-L2 |
| params | 3,106,490 (of which 179,913 never receive an update; see `source/ARCHITECTURE.md`, Known limitations) |
| precision | fp32 throughout (no autocast) |
| software | torch 2.14.1+cu130, numpy 2.5.4, Python 3.12 (`environment.yml`) |
| hardware | one RTX 4090, `batch_size = 1` |
| trained | 2026-10-12 22:42 to 2026-10-14 18:35 CEST |
| repo commit at packaging time | `15876bbb60f34a40dc923980710a25afab8d447b` (the case code was uncommitted working-tree code on top of it) |

## Training history

The run used the exact recipe in `config.toml`. Only the data paths, `use_wandb`, the W&B names
and `ckpt_every` differ; the original run saved every epoch. The process stopped four times
during epochs 2–3. Twice it raised `RuntimeError: Event device type CUDA does not match blocking
stream's device type CPU` (a torch 2.14 issue), and twice it exited without a traceback. Each
time it was resumed from the last saved checkpoint with
`--continue_schedule`, which restores the optimizer, the OneCycle scheduler position and the
epoch, so the learning-rate schedule is the uninterrupted one. After epoch 3 it ran to 400
without interruption. The per-epoch training and validation curves are in
`results/eval_release/train_metrics.csv`.

## Measured results (200 official AirFRANS test cases, own grid, physical space)

Produced by `bash scripts/run_eval.sh` in this folder (logs and arrays in `results/eval_release/`):

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

---

# `norm_stats.npz`

Standardisation statistics (mean and std of the vertex features `x_v` and of the
freestream-non-dimensionalised targets), computed once over the 800 official training cases
(s0 meshes). The two runs produced byte-identical files; this one is
`runs/v57_gated_hodge_vef_v2/airfrans_norm_stats.npz`, and
`runs/v57_gated_hodge/airfrans_norm_stats.npz` has the same sha256
(`1276cf59b08aad87a80d93e0e3844d056544b34e29f2d6184df8d95a7adf7886`). Both evaluation entry points
require it (`--stats checkpoints/norm_stats.npz`) and it applies to both checkpoints. Training
recomputes and saves its own copy.
