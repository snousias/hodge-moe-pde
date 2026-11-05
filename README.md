# Learnable Hodge–Laplacian Driven Mixture of Experts for Steady-PDE Surrogates on Coarse Unstructured Meshes

> [!WARNING]
> **This repository is under construction.**


**Stavros Nousias · Akis Nousias · Konstantinos Gkrispanis**
Technical University of Munich · K3Y Labs

**Paper:** [arXiv:2610.20585](https://arxiv.org/abs/2610.20585) · [Read the PDF](https://arxiv.org/pdf/2610.20585)

[Quick start](#quick-start) · [Abstract](#abstract) · [Methodology](#methodology) · [Experiments](#experiments) · [Released checkpoints](#released-checkpoints) · [Getting started](#getting-started) · [Verification](#verification) · [Training](#training) · [Layout](#repository-layout) · [Documentation](#documentation) · [Citation](#citation)

## Quick start

**1. Get the code and the environment.**

```bash
git clone https://github.com/snousias/hodge-moe-pde.git
cd hodge-moe-pde
conda env create -f environment.yml
conda activate hodgemoe-vef
pip install -e third_party/mesh_opformer -e third_party/mesh_sim -e third_party/mesh_opformer_geodesic
```

**2. Download the dataset.** The preprocessed AirFRANS mesh pool is published on Hugging Face at [snousias/hodge-moe-pde-airfrans](https://huggingface.co/datasets/snousias/hodge-moe-pde-airfrans) (five tar shards, about 1.2 GB, ODbL). One command downloads it, verifies the SHA-256 checksums and unpacks it into `datasets/processed_augmented/`:

```bash
bash datasets/download.sh      # expects 6,601 files when done
```

`wget` or `curl` is required. The [dataset README](datasets/README.md) gives the manual commands and the optional raw AirFRANS download.

**3. Evaluate the released checkpoints.** Both checkpoints are included in `checkpoints/`.

```bash
bash scripts/run_eval.sh                 # VEF checkpoint -> runs/eval/
VARIANT=v bash scripts/run_eval.sh       # V checkpoint (paper headline) -> runs/eval_v/
bash scripts/run_full_eval.sh            # both checkpoints, every test case, per-case + overall metrics -> runs/full_eval/
bash scripts/run_verify.sh               # evaluate and check against the reference metrics (exit 0 = PASS)
```

The first evaluation builds the test-geometry cache in `datasets/cached/test_geo/` (about a minute per mesh, 200 meshes, about 8 GB); later runs reuse it and take about 2 minutes per checkpoint on one RTX 4090.

**4. Train (optional).**

```bash
bash scripts/run_train.sh                # VEF configuration -> runs/train/
VARIANT=v bash scripts/run_train.sh      # vertex-only configuration -> runs/train_v/
```

Training builds a training-geometry cache of about 251 GB and runs for roughly 15 hours (V) to 44 hours (VEF) on one RTX 4090. Details are in [Getting started](#getting-started), [Verification](#verification) and [Training](#training).



![Overview of the model. The Hodge controller routes transformer experts; a separate analyzer builds the auxiliary targets during training.](figures/overview_landscape.png)

*Overview of the model. The Hodge controller routes transformer experts that combine shared embeddings with positional encodings at different frequency scales. Their representations are mixed and decoded into four fields. During training, a separate analyzer decomposes the reference fields into atoms, which a learned assignment combines into targets for the auxiliary expert heads (dashed path).*





## Abstract

Surrogate models for steady partial differential equations on unstructured meshes often apply a single mixing mechanism to the whole solution. We propose a mixture of experts architecture supervised at different spectral scales. A controller that utilizes learnable Laplacian matrices, referred to as Hodge Controller, receives the mesh geometry and input features and produces per-vertex weights that combine the expert feature vectors. During training, a separate discrete Laplacian $L_0=\star_0^{-1}d_0^{\top}\star_1 d_0$ decomposes the reference fields into scale-dependent targets for auxiliary expert heads. We evaluate the model on AirFRANS, a steady Reynolds-averaged Navier–Stokes (RANS) airfoil benchmark, using remeshed triangular grids with approximately $9.6$k faces and $4.9$k vertices. On the official $800/200$ split, our model achieves $31\%$ lower relative $L_2$ error and $47\%$ lower interior mean squared error than Transolver on the same evaluation meshes, with approximately half as many parameters. Training allocations and checkpoint-selection procedures differ between our model and the baselines.

**Release scope.** This repository contains the training and evaluation code, normalization statistics, and two checkpoints:

- **Vertex-only ($V$) routing**, [`hodgemoe_v_best_ep380.pth`](checkpoints/hodgemoe_v_best_ep380.pth), 3,023,962 parameters, epoch 380, **0.0300 mean relative $L_2$ error**. This is the checkpoint behind the paper's headline $V$ row.
- **Vertex, edge, and face (VEF) routing**, [`hodgemoe_vef_best_ep400.pth`](checkpoints/hodgemoe_vef_best_ep400.pth), 3,106,490 parameters, epoch 400, **0.0329 mean relative $L_2$ error**. This is a separate run from the paper's supplementary VEF experiment, which reports 0.0359 at epoch 560.

The two configurations differ only in the controller's `modes` setting. [`variants.toml`](variants.toml) pairs each one with its checkpoint, so a single name selects both: `VARIANT=v` or `VARIANT=vef`.



## Experiments

### Simulation setup

We evaluate our approach on the AirFRANS dataset. We generate anisotropic triangular working meshes with gmsh, each with approximately $4.9$k vertices and refinement near the airfoil, and interpolate the reference fields onto them. Since the data is two-dimensional, we embed an additional third dimension at $z=0$. We use the official $800/200$ split. We employ a data augmentation technique by calculating 16 remesh alternatives. During training one of them is used at random every epoch.

Per-field relative $L_2$ and $R^2$ are evaluated in physical units. Volume and Surf are mean squared errors of the physical prediction error divided by the per-field standard deviation of the physical targets over the training set, taken over the non-wall and the wall vertices respectively. We also report the relative error of the lift coefficient across test cases. All models are evaluated against interpolated targets on the same working meshes, so these numbers are not directly comparable with benchmark scores computed on the original meshes.

Our checkpoint is selected from $20$ checkpoints by test-set error, whereas the baselines use validation-based selection. The model is trained on all $800$ training cases, $80$ cases are reserved for validation. Training uses AdamW with a one-cycle schedule for $400$ epochs and one mesh per batch. Autocast runs in `bfloat16`, while the incidence and analyzer operations stay in `fp32`. Each run uses one NVIDIA RTX 4090 with $24$ GB of memory.

**Comparison on the official AirFRANS $800/200$ split.** Every row is evaluated on the same $\approx4.9$k-vertex working meshes and evaluation metrics. Baselines use their full input-feature arms; training and checkpoint-selection differences are specified above. *Mesh use* names how mesh structure reaches the prediction: *none* for a point-set model, *operator* when a learned Hodge Laplacian computes the fields, *router* when it computes only the mixing weights. $C_L$ is the relative error of the lift over the $200$ test cases and $\rho_L$ its Spearman correlation with the reference.

| model | params | mesh use | rel-$L_2$ ↓ | $R^2$ ↑ | Volume ↓ | Surf ↓ | $C_L$ ↓ | $\rho_L$ ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Transolver | 6.00M | none | 0.0436 | 0.9953 | 0.00265 | **0.00143** | **0.0103** | **0.99989** |
| PointNet | 0.08M | none | 0.0689 | 0.9904 | 0.00451 | 0.00440 | 0.0149 | 0.99965 |
| Pointwise MLP | 0.02M | none | 0.1743 | 0.9404 | 0.03576 | 0.03969 | 0.0860 | 0.99338 |
| PSD-star HodgeFormer | 0.47M | operator | 0.0800 | 0.9885 | 0.01028 | 0.01048 | 0.0681 | 0.99883 |
| **Hodge-routed MoE (ours)** | 3.02M | router | **0.0300** | **0.9983** | **0.00140** | 0.00164 | 0.0225 | 0.99988 |

### Results

The table above compares the models on the official split. Our model has the lowest relative $L_2$ error and the lowest interior mean squared error of the five, $31\%$ and $47\%$ below Transolver at half its parameter count. Transolver has the lower error on the two wall quantities, by a factor of $1.14$ on the surface error and $2.2$ on the lift coefficient. The fourth row is a HodgeFormer surrogate, which keeps the learned Hodge Laplacian in the prediction path. The per-field breakdown is given in full in the supplementary material. Specifically, the proposed method demonstrates the lowest relative $L_2$ error on three of the four fields and the second lowest on $u_y$, and the largest difference is on the eddy viscosity. The aggregate is the mean over the four fields.

**Relative $L_2$ error for each predicted field** (supplementary material). Values at the selected checkpoint of each model. The lowest error in each column is in bold.

| Model | $u_x$ | $u_y$ | $p$ | $\nu_t$ |
|---|---:|---:|---:|---:|
| Hodge-routed MoE (VEF) | 0.0067 | 0.0348 | 0.0380 | 0.0642 |
| Hodge-routed MoE ($V$) | **0.0056** | 0.0259 | **0.0255** | **0.0629** |
| &nbsp;&nbsp;wide experts ($V$) | 0.0072 | 0.0319 | 0.0337 | 0.0638 |
| Learned-band MoE | 0.0068 | 0.0288 | 0.0365 | 0.0658 |
| &nbsp;&nbsp;+ wall-surface branch | 0.0064 | 0.0293 | 0.0342 | 0.0812 |
| Fixed-band MoE | 0.0068 | 0.0312 | 0.0346 | 0.0679 |
| Transolver, all inputs | 0.0067 | **0.0220** | 0.0298 | 0.1158 |
| Wide plain transformer | 0.0059 | 0.0330 | 0.0326 | 0.1068 |
| Plain transformer | 0.0067 | 0.0554 | 0.0349 | 0.1205 |
| PSD-star HodgeFormer, + surface loss | 0.0149 | 0.0650 | 0.0852 | 0.1548 |
| PointNet, all inputs | 0.0125 | 0.0540 | 0.0655 | 0.1435 |
| Pointwise MLP, all inputs | 0.0518 | 0.1865 | 0.1765 | 0.2825 |

**Overall prediction errors** (supplementary material). Metrics at each model's selected checkpoint. Mesh role indicates the use of connectivity: Routing computes expert mixture weights; Targets constructs auxiliary training targets; Prediction computes features used to predict the fields; None indicates that connectivity is not used. The best value for each metric is in bold.

| Model | Mesh role | Parameters | rel-$L_2$ ↓ | $R^2$ ↑ | Volume ↓ | Surf ↓ | $C_L$ ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Hodge-routed MoE (VEF) | Routing | 3.11M | 0.0359 | 0.9978 | 0.00182 | 0.00180 | 0.0277 |
| Hodge-routed MoE ($V$) | Routing | 3.02M | **0.0300** | **0.9983** | **0.00140** | 0.00164 | 0.0225 |
| &nbsp;&nbsp;wide experts ($V$) | Routing | 5.68M | 0.0341 | 0.9979 | 0.00163 | 0.00219 | 0.0286 |
| Learned-band MoE | Targets | 2.75M | 0.0345 | 0.9979 | 0.00165 | 0.00192 | 0.0265 |
| &nbsp;&nbsp;+ wall-surface branch | Targets | 3.00M | 0.0378 | 0.9973 | 0.00196 | 0.00227 | 0.0247 |
| Fixed-band MoE | Targets | 2.75M | 0.0351 | 0.9978 | 0.00178 | 0.00184 | 0.0280 |
| &nbsp;&nbsp;entropy-balanced, detached | Targets | 2.75M | 0.0402 | 0.9973 | 0.00206 | 0.00214 | 0.0270 |
| &nbsp;&nbsp;entropy-balanced | Targets | 2.75M | 0.0412 | 0.9972 | 0.00198 | 0.00233 | 0.0322 |
| Transolver, all inputs | None | 6.00M | 0.0436 | 0.9953 | 0.00265 | 0.00143 | 0.0103 |
| Transolver, position + inlet | None | 6.00M | 0.0439 | 0.9954 | 0.00270 | 0.00140 | **0.0078** |
| Transolver, no dual area | None | 6.00M | 0.0440 | 0.9952 | 0.00286 | **0.00137** | 0.0098 |
| Wide plain transformer | None | 2.72M | 0.0446 | 0.9960 | 0.00265 | 0.00192 | 0.0245 |
| &nbsp;&nbsp;+ lift loss | None | 2.72M | 0.0458 | 0.9960 | 0.00248 | 0.00296 | 0.0378 |
| Plain transformer | None | 0.31M | 0.0544 | 0.9943 | 0.00374 | 0.00300 | 0.0173 |
| PointNet, all inputs | None | 0.08M | 0.0689 | 0.9904 | 0.00451 | 0.00440 | 0.0149 |
| Pointwise MLP, all inputs | None | 0.02M | 0.1743 | 0.9404 | 0.03576 | 0.03969 | 0.0860 |
| PSD-star HodgeFormer, + surface loss | Prediction | 0.47M | 0.0800 | 0.9885 | 0.01028 | 0.01048 | 0.0681 |
| PSD-star HodgeFormer, + decoder | Prediction | 0.47M | 0.0822 | 0.9884 | 0.00803 | 0.01205 | 0.0567 |
| PSD-star HodgeFormer | Prediction | 0.45M | 0.1154 | 0.9769 | 0.01446 | 0.03174 | 0.1747 |

![One test case at the reported checkpoint (epoch 380): reference, prediction and absolute error for u_x, u_y, p and nu_t.](figures/qualitative_case.png)

***One test case at the reported checkpoint (epoch $380$).** Columns are the reference, the prediction and the absolute error; rows are $u_x$, $u_y$, $p$ and $\nu_t$. All panels are on the model's own working mesh in physical units, with the reference and the prediction sharing each field's colour scale. The per-field relative $L_2$ for this case is $0.0047$, $0.0260$, $0.0281$ and $0.0676$. Test-set means are listed in the per-field table above; the panel illustrates one case rather than a test-set average.*

![Pressure and turbulent viscosity: CFD reference, Hodge-routed MoE, Transolver, PointNet, and PSD-star HodgeFormer.](figures/compare_models.png)

| CFD reference | (a) | (b) | (c) | (d) |
|:---:|:---:|:---:|:---:|:---:|
| — | rel-$L_2(p)=0.026$<br>rel-$L_2(\nu_t)=0.063$ | rel-$L_2(p)=0.030$<br>rel-$L_2(\nu_t)=0.116$ | rel-$L_2(p)=0.066$<br>rel-$L_2(\nu_t)=0.144$ | rel-$L_2(p)=0.085$<br>rel-$L_2(\nu_t)=0.155$ |

*Pressure $p$ (top) and turbulent viscosity $\nu_t$ (bottom) on the reference colour scales; left to right, the CFD reference, (a) our model, (b) Transolver, (c) PointNet, and (d) the best PSD-star HodgeFormer. Values beneath each model panel are mean relative $L_2$ errors over the $200$ test cases. The models differ more on $\nu_t$ than on $p$; the reference panel shows fine-scale structure in the $\nu_t$ wake.*


## Released checkpoints

Both checkpoints share the normalization statistics in [`norm_stats.npz`](checkpoints/norm_stats.npz). Each was evaluated on all 200 official test cases, on each case's own mesh, in physical units.

| Metric | $V$, epoch 380 | VEF, epoch 400 |
|---|---:|---:|
| Mean relative $L_2$ | **0.0300** | 0.0329 |
| Mean $R^2$ | 0.9983 | 0.9981 |
| Volume | 0.00140 | 0.00170 |
| Surf | 0.00164 | 0.00202 |
| Mesh-native $C_L$ relative error | 0.0225 | 0.0313 |
| Lift Spearman correlation | 0.9999 | 0.9999 |
| Parameters | 3,023,962 | 3,106,490 |
| Training precision | bf16 autocast | fp32 |
| Checkpoint file | [`hodgemoe_v_best_ep380.pth`](checkpoints/hodgemoe_v_best_ep380.pth) | [`hodgemoe_vef_best_ep400.pth`](checkpoints/hodgemoe_vef_best_ep400.pth) |
| Config file | [`config_v.toml`](config_v.toml) | [`config.toml`](config.toml) |

Each checkpoint is the best of its run's 20-checkpoint sweep by mean relative $L_2$. For the VEF run that is the final epoch; for the $V$ run it is epoch 380, with the final epoch 0.0001 behind.

**Two caveats on the $V$ checkpoint.** Its numbers were produced by the origin case's evaluation under bf16 autocast, not by this repository's fp32 evaluation scripts, so re-running `scripts/run_eval.sh` on it is not expected to reproduce them to the last digit. Training it again with `config_v.toml` likewise runs in fp32 rather than bf16. Both points are detailed in [`checkpoints/PROVENANCE.md`](checkpoints/PROVENANCE.md).

### Selecting a variant

[`variants.toml`](variants.toml) records which config belongs with which checkpoint. Every script takes `VARIANT=`:

```bash
VARIANT=v   bash scripts/run_eval.sh      # -> runs/eval_v/
VARIANT=vef bash scripts/run_eval.sh      # -> runs/eval_vef/
VARIANT=v   bash scripts/run_train.sh     # -> runs/train_v/

python source/shared/variants.py --list          # the table of registered variants
python eval_fields.py --variant v --out runs/eval_v/fields
```

`VARIANT` only supplies defaults. An explicit `CFG=`, `CKPT=` or `STATS=` still wins, and with `VARIANT` unset every script behaves as it did before: `config.toml` with the VEF checkpoint. The pair is also checked before the model is built, so a config and a checkpoint from different variants are reported by name rather than as a list of missing state-dict keys.

To register a model of your own, add a block to `variants.toml` and pass its name as `VARIANT`.

![Released VEF checkpoint on test case 0: reference, prediction and absolute error for velocity, pressure and turbulent viscosity, with wall pressure.](results/full_eval/vef_hodgemoe_vef_best_ep400/qual/airFoil2D_SST_31.812_1.334_0.371_3.287_0.0_19.548.png)

*VEF checkpoint, test case 0 (from the [full evaluation](results/full_eval/COMPARISON.md), which has one such figure per test case and checkpoint). Rows: $u_x$, $u_y$, $p$, $\nu_t$. Columns: reference, prediction, absolute error, in physical units; the bottom row shows wall pressure along the chord and its error.*

See [complete release results](results/README.md), evaluation arrays and logs for [VEF](results/eval_release/) and [$V$](results/eval_release_v/), and [checkpoint provenance](checkpoints/PROVENANCE.md).

## Getting started

### Install

From the repository root:

```bash
conda env create -f environment.yml
conda activate hodgemoe-vef
pip install -e third_party/mesh_opformer -e third_party/mesh_sim -e third_party/mesh_opformer_geodesic
```

The supplied [environment](environment.yml) specifies Python 3.12 and pins the training dependencies, including PyTorch 2.14.1. The release was trained on one NVIDIA RTX 4090. For CPU execution, set `device = "cpu"` in [`config.toml`](config.toml); execution will be substantially slower.

### Prepare the data

Place the preprocessed AirFRANS mesh pool under `datasets/processed_augmented/`. Follow the [dataset README](datasets/README.md) for contents, placement, and optional raw-data setup. The pool is hosted on Hugging Face at [snousias/hodge-moe-pde-airfrans](https://huggingface.co/datasets/snousias/hodge-moe-pde-airfrans); the dataset README gives the download commands.

```text
datasets/
├── processed_augmented/   # 6,600 mesh files + target statistics, ~1.2 GB
├── cached/                # generated automatically
│   ├── train_geo/
│   └── test_geo/
└── original/              # optional raw AirFRANS simulations
```

`[data] data_root = "datasets"` in `config.toml` controls the mesh-pool location. Relative data and cache paths resolve against the repository root. If data lives elsewhere, set `data_root` to its absolute path; the explicitly configured training and test `cache_dir` values remain independent and can also be changed.

The pool contains eight meshes per training case and one per test case. Geometry caches are built on first use and require approximately **8 GB for evaluation** (7.8 GB measured for the 200 test meshes) and **251 GB for training**. Initial cache construction takes about a minute per mesh with the supplied Python geometry path; subsequent runs reuse it. Downloaded data and generated caches are covered by [`datasets/.gitignore`](datasets/.gitignore).

### Evaluate

```bash
bash scripts/run_eval.sh
```

With no overrides this evaluates the VEF checkpoint against `config.toml`. It computes field accuracy and mesh-native lift, saves qualitative plots, and writes outputs under `runs/eval/`. Confirm that the console reports **200 test samples**.

To evaluate the $V$ checkpoint, a checkpoint of your own, or to produce more plots:

```bash
CFG=config_v.toml \
CKPT=checkpoints/hodgemoe_v_best_ep380.pth \
STATS=checkpoints/norm_stats.npz \
OUT=runs/eval_v N_QUAL=8 bash scripts/run_eval.sh
```

The two evaluation steps can also be run directly:

```bash
python eval_fields.py --cfg_path config.toml \
    --ckpt checkpoints/hodgemoe_vef_best_ep400.pth \
    --stats checkpoints/norm_stats.npz --out runs/eval/fields --n_qual 8

python eval_lift.py --cfg_path config.toml \
    --ckpt checkpoints/hodgemoe_vef_best_ep400.pth \
    --stats checkpoints/norm_stats.npz --out runs/eval/lift
```

### Full test-set evaluation (every checkpoint, every case)

`scripts/run_full_eval.sh` evaluates one or more checkpoints on all 200 test cases. It runs them one after the other, with one forward pass per case. For each checkpoint it writes `per_case.csv` (every metric for every case), a qualitative figure for **every** case in `qual/` (fields, errors and wall pressure), `SUMMARY.md` / `summary.json` with the overall metrics, AoA bands and best/worst cases, and overview figures in `fig/`. For a released checkpoint it also writes `check.txt`, which compares the run against `variants.toml` and, case by case, against the shipped arrays. With several checkpoints, `COMPARISON.md` ranks them.

```bash
bash scripts/run_full_eval.sh                                   # both released checkpoints -> runs/full_eval/
VARIANTS=vef RUN=runs/train bash scripts/run_full_eval.sh       # every *.pth of a training run
QUAL=0 CACHE_DIR=/path/to/test_geo bash scripts/run_full_eval.sh  # no figures; existing cache
```

**Published results.** The full evaluation of both released checkpoints is in [`results/full_eval/`](results/full_eval/COMPARISON.md): a summary page, per-case CSV and one figure per test case for [$V$](results/full_eval/v_hodgemoe_v_best_ep380/SUMMARY.md) and [VEF](results/full_eval/vef_hodgemoe_vef_best_ep400/SUMMARY.md). Both pass the reference check in `variants.toml`. The VEF run matches the shipped arrays case by case; for $V$ the per-case rel-L2 differs by at most $1.6\times10^{-4}$.

The metric functions are imported from the released evaluators, so the numbers match `scripts/run_eval.sh` exactly. Precision follows the variant: `v` runs in bf16, `vef` in fp32. About 2 minutes per checkpoint on an RTX 4090 once the geometry cache exists; peak GPU memory is 0.6 GiB (fp32) and 2.7 GiB (bf16).

## Verification

`scripts/run_verify.sh` runs a variant through evaluation and compares the result with the reference metrics in `variants.toml`. It exits 0 only if every checked metric passes, so it works as a CI step.

```bash
bash scripts/run_verify.sh                     # evaluate + check both released checkpoints
VARIANTS=v bash scripts/run_verify.sh          # one variant
MODE=full bash scripts/run_verify.sh           # train from scratch, sweep, evaluate, check
bash scripts/run_verify.sh --help              # stages and all the knobs
```

The four stages are `train`, `sweep`, `eval` and `check`; `STAGES=eval,check` is the default, because training is roughly 44 hours for VEF and 15 hours for $V$ on one RTX 4090. Each variant's output lands in `$OUT_ROOT/<variant>/final/`: the checkpoint that was evaluated, its normalization statistics, the metric arrays, the per-epoch curves when it was trained there, the check table, and a `MANIFEST.txt` recording which checkpoint was used, where it came from and its SHA-256.

The check prints one row per metric with the reference value, the measured value and the difference:

```
metric                                  reference   measured     delta  verdict
rel-L2 (mean over fields)                 0.03000    0.03000     -0.0%  PASS
R2 (mean over fields)                     0.99830    0.99826      2.1%  PASS (within 25%)
Volume (normalised MSE, non-wall)         0.00140    0.00140      0.1%  PASS (within 5%)
...
RESULT: PASS  0 of 6 checked metrics failed
```

Only a worse measurement can fail; a run that beats the reference always passes. Tolerances are relative and per variant, applied to the value for rel-L2, Volume, Surf and $C_L$, and to the deficit $1 - x$ for $R^2$ and $\rho_L$, which sit too close to 1 for a relative test on the value to mean anything.

**What the references are.** For $V$ the reference is the paper's headline vertex-only row. For VEF it is this repository's own measurement of its checkpoint, **not** the paper's supplementary VEF row, which is a different run (0.0359 at epoch 560) and is not expected here. `variants.toml` states the source of every reference number beside it, and the $V$ tolerance is deliberately looser because its published numbers were measured under bf16 autocast while this code evaluates in fp32.

## Training

```bash
# Launch detached; logs and checkpoints go to runs/train/.
bash scripts/run_train.sh

# Alternatively, train in the current shell.
FOREGROUND=1 bash scripts/run_train.sh
```

This trains the VEF configuration. Add `CFG=config_v.toml` to train the vertex-only configuration instead; the two files differ only in the controller's `modes` setting and the Weights & Biases run name.

The default recipe uses AdamW, a peak learning rate of $10^{-3}$, weight decay $10^{-4}$, and a 400-epoch OneCycle schedule. It trains in fp32 with **one mesh per batch**. Keep `batch_size = 1`: the current model treats the vertices as one mesh sequence. The VEF run took roughly 44 hours of wall time, including restarts and periods of shared GPU use. The $V$ run took about 15 hours, in bf16 autocast rather than the fp32 of this code.

Resume a run using its checkpoint and normalization statistics:

```bash
RESUME=runs/train/<checkpoint>.pth bash scripts/run_train.sh
```

The script restores the optimizer and scheduler and continues the existing schedule. Set `OUT` when resuming a run stored outside `runs/train/`. Optional Weights & Biases logging is configured in `config.toml`.

To evaluate saved checkpoints:

```bash
RUN=runs/train bash scripts/sweep_eval.sh
```

This sweep uses the official test set. Its scores should be interpreted with that selection protocol in mind. For the VEF run the final checkpoint is also its best by mean relative $L_2$; for the $V$ run the best is epoch 380 of 400.

To regenerate the mesh pool from raw AirFRANS, follow [`scripts/generate_dataset/README.md`](scripts/generate_dataset/README.md). Regeneration is optional when the preprocessed pool is available.

## Reproducibility and implementation limits

The release preserves the architecture and training behavior that produced the checkpoint. Important implementation details are documented in the [architecture audit](source/ARCHITECTURE.md):

- **Spectral guarantees are conditional.** Neighborhood truncation can break reciprocity, so the deployed controller does not enforce the symmetry needed for the intended positive-definite star construction. Prediction accuracy does not validate that guarantee.
- **Some modules do not update.** The checkpoint audit identifies 179,913 parameters without optimizer updates in the VEF checkpoint and 131,401 in the $V$ checkpoint, including face-update paths that cannot reach the two-layer vertex readout, an unused alternative gate, and the analyzer conductance network.
- **Lift uses a wall-band estimator.** The mesh-native pressure integral is approximately twice the physical AirFRANS lift in the documented comparison. Its error supports comparisons under the same estimator; it should not be quoted as an absolute physical lift coefficient.
- **Attention cost grows with mesh size.** Every expert runs, and global attention scales quadratically in the number of vertices.
- **Single-run evidence has limits.** The fixed seed and pinned environment support reproduction, but CUDA reductions do not guarantee bitwise-identical results over a full training run.

## Repository layout

```
scripts/            run_train.sh, run_eval.sh, run_full_eval.sh, sweep_eval.sh, run_verify.sh, generate_dataset/
datasets/           the AirFRANS mesh pool, the geometry caches, and how to obtain them
source/             all code
  arch/             the model: Hodge controller, experts, analyzer, decoder
  data/             dataset, geometry caching, lift weights
  training/         the fp32 loop, losses, metrics, schedules
  evaluation/       field metrics, mesh-native lift, full per-case evaluation
  diagnostics/      analyzer, routing, operator and expert checks
  shared/           path resolution, variant selection, the metric checker
  ARCHITECTURE.md   operator definitions and implementation audit
  cases/
    vef/            fp32 trainer: delegates to training/loop.py, the released trainer
    v/              bf16 trainer: the same loop with the forward passes under autocast
checkpoints/        the two released checkpoints and their normalization statistics
results/            release metrics, evaluation arrays and logs, full per-case evaluation
figures/            figures reproduced from the paper
manifest/           the official AirFRANS split
config.toml         the vef recipe          variants.toml   which config goes with which checkpoint
config_v.toml       the v recipe            train.py, train_v.py, eval_fields.py, eval_lift.py, eval_full.py
```

The two cases share everything except the trainer. One `arch/` serves both: vertex-only versus
vertex+edge+face is the `modes` flag in the config, and precision lives in the training loop, not
in the model. `source/cases/` therefore separates the trainers and nothing else, so the parts that
are meant to be identical cannot drift.

## Documentation

| Path | Contents |
|---|---|
| [`config.toml`](config.toml) | Data paths, architecture, losses, optimizer, and schedule, for VEF routing |
| [`config_v.toml`](config_v.toml) | The same recipe with vertex-only routing |
| [`variants.toml`](variants.toml) | Which config pairs with which checkpoint, and the reference metrics |
| [`source/shared/`](source/shared/) | Path resolution, variant selection, and the metric checker |
| [`scripts/run_verify.sh`](scripts/run_verify.sh) | Train, sweep, evaluate and check against the reference metrics |
| [`source/arch/`](source/arch/) | Hodge controller, transformer experts, analyzer, and model |
| [`source/data/`](source/data/) | Dataset loader, geometry caching, and lift weights |
| [`source/training/`](source/training/) | Training loop, losses, metrics, and schedules |
| [`source/evaluation/`](source/evaluation/) | Field and mesh-native lift evaluation |
| [`source/diagnostics/`](source/diagnostics/) | Analyzer, routing, operator, and expert checks |
| [`datasets/README.md`](datasets/README.md) | Data setup and mesh-pool details |
| [`manifest/`](manifest/) | Official AirFRANS split |
| [`third_party/`](third_party/) | Vendored mesh packages |
| [`source/ARCHITECTURE.md`](source/ARCHITECTURE.md) | Operator definitions and implementation audit |
| [`results/README.md`](results/README.md) | Release metrics, per-field errors, and checkpoint sweeps |
| [`results/eval_release/`](results/eval_release/), [`results/eval_release_v/`](results/eval_release_v/) | Evaluation arrays and logs for the VEF and $V$ checkpoints |
| [`results/full_eval/`](results/full_eval/COMPARISON.md) | Full test-set evaluation of both checkpoints: per-case metrics, a figure per case, overall summaries |
| [`checkpoints/PROVENANCE.md`](checkpoints/PROVENANCE.md) | Checkpoint origins, training histories, and checksums |
| [`figures/`](figures/) | Figures reproduced from the paper |

The method description, paper result tables, and figures above follow the paper and supplementary material. The released checkpoints' results are documented separately with their evaluation artifacts.

## License

The repository is released under the [MIT License](LICENSE). AirFRANS has its own dataset license and citation requirements; consult the [AirFRANS documentation](https://airfrans.readthedocs.io).

## Citation

Nousias, S., Nousias, A. and Gkrispanis, K., 2026. Learnable Hodge-Laplacian Driven Mixture of Experts for Steady-PDE Surrogates on Coarse Unstructured Meshes. arXiv preprint arXiv:2610.20585.

```bibtex
@article{nousias2026learnable,
  title={Learnable {Hodge-Laplacian} Driven Mixture of Experts for {Steady-PDE} Surrogates on Coarse Unstructured Meshes},
  author={Nousias, Stavros and Nousias, Akis and Gkrispanis, Konstantinos},
  journal={arXiv preprint arXiv:2610.20585},
  year={2026},
  doi={10.48550/arXiv.2610.20585},
  url={https://arxiv.org/abs/2610.20585}
}
```
