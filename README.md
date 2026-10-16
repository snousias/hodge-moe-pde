# HodgeMoE v57: gated-Hodge, `vef` controller

A frequency mixture of experts for steady PDE surrogates on unstructured meshes. Eight vanilla
transformer experts, each biased towards one spatial frequency scale, are blended per vertex by
a gate. A two-layer discrete Hodge-Laplacian branch that reads the simplicial complex computes
that gate. The Hodge branch only routes; the transformers do the fitting.

This release is the **`vef` variant**: the Hodge controller has its own Hodge-Laplacian on each
cochain stream (L₀ on vertices, L₁ = L₁ᵈᵒʷⁿ + L₁ᵘᵖ on edges, L₂ᵈᵒʷⁿ on faces). The HodgePDEMoE
release had a vertex operator only (`modes = "v"`). Everything else (experts, analyzer, loss,
data, 400-epoch schedule) is the same, except that this model trained in fp32 instead of bf16.
Internal name during development: `v57_gated_hodge_vef_v2`.

Benchmark: AirFRANS (2D steady RANS around airfoils; predict `u_x, u_y, p, ν_t`), official
800/200 split.

## 1. Install

```bash
conda env create -f environment.yml
conda activate hodgemoe-vef
pip install -e third_party/mesh_opformer -e third_party/mesh_sim -e third_party/mesh_opformer_geodesic
```

That is the complete dependency set for training and evaluation. PyTorch 2.14.1 (CUDA 13) comes
from pip; see the header of `environment.yml`.

## 2. Data

One setting controls all data paths: `[data] data_root` in `config.toml`, default
`"dataset"`. Relative paths resolve against **this folder**, not the current working directory
(`shared/paths.py`), so the folder can be copied or moved without edits.

```
dataset/
  processed_augmented/   download separately: all 6,600 pool meshes for the full 800/200 split (1.2 GB)
  cached/                geometry cache, built automatically on first use (~271 GB when complete)
  original/              raw AirFRANS, optional (only eval_lift.py --validate and data regeneration)
```

Download the preprocessed model input into `dataset/processed_augmented/` before training
or evaluation; see [dataset setup and download instructions](dataset/README.md). The Dropbox
shared link is pending. Existing checkouts may already contain the full pool. The geometry
cache is derived from it and builds itself on first use. The dataset README also describes
which 8 of the 16 remeshed variants each training case uses.

## 3. Workflows

All scripts run from any directory and use the `python` of the active environment (override
with `PYTHON=...`).

### A. Evaluate the released checkpoint

```bash
bash scripts/run_eval.sh                       # -> runs/eval/{fields,lift}/ + logs
```

This runs field metrics (rel-L2, R², Volume, Surf), qualitative figures and mesh-native lift
(C_L, ρ_L) over the 200 official test cases. It is equivalent to:

```bash
python eval_fields.py --cfg_path config.toml --ckpt checkpoints/<released>.pth \
    --stats checkpoints/norm_stats.npz --out runs/eval/fields --n_qual 6
python eval_lift.py   --cfg_path config.toml --ckpt checkpoints/<released>.pth \
    --stats checkpoints/norm_stats.npz --out runs/eval/lift
```

The console prints `HodgeMoE-vef test samples: N`; check that it reads 200. The first run
builds the test geometry cache (about a minute per mesh, 20 GB). Later runs read it from disk.

### B. Train from scratch

```bash
bash scripts/run_train.sh                      # detached; log at runs/train/train.log
FOREGROUND=1 bash scripts/run_train.sh         # in this shell
RESUME=runs/train/<ckpt>.pth bash scripts/run_train.sh   # resume, continuing the same LR schedule
```

The run is 400 epochs on a single GPU with `batch_size = 1`, in fp32. The original run took
about 5 min per epoch on one RTX 4090, roughly 33 h of training (the original run took 44 h of wall-clock time, including restarts and periods when other jobs shared the GPU). The first epoch is much slower
while it builds the training geometry cache (about 251 GB). Set `use_wandb = true` in
`config.toml` to log to Weights & Biases.

### C. Pick the best checkpoint of a run

```bash
RUN=runs/train bash scripts/sweep_eval.sh      # evaluates ckpt 20, 40, ..., 400; prints the best
```

This is how the released checkpoint was chosen. The sweep scores on the official **test** set,
because this protocol has no validation split. See `docs/RESULTS.md`.

### D. Regenerate the mesh pool from raw AirFRANS (optional)

`scripts/generate_dataset/` rebuilds `processed_augmented/` from `dataset/original/` (see its
README). You do not need this step if you download the preprocessed pool.

### Diagnostics

`diagnostics/` holds checks that never run during training:

| script | checks |
|---|---|
| `verify_analyzer.py` | learned analyzer: L₀ PSD, smoothing iteration stable, atoms telescope to the target exactly, gradient reaches the stars |
| `verify_vef.py` | `vef` vs `v` parameter difference; finite forward pass; perturbing the face features moves the gate under `vef` only |
| `check_psd.py --cfg_path config.toml --ckpt ...` | assembles the learned stars densely on a test mesh: symmetry, extremal eigenvalues, spectrum of L₀ |
| `inspect_experts.py --ckpt ... --stats ...` | gate mass per expert, gate entropy, wall vs interior routing, per-expert band error |
| `diagnose_attention.py --ckpt ... --stats ...` | whether the expert attention uses context (vertex-subset consistency test) |

## 4. Layout

```
config.toml              THE config: data_root, model, training, loss, schedule
train.py                 training entry point          -> training.loop.main
eval_fields.py           field metrics + figures       -> evaluation.fields.main
eval_lift.py             mesh-native lift (C_L, ρ_L)    -> evaluation.lift.main
scripts/                 run_train.sh, run_eval.sh, sweep_eval.sh, generate_dataset/

arch/                    the model only: arch.build_model(**cfg["train"]["model"])
  model.py                 MoEModel: two branches + decoder
  hodge_gate.py            the Hodge spectral controller (produces the gate)
  psd_hodge.py             PSD Hodge-Laplacian layer: L0 / L1 / L2 operators, Gaussian + diagonal stars
  experts.py               8 frequency experts + gate mixing
  analyzer.py              training-only learned Hodge analyzer (defines the target bands)
data/                    dataset (AirfransV39Dataset), geometry cache, C_L weights
training/                loss, metrics, LR schedule, epoch loop (incl. --continue_schedule resume)
evaluation/              field metrics and mesh-native lift
diagnostics/             one-off checks (see above)
shared/                  path resolution, AirFRANS-native metrics
third_party/             vendored mesh packages: mesh_opformer, mesh_sim, mesh_opformer_geodesic
manifest/                official AirFRANS 800/200 split
dataset/                 the data (see section 2)
checkpoints/             released checkpoint, norm stats, PROVENANCE.md
docs/ARCHITECTURE.md     model description, formulas, known limitations (incl. never-trained params)
docs/RESULTS.md          metrics, per-field breakdown, checkpoint sweep, comparison
docs/qualitative/        prediction / error figures on held-out test cases
docs/eval_release/       raw evaluation arrays + logs of the released checkpoint, sweep output, training curves
docs/case_docs/          the original development write-ups (REPORT, ZERO_TO_HERO, figures)
```

## 5. Results

Evaluated on each case's own mesh in physical units, on the 200 official AirFRANS test cases.
The released checkpoint is the final epoch (400), which is also the best checkpoint of the sweep.

| model | params | rel-L2 ↓ | R² ↑ | Volume ↓ | Surf ↓ | C_L ↓ | ρ_L ↑ |
|---|--:|--:|--:|--:|--:|--:|--:|
| **HodgeMoE-vef (this release)** | 3.11M | 0.0329 | 0.9981 | 0.00170 | 0.00202 | 0.0313 | 0.9999 |
| HodgePDEMoE (`modes = "v"`) | 3.02M | **0.0300** | **0.9983** | **0.00140** | 0.00164 | 0.0225 | 0.9999 |
| HodgeMoE-vef, first run (ep 560) | 3.11M | 0.0359 | 0.9978 | 0.00182 | 0.00180 | 0.0277 | 0.9999 |
| Transolver | 6.00M | 0.0436 | – | 0.00265 | **0.00143** | **0.0103** | 0.9999 |
| PointNet | 75k | 0.0689 | – | 0.00451 | 0.00440 | 0.0149 | 0.9996 |
| MLP | 20k | 0.1743 | – | 0.03576 | 0.03969 | 0.0860 | 0.9934 |

`vef` does not improve on the vertex-only HodgePDEMoE: this release is about 10% worse on rel-L2.
Part of the reason is that most of the added edge and face operators never train (see
`docs/ARCHITECTURE.md`, Known limitations). The two `vef` runs are themselves 0.003 apart, so
`v` versus `vef` is within run-to-run spread. Compared with Transolver, at about half its
parameter count, rel-L2 is 25% lower and Volume 36% lower, while Surf is 1.4× higher and C_L
3.0× higher. C_L is the wall-band estimator, about 2× the physical lift, and is valid only for
comparing models. The full per-field breakdown, the checkpoint sweep and the caveats are in
`docs/RESULTS.md`. The raw evaluation arrays are in `docs/eval_release/`.

## 6. Reproducibility

- **Config to checkpoint.** `config.toml` is the recipe that produced the released checkpoint.
  The only differences are the data paths, `use_wandb`, the W&B names and `ckpt_every` (the
  original run saved every epoch). See `checkpoints/PROVENANCE.md`.
- **Code.** This is the case code from the parent repository. The only changes are the import
  bootstrap and the path resolution through `[data] data_root`; no hardcoded paths remain.
  Verified: on the same checkpoint and test cases, `eval_fields.py` here and in the parent
  repository write **bit-identical** metric arrays.
- **Seeding.** `seed = 0` fixes the torch and numpy seeds. With `num_workers = 0`, two short
  runs from this folder gave identical losses. CUDA reduction kernels are still not guaranteed to
  be bit-deterministic over a full 400-epoch run.
- **Environment.** `environment.yml` pins the versions the checkpoint was trained with
  (torch 2.14.1+cu130, numpy 2.5.4, Python 3.12).

## License

MIT, see `LICENSE`. The AirFRANS dataset has its own license and citation terms; see
<https://airfrans.readthedocs.io>.
