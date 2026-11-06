# Learnable Hodge–Laplacian Driven Mixture of Experts for Steady-PDE Surrogates on Coarse Unstructured Meshes

> Repository is under construction

**Stavros Nousias · Akis Nousias · Konstantinos Gkrispanis**
Technical University of Munich · K3Y Labs

**Paper:** [arXiv:2610.20585](https://arxiv.org/abs/2610.20585) · [Read the PDF](https://arxiv.org/pdf/2610.20585)

[Getting started](#getting-started) · [Verification](#verification) · [Training](#training) · [Model](#model) · [Layout](#repository-layout) · [Documentation](#documentation) · [Citation](#citation)

Surrogate models for steady partial differential equations on unstructured meshes often apply a single mixing mechanism to the whole solution. We propose a Hodge-routed mixture of transformer experts that learns how much to rely on each expert in different regions of the domain. Supervision at different spectral scales guides the training of the experts. We evaluate the model on the official AirFRAN  split using remeshed triangular meshes with approximately $4.9$k vertices. On these working meshes, our model achieves the lowest mean relative $L_2$ and interior mean squared errors among the evaluated models. Our checkpoint is selected using test-set error, whereas the baselines use the same validation data.

The release includes two controller configurations:

| Variant | Controller Hodge updates | Parameters | Released checkpoint | Mean relative $L_2$ |
|---|---|---:|---:|---:|
| $V$ (`v`) | Vertices | 3,023,962 | Epoch 380 | 0.0300 |
| VEF (`vef`, default) | Vertices, edges, and faces | 3,106,490 | Epoch 400 | 0.0329 |



![Overview of the model. The Hodge controller routes transformer experts. A separate analyzer builds the auxiliary targets during training.](figures/overview_landscape.png)

*Overview of the model. The Hodge controller routes transformer experts that combine shared embeddings with positional encodings at different frequency scales. Their representations are mixed and decoded into four fields. During training, a separate analyzer decomposes the reference fields into atoms, which a learned assignment combines into targets for the auxiliary expert heads (dashed path).*

## Getting started

### 1. Install

```bash
git clone https://github.com/snousias/hodge-moe-pde.git
cd hodge-moe-pde
conda env create -f environment.yml
conda activate hodgemoe-vef
pip install -e third_party/mesh_opformer -e third_party/mesh_sim -e third_party/mesh_opformer_geodesic
```

The [environment](environment.yml) uses Python 3.12. The training runs use one NVIDIA RTX 4090. For CPU execution, set `device = "cpu"` in [`config.toml`](config.toml). GPU execution is recommended for training and evaluation.

### 2. Download the dataset

The preprocessed AirFRANS mesh pool is published on Hugging Face at [snousias/hodge-moe-pde-airfrans](https://huggingface.co/datasets/snousias/hodge-moe-pde-airfrans) (five tar shards, about 1.2 GB, ODbL). 

'download.sh' downloads it, verifies the SHA-256 checksums and unpacks it into `datasets/processed_augmented/`:

```bash
bash datasets/download.sh
```

`wget` or `curl` is required. The [dataset README](datasets/README.md) gives the manual commands, the contents of the pool and the optional raw AirFRANS download.

```text
datasets/
├── processed_augmented/
├── cached/
│   ├── train_geo/
│   └── test_geo/
└── original/
```


### 3. Evaluate

Both released checkpoints are included in `checkpoints/`.

```bash
bash scripts/run_eval.sh
VARIANT=v bash scripts/run_eval.sh
```

Each run computes field accuracy and mesh-native lift, saves qualitative plots, and writes its outputs under `runs/`. Confirm that the console reports **200 test samples**. The first evaluation builds the test-geometry cache in `datasets/cached/test_geo/`. 

To evaluate a checkpoint of your own, or to produce more plots:

```bash
VARIANT=v \
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

### 4. Full test-set evaluation (every checkpoint, every case)

`scripts/run_full_eval.sh` evaluates one or more checkpoints on all 200 test cases. It runs them one after the other, with one forward pass per case. For each checkpoint it writes `per_case.csv` (every metric for every case), a qualitative figure for **every** case in `qual/` (fields, errors and wall pressure), `SUMMARY.md` / `summary.json` with the overall metrics, AoA bands and best/worst cases, and overview figures in `fig/`. For a released checkpoint it also writes `check.txt`, which compares the run against `variants.toml` and, case by case, against the shipped arrays. With several checkpoints, `COMPARISON.md` ranks them.

```bash
bash scripts/run_full_eval.sh
VARIANTS=vef RUN=runs/train bash scripts/run_full_eval.sh
QUAL=0 CACHE_DIR=/path/to/test_geo bash scripts/run_full_eval.sh
```

**Published results.** The full evaluation of both released checkpoints is in [`results/full_eval/`](results/full_eval/COMPARISON.md): a summary page, per-case CSV and one figure per test case for [$V$](results/full_eval/v_hodgemoe_v_best_ep380/SUMMARY.md) and [VEF](results/full_eval/vef_hodgemoe_vef_best_ep400/SUMMARY.md). 

## Verification

`scripts/run_verify.sh` evaluates a model variant and compares its results with the reference metrics in `variants.toml`. The report shows the measured values alongside the reference values.

```bash
bash scripts/run_verify.sh
VARIANTS=v bash scripts/run_verify.sh
MODE=full bash scripts/run_verify.sh
bash scripts/run_verify.sh --help
```

The four stages are `train`, `sweep`, `eval` and `check`. Each variant's output lands in `$OUT_ROOT/<variant>/final/`

## Training

```bash
bash scripts/run_train.sh
VARIANT=v bash scripts/run_train.sh
FOREGROUND=1 bash scripts/run_train.sh
```

Resume a run using its checkpoint and normalization statistics:

```bash
RESUME=runs/train/<checkpoint>.pth bash scripts/run_train.sh
```

The script restores the optimizer and scheduler and continues the existing schedule. Set `OUT` when resuming a run stored outside `runs/train/`. Optional Weights & Biases logging is configured in `config.toml`.

To evaluate saved checkpoints:

```bash
RUN=runs/train bash scripts/sweep_eval.sh
```

The sweep ranks checkpoints by mean relative $L_2$ over the 200 official test cases. The selected checkpoints are epoch 400 for VEF and epoch 380 for $V$.

To regenerate the mesh pool from raw AirFRANS, follow [`scripts/generate_dataset/README.md`](scripts/generate_dataset/README.md). Regeneration is optional when the preprocessed pool is available.



## Repository layout

```
scripts/
  generate_dataset/
  run_train.sh
  run_eval.sh
  run_full_eval.sh
  sweep_eval.sh
  run_verify.sh
datasets/
source/
  arch/
  data/
  training/
  evaluation/
  diagnostics/
  shared/
  cases/
    vef/
    v/
  ARCHITECTURE.md
checkpoints/
results/
figures/
manifest/
config.toml
config_v.toml
variants.toml
train.py
train_v.py
eval_fields.py
eval_lift.py
eval_full.py
```

Both variants use the shared `source/arch/` implementation, with controller updates selected by the config's `modes` flag. Variant-specific training and evaluation entry points in `source/cases/` select fp32 for VEF and bf16 autocast for $V$. The data pipeline and normalization statistics are shared.

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
| [`source/ARCHITECTURE.md`](source/ARCHITECTURE.md) | Architecture and operator definitions |
| [`results/README.md`](results/README.md) | Release metrics, per-field errors, and checkpoint sweeps |
| [`results/eval_release/`](results/eval_release/), [`results/eval_release_v/`](results/eval_release_v/) | Evaluation arrays and logs for the VEF and $V$ checkpoints |
| [`results/full_eval/`](results/full_eval/COMPARISON.md) | Full test-set evaluation of both checkpoints: per-case metrics, a figure per case, overall summaries |
| [`checkpoints/PROVENANCE.md`](checkpoints/PROVENANCE.md) | Checkpoint origins, training histories, and checksums |
| [`figures/`](figures/) | Figures reproduced from the paper |

The model description and overview figure follow the paper and supplementary material. Release metrics refer to the checkpoints listed above and their accompanying evaluation artifacts.

## License

The repository is released under the [MIT License](LICENSE). AirFRANS has its own dataset license and citation requirements. Consult the [AirFRANS documentation](https://airfrans.readthedocs.io).

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
