# HodgeMoE

### Learnable Hodge–Laplacian Driven Mixture of Experts for Steady-PDE Surrogates on Coarse Unstructured Meshes

**Stavros Nousias · Akis Nousias · Konstantinos Gkrispanis**<br>
Technical University of Munich · K3Y Labs

This repository contains the HodgeMoE implementation for steady-state PDE regression on unstructured meshes.

[Paper](https://arxiv.org/abs/2610.20585) · [PDF](https://arxiv.org/pdf/2610.20585) · [Dataset](https://huggingface.co/datasets/snousias/hodge-moe-pde-airfrans) · [Results](results/full_eval/COMPARISON.md)

![Architecture overview](figures/overview_landscape.png)

[Modules](#modules) · [Experiments](#experiments) · [Pretrained models](#pretrained-models) · [Configuration](#configuration) · [Execution Examples](#execution-examples) · [Documentation](#documentation) · [Citation](#citation)

## Modules

The implementation depends on three Python packages:

- [`mesh_sim`](third_party/mesh_sim/): mesh loading and geometric feature extraction.

- [`mesh_opformer`](third_party/mesh_opformer/): HodgeFormer layers, dataset definitions, and training and evaluation utilities.

- [`mesh_opformer_geodesic`](third_party/mesh_opformer_geodesic/): geodesic-ring neighbourhood extraction and sparse attention supports.

Each package uses a `src` directory layout. Install all three packages in a Python environment using editable installations or wheel files.

[`source/`](source/) contains the HodgeMoE architecture, dataset definitions, and training and evaluation modules. [`scripts/`](scripts/) contains the execution scripts. The configuration files [`config.toml`](config.toml) and [`config_v.toml`](config_v.toml) are in the repository root.

The reported experiments used Python 3.12, PyTorch 2.14.1, and one NVIDIA RTX 4090 GPU.

Clone the repository and create the environment defined in `environment.yml`:

```bash
git clone https://github.com/snousias/hodge-moe-pde.git
cd hodge-moe-pde
conda env create -f environment.yml
conda activate hodgemoe-vef
```

### Installation in development mode

From the repository root, install each package in editable mode:

```bash
cd ./third_party/mesh_sim
pip install -e .

cd ../mesh_opformer
pip install -e .

cd ../mesh_opformer_geodesic
pip install -e .

cd ../..
```

### Installation as wheel packages

Alternatively, build a wheel for each package. Run these commands from the repository root:

```bash
pip install wheel
cd ./third_party/mesh_sim
python setup.py bdist_wheel

cd ../mesh_opformer
python setup.py bdist_wheel

cd ../mesh_opformer_geodesic
python setup.py bdist_wheel

cd ../..
```

Install the generated wheel files:

```bash
pip install third_party/mesh_sim/dist/*.whl
pip install third_party/mesh_opformer/dist/*.whl
pip install third_party/mesh_opformer_geodesic/dist/*.whl
```

## Experiments

The experiments use the AirFRANS dataset for steady-state PDE regression on remeshed triangular domains. The repository includes two model configurations:

- HodgeMoE–V — vertex controller updates: [`config_v.toml`](config_v.toml), [`train_v.py`](train_v.py).
- HodgeMoE–VEF — vertex, edge and face controller updates: [`config.toml`](config.toml), [`train.py`](train.py).

### Pretrained models

[`checkpoints/`](checkpoints/) contains pretrained checkpoints and normalization statistics for both configurations.

| Model | Hodge controller updates | Parameters | Mean relative $L_2$ ↓ | Checkpoint |
|---|---|---:|---:|---|
| HodgeMoE–V | Vertices | 3.02M | 0.0300 | [Epoch 380](checkpoints/hodgemoe_v_best_ep380.pth) |
| HodgeMoE–VEF | Vertices, edges, and faces | 3.11M | 0.0329 | [Epoch 400](checkpoints/hodgemoe_vef_best_ep400.pth) |

The results are computed on the 200 official AirFRANS test cases using remeshed triangular domains with approximately 4,900 vertices and interpolated reference fields. The reported error is averaged across the four predicted fields.

[`checkpoints/PROVENANCE.md`](checkpoints/PROVENANCE.md) documents the checkpoints and training histories.

### Configuration

The TOML configuration files specify dataset paths, preprocessing, model architecture, training, and evaluation parameters. Comments in [`config.toml`](config.toml) and [`config_v.toml`](config_v.toml) describe the settings. To enable Weights & Biases logging, set `use_wandb = true` at the top level and configure the `[wandb]` and `[wandb.init]` sections:

```toml
use_wandb = true

[wandb]
WANDB_MODE = "online"     # Options: 'online', 'offline', 'disabled'
name = "hodgemoe-airfrans"

[wandb.init]
project = "project-name"
entity = "user-wandb-account"
```

### Download the data

```bash
bash datasets/download.sh
```

The script downloads approximately 1.2 GB of preprocessed mesh data. It requires `wget` or `curl`.

Geometry data is cached on first use. The test and training caches require approximately 8 GB and 251 GB, respectively. Initial geometry preparation takes approximately one minute per mesh. The [dataset documentation](datasets/README.md) describes custom paths and manual setup.

### Execution Examples

Run the following training and evaluation commands from the repository root.

#### Inference

Evaluate HodgeMoE–VEF:

```bash
bash scripts/run_eval.sh
```

Evaluate HodgeMoE–V:

```bash
VARIANT=v bash scripts/run_eval.sh
```

Each run computes field error and lift metrics, saves field plots, and writes results to `runs/`. Evaluation with an existing geometry cache takes approximately two minutes per model on an RTX 4090.

#### Training

Train HodgeMoE–VEF with the default configuration:

```bash
bash scripts/run_train.sh
```

For HodgeMoE–V, set `VARIANT=v`. Set `FOREGROUND=1` to run training in the current terminal:

```bash
VARIANT=v FOREGROUND=1 bash scripts/run_train.sh
```

The training configuration uses AdamW, a maximum learning rate of $10^{-3}$, weight decay of $10^{-4}$, and a 400-epoch OneCycle schedule.

Each batch contains one complete mesh (`batch_size = 1`).

Configure data paths, model settings, losses, and logging in [`config.toml`](config.toml) for VEF or [`config_v.toml`](config_v.toml) for V. Training outputs are saved under `runs/train/` and `runs/train_v/`, respectively.

<details>
<summary>Resume training and select a checkpoint</summary>

Resume a VEF run with its saved optimizer, scheduler, and normalization statistics:

```bash
RESUME=runs/train/<checkpoint>.pth bash scripts/run_train.sh
```

For a run in a custom directory, set `OUT` to that directory. For V, also set `VARIANT=v`.

Rank saved checkpoints by mean relative $L_2$ over the 200 official test cases:

```bash
RUN=runs/train bash scripts/sweep_eval.sh
```

The selected release checkpoints are epoch 400 for VEF and epoch 380 for V.

</details>

### Evaluation results

The [model comparison](results/full_eval/COMPARISON.md) summarizes the results for both models. Separate reports for [HodgeMoE–V](results/full_eval/v_hodgemoe_v_best_ep380/SUMMARY.md) and [HodgeMoE–VEF](results/full_eval/vef_hodgemoe_vef_best_ep400/SUMMARY.md) contain aggregate metrics, per-case metrics, and plots of predicted fields, errors, and wall pressure.

To generate complete reports for both pretrained models:

```bash
bash scripts/run_full_eval.sh
```

Results are saved in `runs/full_eval/`, including a comparison report, a CSV of per-case metrics, and a figure for every test case.

<details>
<summary>More evaluation options</summary>

Evaluate all checkpoints from a training run:

```bash
VARIANTS=vef RUN=runs/train bash scripts/run_full_eval.sh
```

Generate metrics without per-case figures:

```bash
QUAL=0 bash scripts/run_full_eval.sh
```

Evaluate a custom checkpoint:

```bash
VARIANT=vef \
CKPT=runs/train/<checkpoint>.pth \
STATS=runs/train/airfrans_norm_stats.npz \
OUT=runs/eval_custom bash scripts/run_eval.sh
```

Compare evaluation scores with the supplied reference metrics:

```bash
bash scripts/run_verify.sh
VARIANTS=v bash scripts/run_verify.sh
```

</details>

## Documentation

| Resource | Contents |
|---|---|
| [Dataset documentation](datasets/README.md) | Downloads, data paths, and dataset contents |
| [Model architecture](source/ARCHITECTURE.md) | Controller, experts, analyzer, and operator definitions |
| [Model configurations](variants.toml) | Configurations, checkpoints, and reference metrics |
| [Release results](results/README.md) | Field errors and checkpoint sweeps |
| [Complete evaluation](results/full_eval/COMPARISON.md) | Model comparison and per-case visualizations |
| [Checkpoint details](checkpoints/PROVENANCE.md) | Training histories and checkpoint checksums |
| [Dataset generation](scripts/generate_dataset/README.md) | Building the mesh pool from raw AirFRANS data |

[`source/`](source/) contains the architecture, data processing, training, and evaluation modules. [`scripts/`](scripts/) contains the execution scripts.

## License

The code is distributed under the [MIT License](LICENSE). The [dataset documentation](datasets/README.md) and [AirFRANS documentation](https://airfrans.readthedocs.io) specify dataset licensing and citation requirements.

## Citation

Cite the following paper when using this work:

Nousias, S., Nousias, A. and Gkrispanis, K., 2026. Learnable Hodge-Laplacian Driven Mixture of Experts for Steady-PDE Surrogates on Coarse Unstructured Meshes. arXiv preprint arXiv:2610.20585.

BibTeX entry:

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
