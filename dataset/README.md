# dataset/

This is where the AirFRANS data lives by default. `[data] data_root = "dataset"` in
`../config.toml` is resolved relative to the deployment folder (see `shared/paths.py`).

```
dataset/
  processed_augmented/     the model input: the preprocessed mesh pool (download separately)
  original/                raw AirFRANS simulations (public dataset), optional, see below
  cached/
    train_geo/             geometry cache, built automatically on first use if missing
    test_geo/
```

## Download the preprocessed dataset

The mesh pool is hosted on Hugging Face:
<https://huggingface.co/datasets/snousias/hodge-moe-pde-airfrans> (ODbL, derived from AirFRANS).
It is five tar shards (1.2 GB). To download, verify and unpack them in one step, run:

```bash
bash dataset/download.sh
```

The script runs from any directory. Equivalent manual steps, from the repository root:

```bash
cd dataset
base=https://huggingface.co/datasets/snousias/hodge-moe-pde-airfrans/resolve/main
wget -q "$base/SHA256SUMS"
for f in train_0 train_1 train_2 train_3 test; do wget -q "$base/$f.tar"; done
sha256sum -c SHA256SUMS
for f in train_0 train_1 train_2 train_3 test; do tar -xf "$f.tar" && rm "$f.tar"; done
```

This creates `dataset/processed_augmented/` with 6,600 `.npz` mesh files plus
`_global_target_std.npy` (6,601 files, about 1.2 GB).

The preprocessed pool is required for training and evaluation. The geometry cache is
built automatically; raw AirFRANS simulations are optional. Downloaded data, caches,
and preprocessing outputs are covered by [`.gitignore`](.gitignore).

## `processed_augmented/`: contents (6,601 files, 1.2 GB)

The preprocessed dataset has every pool file that training and evaluation read for the full official
800/200 split:

| part | files | which |
|---|--:|---|
| 800 train cases | 6,400 | 8 meshes per case: the first 8 of `sorted(glob("<name>__s*.npz"))` |
| 200 test cases | 200 | `<name>__s0.npz` only (evaluation always uses s0) |
| `_global_target_std.npy` | 1 | cached AirFRANS `coef_norm` std over the train s0 meshes (rebuilt if deleted) |

**Which 8 training meshes.** The source pool has 16 remeshed variants (`__s0`…`__s15`) per
training case. `data/dataset.py` keeps `files[:pool_K]` of a *lexicographically* sorted glob,
so with `pool_K = 8` it picks **s0, s1, s10, s11, s12, s13, s14, s15**, not s0–s7. The released pool
contains exactly those files. The released checkpoint was trained on them, and the code reads
nothing else. Training draws one of the 8 at random each time a case is loaded (mesh
augmentation).

Each `.npz` is one remeshed AirFRANS simulation: an anisotropic boundary-layer triangulation
with about 4.9k vertices, with keys `pos`, `faces`, `x_feat`, `y`, `wall`, ... The source is
the parent project's `graph_airfrans_data_v18_3` pool.

## `cached/`: geometry cache, rebuilt automatically

For each mesh, `data/dataset.py` caches the DEC features, incidence matrices `d_0`/`d_1`, the
BFS-ring neighbour supports, the C_L weights and the fixed target bands as
`<name>.geo.npz`, `.hop_ff1.npz`, `.clw.npz` and `.bands2-8-32.npz`. A missing or corrupt cache file triggers a
rebuild, never a crash. The cache is large, about 251 GB for the training pool and 20 GB for
the test set, and it is fully derived from `processed_augmented/` and `config.toml`. Expect
about a minute per mesh with the Python geometry path on the first epoch or evaluation. Every
later run reads it from disk.

To skip the rebuild, copy or symlink an existing cache built with the same pool and window
settings into `cached/train_geo/` and `cached/test_geo/`, or point `cache_dir` in
`config.toml` at it. The files are keyed by mesh name and safe to copy between machines.

## `original/`: raw AirFRANS (optional)

Only `manifest.json` (AirFRANS's own case list) is bundled. The raw `.vtu`/`.vtp` simulations
(about 15 GB) are needed only for:

- `eval_lift.py --validate N`, an optional cross-check against the AirFRANS library's own force
  coefficients (logs NaN for missing cases, never crashes);
- `scripts/generate_dataset/`, which rebuilds `processed_augmented/` from scratch.

They are the public AirFRANS release:

```python
import airfrans
airfrans.dataset.download(root="dataset", file_name="original", unzip=True)
```

(project home: <https://airfrans.readthedocs.io>; dataset paper: Bonnet et al., NeurIPS 2022.)

The train/test split this code reads is `../manifest/airfrans_official_split.json`, the
official AirFRANS `full` split (800/200). It is a different file from `original/manifest.json`.

## Data stored elsewhere

Set `[data] data_root` in `../config.toml` to an absolute path. That folder must hold the same
`processed_augmented/` (and optionally `original/`, `cached/`) layout.
