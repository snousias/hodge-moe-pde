# scripts/generate_dataset/

Builds `dataset/processed_augmented/` (the model's actual input) from `dataset/original/` (the
raw AirFRANS release). Two stages, run in order:

```bash
pip install torch-geometric tqdm gmsh shapely   # optional deps, only for this pipeline — see ../../environment.yml
python 01_convert_raw_to_graphs.py
python 02_build_processed_augmented.py
```

## What each stage does

**`01_convert_raw_to_graphs.py`** reads `dataset/original/` (raw AirFRANS: per-simulation
`.vtp`/`.vtu` files + a top-level `manifest.json`), and for each simulation writes one
intermediate `.pth` file — a `torch_geometric.data.Data` object holding the raw solution in
physical units (`pos`, `x` = inlet velocity + SDF + wall normals, `y` = velocity + pressure +
ν_t). Default output: `dataset/_build/graph_airfrans_data_official/<name>.pth`. Also emits a
regenerated split JSON as a reproducibility check — this should exactly match the bundled
`../../manifest/airfrans_official_split.json`, not replace it.

**`02_build_processed_augmented.py`** reads those `.pth` files and, for each one, builds `--k`
(default 8) independently-randomised anisotropic-boundary-layer meshes via `v16_mesh.py`
(gmsh's `BoundaryLayer` field + a distance-graded far-field size, ~4.9k vertices, airfoil carved
out) — each variant gets its own jittered mesh-sizing parameters (`hfar`, `n_poly`), so the K
meshes for one simulation have genuinely different node positions, not just different labels.
The raw simulation fields are then interpolated onto each new mesh (`interpolate_fields.py`:
linear + nearest-neighbour fallback) and saved as `dataset/processed_augmented/<name>__s{k}.npz`
with keys `pos, faces, x_feat, y, wall, method, bl_thick` — exactly what `data/dataset.py` reads.

Both stages are resumable (existing output files are skipped) and both accept `--start`/`--end`
(stage 2) or `--limit` (stage 1) for sharding a run across machines or smoke-testing on a handful
of cases first. Run either with `--help` for the full option list.

## Provenance

This is a reimplementation, not a verbatim copy, of the pipeline that actually produced the
bundled checkpoint's training data (`cases/v18_3/gen_v18_3_dataset.py` and
`cases/_shared/{v16_mesh,v13_mesh,convert_official_to_graphs}.py` in the parent research repo) —
adapted here for portable paths and a proper CLI instead of hardcoded absolute paths and
positional `sys.argv` parsing. `v16_mesh.py` is vendored verbatim (its meshing logic, unchanged);
`interpolate_fields.py` is `v13_mesh.py` trimmed to the one function this pipeline actually uses,
dropping a dependency (`build_v13_mesh`, an earlier isotropic remesher `v16_mesh.py` superseded,
and its own `_order_airfoil_loop` import) this pipeline never calls.

**Verified**, not just adapted: run end-to-end on the sample cases bundled in
`../../dataset/original/`, producing `.npz` files with the same keys, dtypes, and vertex-count
range as the real bundled `processed_augmented/` files.

## Third-party licenses

`gmsh` is GPL-licensed; `shapely` is BSD. Both are optional runtime dependencies of this one
pipeline (not vendored source, not linked into the deployed model), invoked as ordinary Python
packages — same footing as this deployment's other pip dependencies.
