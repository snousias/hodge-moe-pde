#!/usr/bin/env python
"""
STAGE 2 of 2 — build the mesh POOL (dataset/processed_augmented/): K randomised gmsh
anisotropic-boundary-layer meshes per AirFRANS simulation (~4.9k vertices each), fields
interpolated onto each. This IS mesh-level data augmentation: at train time the dataset randomly
picks one of the K meshes per sample each epoch, so the same node positions are never traversed
twice (`pool_K` in config.toml). Test-time evaluation always uses the __s0 mesh (deterministic).

Each mesh = build_v16_mesh (v16_mesh.py, vendored alongside this script) with per-(name, k)
jittered hfar (+-8%) and n_poly (+-30) — different gmsh sizing parameters, hence different node
positions — fields interpolated from the raw simulation via interpolate_fields.py. Per-mesh cost
is a few seconds; generating the full 800-base training set (K=8) takes on the order of hours.

Output: <out>/<name>__s{k}.npz with keys (pos, faces, x_feat, y, wall, method, bl_thick) — the
exact format data/dataset.py expects. Resumable: existing files are skipped, so an interrupted
run (or a --start/--end sharded run across machines) just continues.

Usage (defaults assume stage 1 already ran with its own defaults):
  python 02_build_processed_augmented.py                        # all sims, K=8
  python 02_build_processed_augmented.py --k 8 --start 0 --end 100   # shard across machines
  python 02_build_processed_augmented.py --rawdir /mnt/data/_build/graph_airfrans_data_official \\
                                          --out    /mnt/data/processed_augmented
"""
import os, sys, time, zlib, argparse, traceback
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from v16_mesh import build_v16_mesh
from interpolate_fields import interpolate_fields
from mesh_opformer.dataset.dataset_airfrans import load_airfrans_graph

K_DEFAULT = 8   # matches config.toml's [dataset] pool_K
# Matches stage 1's default --out: <out>/graph_airfrans_data_official/<name>.pth
RAWDIR_DEFAULT = os.path.join(_DEPLOY_ROOT, "dataset", "_build", "graph_airfrans_data_official")
OUT_DEFAULT = os.path.join(_DEPLOY_ROOT, "dataset", "processed_augmented")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rawdir", default=RAWDIR_DEFAULT, help="stage 1's output: <rawdir>/<name>.pth")
    ap.add_argument("--out", default=OUT_DEFAULT, help="writes <out>/<name>__s{k}.npz")
    ap.add_argument("--k", type=int, default=K_DEFAULT, help="augmentation variants per simulation")
    ap.add_argument("--start", type=int, default=0, help="shard: first sim index (sorted by name)")
    ap.add_argument("--end", type=int, default=0, help="shard: last sim index, exclusive (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not os.path.isdir(args.rawdir):
        raise FileNotFoundError(
            f"{args.rawdir} not found — run 01_convert_raw_to_graphs.py first (or pass --rawdir).")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(args.rawdir) if f.endswith(".pth"))
    end = args.end or len(names)
    names = names[args.start:end]
    print(f"mesh pool gen: {len(names)} samples x K={args.k} [{args.start}:{end}] -> {args.out}", flush=True)

    t0 = time.time(); ok = 0; skip = 0; fail = 0
    for i, name in enumerate(names):
        pos = x = None
        for k in range(args.k):
            outp = os.path.join(args.out, f"{name}__s{k}.npz")
            if os.path.isfile(outp):
                skip += 1; continue
            try:
                if pos is None:
                    pos, x, y = load_airfrans_graph(os.path.join(args.rawdir, name + ".pth"))
                rng = np.random.default_rng(zlib.crc32(name.encode()) ^ (k + 1))
                hfar = 0.13 * (1.0 + rng.uniform(-0.08, 0.08))
                n_poly = int(220 + rng.integers(-30, 31))
                V, F, wall_v, info = build_v16_mesh(pos, x[:, 2], hfar=hfar, n_poly=n_poly)
                xf = interpolate_fields(pos, x[:, :5], V)
                yv = interpolate_fields(pos, y[:, :4], V)
                np.savez_compressed(outp, pos=V.astype("float32"), faces=F.astype("int32"),
                                    x_feat=xf.astype("float32"), y=yv.astype("float32"),
                                    wall=wall_v.astype("bool"), method=info["method"],
                                    bl_thick=(np.nan if info["bl_thick"] is None else info["bl_thick"]))
                ok += 1
            except Exception:
                fail += 1
                print(f"  FAIL {name}__s{k}\n{traceback.format_exc()}", flush=True)
        if (i + 1) % 10 == 0:
            el = time.time() - t0; rate = (ok + fail) / el if el > 0 else 0
            print(f"  [{i+1}/{len(names)}] ok={ok} skip={skip} fail={fail}  {rate:.2f} mesh/s  "
                  f"ETA {(len(names)-i-1)*args.k/max(rate,1e-9)/60:.1f} min", flush=True)
    print(f"DONE: ok={ok} skip={skip} fail={fail}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
