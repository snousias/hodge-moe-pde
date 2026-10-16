#!/usr/bin/env python
"""
STAGE 1 of 2 — convert the raw AirFRANS release into the intermediate .pth graph format the
mesh-pool generator (02_build_processed_augmented.py) reads.

Each output is a torch_geometric Data object, in PHYSICAL units, named by simulation:
  pos : (N, 2)  node positions [m]
  x   : (N, 5)  [inlet_vx, inlet_vy, sdf, normal_x, normal_y]  (physical)
  y   : (N, 4)  [u, v, p/rho, nu_t]                            (physical)

Also emits the official train/test split JSON, keyed by the converted filenames — this should
match dataset's bundled manifest/airfrans_official_split.json exactly (both are built from the
same public manifest.json); regenerating it is only useful as a reproducibility check, not a
prerequisite (the bundled one is already correct).

Requires `dataset/original/` to be populated (see dataset/README.md) and `manifest.json` in it
(ships with the standard `airfrans.dataset.download(...)` — see the main README's "Getting the
data"). Resumable: existing .pth files are skipped.

Usage (defaults assume you're running from this script's own directory):
  python 01_convert_raw_to_graphs.py
  python 01_convert_raw_to_graphs.py --afroot /mnt/data/airfrans/original --out /mnt/data/airfrans/_build
"""
import os, sys, json, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_ROOT = os.path.dirname(os.path.dirname(_HERE))


def build_data(sim):
    import numpy as np
    import torch
    from torch_geometric.data import Data
    pos = sim.position.astype("float32")                                                        # (N,2)
    x = np.concatenate([sim.input_velocity, sim.sdf, sim.normals], axis=-1).astype("float32")    # (N,5)
    y = np.concatenate([sim.velocity, sim.pressure, sim.nu_t], axis=-1).astype("float32")         # (N,4)
    return Data(pos=torch.from_numpy(pos), x=torch.from_numpy(x), y=torch.from_numpy(y))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--afroot", default=os.path.join(_DEPLOY_ROOT, "dataset", "original"),
                    help="raw AirFRANS root (must contain manifest.json + one folder per simulation)")
    ap.add_argument("--out", default=os.path.join(_DEPLOY_ROOT, "dataset", "_build"),
                    help="writes <out>/graph_airfrans_data_official/<name>.pth")
    ap.add_argument("--split", default=os.path.join(_DEPLOY_ROOT, "dataset", "_build", "airfrans_official_split.json"),
                    help="where to write the regenerated split JSON (does NOT touch the bundled "
                         "manifest/airfrans_official_split.json)")
    ap.add_argument("--task", default="full")
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N sims (smoke test)")
    args = ap.parse_args()

    import numpy as np
    import airfrans as af
    from tqdm import tqdm

    outdir = os.path.join(args.out, "graph_airfrans_data_official")
    os.makedirs(outdir, exist_ok=True)

    manifest_path = os.path.join(args.afroot, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found. --afroot must point at a populated AirFRANS release "
            f"(see dataset/README.md \"Populating each folder\" -> original/).")
    with open(manifest_path) as f:
        manifest = json.load(f)
    train_names = manifest[f"{args.task}_train"]
    test_names = manifest[f"{args.task}_test"]
    all_names = train_names + test_names
    if args.limit:
        all_names = all_names[:args.limit]

    import torch
    bad = []
    for name in tqdm(all_names, desc="Converting sims"):
        out_path = os.path.join(outdir, name + ".pth")
        if os.path.exists(out_path):
            continue
        try:
            sim = af.Simulation(root=args.afroot, name=name)
            ns = int(sim.surface.sum())
            sdf_surf = float(np.abs(sim.sdf[sim.surface]).max()) if ns else 0.0
            data = build_data(sim)
            torch.save(data, out_path)
            if sdf_surf > 1e-6:
                bad.append((name, "sdf_surf", sdf_surf))
        except Exception as e:
            bad.append((name, "error", str(e)))

    def base(n): return n + ".pth"
    split = {
        "task": args.task,
        "afroot": os.path.abspath(args.afroot),
        "train": [base(n) for n in train_names],
        "test": [base(n) for n in test_names],
        "graph_to_name": {base(n): n for n in all_names},
        "name_to_graph": {n: base(n) for n in all_names},
    }
    os.makedirs(os.path.dirname(args.split), exist_ok=True)
    with open(args.split, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\nConverted {len(all_names)} sims -> {outdir}")
    print(f"Split: {len(split['train'])} train / {len(split['test'])} test -> {args.split}")
    if bad:
        print(f"  !! {len(bad)} anomalies (e.g. {bad[:3]})")


if __name__ == "__main__":
    main()
