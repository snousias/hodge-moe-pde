"""
Single-source-of-truth data path resolution for this deployment.

The config file (`config.toml`) carries exactly ONE user-facing path:

    [data]
    data_root = "dataset"          # relative -> resolved against THIS FOLDER, not the CWD

Everything the training/eval code needs — the mesh pool, the geometry cache, the raw
AirFRANS simulation folder (only used by eval_lift.py's library cross-check) — is a fixed
subpath under that one root. `resolve_data_paths` mutates the loaded cfg dict in place so the
rest of the code (unchanged from the original case) keeps reading `cfg["dataset"]["data_root"]`,
`cfg["dataset"]["train"]["kw"]["cache_dir"]`, etc. exactly as before.

PORTABILITY. `data_root` and `cache_dir` may each be given as a RELATIVE path (the default,
`dataset` and `dataset/cached/{train,test}_geo`) or an ABSOLUTE one (`/mnt/big-disk/airfrans`,
`~/data/airfrans`). A relative path is resolved against `deploy_root` — the deployment root, where
config.toml lives — NEVER against the process's current working directory. This is what makes
`cp -r HodgePDEMoE /anywhere/else` just work with zero edits when you use the bundled `dataset/`
folder: nothing here hardcodes this machine's absolute path. Point `data_root` at an absolute path
instead if your data lives outside the deployment folder (a big shared disk, a separate mount, ...).

Expected layout under data_root (see README.md "Getting the data", dataset/README.md):

    <data_root>/
      processed_augmented/     the actual model input: preprocessed mesh pool,
                                <name>__s{0..7}.npz (train augmentation) + <name>__s0.npz
                                for every official test case
      original/                 raw AirFRANS simulation folders — NOT needed for training or
                                eval_fields.py; only for eval_lift.py's optional library
                                cross-check
      cached/                    geometry cache, derived from processed_augmented/, built
                                automatically on first run if missing (~260 GB for the full
                                800-base train split; ~2h to build on one CPU)
        train_geo/
        test_geo/

The `official_split_json` manifest is bundled with the code (manifest/), not treated as "data".

`cache_dir` is the one exception to "everything derives from data_root": config.toml sets it
EXPLICITLY (to the exact cache the released checkpoint was trained/evaluated against). An explicit
value always wins over the data_root-derived default; delete it from config.toml to fall back to
<data_root>/cached/{train,test}_geo.
"""
import os


def _resolve(path, deploy_root):
    """Absolute (or `~`) paths are used as-is; anything else is resolved against `deploy_root`,
    never the current working directory — see module docstring "PORTABILITY"."""
    p = os.path.expanduser(path)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(deploy_root, p))


def resolve_data_paths(cfg, deploy_root):
    """Mutate `cfg` in place: expand cfg['data']['data_root'] into every concrete path the
    dataset/eval code reads. Call this immediately after `toml.load(cfg_path)`, before the
    config dict is handed to anything else. Returns `cfg` for convenience (`cfg = resolve_data_paths(...)`).
    """
    data_root = _resolve(cfg["data"]["data_root"], deploy_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"data_root does not exist: {data_root!r}. Edit [data] data_root in your config.toml "
            f"to point at the folder that contains processed_augmented/ and cached/ "
            f"(see README.md \"Getting the data\").")
    cfg["data"]["data_root"] = data_root   # write back the resolved absolute form

    mesh_pool_dir = os.path.join(data_root, "processed_augmented")
    cache_train_default = os.path.join(data_root, "cached", "train_geo")
    cache_test_default = os.path.join(data_root, "cached", "test_geo")
    raw_dataset_root = os.path.join(data_root, "original")

    dcfg = cfg["dataset"]
    dcfg["data_root"] = mesh_pool_dir
    dcfg["official_split_json"] = os.path.join(deploy_root, "manifest", "airfrans_official_split.json")

    train_kw = dcfg.setdefault("train", {}).setdefault("kw", {})
    test_kw = dcfg.setdefault("test", {}).setdefault("kw", {})
    # Resolve whatever ended up in cache_dir (explicit relative/absolute value, or our own
    # data_root-derived default) the SAME way as data_root — so a relative cache_dir in
    # config.toml is also copy-anywhere-safe, not just data_root.
    train_kw["cache_dir"] = _resolve(train_kw.get("cache_dir", cache_train_default), deploy_root)
    test_kw["cache_dir"] = _resolve(test_kw.get("cache_dir", cache_test_default), deploy_root)

    # Stashed for eval_lift.py's library cross-check (af.Simulation(root=...)); not read by
    # training or eval_fields.py, which never touch the raw AirFRANS original/ folder.
    cfg["data"]["_raw_dataset_root"] = raw_dataset_root
    cfg["data"]["_resolved"] = True
    return cfg
