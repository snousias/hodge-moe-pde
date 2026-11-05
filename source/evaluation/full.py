# ========================================================================================
# EVALUATION ENTRY POINT
#   main()  — full test-set evaluation of one or more checkpoints:
#             per-case quantitative results, qualitative figures for every case, and the
#             overall metrics, in one forward pass per case.
# ========================================================================================
# __deployment_path_bootstrap__
import os as _o, sys as _s
_ROOT = _o.path.dirname(_o.path.abspath(__file__))
while _ROOT != _o.path.dirname(_ROOT) and not _o.path.isfile(
        _o.path.join(_ROOT, "variants.toml")):
    _ROOT = _o.path.dirname(_ROOT)                      # deployment root (holds variants.toml)
for _d in (_o.path.join(_ROOT, "source", "shared"), _o.path.join(_ROOT, "source"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)
#!/usr/bin/env python
"""
Full test-set evaluation: every checkpoint × every official test case.

For each checkpoint this writes, under <out>/<checkpoint-name>/:

    per_case.csv          one row per test case: flow condition, mesh size, per-field rel-L2 /
                          R² / MSE / MAE / RMSE, AirFRANS Volume / Surf MSE per field, C_L true /
                          predicted / error
    test_metrics.npz      same layout as evaluation/fields.py  } so source/shared/check_metrics.py
    cl_native.npz         same layout as evaluation/lift.py    } reads this folder unchanged
    summary.json          overall metrics, per-field means, AoA bands, best / worst cases
    SUMMARY.md            the same, as tables
    fig/                  overall figures: per-case error histogram, error vs AoA, C_L scatter
    qual/<case>.png       qualitative figure for EVERY test case (fields, errors, wall pressure)
    pred/<case>.npz       physical prediction + target per case (--save_pred)
    check.txt             comparison with the [<variant>.expect] block of variants.toml, when the
                          checkpoint is a registered variant's checkpoint

With more than one checkpoint, <out>/COMPARISON.md and <out>/comparison.csv rank them.

The metric definitions are imported from the released evaluators, not re-implemented:
per_channel_metrics (evaluation/fields.py), vol_surf_mse / global_target_std / spearman
(shared/airfrans_native.py) and mesh_cl (evaluation/lift.py). The numbers therefore match
scripts/run_eval.sh exactly; the difference is that this script runs ONE forward pass per case
for both, keeps every per-case value, and draws every case.

Precision follows the variant: `v` is evaluated under bf16 autocast (how its published numbers
were measured), `vef` in fp32. Override with --precision.

    python eval_full.py --variant v --variant vef --out runs/full_eval
    python eval_full.py --variant vef --ckpt runs/train/*_ckpt_*.pth --out runs/full_eval_sweep
"""
import os, sys, csv, json, glob, time, hashlib, argparse
import numpy as np
import torch
import toml
from torch.utils.data import DataLoader

from data.dataset import AirfransV39Dataset as AirfransDataset, redim_targets, uinf
from mesh_opformer.dataset import collate_batch_pde_fn
from arch.model import build_v57_gated_hodge_vef_model as build_model
from mesh_opformer.training.utils import validate_device
from airfrans_native import vol_surf_mse, global_target_std, spearman
from evaluation.fields import per_channel_metrics, FIELD_KEYS
from evaluation.lift import mesh_cl, order_wall_loop
from paths import resolve_data_paths
from variants import load_registry, resolve_variant, check_pairing_or_exit

FIELD_TEX = [r"$u_x$", r"$u_y$", r"$p$", r"$\nu_t$"]
UNITS = ["m/s", "m/s", "m²/s²", "m²/s"]
QUANT = ["mse", "mae", "rmse", "r2", "l2rel", "l2rel_c"]      # column order of test_metrics.npz
AOA_BANDS = [(-np.inf, 0.0, "AoA < 0°"), (0.0, 5.0, "0–5°"), (5.0, 10.0, "5–10°"),
             (10.0, np.inf, "≥ 10°")]


# ── helpers ───────────────────────────────────────────────────────────────────────────────
def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def flow_condition(name):
    """airFoil2D_SST_<U_inf>_<AoA>_<NACA digits...> -> (U_inf m/s, AoA deg)."""
    parts = name.split("_")
    return float(parts[2]), float(parts[3])


def repo_rel(path, root):
    """Path relative to the repository root when inside it (published outputs carry no local paths)."""
    p = os.path.abspath(path)
    r = os.path.relpath(p, root)
    return p if r.startswith("..") else r.replace(os.sep, "/")


def ckpt_label(path):
    """Short folder name for a checkpoint: file stem without the long run prefix."""
    stem = os.path.splitext(os.path.basename(path))[0]
    for pre in ("airfrans_",):
        if stem.startswith(pre):
            stem = stem[len(pre):]
    return stem


def registered_variant_of(ckpt, registry, root):
    """Name of the variant whose released checkpoint IS this file (same SHA-256), else None."""
    h = None
    for name, v in registry.items():
        ref = v.get("checkpoint")
        if not ref:
            continue
        ref = ref if os.path.isabs(ref) else os.path.join(root, ref)
        if os.path.isfile(ref) and os.path.getsize(ref) == os.path.getsize(ckpt):
            h = h or sha256(ckpt)
            if sha256(ref) == h:
                return name
    return None


# ── qualitative figure (runs in worker processes) ─────────────────────────────────────────
def draw_case(job):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    pred_file, png, title = job
    z = np.load(pred_file)
    pos, faces, wall = z["pos"], z["faces"], z["wall"].astype(bool)
    pred, targ = z["pred"], z["target"]
    tri = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    # view: the airfoil and its near wake, where the fields vary
    xlim, ylim = (-0.4, 1.6), (-0.45, 0.45)
    near = (pos[:, 0] > xlim[0]) & (pos[:, 0] < xlim[1]) & (pos[:, 1] > ylim[0]) & (pos[:, 1] < ylim[1])

    fig = plt.figure(figsize=(17, 17))
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 1.15], hspace=0.32, wspace=0.12)
    for c in range(4):
        gt, pr = targ[:, c], pred[:, c]
        err = np.abs(pr - gt)
        lo, hi = np.percentile(gt[near], [1, 99])
        emax = max(np.percentile(err[near], 99), 1e-12)
        rel = np.linalg.norm(pr - gt) / (np.linalg.norm(gt) + 1e-8)
        for j, (val, ttl, cmap, vmin, vmax) in enumerate([
                (gt, "reference", "viridis", lo, hi),
                (pr, "prediction", "viridis", lo, hi),
                (err, f"|error|   (case rel-L2 {rel:.4f})", "magma", 0.0, emax)]):
            ax = fig.add_subplot(gs[c, j])
            tpc = ax.tripcolor(tri, val, cmap=cmap, shading="gouraud", vmin=vmin, vmax=vmax)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{FIELD_TEX[c]} {ttl}", fontsize=11)
            cb = fig.colorbar(tpc, ax=ax, shrink=0.8, pad=0.01)
            cb.ax.tick_params(labelsize=8)
            if j == 0:
                ax.set_ylabel(UNITS[c], fontsize=9)

    # wall pressure along the airfoil, both sides, reference vs prediction
    widx = np.where(wall)[0]
    order = order_wall_loop(pos[widx])
    w = widx[order]
    x = pos[w, 0]
    i_te = int(np.argmax(x))
    sides = [(slice(0, i_te + 1), "side 1"), (slice(i_te + 1, None), "side 2")]
    ax = fig.add_subplot(gs[4, 0:2])
    for (sl, lab), col in zip(sides, ["#1f77b4", "#d62728"]):
        ax.plot(x[sl], targ[w, 2][sl], "-", color=col, lw=2, label=f"reference, {lab}")
        ax.plot(x[sl], pred[w, 2][sl], "--", color=col, lw=1.5, label=f"prediction, {lab}")
    ax.invert_yaxis()
    ax.set_xlabel("x (chord)"); ax.set_ylabel("wall p  (m²/s², axis inverted)")
    ax.set_title("pressure on the airfoil wall", fontsize=11)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[4, 2])
    ax.plot(x, np.abs(pred[w, 2] - targ[w, 2]), ".", ms=3, color="#444444")
    ax.set_xlabel("x (chord)"); ax.set_ylabel("|p error|  (m²/s²)")
    ax.set_title("wall pressure error", fontsize=11); ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.savefig(png, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return png


# ── overall figures ───────────────────────────────────────────────────────────────────────
def overall_figures(rows, cl_true, cl_pred, fig_dir, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    rel = np.array([r["rel_l2_mean"] for r in rows])
    aoa = np.array([r["aoa_deg"] for r in rows])
    per_field = np.array([[r[f"{k}_l2rel"] for k in FIELD_KEYS] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    ax = axes[0]
    ax.hist(rel, bins=30, color="#4c72b0", edgecolor="white")
    ax.axvline(rel.mean(), color="k", ls="--", lw=1, label=f"mean {rel.mean():.4f}")
    ax.axvline(np.median(rel), color="k", ls=":", lw=1, label=f"median {np.median(rel):.4f}")
    ax.set_xlabel("per-case rel-L2 (mean of 4 fields)"); ax.set_ylabel("test cases")
    ax.legend(); ax.set_title("distribution over the test set")
    ax = axes[1]
    ax.scatter(aoa, rel, s=12, color="#4c72b0")
    ax.set_xlabel("angle of attack (°)"); ax.set_ylabel("per-case rel-L2")
    ax.set_title("error vs angle of attack"); ax.grid(alpha=0.3)
    ax = axes[2]
    lo, hi = min(cl_true.min(), cl_pred.min()), max(cl_true.max(), cl_pred.max())
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=1)
    ax.scatter(cl_true, cl_pred, s=12, color="#4c72b0")
    ax.set_xlabel("C_L reference (mesh-native)"); ax.set_ylabel("C_L predicted")
    ax.set_title("lift coefficient"); ax.grid(alpha=0.3)
    fig.suptitle(label); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "overview.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([per_field[:, c] for c in range(4)], showfliers=True)
    ax.set_xticks(range(1, 5), FIELD_TEX); ax.set_yscale("log")
    ax.set_ylabel("per-case rel-L2"); ax.set_title(f"{label}: per-field error over the test set")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "per_field_box.png"), dpi=110); plt.close(fig)


# ── one checkpoint ────────────────────────────────────────────────────────────────────────
def evaluate_checkpoint(ckpt, cfg_path, stats, out, precision, cache_dir, qual, save_pred,
                        workers, variant_tag, registry, root):
    t0 = time.time()
    cfg = resolve_data_paths(toml.load(cfg_path), root)
    device = validate_device(cfg["device"])
    dcfg = cfg["dataset"]
    split = json.load(open(dcfg["official_split_json"]))
    pool_dir = dcfg["data_root"]; K = dcfg.get("pool_K", 8)
    has_pool = lambda b: os.path.isfile(os.path.join(pool_dir, os.path.splitext(b)[0] + "__s0.npz"))
    test_names = [os.path.splitext(b)[0] for b in split["test"] if has_pool(b)]
    train_names = [os.path.splitext(b)[0] for b in split["train"] if has_pool(b)]
    test_paths = [os.path.join(pool_dir, n + "__s0.npz") for n in test_names]
    train_paths = [os.path.join(pool_dir, n + "__s0.npz") for n in train_names]
    if len(test_names) != len(split["test"]):
        print(f"WARNING: only {len(test_names)} of {len(split['test'])} test meshes found in {pool_dir}")
    std_glob = global_target_std(train_paths)

    st = np.load(stats)
    tm, ts = st["target_mean"], st["target_std"]
    cache = cache_dir or dcfg["test"]["kw"].get("cache_dir")
    ds = AirfransDataset(
        test_names, pool_dir, K=K, train_augment=False,
        feature_cols=dcfg.get("feature_cols"), target_cols=dcfg.get("target_cols"),
        feat_mean=st["feat_mean"], feat_std=st["feat_std"], target_mean=tm, target_std=ts,
        extract_kw=dcfg["test"]["kw"]["extract_kw"], cache_dir=cache, use_bc_injection=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=collate_batch_pde_fn)

    model = build_model(**cfg["train"]["model"]).to(device.type)
    ck = torch.load(ckpt, map_location=device.type, weights_only=False)
    state = ck["state"] if "state" in ck else ck["model_state_dict"]
    check_pairing_or_exit(cfg, state, cfg_path, ckpt, root)
    model.load_state_dict(state); model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    epoch = ck.get("epoch") if isinstance(ck, dict) else None
    use_bf16 = precision == "bf16"
    modes = str(cfg["train"]["model"].get("hodge_gate_kw", {}).get(
        "modes", cfg["train"]["model"].get("modes", "v")))
    label = f"HodgeMoE-{modes} · {os.path.basename(ckpt)}"
    print(f"\n=== {label}\n    cfg {cfg_path} | {precision} | {n_params:,} params | "
          f"{len(test_names)} test cases | cache {cache}", flush=True)

    os.makedirs(out, exist_ok=True)
    pred_dir = os.path.join(out, "pred"); os.makedirs(pred_dir, exist_ok=True)
    C = int(cfg["train"]["model"]["out_channels"])

    rows, metrics_arr, vol_arr, surf_arr, cl_t, cl_p = [], [], [], [], [], []
    for i, (tensors, target) in enumerate(loader):
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            pred = model(*[t.to(device.type) for t in tensors], mask=None)
        pred = pred[0].float().cpu().numpy()
        targ = target[0].cpu().numpy()[:, :C]
        z = np.load(test_paths[i])
        xf, wall, pos = z["x_feat"], z["wall"], z["pos"]
        U = uinf(xf)
        pred_phys = redim_targets(pred * ts + tm, U)            # as evaluation/fields.py
        targ_phys = redim_targets(targ * ts + tm, U)
        m = per_channel_metrics(pred_phys, targ_phys)
        vf, sf = vol_surf_mse(pred_phys, targ_phys, wall, std_glob)
        inlet = np.array([xf[0, 0], xf[0, 1]]); normals = xf[:, 3:5]   # as evaluation/lift.py
        c_p = mesh_cl(pos, wall, pred_phys[:, 2], normals, inlet)
        c_t = mesh_cl(pos, wall, z["y"][:, 2], normals, inlet)

        name = test_names[i]
        u_inf, aoa = flow_condition(name)
        r = dict(idx=i, case=name, u_inf=u_inf, aoa_deg=aoa,
                 n_vertices=int(pos.shape[0]), n_wall=int(np.asarray(wall).sum()))
        for k in FIELD_KEYS:
            for q in QUANT:
                r[f"{k}_{q}"] = m[k][q]
        for c, k in enumerate(FIELD_KEYS):
            r[f"{k}_vol_mse"] = float(vf[c]); r[f"{k}_surf_mse"] = float(sf[c])
        r["rel_l2_mean"] = float(np.mean([m[k]["l2rel"] for k in FIELD_KEYS]))
        r["r2_mean"] = float(np.mean([m[k]["r2"] for k in FIELD_KEYS]))
        r["volume_mse"] = float(np.mean(vf)); r["surf_mse"] = float(np.mean(sf))
        r["cl_true"] = c_t; r["cl_pred"] = c_p; r["cl_abs_err"] = abs(c_p - c_t)
        r["cl_rel_err"] = abs(c_p - c_t) / (abs(c_t) + 1e-12)
        rows.append(r)
        metrics_arr.append([[m[k][q] for q in QUANT] for k in FIELD_KEYS])
        vol_arr.append(vf); surf_arr.append(sf); cl_t.append(c_t); cl_p.append(c_p)

        np.savez_compressed(os.path.join(pred_dir, name + ".npz"), pred=pred_phys.astype("float32"),
                            target=targ_phys.astype("float32"), pos=pos, faces=z["faces"], wall=wall)
        if (i + 1) % 40 == 0 or i + 1 == len(test_names):
            print(f"  [{i+1}/{len(test_names)}] running mean rel-L2 "
                  f"{np.mean([q['rel_l2_mean'] for q in rows]):.4f}", flush=True)

    t_inf = time.time() - t0
    cl_t, cl_p = np.array(cl_t), np.array(cl_p)
    vol, surf = np.mean(np.array(vol_arr), axis=0), np.mean(np.array(surf_arr), axis=0)
    cl_rel = float(np.linalg.norm(cl_t - cl_p) / (np.linalg.norm(cl_t) + 1e-12))
    rho = spearman(cl_t, cl_p)

    # same files and layout as the released evaluators -> check_metrics.py works on this folder
    np.savez(os.path.join(out, "test_metrics.npz"), vol_mse=vol, surf_mse=surf,
             metrics=np.array(metrics_arr), field_keys=FIELD_KEYS, case_names=np.array(test_names))
    np.savez(os.path.join(out, "cl_native.npz"), cl_true=cl_t, cl_pred=cl_p, relL2=cl_rel,
             rho_L=rho, case_names=np.array(test_names))

    with open(os.path.join(out, "per_case.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    per_field = {k: {q: float(np.mean([r[f"{k}_{q}"] for r in rows])) for q in QUANT}
                 for k in FIELD_KEYS}
    for c, k in enumerate(FIELD_KEYS):
        per_field[k]["vol_mse"] = float(vol[c]); per_field[k]["surf_mse"] = float(surf[c])
    rel = np.array([r["rel_l2_mean"] for r in rows]); aoa = np.array([r["aoa_deg"] for r in rows])
    bands = []
    for lo, hi, lab in AOA_BANDS:
        sel = (aoa >= lo) & (aoa < hi)
        if sel.any():
            bands.append(dict(band=lab, n=int(sel.sum()),
                              rel_l2=float(rel[sel].mean()),
                              p_l2rel=float(np.mean([rows[i]["p_l2rel"] for i in np.where(sel)[0]])),
                              nut_l2rel=float(np.mean([rows[i]["nut_l2rel"] for i in np.where(sel)[0]])),
                              cl_rel_l2=float(np.linalg.norm(cl_t[sel] - cl_p[sel])
                                              / (np.linalg.norm(cl_t[sel]) + 1e-12))))
    order = np.argsort(rel)
    pick = lambda idx: [dict(case=rows[i]["case"], aoa_deg=rows[i]["aoa_deg"],
                             u_inf=rows[i]["u_inf"], rel_l2=rows[i]["rel_l2_mean"],
                             cl_abs_err=rows[i]["cl_abs_err"]) for i in idx]
    summary = dict(
        checkpoint=repo_rel(ckpt, root), checkpoint_sha256=sha256(ckpt), epoch=epoch,
        config=repo_rel(cfg_path, root), stats=repo_rel(stats, root), variant=variant_tag,
        modes=modes, precision=precision, params=n_params, n_cases=len(rows),
        overall=dict(rel_l2=float(rel.mean()), rel_l2_median=float(np.median(rel)),
                     rel_l2_p90=float(np.percentile(rel, 90)),
                     r2=float(np.mean([r["r2_mean"] for r in rows])),
                     volume=float(vol.mean()), surf=float(surf.mean()),
                     cl_rel_l2=cl_rel, rho_l=rho,
                     cl_mean_abs_err=float(np.mean(np.abs(cl_p - cl_t)))),
        per_field=per_field, aoa_bands=bands,
        best_cases=pick(order[:5]), worst_cases=pick(order[::-1][:10]),
        seconds_inference=round(t_inf, 1))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    write_summary_md(summary, os.path.join(out, "SUMMARY.md"), label)
    overall_figures(rows, cl_t, cl_p, os.path.join(out, "fig"), label)

    # qualitative figure for every case (or the first `qual`)
    n_q = len(rows) if qual < 0 else min(qual, len(rows))
    if n_q:
        qdir = os.path.join(out, "qual"); os.makedirs(qdir, exist_ok=True)
        jobs = []
        for r in rows[:n_q]:
            title = (f"{label}\n{r['case']}   U∞ {r['u_inf']:.1f} m/s · AoA {r['aoa_deg']:.2f}° · "
                     f"rel-L2 {r['rel_l2_mean']:.4f} · C_L {r['cl_true']:.3f} → {r['cl_pred']:.3f}")
            jobs.append((os.path.join(pred_dir, r["case"] + ".npz"),
                         os.path.join(qdir, r["case"] + ".png"), title))
        import multiprocessing as mp
        t1 = time.time()
        with mp.get_context("spawn").Pool(workers) as pool:
            for k, _ in enumerate(pool.imap_unordered(draw_case, jobs), 1):
                if k % 50 == 0 or k == len(jobs):
                    print(f"  qualitative {k}/{len(jobs)}", flush=True)
        print(f"  figures {time.time() - t1:.0f}s -> {qdir}")
    if not save_pred:
        import shutil
        shutil.rmtree(pred_dir, ignore_errors=True)

    # reference check for registered checkpoints
    if variant_tag:
        try:
            import io
            from check_metrics import report
            buf = io.StringIO()
            failed = report(variant_tag, repo_rel(out, root), root, stream=buf)
            # per-case agreement with the shipped evaluation arrays of this variant (same split order)
            ref_dir = registry[variant_tag].get("eval_dir")
            ref_dir = ref_dir and (ref_dir if os.path.isabs(ref_dir) else os.path.join(root, ref_dir))
            if ref_dir and os.path.isfile(os.path.join(ref_dir, "test_metrics.npz")):
                rm = np.load(os.path.join(ref_dir, "test_metrics.npz"))["metrics"]
                mm = np.array(metrics_arr)
                buf.write(f"\nPER-CASE agreement with {repo_rel(ref_dir, root)}\n")
                if rm.shape == mm.shape:
                    d = np.abs(rm[:, :, 4].mean(1) - mm[:, :, 4].mean(1))
                    buf.write(f"  per-case rel-L2: max |Δ| {d.max():.2e}, mean |Δ| {d.mean():.2e} "
                              f"over {len(d)} cases\n")
                else:
                    buf.write(f"  shapes differ: reference {rm.shape} vs measured {mm.shape}\n")
                cf = os.path.join(ref_dir, "cl_native.npz")
                if os.path.isfile(cf):
                    rc = np.load(cf)["cl_pred"]
                    if rc.shape == cl_p.shape:
                        buf.write(f"  per-case C_L pred: max |Δ| {np.abs(rc - cl_p).max():.2e}\n")
            open(os.path.join(out, "check.txt"), "w").write(buf.getvalue())
            print(buf.getvalue())
            summary["check_failed"] = int(failed) if failed is not None else None
            json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
        except Exception as e:                                   # never lose the evaluation
            print(f"  reference check skipped: {e}")

    o = summary["overall"]
    print(f"  OVERALL rel-L2 {o['rel_l2']:.4f} | R² {o['r2']:.4f} | Volume {o['volume']:.5f} | "
          f"Surf {o['surf']:.5f} | C_L rel-L2 {o['cl_rel_l2']:.4f} | rho_L {o['rho_l']:.5f}  "
          f"({time.time() - t0:.0f}s) -> {out}", flush=True)
    del model; torch.cuda.empty_cache()
    return summary


def write_summary_md(s, path, label):
    o = s["overall"]; L = []
    A = L.append
    A(f"# Full test-set evaluation — {label}\n")
    A(f"- Checkpoint: `{s['checkpoint']}` (epoch {s['epoch']}, SHA-256 `{s['checkpoint_sha256'][:16]}…`)")
    A(f"- Config: `{s['config']}` · modes `{s['modes']}` · precision **{s['precision']}** · "
      f"{s['params']:,} parameters")
    A(f"- Test cases: **{s['n_cases']}** (official AirFRANS `full` split, fixed `__s0` meshes)\n")
    A("## Overall\n")
    A("| rel-L2 (mean) | rel-L2 (median) | rel-L2 (90th pct) | R² | Volume | Surf | C_L rel-L2 | ρ_L | mean abs C_L error |")
    A("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    A(f"| **{o['rel_l2']:.4f}** | {o['rel_l2_median']:.4f} | {o['rel_l2_p90']:.4f} | {o['r2']:.4f} | "
      f"{o['volume']:.5f} | {o['surf']:.5f} | {o['cl_rel_l2']:.4f} | {o['rho_l']:.5f} | "
      f"{o['cl_mean_abs_err']:.4f} |\n")
    A("rel-L2 and R² are per-case values in physical units, averaged over the 4 fields and the test "
      "cases. Volume / Surf are the AirFRANS-native MSE of fields normalised by the global training "
      "std, on non-wall / wall vertices. C_L is the mesh-native pressure-integral lift (see README: "
      "about twice the physical AirFRANS lift; compare only like with like).\n")
    A("## Per field\n")
    A("| Field | rel-L2 | R² | MSE | MAE | RMSE | Volume MSE | Surf MSE |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, t in zip(FIELD_KEYS, ["u_x", "u_y", "p", "ν_t"]):
        f = s["per_field"][k]
        A(f"| {t} | {f['l2rel']:.4f} | {f['r2']:.4f} | {f['mse']:.4g} | {f['mae']:.4g} | "
          f"{f['rmse']:.4g} | {f['vol_mse']:.5f} | {f['surf_mse']:.5f} |")
    A("\n## By angle of attack\n")
    A("| AoA band | cases | rel-L2 | p rel-L2 | ν_t rel-L2 | C_L rel-L2 |")
    A("|---|---:|---:|---:|---:|---:|")
    for b in s["aoa_bands"]:
        A(f"| {b['band']} | {b['n']} | {b['rel_l2']:.4f} | {b['p_l2rel']:.4f} | "
          f"{b['nut_l2rel']:.4f} | {b['cl_rel_l2']:.4f} |")
    for title, key in [("Worst 10 cases", "worst_cases"), ("Best 5 cases", "best_cases")]:
        A(f"\n## {title}\n")
        A("| Case | U∞ (m/s) | AoA (°) | rel-L2 | abs C_L error | figure |")
        A("|---|---:|---:|---:|---:|---|")
        for c in s[key]:
            A(f"| `{c['case']}` | {c['u_inf']:.1f} | {c['aoa_deg']:.2f} | {c['rel_l2']:.4f} | "
              f"{c['cl_abs_err']:.4f} | [png](qual/{c['case']}.png) |")
    A("\n## Files\n")
    A("- `per_case.csv`: every metric for every case · `summary.json`: this page as data")
    A("- `test_metrics.npz`, `cl_native.npz`: same layout as `scripts/run_eval.sh` "
      "(readable by `source/shared/check_metrics.py`)")
    A("- `fig/overview.png`, `fig/per_field_box.png` · `qual/<case>.png`: one figure per test case")
    open(path, "w").write("\n".join(L) + "\n")


def write_comparison(summaries, out):
    keys = ["rel_l2", "r2", "volume", "surf", "cl_rel_l2", "rho_l"]
    best = {k: (max if k in ("r2", "rho_l") else min)(s["overall"][k] for s in summaries) for k in keys}
    with open(os.path.join(out, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder", "variant", "modes", "precision", "epoch", "params"] + keys
                   + [f"{k}_l2rel" for k in FIELD_KEYS])
        for s in summaries:
            w.writerow([s["_folder"], s["variant"], s["modes"], s["precision"], s["epoch"], s["params"]]
                       + [s["overall"][k] for k in keys]
                       + [s["per_field"][k]["l2rel"] for k in FIELD_KEYS])
    L = ["# Checkpoint comparison — full official test set\n",
         "Bold = best in column. Each row links to its own SUMMARY.md.\n",
         "| Checkpoint | modes | precision | epoch | rel-L2 | R² | Volume | Surf | C_L rel-L2 | ρ_L | "
         "u_x | u_y | p | ν_t |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = dict(rel_l2="{:.4f}", r2="{:.4f}", volume="{:.5f}", surf="{:.5f}", cl_rel_l2="{:.4f}",
               rho_l="{:.5f}")
    for s in summaries:
        cells = []
        for k in keys:
            v = fmt[k].format(s["overall"][k])
            cells.append(f"**{v}**" if s["overall"][k] == best[k] else v)
        L.append(f"| [{s['_folder']}]({s['_folder']}/SUMMARY.md) | {s['modes']} | {s['precision']} | "
                 f"{s['epoch']} | " + " | ".join(cells) + " | "
                 + " | ".join(f"{s['per_field'][k]['l2rel']:.4f}" for k in FIELD_KEYS) + " |")
    open(os.path.join(out, "COMPARISON.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", default=[],
                    help="registered variant (v, vef); repeatable. Evaluates its released "
                         "checkpoint, or the --ckpt files with its config.")
    ap.add_argument("--cfg_path", default=None, help="config for --ckpt (instead of --variant)")
    ap.add_argument("--ckpt", nargs="*", default=[],
                    help="checkpoint files, globs or run folders (all *.pth inside)")
    ap.add_argument("--stats", default=None, help="norm stats (default: the variant's / checkpoints/norm_stats.npz)")
    ap.add_argument("--out", default="runs/full_eval")
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto",
                    help="auto: bf16 for variant v (its published precision), fp32 otherwise")
    ap.add_argument("--cache_dir", default=None, help="geometry cache for the test meshes (overrides the config)")
    ap.add_argument("--qual", type=int, default=-1, help="qualitative figures: -1 = every case (default), N = first N, 0 = none")
    ap.add_argument("--save_pred", action="store_true", help="keep pred/<case>.npz (≈0.1 MB per case)")
    ap.add_argument("--workers", type=int, default=8, help="processes drawing the figures")
    args = ap.parse_args()

    root = _ROOT
    registry, default = load_registry(root)
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)

    ckpts = []
    for c in args.ckpt:
        c = c if os.path.isabs(c) else os.path.join(os.getcwd(), c)
        if os.path.isdir(c):
            found = sorted(glob.glob(os.path.join(c, "*.pth")))
        else:
            found = sorted(glob.glob(c))
        if not found:
            sys.exit(f"no checkpoint matches {c}")
        ckpts += [f for f in found if not f.endswith("_resume.pth")]

    jobs = []                                     # (ckpt, cfg, stats, variant)
    if not args.variant and not ckpts:
        args.variant = sorted(registry)           # nothing given: every released checkpoint
    if args.variant and not ckpts:
        for v in args.variant:
            r = resolve_variant(v, root)
            jobs.append((r["checkpoint"], r["config"], args.stats or r["stats"], v))
    else:
        if args.cfg_path:
            cfg_path, vname = args.cfg_path, None
        elif len(args.variant) == 1:
            r = resolve_variant(args.variant[0], root, require_checkpoint=False)
            cfg_path, vname = r["config"], args.variant[0]
        else:
            sys.exit("--ckpt needs exactly one --variant or a --cfg_path to build the model")
        stats = args.stats or os.path.join(root, "checkpoints", "norm_stats.npz")
        for c in ckpts:
            jobs.append((c, cfg_path, stats, vname))

    os.chdir(root)                       # relative paths in the reports resolve against the repo root
    summaries = []
    for ck, cfg_path, stats, vname in jobs:
        cfg_path = cfg_path if os.path.isabs(cfg_path) else os.path.join(root, cfg_path)
        stats = stats if os.path.isabs(stats) else os.path.join(root, stats)
        reg = registered_variant_of(ck, registry, root)         # the released file itself?
        if args.precision != "auto":
            prec = args.precision
        else:
            p = registry.get(vname or reg or "", {}).get("precision", "fp32")
            prec = "bf16" if "bf16" in str(p) else "fp32"
        folder = (f"{reg}_{ckpt_label(ck)}" if reg else ckpt_label(ck))
        s = evaluate_checkpoint(ck, cfg_path, stats, os.path.join(out, folder), prec, args.cache_dir,
                                args.qual, args.save_pred, args.workers, reg, registry, root)
        s["_folder"] = folder
        summaries.append(s)

    os.makedirs(out, exist_ok=True)
    write_comparison(summaries, out)
    print(f"\nall {len(summaries)} checkpoint(s) done -> {out}/COMPARISON.md")


if __name__ == "__main__":
    main()
