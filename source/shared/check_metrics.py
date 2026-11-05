"""
Compare a finished evaluation against the reference metrics in `variants.toml`.

`scripts/run_eval.sh` writes `test_metrics.npz` (per-case, per-field) and `cl_native.npz`
(per-case lift) into its output folder. This module reduces those arrays to the six headline
numbers and checks each one against the `[<variant>.expect]` block, printing one row per metric
and exiting non-zero if any row fails.

    python source/shared/check_metrics.py --variant vef --eval-dir runs/eval_vef
    python source/shared/check_metrics.py --variant v --eval-dir results/eval_release_v   # the shipped arrays

The arrays are the source of truth, not the console logs: the logs round to four decimals, which
is coarser than some of the tolerances.

Tolerances. `variants.toml` carries two relative tolerances per variant.

`tol` applies to `rel_l2`, `volume`, `surf` and `cl_rel_l2`, compared on the value itself.

`tol_deficit` applies to `r2` and `rho_l`, compared on the DEFICIT, 1 - value. Those sit at
0.998 and 0.9999, where a 2% test on the value would accept a model that had collapsed. It is
looser than `tol` by default because the published references are rounded to four or five
digits, and at a deficit of 0.0017 that rounding alone is worth about 2%: a tight test there
would measure the rounding rather than the model.

A measurement better than the reference never fails; only a worse one can.
"""
import os


def _posix(path):
    """Forward slashes on every platform; see source/shared/variants.py:posix."""
    return str(path).replace(os.sep, "/").replace("\\", "/")

try:                                  # Python 3.11+
    import tomllib as _toml_r

    def _load_toml(path):
        with open(path, "rb") as f:
            return _toml_r.load(f)
except ImportError:
    import toml as _toml_r

    def _load_toml(path):
        return _toml_r.load(path)

FIELD_KEYS = ["u_x", "u_y", "p", "nu_t"]

# name -> (label, "value" | "deficit", lower_is_better)
METRICS = (
    ("rel_l2",    ("rel-L2 (mean over fields)",            "value",   True)),
    ("r2",        ("R2 (mean over fields)",                "deficit", False)),
    ("volume",    ("Volume (normalised MSE, non-wall)",    "value",   True)),
    ("surf",      ("Surf (normalised MSE, wall)",          "value",   True)),
    ("cl_rel_l2", ("C_L rel-L2 (wall-band contour)",       "value",   True)),
    ("rho_l",     ("rho_L (Spearman, lift)",               "deficit", False)),
)


class CheckError(Exception):
    """The evaluation output needed for a check is missing or unreadable."""


def read_measured(eval_dir):
    """Reduce an evaluation output folder to the headline metrics.

    Accepts either the layout `scripts/run_eval.sh` produces (`fields/test_metrics.npz`,
    `lift/cl_native.npz`) or a flat folder holding both files, which is how the shipped
    `results/eval_release*/` directories are arranged. Lift metrics are optional: a `--n_qual 0`
    field-only run still yields the four field metrics.
    """
    import numpy as np

    def find(name):
        for rel in (name, os.path.join("fields", name), os.path.join("lift", name)):
            p = os.path.join(eval_dir, rel)
            if os.path.isfile(p):
                return _posix(p)
        return None

    fp = find("test_metrics.npz")
    if fp is None:
        raise CheckError(
            "no test_metrics.npz under %s.\n"
            "  Run the evaluation first:  VARIANT=<name> bash scripts/run_eval.sh" % eval_dir)

    z = np.load(fp, allow_pickle=True)
    m = z["metrics"]                        # (cases, 4 fields, 6 quantities)
    if m.ndim != 3 or m.shape[1] != 4 or m.shape[2] != 6:
        raise CheckError("%s has metrics of shape %s; expected (cases, 4, 6)." % (fp, m.shape))
    # Column order is fixed by evaluation/fields.py: mse, mae, rmse, r2, l2rel, l2rel_c.
    out = {
        "rel_l2": float(m[:, :, 4].mean()),
        "r2": float(m[:, :, 3].mean()),
        "volume": float(np.mean(z["vol_mse"])),
        "surf": float(np.mean(z["surf_mse"])),
        "rel_l2_per_field": [float(x) for x in m[:, :, 4].mean(axis=0)],
        "n_cases": int(m.shape[0]),
        "_files": [fp],
    }

    cp = find("cl_native.npz")
    if cp is not None:
        c = np.load(cp)
        out["cl_rel_l2"] = float(c["relL2"])
        out["rho_l"] = float(c["rho_L"])
        out["_files"].append(cp)
    return out


def expectations(variant, deploy_root):
    reg = _load_toml(os.path.join(deploy_root, "variants.toml"))
    if variant not in reg or not isinstance(reg[variant], dict):
        raise CheckError("variants.toml has no variant %r." % variant)
    exp = reg[variant].get("expect")
    if not exp:
        raise CheckError("variant %r in variants.toml has no [%s.expect] block, so there is "
                         "nothing to check against." % (variant, variant))
    return exp


def compare(measured, exp):
    """Return (rows, n_failed). Each row is (label, expected, measured, delta%, verdict)."""
    tol_value = float(exp.get("tol", 0.02))
    tol_deficit = float(exp.get("tol_deficit", 0.25))
    rows, failed = [], 0
    for key, (label, kind, lower_better) in METRICS:
        if key not in exp:
            continue
        tol = tol_deficit if kind == "deficit" else tol_value
        want = float(exp[key])
        if key not in measured:
            rows.append((label, want, None, None, "SKIP (not measured)"))
            continue
        got = float(measured[key])

        if kind == "deficit":
            # Compare 1 - value; a smaller deficit is better.
            ref, cur = 1.0 - want, 1.0 - got
            worse = cur - ref
            rel = worse / abs(ref) if ref else (0.0 if worse <= 0 else float("inf"))
        else:
            worse = (got - want) if lower_better else (want - got)
            rel = worse / abs(want) if want else (0.0 if worse <= 0 else float("inf"))

        if rel <= tol:
            verdict = "PASS" if rel <= 0 else "PASS (within %.0f%%)" % (tol * 100)
        else:
            verdict = "FAIL (worse by %.1f%%, tol %.0f%%)" % (rel * 100, tol * 100)
            failed += 1
        rows.append((label, want, got, rel * 100.0, verdict))
    return rows, failed


def report(variant, eval_dir, deploy_root, stream=None):
    """Print the comparison table. Returns the number of failed metrics."""
    import sys
    out = stream or sys.stdout
    w = out.write

    measured = read_measured(eval_dir)
    exp = expectations(variant, deploy_root)
    rows, failed = compare(measured, exp)

    w("\n" + "=" * 86 + "\n")
    w("METRIC CHECK  variant %r   %d test cases\n" % (variant, measured["n_cases"]))
    w("  measured in  %s\n" % _posix(eval_dir))
    for f in measured["_files"]:
        w("    from       %s\n" % _posix(f))
    w("  reference    %s\n" % exp.get("source", "variants.toml"))
    w("=" * 86 + "\n")
    w("%-38s %10s %10s %9s  %s\n" %
      ("metric", "reference", "measured", "delta", "verdict"))
    w("-" * 86 + "\n")
    for label, want, got, delta, verdict in rows:
        if got is None:
            w("%-38s %10.5f %10s %9s  %s\n" % (label, want, "-", "-", verdict))
        else:
            w("%-38s %10.5f %10.5f %8.1f%%  %s\n" % (label, want, got, delta, verdict))

    if "rel_l2_per_field" in exp and "rel_l2_per_field" in measured:
        w("\nper-field rel-L2 (informational, not part of the verdict)\n")
        w("  %-6s %10s %10s\n" % ("field", "reference", "measured"))
        for k, a, b in zip(FIELD_KEYS, exp["rel_l2_per_field"], measured["rel_l2_per_field"]):
            w("  %-6s %10.4f %10.4f\n" % (k, a, b))

    if measured["n_cases"] != 200:
        w("\nNOTE: %d test cases, not the full official 200. The means are over whatever was\n"
          "      present, so this is not a comparison against the published numbers.\n"
          % measured["n_cases"])
    if "cl_rel_l2" not in measured:
        w("\nNOTE: no cl_native.npz found, so the two lift metrics were not checked.\n")

    w("\n%s  %d of %d checked metrics failed\n"
      % ("RESULT: FAIL" if failed else "RESULT: PASS", failed, len([r for r in rows if r[2] is not None])))
    w("=" * 86 + "\n")
    return failed


def _main(argv=None):
    import argparse
    import sys

    deploy_root = os.path.dirname(os.path.abspath(__file__))
    while deploy_root != os.path.dirname(deploy_root) and not os.path.isfile(
            os.path.join(deploy_root, "variants.toml")):
        deploy_root = os.path.dirname(deploy_root)
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--variant", required=True, help="variant name in variants.toml")
    ap.add_argument("--eval-dir", required=True,
                    help="folder holding test_metrics.npz and cl_native.npz, or a run_eval.sh "
                         "output folder containing fields/ and lift/")
    ap.add_argument("--root", default=deploy_root, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    d = args.eval_dir
    if not os.path.isabs(d):
        d = os.path.normpath(os.path.join(args.root, d)) if not os.path.isdir(d) else d
    d = _posix(d)
    try:
        return 1 if report(args.variant, d, args.root) else 0
    except CheckError as e:
        print("metric check error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
