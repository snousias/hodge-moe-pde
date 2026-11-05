"""
Variant selection for this deployment.

The release ships two models that differ only in the controller's `modes` flag: `"v"`
(vertex-only) and `"vef"` (vertex, edge and face). They share the data pipeline, the geometry
cache and `norm_stats.npz`. Each has its own config file and its own checkpoint, and the two
must be used as a pair.

`variants.toml` at the deployment root is the registry that holds those pairs. This module
reads it and resolves a variant NAME into absolute paths, so one name selects config and
checkpoint together:

    from variants import resolve_variant, check_pairing

    v = resolve_variant("v", deploy_root)     # -> Variant(config=..., checkpoint=..., ...)

`VARIANT` only supplies defaults. The shell scripts still honour an explicit `CFG=`, `CKPT=` or
`STATS=`, and with `VARIANT` unset they behave exactly as they did before this file existed.

Run as a script for the shell scripts and for a human-readable listing:

    python source/shared/variants.py --list
    python source/shared/variants.py --variant v --shell     # V_CFG=... V_CKPT=... V_STATS=...
    python source/shared/variants.py --variant v --print config

`check_pairing` is the other half: it compares a checkpoint's state-dict keys with the `modes`
the config asks for and fails with a readable message, instead of letting `load_state_dict`
raise a wall of missing keys. It works on any checkpoint, registered or not.
"""
import os

try:                                  # Python 3.11+
    import tomllib as _toml_r

    def _load_toml(path):
        with open(path, "rb") as f:
            return _toml_r.load(f)
except ImportError:                   # fall back to the `toml` package the rest of the code uses
    import toml as _toml_r

    def _load_toml(path):
        return _toml_r.load(path)


REGISTRY_FILE = "variants.toml"

# Keys the scripts depend on. Everything else in a variant block is descriptive.
_REQUIRED = ("config", "checkpoint")
_PATH_KEYS = ("config", "checkpoint", "stats", "eval_dir", "trainer")


class VariantError(Exception):
    """A variant name, registry entry or config/checkpoint pairing that cannot be used."""


class Variant(dict):
    """A resolved variant. Paths are absolute; the descriptive keys are passed through."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def posix(path):
    """Forward slashes, on every platform.

    This repository targets Linux and is only edited on Windows, where os.path.join and
    os.path.relpath produce backslashes. Python accepts forward slashes on Windows, so
    normalising here keeps every path the tooling prints, stores or passes to a shell
    identical across platforms.
    """
    return str(path).replace(os.sep, "/").replace("\\", "/")


def registry_path(deploy_root):
    return posix(os.path.join(deploy_root, REGISTRY_FILE))


def load_registry(deploy_root):
    """Read `variants.toml`. Returns (variants_dict, default_name)."""
    path = registry_path(deploy_root)
    if not os.path.isfile(path):
        raise VariantError(
            "no variant registry at %s.\n"
            "Pass --cfg_path/--ckpt explicitly (or CFG=/CKPT= to the scripts) instead of a "
            "variant name." % path)
    raw = _load_toml(path)
    default = raw.pop("default", None)
    variants = {k: v for k, v in raw.items() if isinstance(v, dict)}
    if not variants:
        raise VariantError("%s defines no variants." % path)
    if default is not None and default not in variants:
        raise VariantError("%s sets default = %r, which is not one of: %s"
                           % (path, default, ", ".join(sorted(variants))))
    return variants, default


def available(deploy_root):
    """Sorted variant names, for error messages and `--list`."""
    return sorted(load_registry(deploy_root)[0])


def resolve_variant(name, deploy_root, require_checkpoint=True):
    """Resolve a variant name into a `Variant` with absolute `config`/`checkpoint`/`stats`.

    `name` may be None, in which case the registry's `default` is used. Relative paths in the
    registry resolve against `deploy_root`, never the current working directory — the same rule
    `shared/paths.py` applies to data paths.
    """
    variants, default = load_registry(deploy_root)
    if name is None:
        name = default
        if name is None:
            raise VariantError("no variant given and %s sets no `default`. Choose one of: %s"
                               % (registry_path(deploy_root), ", ".join(sorted(variants))))
    key = str(name).strip()
    if key not in variants:
        lowered = {k.lower(): k for k in variants}
        if key.lower() in lowered:                      # accept "V" / "VEF"
            key = lowered[key.lower()]
        else:
            raise VariantError("unknown variant %r. Available: %s"
                               % (name, ", ".join(sorted(variants))))

    entry = dict(variants[key])
    for k in _REQUIRED:
        if not entry.get(k):
            raise VariantError("variant %r in %s has no `%s`."
                               % (key, registry_path(deploy_root), k))
    entry.setdefault("stats", "checkpoints/norm_stats.npz")
    # The fp32 loop is the default: a variant that does not name a trainer trains like vef.
    entry.setdefault("trainer", "train.py")

    out = Variant(entry)
    out["name"] = key
    for k in _PATH_KEYS:
        if entry.get(k):
            p = os.path.expanduser(str(entry[k]))
            out[k] = posix(p if os.path.isabs(p)
                           else os.path.normpath(os.path.join(deploy_root, p)))

    missing = [(k, out[k]) for k in ("config", "stats", "trainer") if not os.path.isfile(out[k])]
    if require_checkpoint and not os.path.isfile(out["checkpoint"]):
        missing.append(("checkpoint", out["checkpoint"]))
    if missing:
        raise VariantError("variant %r refers to files that are not here:\n%s"
                           % (key, "\n".join("  %-10s %s" % (k, p) for k, p in missing)))
    return out


# ── argparse integration ─────────────────────────────────────────────────────────

# `--variant` fills in whichever of --cfg_path/--ckpt/--stats the caller left out. An explicit
# value always wins, so every command line that worked before `variants.toml` existed still
# works, with or without a variant name.

_ARG_TO_FIELD = {"cfg_path": "config", "ckpt": "checkpoint", "stats": "stats"}


def fill_args(args, deploy_root, need=("cfg_path", "ckpt", "stats")):
    """Fill the unset entries of `need` on an argparse namespace from `args.variant`.

    Without `--variant`, every name in `need` must already be set; otherwise this raises
    `VariantError` listing what is missing and which variants exist. Returns `args`.
    """
    name = getattr(args, "variant", None)
    if name:
        v = resolve_variant(name, deploy_root)
        for a in need:
            if getattr(args, a, None) is None:
                setattr(args, a, v[_ARG_TO_FIELD[a]])
        setattr(args, "variant", v["name"])

    missing = [a for a in need if getattr(args, a, None) is None]
    if missing:
        try:
            names = ", ".join(available(deploy_root))
        except VariantError:
            names = "<no variants.toml>"
        raise VariantError(
            "missing %s.\n"
            "  Give them explicitly, or name a variant with --variant (available: %s)."
            % (" and ".join("--" + a.replace("_", "-") for a in missing), names))
    return args


def fill_args_or_exit(args, deploy_root, need=("cfg_path", "ckpt", "stats")):
    """`fill_args`, reporting a failure as a clean message and exit 2 rather than a traceback."""
    import sys
    try:
        return fill_args(args, deploy_root, need)
    except VariantError as e:
        print("variant error: %s" % e, file=sys.stderr)
        raise SystemExit(2)


# ── config / checkpoint pairing ─────────────────────────────────────────────────────────────
# The two variants differ in which Hodge operators the controller builds. A `"vef"` checkpoint
# carries edge- and face-operator tensors that a `"v"` model does not define, and a `"v"`
# checkpoint lacks tensors a `"vef"` model requires. Either way `load_state_dict(strict=True)`
# refuses the load — correctly, but with dozens of key names and no indication of the cause.
# `check_pairing` reads the modes off the checkpoint itself and says which two files disagree.

def modes_of_state(state):
    """Infer the controller `modes` a checkpoint was trained with, from its state-dict keys."""
    has_e = any(".e_attn." in k for k in state)
    has_f = any(".f_attn." in k for k in state)
    return "v" + ("e" if has_e else "") + ("f" if has_f else "")


def modes_of_cfg(cfg):
    """The controller `modes` a loaded config asks for."""
    model = (cfg or {}).get("train", {}).get("model", {})
    return str(model.get("hodge_gate_kw", {}).get("modes", model.get("modes", "v")))


def check_pairing(cfg, state, cfg_path=None, ckpt_path=None, deploy_root=None):
    """Raise `VariantError` if `state` was not trained with the `modes` `cfg` asks for.

    Call it after `torch.load` and before `load_state_dict`. A checkpoint that matches is
    returned unchanged, so this can be used inline.
    """
    want, got = modes_of_cfg(cfg), modes_of_state(state)
    if want == got:
        return state

    hint = ""
    if deploy_root:
        try:
            names = [n for n in available(deploy_root)
                     if resolve_variant(n, deploy_root, require_checkpoint=False)
                     .get("modes", n) == got]
            if names:
                hint = ("\n  This checkpoint belongs to variant %r. Either run it with that "
                        "variant\n    VARIANT=%s bash scripts/run_eval.sh\n"
                        "  or pass its own config with --cfg_path." % (names[0], names[0]))
        except VariantError:
            pass

    raise VariantError(
        "config and checkpoint do not match.\n"
        "  config     %s\n"
        "             asks for a controller with modes = %r\n"
        "  checkpoint %s\n"
        "             was trained with modes = %r%s"
        % (cfg_path or "<config>", want, ckpt_path or "<checkpoint>", got, hint))


def format_table(deploy_root):
    """The `--list` output: one row per registered variant."""
    variants, default = load_registry(deploy_root)
    rows = []
    for name in sorted(variants):
        try:
            v = resolve_variant(name, deploy_root, require_checkpoint=False)
            present = "yes" if os.path.isfile(v["checkpoint"]) else "MISSING"
            rows.append((name + ("  (default)" if name == default else ""),
                         posix(os.path.relpath(v["config"], deploy_root)),
                         posix(os.path.relpath(v["trainer"], deploy_root)),
                         posix(os.path.relpath(v["checkpoint"], deploy_root)),
                         str(v.get("modes", "")), str(v.get("precision", "")),
                         str(v.get("epoch", "")),
                         ("%.4f" % v["rel_l2"]) if "rel_l2" in v else "", present))
        except VariantError as e:
            rows.append((name, "<unresolved: %s>" % e, "", "", "", "", "", "", "no"))
    hdr = ("variant", "config", "trainer", "checkpoint", "modes", "precision", "epoch",
           "rel-L2", "file")
    w = [max(len(hdr[i]), *(len(r[i]) for r in rows)) for i in range(len(hdr))]
    line = lambda r: "  ".join(str(r[i]).ljust(w[i]) for i in range(len(hdr))).rstrip()
    return "\n".join([line(hdr), "  ".join("-" * x for x in w)] + [line(r) for r in rows])


def _relative_to(path, root):
    """`path` relative to `root` with forward slashes, or `path` unchanged if it is outside."""
    try:
        rel = os.path.relpath(path, root)
    except ValueError:                      # different drive on Windows
        return posix(path)
    if rel.startswith(os.pardir):
        return posix(path)
    return posix(rel)


def _main(argv=None):
    import argparse
    import sys

    deploy_root = os.path.dirname(os.path.abspath(__file__))
    while deploy_root != os.path.dirname(deploy_root) and not os.path.isfile(
            os.path.join(deploy_root, "variants.toml")):
        deploy_root = os.path.dirname(deploy_root)
    ap = argparse.ArgumentParser(
        description="Resolve a release variant into its config and checkpoint.")
    ap.add_argument("--variant", default=None, help="variant name; omit for the registry default")
    ap.add_argument("--list", action="store_true", help="print the registered variants and exit")
    ap.add_argument("--shell", action="store_true",
                    help="print V_CFG/V_CKPT/V_STATS/V_NAME assignments for `eval` in a shell")
    ap.add_argument("--print", dest="field", default=None,
                    help="print one field of the variant (config, checkpoint, stats, modes, ...)")
    ap.add_argument("--root", default=deploy_root, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    try:
        if args.list:
            print(format_table(args.root))
            return 0
        v = resolve_variant(args.variant, args.root,
                            require_checkpoint=not (args.field and args.field != "checkpoint"))
        if args.field:
            if args.field not in v:
                raise VariantError("variant %r has no field %r. It has: %s"
                                   % (v["name"], args.field, ", ".join(sorted(v))))
            print(v[args.field])
        elif args.shell:
            # The shell scripts `cd` to the deployment root first, so emit paths relative to it
            # (and with forward slashes) to match the literal defaults they already carry. A path
            # outside the root is emitted absolute. Single-quoted for `eval`: paths may contain
            # spaces.
            for key, field in (("V_NAME", "name"), ("V_CFG", "config"),
                               ("V_CKPT", "checkpoint"), ("V_STATS", "stats"),
                               ("V_TRAINER", "trainer")):
                val = str(v[field])
                if field != "name":
                    val = _relative_to(val, args.root)
                print("%s='%s'" % (key, val.replace("'", "'\\''")))
        else:
            print(format_table(args.root))
    except VariantError as e:
        print("variant error: %s" % e, file=sys.stderr)
        return 2
    return 0


def check_pairing_or_exit(cfg, state, cfg_path=None, ckpt_path=None, deploy_root=None):
    """`check_pairing`, reporting a mismatch as a clean message and exit 2."""
    import sys
    try:
        return check_pairing(cfg, state, cfg_path, ckpt_path, deploy_root)
    except VariantError as e:
        print("variant error: %s" % e, file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(_main())
