"""The two released cases, and the code that is specific to each.

Everything the two share — the architecture, the data pipeline, the losses, the metrics, the
schedule, the evaluation — lives once, in `source/arch`, `source/data`, `source/training` and
`source/evaluation`. This package holds only what genuinely differs between them, so the two
cases cannot drift apart in the parts that are meant to be identical.

    cases/vef/trainer.py    fp32. Delegates straight to `training.loop.main`. This is the
                            released trainer, unmodified: it produced hodgemoe_vef_best_ep400.pth.
    cases/v/trainer.py      bf16 autocast. The same loop with the forward passes wrapped, which
                            is how hodgemoe_v_best_ep380.pth was trained.

What does NOT differ, and therefore is not duplicated here: the model. One `arch/` package
serves both; a case selects vertex-only or vertex+edge+face through the `modes` flag in its
config, and `modes` is the only architectural difference between them.

Precision is a property of the training loop, not of the architecture. `torch.autocast` casts
the inputs of individual ops at run time and never changes `param.dtype`, so both released
checkpoints are fp32 on disk and both load into the same `arch/`. That is why the split here is
by trainer rather than by model.

`variants.toml` maps a case name to its config and its entry point. See `source/shared/variants.py`.
"""
import os as _o
import os

# Deployment root: the directory holding variants.toml, located by walking up.
_ROOT = _o.path.dirname(_o.path.abspath(__file__))
while _ROOT != _o.path.dirname(_ROOT) and not _o.path.isfile(
        _o.path.join(_ROOT, "variants.toml")):
    _ROOT = _o.path.dirname(_ROOT)


def select(argv=None, default=None):
    """Return the case package named by `--variant` on the command line.

    The evaluators parse `--variant` themselves, but the choice of WHICH evaluator to run has to
    be made before that happens, because the two cases differ in precision. So the launcher peeks
    at argv. An unrecognised or absent name falls back to the registry default, which keeps the
    old command lines — `python eval_fields.py --cfg_path ... --ckpt ...` with no variant —
    behaving exactly as they did.
    """
    import importlib
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    name = None
    for i, a in enumerate(argv):
        if a == "--variant" and i + 1 < len(argv):
            name = argv[i + 1]
        elif a.startswith("--variant="):
            name = a.split("=", 1)[1]
    if name is None:
        name = os.environ.get("VARIANT") or None
    if name is None:
        try:
            from variants import load_registry
            name = load_registry(_ROOT)[1]
        except Exception:
            name = default or "vef"
    try:
        from variants import resolve_variant
        name = resolve_variant(name, _ROOT, require_checkpoint=False)["name"]
    except Exception:
        pass
    try:
        return importlib.import_module("cases.%s" % name)
    except ImportError:
        return importlib.import_module("cases.%s" % (default or "vef"))


def evaluator(argv=None):
    """The case's evaluator module: bf16 for v, the released fp32 code for vef."""
    import importlib
    pkg = select(argv)
    return importlib.import_module(pkg.__name__ + ".evaluator")
