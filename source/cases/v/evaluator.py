# ========================================================================================
# EVALUATORS, bf16   —   the precision the vertex-only metrics were measured in
# ========================================================================================
"""bf16-autocast evaluation, for `config_v.toml` / `VARIANT=v`.

`source/evaluation/fields.py` and `source/evaluation/lift.py` are the released fp32 evaluators
and are left exactly as they are. They produced the vef numbers in `results/eval_release/`.

The published vertex-only metrics — mean relative L2 **0.0300**, R2 0.9983, Volume 0.00140,
Surf 0.00164, C_L 0.0225 — were measured in the origin case `v57_gated_hodge`, whose evaluators
run the forward pass under `torch.autocast(bfloat16)`:

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        pred = model(*tensors, mask=None)

Evaluating those same weights in fp32 is a valid measurement, but it is a different one, and it
is not expected to land on 0.0300. This module restores the published precision.

HOW, without touching the released evaluators. Both of them build the network through a single
module-level name (`fields.build_model`, `lift.build_mesh_opformer_model`) and then call the
result. Rebinding that name to a factory that returns an autocast-wrapping proxy puts the whole
forward under autocast, which is exactly what wrapping the call site does: autocast affects how
individual ops dispatch inside the forward, nothing else. The rebinding is scoped to the call and
undone afterwards, including on an exception.

The proxy deliberately does NOT subclass nn.Module. A submodule would prefix every key with
"model.", and `load_state_dict` on the released checkpoints would fail.

Precision is not an architectural property: autocast never changes `param.dtype`, both released
checkpoints are fp32 on disk, and both load into the same `arch/`. See checkpoints/PROVENANCE.md.

    python eval_fields.py --variant v --out runs/eval_v/fields
    VARIANT=v bash scripts/run_eval.sh
"""
# __deployment_path_bootstrap__
import os as _o, sys as _s
_ROOT = _o.path.dirname(_o.path.abspath(__file__))
while _ROOT != _o.path.dirname(_ROOT) and not _o.path.isfile(
        _o.path.join(_ROOT, "variants.toml")):
    _ROOT = _o.path.dirname(_ROOT)                      # deployment root (holds variants.toml)
for _d in (_o.path.join(_ROOT, "source", "shared"), _o.path.join(_ROOT, "source"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)

import contextlib

import torch

__all__ = ["fields_main", "lift_main", "AutocastModel"]

DTYPE = torch.bfloat16


class AutocastModel(object):
    """Runs the wrapped model's forward under autocast; everything else passes through.

    Not an `nn.Module` on purpose — see the module docstring. `to()`, `eval()` and `train()`
    return the proxy rather than the inner model, so the released
    `model = build(...).to(device.type)` and `model.eval()` keep working on the proxy.
    """

    def __init__(self, model, dtype=DTYPE):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_dtype", dtype)

    def __call__(self, *args, **kwargs):
        # Device read from the parameters, so this is correct whether the model was moved
        # before or after wrapping, and on CPU as well as CUDA.
        try:
            device_type = next(self._model.parameters()).device.type
        except StopIteration:
            device_type = "cpu"
        with torch.autocast(device_type, dtype=self._dtype):
            return self._model(*args, **kwargs)

    # keep the proxy in place across the fluent calls the evaluators make
    def to(self, *a, **k):
        self._model.to(*a, **k)
        return self

    def eval(self):
        self._model.eval()
        return self

    def train(self, mode=True):
        self._model.train(mode)
        return self

    def __getattr__(self, name):          # parameters(), load_state_dict(), analyzer, aux, ...
        return getattr(object.__getattribute__(self, "_model"), name)

    def __setattr__(self, name, value):
        setattr(self._model, name, value)

    def __repr__(self):
        return "AutocastModel(%s, dtype=%s)" % (type(self._model).__name__, self._dtype)


@contextlib.contextmanager
def _builder_patched(module, attr, dtype=DTYPE):
    """Make `module.attr(...)` return an autocast-wrapped model for the duration of the block."""
    original = getattr(module, attr)

    def build_wrapped(*a, **k):
        return AutocastModel(original(*a, **k), dtype=dtype)

    setattr(module, attr, build_wrapped)
    try:
        yield
    finally:
        setattr(module, attr, original)


def fields_main():
    """Field metrics (rel-L2, R2, Volume, Surf) with the forward in bf16."""
    from evaluation import fields
    print("[cases.v] bf16 autocast evaluation (forward pass only)", flush=True)
    with _builder_patched(fields, "build_model"):
        return fields.main()


def lift_main():
    """Mesh-native lift (C_L rel-L2, rho_L) with the forward in bf16."""
    from evaluation import lift
    print("[cases.v] bf16 autocast evaluation (forward pass only)", flush=True)
    with _builder_patched(lift, "build_mesh_opformer_model"):
        return lift.main()
