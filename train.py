#!/usr/bin/env python
# ========================================================================================
# TRAINING ENTRY POINT   —   python train.py --cfg_path config/<cfg>.toml --out runs/<case>/
# ========================================================================================
"""Launcher only. The implementation lives in `training/`; the model in `arch/`.

    arch/         the model, and nothing else      ENTRY: arch.build_model(**cfg["train"]["model"])
    data/         dataset + per-mesh quantities    ENTRY: data.dataset.AirfransV39Dataset
    training/     losses, metrics, schedules, loops ENTRY: training.loop.main   (this file)
    evaluation/   metrics for a checkpoint         ENTRY: eval_fields.py, eval_lift.py
    diagnostics/  one-off checks
    config/       the TOML
"""
import os as _o, sys as _s
_ROOT = _o.path.dirname(_o.path.abspath(__file__))
for _d in (_o.path.join(_ROOT, "shared"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)

from training.loop import main

if __name__ == "__main__":
    main()
