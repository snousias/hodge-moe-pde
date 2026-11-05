#!/usr/bin/env python
# ========================================================================================
# TRAINING ENTRY POINT, bf16   —   python train_v.py --variant v --out runs/train_v/
# ========================================================================================
"""Launcher for the bf16 trainer; implementation in `source/cases/v/trainer.py`.

`train.py` is the fp32 entry point and is unchanged: it trains the recipe that produced
`hodgemoe_vef_best_ep400.pth`. This one runs the same loop with the forward passes under
bf16 autocast, which is the recipe that produced `hodgemoe_v_best_ep380.pth`.

Use it for the vertex-only variant. `scripts/run_train.sh` picks the right entry point from
`variants.toml`, so `VARIANT=v bash scripts/run_train.sh` already comes here.

The architecture is the same either way — precision is not an architectural property, and both
checkpoints are fp32 on disk. What differs is the arithmetic the forward pass runs in. See
`source/cases/v/trainer.py` and `checkpoints/PROVENANCE.md`.
"""
import os as _o, sys as _s
_ROOT = _o.path.dirname(_o.path.abspath(__file__))
for _d in (_o.path.join(_ROOT, "source", "shared"), _o.path.join(_ROOT, "source"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)

from cases.v.trainer import main

if __name__ == "__main__":
    main()
