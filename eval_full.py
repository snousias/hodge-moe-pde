#!/usr/bin/env python
# ========================================================================================
# EVALUATION ENTRY POINT   —   every checkpoint × every test case: per-case metrics,
#                              qualitative figures for every case, overall metrics
# ========================================================================================
"""Launcher only; implementation in `source/evaluation/full.py`.

    python eval_full.py                                   # both released checkpoints (v, vef)
    python eval_full.py --variant vef --ckpt runs/train   # every *.pth in a training run
"""
import os as _o, sys as _s
_ROOT = _o.path.dirname(_o.path.abspath(__file__))
for _d in (_o.path.join(_ROOT, "source", "shared"), _o.path.join(_ROOT, "source"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)

from evaluation.full import main

if __name__ == "__main__":
    main()
