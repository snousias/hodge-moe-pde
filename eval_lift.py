#!/usr/bin/env python
# ========================================================================================
# EVALUATION ENTRY POINT   —   mesh-native C_L and rho_L
# ========================================================================================
"""Launcher only; implementation in `evaluation/lift.py`."""
import os as _o, sys as _s
_ROOT = _o.path.dirname(_o.path.abspath(__file__))
for _d in (_o.path.join(_ROOT, "source", "shared"), _o.path.join(_ROOT, "source"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)

# See eval_fields.py: the evaluator is chosen from the variant before argument parsing.
from cases import evaluator

if __name__ == "__main__":
    evaluator().lift_main()
