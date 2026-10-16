#!/usr/bin/env python
# ========================================================================================
# EVALUATION ENTRY POINT   —   mesh-native C_L and rho_L
# ========================================================================================
"""Launcher only; implementation in `evaluation/lift.py`."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))

from evaluation.lift import main

if __name__ == "__main__":
    main()
