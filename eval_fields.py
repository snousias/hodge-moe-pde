#!/usr/bin/env python
# ========================================================================================
# EVALUATION ENTRY POINT   —   field metrics + qualitative figures
# ========================================================================================
"""Launcher only; implementation in `evaluation/fields.py`."""
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))

from evaluation.fields import main

if __name__ == "__main__":
    main()
