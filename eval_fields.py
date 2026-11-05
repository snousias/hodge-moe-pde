#!/usr/bin/env python
# ========================================================================================
# EVALUATION ENTRY POINT   —   field metrics + qualitative figures
# ========================================================================================
"""Launcher only; implementation in `evaluation/fields.py`."""
import os as _o, sys as _s
_ROOT = _o.path.dirname(_o.path.abspath(__file__))
for _d in (_o.path.join(_ROOT, "source", "shared"), _o.path.join(_ROOT, "source"), _ROOT):
    if _d not in _s.path:
        _s.path.insert(0, _d)

# The two cases evaluate at different precisions, so the evaluator is chosen from the
# variant before argument parsing. With no --variant this is the registry default, which
# keeps every pre-existing command line behaving as it did.
from cases import evaluator

if __name__ == "__main__":
    evaluator().fields_main()
