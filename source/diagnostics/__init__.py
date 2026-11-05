"""DIAGNOSTICS — one-off checks. Never imported by training or evaluation.

    verify_vef.py        the face stream reaches the gate only under modes='vef'
    verify_analyzer.py   PSD / stability / completeness of the learned Hodge analyzer
    inspect_experts.py   did the experts specialise, did the gate collapse
    diagnose_attention.py  is the model actually using cross-vertex context
    check_psd.py         symmetry / spectrum of the learned Hodge stars
"""
