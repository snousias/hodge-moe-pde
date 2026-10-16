"""ARCHITECTURE — the model, and nothing else.

    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │  ENTRY POINT:   build_model(**cfg["train"]["model"])  ->  MoEModel               │
    │                 (alias of arch.model.build_v57_gated_hodge_vef_model)                │
    └──────────────────────────────────────────────────────────────────────────────────┘

Everything else in this package is a COMPONENT used by that builder, never called directly by
training or evaluation:

    model.py       MoEModel (the assembled network) + FieldAttnDecoder + the builder
    hodge_gate.py  HodgeGate      — the spectral controller: 2 PSD-Hodge layers -> gate logits
    experts.py     FrequencyMoE   — the K band-specialised vanilla transformers + the gate mixing,
                                    LearnableAssignment, VanillaTransformerBlock, FourierFeatures
    analyzer.py    LearnedHodgeAnalyzer — defines the frequency bands of the TARGET (supervision
                                    side; used by training/, not part of the forward pass)
    psd_hodge.py   PSDHodgeLayer  — the Hodge-Laplacian attention layer the controller is built from

This package imports NOTHING from data/, training/ or evaluation/. That is the invariant that keeps
the architecture reusable and separately testable; `diagnostics/verify.py` checks it.
"""

from .model import build_v57_gated_hodge_vef_model as build_model

# The one public name. Components are reachable as arch.model / arch.experts / ... for inspection,
# but the builder is the only supported way to construct the network.
__all__ = ["build_model"]
