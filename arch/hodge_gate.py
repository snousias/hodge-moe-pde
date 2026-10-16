# ========================================================================================
# COMPONENT of arch/ — the spectral controller. Built by arch.model, not used directly.
# ========================================================================================
#!/usr/bin/env python
"""v57 — the Hodge operator as a SPECTRAL CONTROLLER for the vanilla experts.

Every attempt to use the discrete Hodge operator as the *fitting* engine has lost: on matched
settings a 309k four-layer softmax transformer beats every HodgeFormer variant on every metric, and
the mechanism is understood — `d_0^T *_1 d_0` is the Dirichlet-energy Laplacian, so the layer carries
a smoothing / low-graph-frequency prior that mis-specifies sharp near-wall RANS structure.

v57 turns that liability into the job it is actually suited for. A diffusion operator is a natural
*frequency analyser*: applying it tells you how much high-frequency content sits at a vertex. So:

    x_v ──┬─► HODGE BRANCH   2 PSD-Hodge layers → read-out → per-node logits over K experts
          │                   (this decides WHICH SPECTRAL BAND each vertex should be served by)
          │                                   │
          │                                   ▼  gate g(v) ∈ simplex^K
          └─► EXPERT BRANCH  K vanilla transformers, band-specialised ──► Σ_k g_k · proj_k(z_k)

Both branches read the SAME input. The Hodge side never predicts the field; it only routes. The
vanilla side never sees the mesh; it only fits. That is the division of labour the measurements have
been pointing at all along.

Contrast with v53–v56, where the gate was a small MLP on the vanilla latent — a gate with no
geometric information deciding which *geometric* frequency band a vertex needs.
"""
import torch
import torch.nn as nn

from mesh_opformer.layers.embedding import NeighborEmbedding
from arch.psd_hodge import PSDHodgeLayer


class HodgeGate(nn.Module):
    """Two PSD-Hodge layers → per-node gate logits over the K experts.

    Output is logits, not a field: the branch is a router. It sees the full simplicial complex
    (vertex/edge/face streams, incidence, neighbour supports) — everything the expert branch does
    not — and its only channel of influence is the K-way weight it assigns per vertex.
    """

    def __init__(self, v_in, e_in, f_in, n_experts, d=64, d_hidden=256, h=4, n_layers=2,
                 dropout=0.1, embed_kw=None, readout_hidden=64, activation="gelu", window=64,
                 modes="v", window_e=None):
        super().__init__()
        A = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        self.embed = NeighborEmbedding(v_in=v_in, e_in=e_in, f_in=f_in, d_v=d, d_e=d, d_f=d,
                                       **(embed_kw or {}))
        self.hodge = nn.ModuleList([
            PSDHodgeLayer(d_v=d, d_e=d, d_f=d, d_hidden=d_hidden, h=h, dropout=dropout,
                          modes=modes)
            for _ in range(int(n_layers))])
        self.norm = nn.LayerNorm(d)
        self.readout = nn.Sequential(nn.Linear(d, readout_hidden), A(),
                                     nn.Linear(readout_hidden, int(n_experts)))
        # The cached supports are 256 wide, and those gathers dominate memory: at full width the
        # model needs ~19.9 GB to train. The controller only has to decide WHICH BAND a vertex
        # needs, not reconstruct a field, so it runs on the nearest `window` neighbours. The
        # geodesic rings are ordered nearest-first, so this is a truncation to the inner rings.
        self.window = int(window)
        # The edge support is ~3x as long as the vertex support, and modes with "e"/"f" run stars
        # over it; `window_e` truncates it independently so the edge operators stay affordable.
        self.window_e = int(window_e) if window_e is not None else int(window)
        self.modes = str(modes)

    def forward(self, x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx, mask=None):
        xv, xe, xf = self.embed(x_v, x_e, x_f, d_0, d_1)
        w = self.window
        vi = v_idx[..., :w] if v_idx is not None and v_idx.shape[-1] > w else v_idx
        we = self.window_e
        ei = e_idx[..., :we] if e_idx is not None and e_idx.shape[-1] > we else e_idx
        for lyr in self.hodge:
            xv, xe, xf = lyr(xv, xe, xf, d_0, d_1, vi, ei, f_idx, mask)
        return self.readout(self.norm(xv))                     # (B, N, K) logits
