"""
hodge_decoder.py
================
A **Hodge decoder** — it produces the final per-vertex output fields by decoding the encoder latent
*through the discrete Hodge–Laplacian* (the encoder's core operator), instead of a plain pointwise MLP
or a cross-field self-attention head.

Insight reused from the encoder
-------------------------------
The PSD-Hodge encoder evolves the vertex stream by the discrete Hodge–Laplacian
``L₀ = ⋆₀⁻¹ · d₀ᵀ · ⋆₁ · d₀`` with learned **symmetric-positive-definite Gaussian (RBF) Hodge stars**
(tied query=key, ``⋆ = (1−β)I + βŜ``, ``Ŝ = D^{-1/2} W D^{-1/2}``, ``β = ½·sigmoid(θ) ∈ (0,½)`` → SPD).
This decoder applies the *same* operator as a geometry-aware **decode refinement** of the latent, so the
predicted fields inherit the mesh's discrete differential structure (a smooth, geometry-consistent
solution rather than an unconstrained pointwise regression), and only then reads out the fields.

It is self-contained (the ``psd_star`` helper and the Laplace refinement live here) so the package has no
dependency on any experiment/case file. Purely **additive** — it does not touch any existing module.

Interface
---------
``HodgeDecoder(d, d_e, out_channels=4, …).forward(h, x_e, d_0, v_idx, e_idx) -> (B, N, out_channels)``
where ``h`` is the encoder's per-vertex latent, ``x_e`` the edge-stream latent (feeds the ⋆₁ kernel),
``d_0`` the vertex→edge incidence (sparse), and ``v_idx``/``e_idx`` the neighbour-window supports.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mesh_opformer.layers.transformer import init_layer__transformer

_EPS = 1e-8
_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


def _gather(x, idx):
    """x:(b,h,N,d), idx:(N,W) long (−1 pad) → (b,h,N,W,d) via advanced indexing on the N axis."""
    safe = idx.clamp(min=0)
    return x[:, :, safe, :]


def psd_star(x, phi, idx, log_sigma, theta_beta):
    """One symmetric-PSD Gaussian Hodge star applied to values ``x`` (identical to the encoder's).

    x         : (b,h,N,d)  values to smooth (a k-form on the N k-cells).
    phi       : (b,h,N,p)  tied query=key kernel features.
    idx       : (N,W)      neighbour indices, −1 padded; self included (ring 0) so w_ii = 1.
    log_sigma : (h,)       per-head log bandwidth  (σ = softplus(log_sigma)).
    theta_beta: (h,)       per-head mixing logit   (β = ½·sigmoid(θ) ∈ (0,½) → SPD).
    """
    b, h, N, d = x.shape
    W = idx.shape[1]
    pad = (idx < 0).view(1, 1, N, W)

    phi_j = _gather(phi, idx)                                     # (b,h,N,W,p)
    sq = ((phi.unsqueeze(3) - phi_j) ** 2).sum(-1)               # (b,h,N,W)  ‖φ_i−φ_j‖²
    sigma2 = (F.softplus(log_sigma) + _EPS) ** 2                  # (h,)
    w = torch.exp(-sq / (2.0 * sigma2.view(1, h, 1, 1)))         # RBF, in (0,1]
    w = w.masked_fill(pad, 0.0)

    D = w.sum(-1)                                                 # (b,h,N)   degrees (incl self)
    D_j = _gather(D.unsqueeze(-1), idx).squeeze(-1)              # (b,h,N,W)
    Shat = w / torch.sqrt(D.unsqueeze(-1) * D_j + _EPS)          # D^{-1/2} W D^{-1/2}

    x_j = _gather(x, idx)                                         # (b,h,N,W,d)
    agg = (Shat.unsqueeze(-1) * x_j).sum(3)                      # Ŝ x

    beta = 0.5 * torch.sigmoid(theta_beta).view(1, h, 1, 1)      # ∈ (0,½)
    return (1.0 - beta) * x + beta * agg                         # (1−β)x + βŜx  (SPD)


class HodgeLaplaceRefine(nn.Module):
    """One discrete Hodge–Laplacian refinement of a vertex stream:
    ``x_v ↦ ⋆₀⁻¹ · d₀ᵀ · ⋆₁ · d₀ · x_v`` with learned symmetric-PSD Gaussian stars.

    This is the encoder's vertex operator reused as a decode step. The two ``d₀`` lifts are exact
    (sparse, per head, fp32); the two stars (``⋆₁`` on edges, ``⋆₀⁻¹`` on vertices) are learned RBF
    diffusions. The edge kernel features come from the edge-stream latent ``x_e``.
    """

    def __init__(self, d_v, d_e, h=4, p_head=None, dropout=0.1):
        super().__init__()
        assert d_v % h == 0, "d_v must be divisible by heads"
        self.h = h
        self.dh = d_v // h
        self.p = p_head or self.dh
        self.v_phi_inv = nn.Linear(d_v, h * self.p, bias=False)   # ⋆₀⁻¹ vertex feature
        self.e_phi = nn.Linear(d_e, h * self.p, bias=False)       # ⋆₁ edge feature
        self.v_V = nn.Linear(d_v, h * self.dh, bias=False)
        self.linear_out = nn.Linear(d_v, d_v)
        self.e_log_sigma = nn.Parameter(torch.full((h,), 0.5413))  # σ ≈ 1
        self.v_log_sigma = nn.Parameter(torch.full((h,), 0.5413))
        self.e_theta_beta = nn.Parameter(torch.full((h,), -0.85))
        self.v_theta_beta = nn.Parameter(torch.full((h,), -0.85))
        self.drop = nn.Dropout(dropout)

    def _heads(self, lin, x, dim):
        b, n, _ = x.shape
        return lin(x).view(b, n, self.h, dim).transpose(1, 2)     # (b,h,n,dim)

    def forward(self, x_v, x_e, d_0, v_idx, e_idx):
        h, dh = self.h, self.dh
        if v_idx.dim() == 3:
            v_idx = v_idx[0]
        if e_idx.dim() == 3:
            e_idx = e_idx[0]
        v_V = self._heads(self.v_V, x_v, dh)                      # (b,h,N_v,dh)
        e_phi = self._heads(self.e_phi, x_e, self.p)             # (b,h,N_e,p)
        v_phi_inv = self._heads(self.v_phi_inv, x_v, self.p)     # (b,h,N_v,p)

        _x = v_V
        # d₀ : lift 0-form (vertices) -> 1-form (edges), exact sparse op, per head, fp32.
        with torch.autocast(_x.device.type, dtype=torch.bfloat16, enabled=False):
            _x = _x.float()
            _x = torch.stack([torch.bmm(d_0, _x[:, i, ...]) for i in range(h)], dim=1)
        # ⋆₁ : learned symmetric-PSD edge star.
        _x = psd_star(_x, e_phi, e_idx, self.e_log_sigma, self.e_theta_beta)
        # d₀ᵀ : co-boundary back to vertices.
        with torch.autocast(_x.device.type, dtype=torch.bfloat16, enabled=False):
            _x = _x.float()
            _x = torch.stack([torch.bmm(d_0.transpose(-2, -1), _x[:, i, ...]) for i in range(h)], dim=1)
        # ⋆₀⁻¹ : learned symmetric-PSD vertex star (own feature φ').
        _x = psd_star(_x, v_phi_inv, v_idx, self.v_log_sigma, self.v_theta_beta)

        out = _x.transpose(1, 2).contiguous().view(x_v.size(0), x_v.size(1), h * dh)
        return self.linear_out(self.drop(out))


class _CrossFieldBlock(nn.Module):
    """Pre-norm multi-head self-attention over the field-token axis + position-wise FFN.

    Optional field-coupling stage: the ``out_channels`` fields (u_x, u_y, p, ν_t on the RANS task) are
    physically coupled, so after the geometry-aware Hodge refinement we let their per-vertex tokens
    attend to each other before read-out.
    """

    def __init__(self, d, heads, d_hidden, dropout=0.1, activation="gelu"):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        A = _ACT[activation]
        self.ff = nn.Sequential(nn.Linear(d, d_hidden), A(), nn.Linear(d_hidden, d))

    def forward(self, x):                                         # x: (B*N, C, d)
        y = self.n1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.ff(self.n2(x))
        return x


class HodgeDecoder(nn.Module):
    """Decode the encoder latent into ``out_channels`` per-vertex fields through the Hodge–Laplacian.

    Pipeline: ``n_hodge`` pre-norm-residual Hodge–Laplacian refinements of the vertex latent
    (geometry-aware spatial coupling, reusing the encoder's SPD-Hodge operator) → per-vertex field
    tokens (latent + learned per-field embedding) → optional cross-field attention (``cross_field``)
    → shared read-out MLP → ``(B, N, out_channels)``.

    Parameters
    ----------
    d          : encoder latent width (vertex stream).
    d_e        : edge-stream latent width (feeds the ⋆₁ kernel features).
    out_channels : number of output fields.
    n_hodge    : number of Hodge–Laplacian decode refinements.
    h          : heads inside each Hodge star.
    cross_field: if True, add one cross-field attention block over the field tokens.
    dec_heads / d_hidden : cross-field attention heads / FFN width (used iff cross_field).
    readout_hidden : hidden width of the shared per-field read-out MLP.
    """

    def __init__(self, d, d_e, out_channels=4, n_hodge=2, h=4, dropout=0.1, activation="gelu",
                 cross_field=True, dec_heads=4, d_hidden=256, readout_hidden=64):
        super().__init__()
        self.C = out_channels
        self.d = d
        self.refine = nn.ModuleList([HodgeLaplaceRefine(d, d_e, h=h, dropout=dropout)
                                     for _ in range(n_hodge)])
        self.refine_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_hodge)])
        self.field_emb = nn.Parameter(torch.empty(out_channels, d))
        nn.init.normal_(self.field_emb, std=0.02)                # distinct field identities
        self.cross_field = cross_field
        if cross_field:
            self.cross = _CrossFieldBlock(d, dec_heads, d_hidden, dropout, activation)
        self.norm = nn.LayerNorm(d)
        A = _ACT[activation]
        self.readout = nn.Sequential(nn.Linear(d, readout_hidden), A(),
                                     nn.Linear(readout_hidden, 1))  # shared across field tokens

    def forward(self, h, x_e, d_0, v_idx, e_idx):                # h: (B, N, d)
        for blk, nrm in zip(self.refine, self.refine_norm):
            h = h + blk(nrm(h), x_e, d_0, v_idx, e_idx)          # pre-norm residual Hodge refinement
        B, N, d = h.shape
        tok = h.unsqueeze(2) + self.field_emb.view(1, 1, self.C, d)   # (B, N, C, d)
        if self.cross_field:
            tok = tok.reshape(B * N, self.C, d)
            tok = self.cross(tok)
            tok = tok.reshape(B, N, self.C, d)
        tok = self.norm(tok)
        return self.readout(tok).squeeze(-1)                     # (B, N, C)


class HodgeToTransformerDecoder(nn.Module):
    """**Hodge → linear-transformer** decode pipeline: *smooth (variational) → sharpen (precise)*.

    Rationale
    ---------
    The Hodge–Laplacian refinement ``⋆₀⁻¹ d₀ᵀ ⋆₁ d₀`` is a **diffusion**: it minimises a Dirichlet
    energy, so it yields a smooth, geometry-consistent field but **over-smooths** the sharp structures
    that dominate the error (suction-peak pressure, wake ν_t) — exactly the failure of the pure
    ``HodgeDecoder``. We therefore split the decode into two stages, moving from a *variational* to a
    *precise* solution:

        h → [ n_hodge Hodge–Laplacian refinements ]              # VARIATIONAL — smooth geometry-aware prior
          → [ n_transformer_dec global LINEAR-transformer blocks ]  # PRECISE — data-driven residual that
                                                                    #   restores the high-frequency detail
                                                                    #   the diffusion washed out
          → per-vertex field tokens (+ optional cross-field attention) → shared read-out → (B, N, C)

    Stage 1 (the encoder's SPD-Hodge operator, reused) produces a physically-consistent low-frequency
    field; stage 2 is the **same linear transformer the encoder uses**
    (``attn_type='full-linear-fast-transformers'``), here as a decode-side correction that carries the
    smooth prior toward a precise field. Purely additive — ``HodgeDecoder`` is untouched.

    Parameters (differences from ``HodgeDecoder``)
    ----------
    n_transformer_dec : number of global linear-transformer decode blocks (the sharpening stage).
    attn_basic_kw     : attention kwargs for those blocks; defaults to the encoder's linear-attention
                        config (``{'h': dec_heads, 'attn_type': 'full-linear-fast-transformers'}``).
    """

    def __init__(self, d, d_e, out_channels=4, n_hodge=2, h=4,
                 n_transformer_dec=2, attn_basic_kw=None, d_hidden=256,
                 dropout=0.1, activation="gelu",
                 cross_field=True, dec_heads=4, readout_hidden=64):
        super().__init__()
        self.C = out_channels
        self.d = d
        # Stage 1 — Hodge–Laplacian smoothing (encoder operator reused), pre-norm residual.
        self.refine = nn.ModuleList([HodgeLaplaceRefine(d, d_e, h=h, dropout=dropout)
                                     for _ in range(n_hodge)])
        self.refine_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_hodge)])
        # Stage 2 — global linear-transformer decode (sharpening / precision correction). Matches the
        # encoder's linear attention by default so the decode operator == the encode operator.
        attn_basic_kw = dict(attn_basic_kw or {"h": dec_heads,
                                               "attn_type": "full-linear-fast-transformers"})
        self.transformers = nn.ModuleList([
            init_layer__transformer(d=d, d_hidden=d_hidden, dropout=dropout, attn_kw=attn_basic_kw)
            for _ in range(int(n_transformer_dec))])
        self.mid_norm = nn.LayerNorm(d)
        # Field read-out (+ optional cross-field coupling) — identical head to HodgeDecoder.
        self.field_emb = nn.Parameter(torch.empty(out_channels, d))
        nn.init.normal_(self.field_emb, std=0.02)                # distinct field identities
        self.cross_field = cross_field
        if cross_field:
            self.cross = _CrossFieldBlock(d, dec_heads, d_hidden, dropout, activation)
        self.norm = nn.LayerNorm(d)
        A = _ACT[activation]
        self.readout = nn.Sequential(nn.Linear(d, readout_hidden), A(),
                                     nn.Linear(readout_hidden, 1))  # shared across field tokens

    def forward(self, h, x_e, d_0, v_idx, e_idx):                # h: (B, N, d)
        # Stage 1 — Hodge smoothing (variational, geometry-aware).
        for blk, nrm in zip(self.refine, self.refine_norm):
            h = h + blk(nrm(h), x_e, d_0, v_idx, e_idx)          # pre-norm residual Hodge refinement
        # Stage 2 — linear-transformer sharpening (global, over vertices; same op as the encoder).
        for tf in self.transformers:
            h = tf(h, idx=v_idx, mask=None)
        h = self.mid_norm(h)
        # Field read-out.
        B, N, d = h.shape
        tok = h.unsqueeze(2) + self.field_emb.view(1, 1, self.C, d)   # (B, N, C, d)
        if self.cross_field:
            tok = tok.reshape(B * N, self.C, d)
            tok = self.cross(tok)
            tok = tok.reshape(B, N, self.C, d)
        tok = self.norm(tok)
        return self.readout(tok).squeeze(-1)                     # (B, N, C)
