# ========================================================================================
# COMPONENT of arch/ — the Hodge-Laplacian attention layer used by the controller.
# ========================================================================================
#!/usr/bin/env python
"""
v34 — SELF-CONTAINED PSD Hodge attention (Gaussian-kernel Hodge stars).

Design: runs/v34/scenario/scenario.md. Goal: make the learned Hodge stars ⋆₀, ⋆₁ genuine
symmetric positive-semidefinite (PSD) operators, so that L₀ = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀ is a real discrete
Hodge–Laplacian (real, non-negative spectrum) instead of the row-stochastic attention approximation
in the stock kernel (`sparse_attention`, dot-product + row-softmax → asymmetric, non-PSD).

Nothing here touches the shared `mesh_opformer` package, so a concurrently-training v33 is unaffected.

The star (Variant A — symmetric-normalized Gaussian diffusion):
    Q = K  ->  a single per-node kernel feature  φ_i = W_φ x_i          (tied query/key)
    w_ij   = exp( −‖φ_i − φ_j‖² / (2 σ²) )   over the neighbour support   (RBF = PD kernel)
    Ŝ      = D^{-1/2} W D^{-1/2},   D_i = Σ_j w_ij                        (symmetric normalization)
    ⋆ x    = (1−β) x + β Ŝ x,      β = ½·sigmoid(θ) ∈ (0, ½)             (PSD: eig ∈ [1−2β, 1] > 0)

`Ŝ` is symmetric (given a symmetric neighbour support, which the v34 dataset guarantees), with
eigenvalues in [−1, 1]; so ⋆ is symmetric with eigenvalues in [1−2β, 1] ⊂ (0, 1] — SPD for β < ½.
No softmax (softmax would re-introduce row-stochastic asymmetry).

⋆₀⁻¹ gets its OWN kernel feature φ' (a separate SPD star, not the literal inverse — see §2/§4 of the
scenario: only SPD-ness of the two stars is needed for L₀ to have a real non-negative spectrum).
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


def _gather(x, idx):
    """x:(b,h,N,d), idx:(N,W) long (−1 pad) → (b,h,N,W,d) via advanced indexing on the N axis."""
    safe = idx.clamp(min=0)                       # (N,W); pad handled by the caller's weight mask
    return x[:, :, safe, :]                        # (b,h,N,W,d)


def sparse_apply(op, z):
    """Apply an incidence operator `op` (b,M,N) to per-head values `z` (b,h,N,d) → (b,h,M,d), fp32.

    The stock formulation loops `torch.bmm(op, z[:, k])` over heads — h separate sparse kernels per
    lift. modes="vef" issues six lifts per layer (d₀ᵀ, d₀, d₁, d₁ᵀ for edges; d₁ᵀ, d₁ for faces), and
    measured, that loop was the single largest cost in the step. Flattening the head axis into the
    column axis turns it into ONE `torch.sparse.mm` over (N, h·d): identical arithmetic (checked
    allclose, max |Δ| 2.4e-6) and ~50x faster on these shapes (3.63 ms → 0.07 ms at N_e=13896, h=4).

    Kept in fp32 (vef_v2 trains in plain fp32, no autocast) — `bmm_sparse_cuda` has no bfloat16 kernel, and the incidence
    lifts are exact operators that should not be approximated anyway.
    """
    z = z.float()
    b, h, N, d = z.shape
    op2 = op[0] if op.dim() == 3 else op
    if b == 1 and op2.dim() == 2:                                  # batch_size=1 throughout
        flat = z[0].permute(1, 0, 2).reshape(N, h * d)             # (N, h·d)
        y = torch.sparse.mm(op2, flat) if op2.is_sparse else op2 @ flat
        return y.view(-1, h, d).permute(1, 0, 2).unsqueeze(0)
    return torch.stack([torch.bmm(op, z[:, k, ...]) for k in range(h)], dim=1)


def psd_diag_star(x, phi, log_sigma, theta_beta):
    """Apply one symmetric-PSD DIAGONAL Hodge star — the star for a cell type with no neighbour list.

    Needed for ⋆₂. The cached face support `f_idx` is ONE column wide (self only), and on a width-1
    support the Gaussian star above is provably the identity: w_ii = exp(0) = 1 ⇒ Ŝ = 1 ⇒
    (1−β)x + βx = x, for every φ. So a Gaussian ⋆₂ would silently erase the face stream — the star
    would not depend on its own kernel features at all.

    A diagonal star is the classical DEC form for ⋆₂ anyway (⋆₂ = 1/area per face), so the learned
    version is a per-face positive scaling — a learned area. Positivity is what makes it SPD:

        z_i = ‖φ_i‖² / (p σ²) ≥ 0        (tied Q=K, same exponent as the Gaussian kernel)
        λ_i = exp(−z_i) ∈ (0, 1]
        ⋆ x = (1−β) x + β λ ⊙ x,   β = ½·sigmoid(θ) ∈ (0,½)

    Eigenvalues are 1−β+βλ_i ∈ [1−β, 1] ⊂ (½, 1] — SPD, bounded, and ≈ identity at init, matching
    the Gaussian star's conditioning so the composed Laplacian starts in the same regime.
    """
    h, p = x.shape[1], phi.shape[-1]
    eps, pp = log_sigma.new_tensor(_EPS), log_sigma.new_tensor(float(p))   # vef_v2: no Python scalars on
    sigma2 = (F.softplus(log_sigma) + eps) ** 2                  # (h,)    the per-head params (torch 2.14)
    z = (phi ** 2).sum(-1) / (pp * sigma2.view(1, h, 1))         # (b,h,N)
    lam = torch.exp(-z).unsqueeze(-1)                            # (b,h,N,1) ∈ (0,1]
    # vef_v2: constants as 0-dim tensors on theta_beta's device, not Python floats. Under torch 2.14.1 a
    # Python scalar here becomes a CPU wrapped-number in the autograd graph and the backward engine
    # intermittently fails with "Event device type CUDA does not match blocking stream's device type CPU"
    # (caught by anomaly detection at this MulBackward0). Same values, same arithmetic.
    half, one = theta_beta.new_tensor(0.5), theta_beta.new_tensor(1.0)
    beta = half * torch.sigmoid(theta_beta).view(1, h, 1, 1)
    return (one - beta) * x + beta * lam * x


def psd_star(x, phi, idx, log_sigma, theta_beta):
    """Apply one symmetric-PSD Gaussian Hodge star to values `x`.

    Falls back to `psd_diag_star` when the support is a single column (self only), where the
    Gaussian form is exactly the identity and would drop `phi` — see that function.

    x         : (b,h,N,d)  values to smooth (a k-form on the N k-cells).
    phi       : (b,h,N,p)  tied query=key kernel features.
    idx       : (N,W)      neighbour indices, −1 padded; self MUST be included (ring 0) so w_ii=1.
    log_sigma : (h,)       per-head log bandwidth  (σ = softplus(log_sigma)).
    theta_beta: (h,)       per-head mixing logit   (β = ½·sigmoid(θ) ∈ (0,½) → SPD).
    """
    b, h, N, d = x.shape
    W = idx.shape[1]
    if W < 2:
        return psd_diag_star(x, phi, log_sigma, theta_beta)
    pad = (idx < 0).view(1, 1, N, W)                              # (1,1,N,W)

    phi_j = _gather(phi, idx)                                     # (b,h,N,W,p)
    sq = ((phi.unsqueeze(3) - phi_j) ** 2).sum(-1)               # (b,h,N,W)  ‖φ_i−φ_j‖²
    eps, two = log_sigma.new_tensor(_EPS), log_sigma.new_tensor(2.0)   # vef_v2: see psd_diag_star
    sigma2 = (F.softplus(log_sigma) + eps) ** 2                  # (h,)
    w = torch.exp(-sq / (two * sigma2.view(1, h, 1, 1)))         # (b,h,N,W)  RBF, in (0,1]
    w = w.masked_fill(pad, 0.0)

    D = w.sum(-1)                                                 # (b,h,N)   row sums (incl self)
    D_j = _gather(D.unsqueeze(-1), idx).squeeze(-1)              # (b,h,N,W)  neighbour degrees
    Shat = w / torch.sqrt(D.unsqueeze(-1) * D_j + eps)          # (b,h,N,W)  D^{-1/2} W D^{-1/2}

    x_j = _gather(x, idx)                                         # (b,h,N,W,d)
    agg = (Shat.unsqueeze(-1) * x_j).sum(3)                      # (b,h,N,d)   Ŝ x

    half, one = theta_beta.new_tensor(0.5), theta_beta.new_tensor(1.0)   # vef_v2: see psd_diag_star
    beta = half * torch.sigmoid(theta_beta).view(1, h, 1, 1)     # (1,h,1,1) ∈ (0,½)
    return (one - beta) * x + beta * agg                         # (1−β)x + βŜx  (SPD star)


class PSDHodgeAttentionVertices(nn.Module):
    """Vertex-mode PSD Hodge-Laplacian: x_v ↦ ⋆₀⁻¹ · d₀ᵀ · ⋆₁ · d₀ · x_v with Gaussian PSD stars.

    Mirrors the stock `mesh_attention_v` structure (the two d₀ lifts are exact; only the two stars
    are learned), but each star is the symmetric-PSD Gaussian diffusion above with tied Q=K.
    """
    def __init__(self, d_v, d_e, h=4, p_head=None, dropout=0.1):
        super().__init__()
        assert d_v % h == 0, "d_v must be divisible by heads"
        self.h = h
        self.dh = d_v // h                                        # per-head value dim
        self.p = p_head or self.dh                                # per-head kernel-feature dim
        # tied query=key kernel features (φ), one per stream; plus the vertex value projection.
        self.v_phi = nn.Linear(d_v, h * self.p, bias=False)      # ⋆₀⁻¹ vertex feature
        self.v_phi_inv = nn.Linear(d_v, h * self.p, bias=False)  # separate feature for the inverse star
        self.e_phi = nn.Linear(d_e, h * self.p, bias=False)      # ⋆₁   edge feature
        self.v_V = nn.Linear(d_v, h * self.dh, bias=False)
        self.linear_out = nn.Linear(d_v, d_v)
        # per-head bandwidths σ and mixing β (learnable). σ init ~1 (softplus(0.54)≈1); β init ≈0.30.
        self.e_log_sigma = nn.Parameter(torch.full((h,), 0.5413))
        self.v_log_sigma = nn.Parameter(torch.full((h,), 0.5413))
        self.e_theta_beta = nn.Parameter(torch.full((h,), -0.85))  # ½·sigmoid(−0.85)≈0.15... set ≈0.30
        self.v_theta_beta = nn.Parameter(torch.full((h,), -0.85))
        self.drop = nn.Dropout(dropout)

    def _heads(self, lin, x, dim):
        b, n, _ = x.shape
        return lin(x).view(b, n, self.h, dim).transpose(1, 2)    # (b,h,n,dim)

    def forward(self, x_v, x_e, d_0, v_idx, e_idx):
        h, dh = self.h, self.dh
        # collate adds a batch axis: idx is (B,N,W). This project is batch_size=1 throughout, so
        # use the single mesh's (N,W) support for the sparse gather.
        if v_idx.dim() == 3:
            v_idx = v_idx[0]
        if e_idx.dim() == 3:
            e_idx = e_idx[0]
        v_V = self._heads(self.v_V, x_v, dh)                      # (b,h,N_v,dh)
        e_phi = self._heads(self.e_phi, x_e, self.p)             # (b,h,N_e,p)
        v_phi_inv = self._heads(self.v_phi_inv, x_v, self.p)     # (b,h,N_v,p)

        _x = v_V
        # d₀ : lift 0-form (vertices) to 1-form (edges) — exact sparse op, fp32.
        _x = sparse_apply(d_0, _x)
        # ⋆₁ : learned symmetric-PSD edge star.
        _x = psd_star(_x, e_phi, e_idx, self.e_log_sigma, self.e_theta_beta)
        # d₀ᵀ : co-boundary back to vertices.
        _x = sparse_apply(d_0.transpose(-2, -1), _x)
        # ⋆₀⁻¹ : learned symmetric-PSD vertex star (own feature φ').
        _x = psd_star(_x, v_phi_inv, v_idx, self.v_log_sigma, self.v_theta_beta)

        out = _x.transpose(1, 2).contiguous().view(x_v.size(0), x_v.size(1), h * dh)
        return self.linear_out(self.drop(out))


class PSDHodgeAttentionEdges(nn.Module):
    """Edge-mode PSD Hodge-Laplacian on 1-forms:  x_e ↦ d₀ ⋆₀⁻¹ d₀ᵀ ⋆₁ x_e  +  ⋆₁⁻¹ d₁ᵀ ⋆₂ d₁ x_e.

    This is the DOWN + UP decomposition L₁ = L₁ᵈᵒʷⁿ + L₁ᵘᵖ, and it is the same operator sequence the
    stock `mesh_attention_e` builds (attention_hodge_laplace.py: `_y` runs ⋆₁ → d₀ᵀ → ⋆₀⁻¹ → d₀,
    `_x` runs d₁ → ⋆₂ → d₁ᵀ → ⋆₁⁻¹, and it returns `_y + _x`). Only the stars differ: each is the
    symmetric-PSD Gaussian diffusion of `psd_star` rather than a row-softmax attention, so the
    composed operator keeps a real, non-negative spectrum.

    Four stars ⇒ four kernel features: ⋆₁ (edges, down), ⋆₀⁻¹ (vertices), ⋆₂ (faces), ⋆₁⁻¹ (edges,
    up). Each carries its own σ and β, so the up and down halves can weight themselves differently.
    """
    def __init__(self, d_v, d_e, d_f, h=4, p_head=None, dropout=0.1):
        super().__init__()
        assert d_e % h == 0, "d_e must be divisible by heads"
        self.h = h
        self.dh = d_e // h
        self.p = p_head or self.dh
        self.e_V = nn.Linear(d_e, h * self.dh, bias=False)
        self.e_phi = nn.Linear(d_e, h * self.p, bias=False)        # ⋆₁    (down, first)
        self.v_phi_inv = nn.Linear(d_v, h * self.p, bias=False)    # ⋆₀⁻¹  (down)
        self.f_phi = nn.Linear(d_f, h * self.p, bias=False)        # ⋆₂    (up)
        self.e_phi_inv = nn.Linear(d_e, h * self.p, bias=False)    # ⋆₁⁻¹  (up)
        self.linear_out = nn.Linear(d_e, d_e)
        for nm in ("e", "v", "f", "ei"):
            self.register_parameter(f"{nm}_log_sigma", nn.Parameter(torch.full((h,), 0.5413)))
            self.register_parameter(f"{nm}_theta_beta", nn.Parameter(torch.full((h,), -0.85)))
        self.drop = nn.Dropout(dropout)

    def _heads(self, lin, x, dim):
        b, n, _ = x.shape
        return lin(x).view(b, n, self.h, dim).transpose(1, 2)

    def forward(self, x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx, mask=None):
        h = self.h
        if v_idx is not None and v_idx.dim() == 3:
            v_idx = v_idx[0]
        if e_idx is not None and e_idx.dim() == 3:
            e_idx = e_idx[0]
        if f_idx is not None and f_idx.dim() == 3:
            f_idx = f_idx[0]

        e_V = self._heads(self.e_V, x_e, self.dh)                     # (b,h,N_e,dh)
        e_phi = self._heads(self.e_phi, x_e, self.p)
        v_phi_inv = self._heads(self.v_phi_inv, x_v, self.p)
        f_phi = self._heads(self.f_phi, x_f, self.p)
        e_phi_inv = self._heads(self.e_phi_inv, x_e, self.p)

        _bmm = sparse_apply                                           # exact sparse lift, fp32

        # DOWN: ⋆₁ → d₀ᵀ (edges→vertices) → ⋆₀⁻¹ → d₀ (vertices→edges)
        _y = psd_star(e_V, e_phi, e_idx, self.e_log_sigma, self.e_theta_beta)
        _y = _bmm(d_0.transpose(-2, -1), _y)
        _y = psd_star(_y, v_phi_inv, v_idx, self.v_log_sigma, self.v_theta_beta)
        _y = _bmm(d_0, _y)
        # UP: d₁ (edges→faces) → ⋆₂ → d₁ᵀ (faces→edges) → ⋆₁⁻¹
        _x = _bmm(d_1, e_V)
        _x = psd_star(_x, f_phi, f_idx, self.f_log_sigma, self.f_theta_beta)
        _x = _bmm(d_1.transpose(-2, -1), _x)
        _x = psd_star(_x, e_phi_inv, e_idx, self.ei_log_sigma, self.ei_theta_beta)

        out = (_y + _x).transpose(1, 2).contiguous().view(x_e.size(0), x_e.size(1), h * self.dh)
        return self.linear_out(self.drop(out))


class PSDHodgeAttentionFaces(nn.Module):
    """Face-mode PSD Hodge-Laplacian on 2-forms:  x_f ↦ d₁ ⋆₁⁻¹ d₁ᵀ ⋆₂ x_f.

    The down-Laplacian L₂ᵈᵒʷⁿ — the only one that exists on a 2-complex, since there are no 3-cells.
    Same sequence as the stock `mesh_attention_f` (⋆₂ → d₁ᵀ → ⋆₁⁻¹ → d₁), PSD stars substituted.
    """
    def __init__(self, d_e, d_f, h=4, p_head=None, dropout=0.1):
        super().__init__()
        assert d_f % h == 0, "d_f must be divisible by heads"
        self.h = h
        self.dh = d_f // h
        self.p = p_head or self.dh
        self.f_V = nn.Linear(d_f, h * self.dh, bias=False)
        self.f_phi = nn.Linear(d_f, h * self.p, bias=False)        # ⋆₂
        self.e_phi_inv = nn.Linear(d_e, h * self.p, bias=False)    # ⋆₁⁻¹
        self.linear_out = nn.Linear(d_f, d_f)
        self.f_log_sigma = nn.Parameter(torch.full((h,), 0.5413))
        self.e_log_sigma = nn.Parameter(torch.full((h,), 0.5413))
        self.f_theta_beta = nn.Parameter(torch.full((h,), -0.85))
        self.e_theta_beta = nn.Parameter(torch.full((h,), -0.85))
        self.drop = nn.Dropout(dropout)

    def _heads(self, lin, x, dim):
        b, n, _ = x.shape
        return lin(x).view(b, n, self.h, dim).transpose(1, 2)

    def forward(self, x_e, x_f, d_1, e_idx, f_idx, mask=None):
        h = self.h
        if e_idx is not None and e_idx.dim() == 3:
            e_idx = e_idx[0]
        if f_idx is not None and f_idx.dim() == 3:
            f_idx = f_idx[0]
        f_V = self._heads(self.f_V, x_f, self.dh)                     # (b,h,N_f,dh)
        f_phi = self._heads(self.f_phi, x_f, self.p)
        e_phi_inv = self._heads(self.e_phi_inv, x_e, self.p)

        _bmm = sparse_apply

        _y = psd_star(f_V, f_phi, f_idx, self.f_log_sigma, self.f_theta_beta)
        _y = _bmm(d_1.transpose(-2, -1), _y)                          # faces → edges
        _y = psd_star(_y, e_phi_inv, e_idx, self.e_log_sigma, self.e_theta_beta)
        _y = _bmm(d_1, _y)                                            # edges → faces

        out = _y.transpose(1, 2).contiguous().view(x_f.size(0), x_f.size(1), h * self.dh)
        return self.linear_out(self.drop(out))


class PSDHodgeLayer(nn.Module):
    """Pre-norm residual block: PSD-Hodge attention on each stream in `modes` + position-wise FFNs.

    Matches the (x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx, mask) interface of the stock
    MeshHodgeFormerLayer so it drops into the v30/v23-style composition.

      modes='v'    only the vertex stream is evolved by a Hodge-Laplacian (L₀); the edge/face streams
                   get a light FFN refinement and feed the vertex operator through the edge star ⋆₁.
      modes='vef'  every stream is evolved by its own Hodge-Laplacian — L₀ on vertices, L₁ = down+up
                   on edges, L₂ᵈᵒʷⁿ on faces — exactly the mode structure of the original HodgeFormer
                   (`MeshHodgeFormerAttention`, which switches its v/e/f operators on `modes`).

    Any subset works ('ve', 'ef', …): a stream not named in `modes` keeps only its FFN.
    """
    def __init__(self, d_v, d_e, d_f, d_hidden, h=4, dropout=0.1, modes="v"):
        super().__init__()
        self.modes = str(modes)
        self.attn = (PSDHodgeAttentionVertices(d_v, d_e, h=h, dropout=dropout)
                     if "v" in self.modes else None)
        self.e_attn = (PSDHodgeAttentionEdges(d_v, d_e, d_f, h=h, dropout=dropout)
                       if "e" in self.modes else None)
        self.f_attn = (PSDHodgeAttentionFaces(d_e, d_f, h=h, dropout=dropout)
                       if "f" in self.modes else None)
        self.n_v1 = nn.LayerNorm(d_v); self.n_e1 = nn.LayerNorm(d_e); self.n_f1 = nn.LayerNorm(d_f)
        self.n_v2 = nn.LayerNorm(d_v); self.n_e2 = nn.LayerNorm(d_e); self.n_f2 = nn.LayerNorm(d_f)
        self.v_ff = self._ff(d_v, d_hidden)
        self.e_ff = self._ff(d_e, d_hidden)
        self.f_ff = self._ff(d_f, d_hidden)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _ff(d, d_hidden):
        return nn.Sequential(nn.Linear(d, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d))

    def forward(self, x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx, mask=None):
        # sublayer 1: PSD Hodge attention. All three operators read the SAME pre-update streams —
        # this is how the stock MeshHodgeFormerAttention composes its modes (v/e/f are computed from
        # x_v, x_e, x_f and returned together), so the layer stays a single simultaneous step rather
        # than a v→e→f cascade whose result would depend on stream ordering.
        nv, ne, nf = self.n_v1(x_v), self.n_e1(x_e), self.n_f1(x_f)
        if self.attn is not None:
            x_v = x_v + self.drop(self.attn(nv, ne, d_0, v_idx, e_idx))
        if self.e_attn is not None:
            x_e = x_e + self.drop(self.e_attn(nv, ne, nf, d_0, d_1, v_idx, e_idx, f_idx, mask))
        if self.f_attn is not None:
            x_f = x_f + self.drop(self.f_attn(ne, nf, d_1, e_idx, f_idx, mask))
        # sublayer 2: position-wise FFNs.
        x_v = x_v + self.drop(self.v_ff(self.n_v2(x_v)))
        x_e = x_e + self.drop(self.e_ff(self.n_e2(x_e)))
        x_f = x_f + self.drop(self.f_ff(self.n_f2(x_f)))
        return x_v, x_e, x_f
