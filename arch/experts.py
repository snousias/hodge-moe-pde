# ========================================================================================
# COMPONENT of arch/ — the frequency experts and the gate mixing. Built by arch.model.
# ========================================================================================
#!/usr/bin/env python
"""v57_gated_hodge_vef — frequency-banded mixture of experts (vanilla transformers only).

Motivation. The whole v50-v52 line is one undifferentiated stack that must represent the smooth
far-field and the sharp near-wall structure with the same weights. The measured error is wildly
uneven across that spectrum: nu_t sits at rel-L2 ~0.10 while u_x is ~0.006, and the suction peak /
boundary layer are exactly the high-graph-frequency content that the Hodge line's smoothing bias
mis-specified (docs/DISCUSSION.md 3.5). v57_gated_hodge_vef gives each frequency band its own expert.

Two pieces:

  1. BANDS — `band_decompose` splits the target on the MESH GRAPH using the symmetric normalised
     adjacency `S = D^-1/2 (A + I) D^-1/2`, a low-pass operator:

         b_0 = S^m0 y                     lowest  (far-field / bulk)
         b_k = S^mk y - S^m(k-1) y        middle bands
         b_K = y - S^mK y                 highest (boundary layer, suction peak, wake)

     with a decreasing smoothing ladder m0 > m1 > ... The bands sum to y EXACTLY, so the
     decomposition is lossless and the auxiliary loss cannot bias the total.

     Note the mesh is used only to DEFINE the target bands (a preprocessing step). The model itself
     never sees mesh structure — v57_gated_hodge_vef stays a pure point transformer like v50/v51/v52.

  2. EXPERTS — `FrequencyMoE`: K vanilla-transformer stacks, each fed Fourier positional features at
     its own spatial scale, plus a per-node softmax gate. Expert k is pinned to band k by an
     auxiliary head; the gated sum is what the field loss sees.

Why the gate AND the auxiliary loss: the gate alone gives a mixture with no reason for any expert to
specialise (experts collapse to near-identical functions), and the auxiliary loss alone gives a fixed
additive decomposition with no ability to vary the mix per node. Together the experts specialise and
the model still chooses how much of each band a given vertex needs.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from scipy.sparse import coo_matrix

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


# ── band decomposition ────────────────────────────────────────────────────────────────────
def smoothing_operator(faces, n_vert):
    """Symmetric normalised adjacency with self-loops, S = D^-1/2 (A + I) D^-1/2 (CSR).
    Spectrum in (-1, 1]; repeated application is a low-pass filter on the mesh graph."""
    f = np.asarray(faces, dtype="int64")
    e0 = np.concatenate([f[:, 0], f[:, 1], f[:, 2], np.arange(n_vert)])
    e1 = np.concatenate([f[:, 1], f[:, 2], f[:, 0], np.arange(n_vert)])
    A = coo_matrix((np.ones(e0.size), (e0, e1)), shape=(n_vert, n_vert)).tocsr()
    A = ((A + A.T) > 0).astype("float64")
    d = np.asarray(A.sum(axis=1)).ravel()
    dm = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return (A.multiply(dm.reshape(-1, 1))).multiply(dm.reshape(1, -1)).tocsr()


def band_decompose(faces, y, ladder=(2, 8, 32)):
    """Split y (N,C) into len(ladder)+1 frequency bands that sum to y EXACTLY.

    `ladder` = smoothing counts, ASCENDING. With (2, 8, 32):

        L2 = S^2 y,  L8 = S^8 y,  L32 = S^32 y
        b0 = L32            lowest frequency (bulk / far field)
        b1 = L8  - L32
        b2 = L2  - L8
        b3 = y   - L2       highest frequency (boundary layer, suction peak, wake)

    Returns (N, C, K) with band 0 the smoothest. Smoothing is applied INCREMENTALLY, so the cost is
    max(ladder) sparse mat-vecs, not their sum.
    """
    y = np.asarray(y, dtype="float64")
    S = smoothing_operator(faces, y.shape[0])
    ladder = sorted(int(m) for m in ladder)
    lows, cur, done = [], y.copy(), 0
    for m in ladder:
        for _ in range(m - done):
            cur = S @ cur
        done = m
        lows.append(cur.copy())                       # lows ascending in smoothing: [L2, L8, L32]
    bands = [lows[-1]]                                # lowest frequency
    for i in range(len(lows) - 2, -1, -1):
        bands.append(lows[i] - lows[i + 1])           # band-pass differences
    bands.append(y - lows[0])                         # highest frequency residual
    return np.stack(bands, axis=-1).astype("float32")  # (N, C, K)


class LearnableAssignment(torch.nn.Module):
    """Learns WHICH frequencies each expert owns.

    v53 hard-wired expert k to band k of a hand-picked ladder (2, 8, 32). There is no reason the data
    should agree with that split. v57_gated_hodge_vef instead decomposes the target into M FINE atoms (a dyadic
    ladder) and learns a K x M assignment: expert k's target is

        t_k = sum_m P[k, m] * atom_m

    with `P = softmax over EXPERTS (dim 0)`, i.e. each COLUMN sums to 1. That is the key property:
    every atom's mass is fully distributed across experts, so

        sum_k t_k = sum_m atom_m = y      EXACTLY, for any value of the learned parameters.

    Completeness therefore never depends on training — only WHO owns WHAT does. An expert may take
    several adjacent atoms, split one with a neighbour, or (if `entropy_lambda` > 0) be pushed
    towards crisp ownership.
    """

    def __init__(self, n_experts, n_atoms, init_groups=None, temperature=1.0):
        super().__init__()
        logits = torch.zeros(n_experts, n_atoms)
        if init_groups:                       # start from a sensible grouping, e.g. v53's
            for k, atoms in enumerate(init_groups):
                for m in atoms:
                    if 0 <= m < n_atoms:
                        logits[k, m] = 2.0
        self.logits = torch.nn.Parameter(logits)
        self.temperature = float(temperature)

    @property
    def P(self):                              # (K, M), columns sum to 1
        return torch.softmax(self.logits / self.temperature, dim=0)

    def targets(self, atoms):
        """atoms (B,N,C,M) -> per-expert targets (B,N,C,K)."""
        return torch.einsum("bncm,km->bnck", atoms, self.P.to(atoms.dtype))

    def entropy(self):
        """Mean column entropy in nats; 0 = every atom owned by exactly one expert."""
        P = self.P.clamp_min(1e-9)
        return float(-(P * P.log()).sum(dim=0).mean())


# ── model ─────────────────────────────────────────────────────────────────────────────────
class FourierFeatures(nn.Module):
    """Fixed sin/cos positional features at ONE spatial scale — the frequency prior handed to an
    expert. Scale sigma sets the wavelength the expert can resolve cheaply."""
    def __init__(self, sigma, n_freq=8, in_dim=2):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_freq) * sigma)

    def forward(self, pos):                                  # (B,N,2) -> (B,N,2*n_freq)
        proj = 2.0 * np.pi * (pos @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    @property
    def out_dim(self):
        return 2 * self.B.shape[1]


class VanillaTransformerBlock(nn.Module):
    """Standard pre-norm softmax self-attention + FFN (identical to v50-v52's block)."""
    def __init__(self, d, heads, d_hidden, dropout=0.1, activation="gelu"):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        A = _ACT[activation]
        self.ff = nn.Sequential(nn.Linear(d, d_hidden), A(), nn.Linear(d_hidden, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.n1(x)
        x = x + self.drop(self._attend(y))
        return x + self.drop(self.ff(self.n2(x)))

    def _attend(self, y):
        """Exactly self.attn(y, y, y, need_weights=False), with the same parameters, in fp32 without the
        N x N matrix. vef_v2 trains in fp32 (no autocast): FlashAttention has no fp32 kernel, so SDPA fell
        back to the math path (17.5 GiB OOM at N_v=4755). The memory-efficient kernel works in fp32 but
        needs head_dim % 4 == 0 and here head_dim = 88/8 = 11, so q/k/v are zero-padded to 12 — a zero
        column adds nothing to q·k or to the weighted sum of v — with the scale kept at 1/sqrt(11)."""
        mha = self.attn
        B, N, d = y.shape
        h = mha.num_heads; hd = d // h
        q, k, v = nn.functional.linear(y, mha.in_proj_weight, mha.in_proj_bias) \
                    .view(B, N, 3, h, hd).permute(2, 0, 3, 1, 4)          # each (B,h,N,hd)
        pad = (-hd) % 4
        if pad:
            q, k, v = (nn.functional.pad(t, (0, pad)) for t in (q, k, v))
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            o = nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=mha.dropout if self.training else 0.0, scale=hd ** -0.5)
        return mha.out_proj(o[..., :hd].transpose(1, 2).reshape(B, N, d))


class Expert(nn.Module):
    """One frequency expert: Fourier features at its scale -> project -> vanilla transformer stack.
    `band_head` reads its own band prediction out, which is what the auxiliary loss pins."""
    def __init__(self, d_model, d_expert, n_blocks, heads, d_hidden, sigma, n_freq,
                 out_channels, dropout=0.1, activation="gelu"):
        super().__init__()
        self.fourier = FourierFeatures(sigma, n_freq)
        self.proj_in = nn.Linear(d_model + self.fourier.out_dim, d_expert)
        self.blocks = nn.ModuleList([
            VanillaTransformerBlock(d_expert, heads, d_hidden, dropout, activation)
            for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_expert)
        self.proj_out = nn.Linear(d_expert, d_model)
        self.band_head = nn.Linear(d_expert, out_channels)

    def forward(self, h, pos):
        z = self.proj_in(torch.cat([h, self.fourier(pos)], dim=-1))
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        return self.proj_out(z), self.band_head(z)          # (B,N,d_model), (B,N,C)


class FrequencyMoE(nn.Module):
    """K frequency experts + a per-node softmax gate.

    Returns (h_mixed, band_preds, gate) where band_preds is (B,N,C,K) — the per-expert band outputs
    the auxiliary loss supervises — and gate is (B,N,K) for load-balance monitoring.
    """
    def __init__(self, d_model, experts, gate_hidden=64, dropout=0.1, activation="gelu"):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        A = _ACT[activation]
        self.gate = nn.Sequential(nn.Linear(d_model, gate_hidden), A(),
                                  nn.Linear(gate_hidden, len(experts)))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, pos, gate_logits=None):
        # v57: the gate can be supplied by the HODGE BRANCH instead of the internal MLP. The internal
        # gate is kept so the module still works standalone (and so v55 vs v57 differs only in WHO
        # decides the routing, not in what the experts are).
        g = torch.softmax(self.gate(h) if gate_logits is None else gate_logits, dim=-1)
        outs, bands = [], []
        for k, ex in enumerate(self.experts):
            o, b = ex(h, pos)
            outs.append(o); bands.append(b)
        H = torch.stack(outs, dim=-1)                              # (B,N,d,K)
        mixed = (H * g.unsqueeze(-2)).sum(dim=-1)                  # gated combination
        return self.drop(mixed), torch.stack(bands, dim=-1), g     # (B,N,C,K), (B,N,K)


def build_experts(n_experts, d_model, d_expert, n_blocks, heads, d_hidden, sigmas, n_freq,
                  out_channels, dropout=0.1, activation="gelu"):
    if sigmas is None or len(sigmas) != n_experts:
        sigmas = [2.0 ** i for i in range(n_experts)]              # geometric scale ladder
    return [Expert(d_model, d_expert, n_blocks, heads, d_hidden, s, n_freq, out_channels,
                   dropout, activation) for s in sigmas]
