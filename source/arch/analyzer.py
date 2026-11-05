# ========================================================================================
# COMPONENT — defines the frequency bands of the TARGET. Used by training/, not in the forward pass.
# ========================================================================================
#!/usr/bin/env python
"""v57_gated_hodge_vef — the frequency analyzer is a LEARNED Hodge Laplacian.

v53 fixed what "frequency" means: powers of the combinatorial adjacency, with a hand-picked ladder
(2, 8, 32) and a hard-wired expert<->band map. v57_gated_hodge_vef learns the operator itself.

THE OPERATOR. A discrete Hodge Laplacian on 0-forms with LEARNED, DIAGONAL stars:

    L_0 = *_0^-1 d_0^T *_1 d_0
        *_1 = diag(w_e),  w_e = softplus(MLP(x_e))    learned edge conductance (1-form star)
        *_0 = diag(a_v),  a_v = softplus(MLP(x_v))    learned vertex mass     (0-form star)

Diagonal stars are a deliberate choice over v34_2's attention-style stars. They give exactly what
that construction promised but failed to deliver: with w_e >= 0 the quadratic form is

    x^T d_0^T *_1 d_0 x = sum_e w_e (x_i - x_j)^2 >= 0

so L_0 is PSD **by construction**, self-adjoint in the *_0 inner product, with a real non-negative
spectrum — no window truncation to break the symmetry, no complex eigenvalues
(cf. project_v34_2_spd_claim_fails). The learnable content is WHICH EDGES CONDUCT, which is precisely
the notion of distance that defines frequency, and it is what the fixed adjacency operator gets wrong
on a graded mesh.

STABILITY. Smoothing is the explicit `(I - tau L_0)^m`, whose spectrum stays in [0, 1] iff
tau <= 1/lambda_max. lambda_max is estimated per sample by a few DETACHED power iterations of the
same mat-vec (a Gershgorin bound would need |d_0|, which the signed sparse tensor does not expose),
with a 1.1 safety factor. No solves, no tuning, stable however the stars move during training.

COMPLETENESS IS FREE. Bands telescope, so sum_k b_k = y EXACTLY for ANY value of the learned
parameters. The analyzer can change what each band CONTAINS; it can never change the total.

THE CHEAT, AND THE BLOCK. Because the band targets now depend on the model's own parameters, there is
a degenerate optimum: drive w_e -> 0, so L_0 -> 0, every low-pass becomes the identity, and the bands
collapse to (y, 0, 0, 0). Experts 1..K then have trivial zero targets and the auxiliary loss vanishes
without anything being learned. `band_energy_entropy` measures exactly that collapse (entropy of the
per-band energy shares) and the trainer maximises it, which makes the degenerate solution the WORST
point of that term rather than the best.
"""
import torch
import torch.nn as nn


class LearnedHodgeAnalyzer(nn.Module):
    """Learned diagonal-star Hodge Laplacian + telescoping band decomposition.

    forward-style entry point is `bands(y, x_v, x_e, d_0)`; nothing here touches the prediction path,
    so v57_gated_hodge_vef's predictor stays the pure point-transformer of v50-v53 — the mesh enters only through the
    analyzer that defines the supervision.
    """

    def __init__(self, e_in, v_in, hidden=64, ladder=(2, 4, 6, 8, 12, 16, 24, 32), learn_mass=False,
                 w_floor=1e-4, activation="gelu"):
        super().__init__()
        A = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        self.edge_mlp = nn.Sequential(nn.Linear(e_in, hidden), A(), nn.Linear(hidden, 1))
        self.mass_mlp = (nn.Sequential(nn.Linear(v_in, hidden), A(), nn.Linear(hidden, 1))
                         if learn_mass else None)
        self.ladder = tuple(sorted(int(m) for m in ladder))
        self.w_floor = float(w_floor)

    # ── the operator ──────────────────────────────────────────────────────────────────────
    def stars(self, x_v, x_e, d_0):
        """(*_1 edge conductance (B,E,1), *_0 vertex mass (B,N,1)), both strictly positive.

        *_0 is the LEARNED DEGREE, a_i = sum_{e incident to i} w_e, not a free per-vertex MLP. That
        choice is what bounds the spectrum: with it, L_0 = *_0^-1 d_0^T *_1 d_0 is the random-walk
        Laplacian, whose eigenvalues lie in [0, 2] for ANY positive conductances. A free mass makes
        lambda_max scale with the grid grading (measured ~1e6 here), and then no affordable number of
        explicit smoothing steps resolves the low end of the spectrum — every band but the first
        collapses to zero before training even starts.

        `learn_mass=True` restores the free mass for ablation; it is off by default for that reason.
        """
        w = nn.functional.softplus(self.edge_mlp(x_e)) + self.w_floor
        deg = self._apply_d0T(torch.abs(d_0), w)                 # |d_0|^T w = sum of incident w
        if self.mass_mlp is None:
            a = deg.clamp_min(self.w_floor)
        else:
            a = (deg * nn.functional.softplus(self.mass_mlp(x_v))).clamp_min(self.w_floor)
        return w, a

    @staticmethod
    def _apply_d0(d_0, x):
        """d_0 @ x for a (possibly batched) sparse d_0 of shape (B,E,N) or (E,N)."""
        if d_0.dim() == 3:
            return torch.bmm(d_0, x)
        return torch.matmul(d_0, x)

    @staticmethod
    def _apply_d0T(d_0, x):
        if d_0.dim() == 3:
            return torch.bmm(d_0.transpose(1, 2), x)
        return torch.matmul(d_0.transpose(0, 1), x)

    def laplacian(self, x, w, a, d_0):
        """L_0 x = *_0^-1 d_0^T *_1 d_0 x — PSD by construction (w >= 0)."""
        return self._apply_d0T(d_0, w * self._apply_d0(d_0, x)) / a

    def tau(self, w, a, d_0):
        """Smoothing step size. Constant 1/2, from THEORY, not estimation.

        With *_0 = learned degree, L_0 is the random-walk Laplacian, whose spectrum lies in [0, 2]
        for any positive conductances. So `I - L/2` has spectrum in [0, 1] unconditionally and
        tau = 1/2 is always stable — no power iteration needed.

        This matters beyond elegance: an estimated tau makes the bands NON-DETERMINISTIC (a random
        probe gives a slightly different lambda_max each call, measured 6.5e-4 relative drift in the
        bands), which would put noise into the training target and break scale invariance. A constant
        is exactly reproducible.
        """
        return torch.as_tensor(0.5, device=w.device, dtype=w.dtype).view(1, 1, 1)

    def lambda_max_estimate(self, w, a, d_0, n_iter=8):
        """Detached power-iteration estimate of lambda_max — diagnostics only, never in the band
        path. Should sit at or below the random-walk bound of 2."""
        with torch.no_grad():
            v = torch.randn(w.shape[0], d_0.shape[-1], 1, device=w.device, dtype=w.dtype)
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
            lam = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
            for _ in range(n_iter):
                Lv = self.laplacian(v, w, a, d_0)
                n = Lv.norm(dim=1, keepdim=True).clamp_min(1e-12)
                lam = n.view(-1); v = Lv / n
        return lam

    # ── bands ─────────────────────────────────────────────────────────────────────────────
    def bands(self, y, x_v, x_e, d_0):
        """Telescoping frequency ATOMS of y (B,N,C) -> (B,N,C,M), M = len(ladder)+1.

        v57_gated_hodge_vef uses a FINER ladder than v54 at the SAME cost — the price is set by max(ladder), which is
        unchanged at 32, so adding intermediate rungs is nearly free. The atoms are then combined
        into per-expert bands by a learned assignment (moe_mod.LearnableAssignment), which is what
        makes the band BOUNDARIES learnable rather than just the operator.

        sum_k bands[..., k] == y exactly, whatever the learned stars are.

        The operator is used in SPECTRALLY NORMALISED form, `L / lambda_max`, so one smoothing step
        is `I - L/lambda_max` with spectrum in [0, 1]. Two reasons, both load-bearing:

        * These grids are graded ~14:1, so lambda_max is huge and the stable explicit step
          `tau = 1/lambda_max` is ~1e-6. Un-normalised, 32 steps smooth essentially NOTHING and every
          band but the first collapses to zero — measured entropy 0.006 of a possible 1.386, i.e. the
          decomposition degenerates for numerical reasons before any learning happens.
        * Normalisation also removes the obvious cheat: scaling all conductances down (w -> 0) now
          leaves `L / lambda_max` unchanged, so the analyzer cannot buy an easy auxiliary loss by
          weakening the operator. What it can still learn is the SHAPE — which edges conduct
          relative to which — and that is exactly the notion of distance that defines frequency.
        """
        w, a = self.stars(x_v, x_e, d_0)
        t = self.tau(w, a, d_0)
        lows, cur, done = [], y, 0
        for m in self.ladder:
            for _ in range(m - done):
                cur = cur - t * self.laplacian(cur, w, a, d_0)      # one explicit smoothing step
            done = m
            lows.append(cur)
        out = [lows[-1]]
        for i in range(len(lows) - 2, -1, -1):
            out.append(lows[i] - lows[i + 1])
        out.append(y - lows[0])
        return torch.stack(out, dim=-1)

    # ── the anti-collapse guard ───────────────────────────────────────────────────────────
    @staticmethod
    def band_floor_penalty(bands, min_share=None, eps=1e-9):
        """ONE-SIDED guard: penalise only bands whose energy share falls BELOW a floor.

        v54a/v54b used entropy maximisation here, and that was a design error worth recording. It
        blocks the collapse, but it also *forces balance* — both runs converged to entropy ~1.15 of a
        possible 1.386 and lost to the fixed ladder, whose shares are [0.96, 0.03, 0.01, 0.005]. The
        analyzer was never free to choose a lopsided decomposition, so those runs did not test
        "learned vs fixed"; they tested "balance-forced vs lopsided".

        A floor blocks the degenerate solution (a band with no content gives its expert nothing to
        learn) while leaving every non-degenerate distribution — including a very lopsided one —
        completely unpenalised.
        """
        e = (bands ** 2).sum(dim=(1, 2))                     # (B,K)
        p = e / e.sum(dim=-1, keepdim=True).clamp_min(eps)
        K = p.shape[-1]
        floor = (0.5 / K) if min_share is None else float(min_share)
        return (torch.relu(floor - p) ** 2).sum(-1).mean(), p.mean(0)

    @staticmethod
    def band_energy_entropy(bands, eps=1e-9):
        """Entropy (nats) of the per-band energy shares. Maximal (log K) when every band carries an
        equal share; -> 0 when the analyzer collapses everything into one band, which is exactly the
        degenerate optimum of the auxiliary loss. Returned with its shares for logging."""
        e = (bands ** 2).sum(dim=(1, 2))                     # (B,K)
        p = e / e.sum(dim=-1, keepdim=True).clamp_min(eps)
        h = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(-1)
        return h.mean(), p.mean(0)
