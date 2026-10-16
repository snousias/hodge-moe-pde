# ========================================================================================
# MODEL ENTRY POINT
#   build_v57_gated_hodge_vef_model(**cfg['train']['model']) -> MoEModel
#   This is the ONLY supported way to construct the network. Everything else in arch/ is a
#   component of it.
# ========================================================================================
#!/usr/bin/env python
"""
v57_gated_hodge_vef — frequency mixture-of-experts of VANILLA transformers.

    x    = Linear(x_v)                                  # (B,N,9) -> (B,N,d_model) tokens
    h,bands,gate = FrequencyMoE(x, pos)                 # K experts, per-node softmax gate
    out  = FieldAttnDecoder(h)                           # (B,N,4)

Each expert is a plain softmax transformer stack (no Hodge, no mesh structure in the model) fed
Fourier positional features at its own spatial scale, and is pinned to one frequency band of the
target by an auxiliary head. The gate chooses, per vertex, how much of each band to use.

Interface note: `forward` returns ONLY the field tensor, so every existing eval script works
unchanged. The auxiliary outputs the training loss needs (per-expert band predictions and the gate)
are stashed on `self.aux` by the same call — see `airfrans_regression_v57_gated_hodge_vef.train_epoch`.
"""
import torch
import torch.nn as nn

from arch.experts import FrequencyMoE, build_experts, LearnableAssignment
from arch.analyzer import LearnedHodgeAnalyzer
from arch.hodge_gate import HodgeGate

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


class _DecoderBlock(nn.Module):
    """Pre-norm multi-head self-attention over the field-token axis + position-wise FFN."""
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
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x)))


class FieldAttnDecoder(nn.Module):
    """Cross-field attention decoder (identical to v45_2 / v50-v52)."""
    def __init__(self, d, out_channels=4, n_dec_layers=2, dec_heads=4, d_hidden=256,
                 dropout=0.1, activation="gelu", readout_hidden=64):
        super().__init__()
        self.C = out_channels
        self.field_emb = nn.Parameter(torch.empty(out_channels, d))
        nn.init.normal_(self.field_emb, std=0.02)
        self.blocks = nn.ModuleList([
            _DecoderBlock(d, dec_heads, d_hidden, dropout, activation) for _ in range(n_dec_layers)])
        self.norm = nn.LayerNorm(d)
        A = _ACT[activation]
        self.readout = nn.Sequential(nn.Linear(d, readout_hidden), A(),
                                     nn.Linear(readout_hidden, 1))

    def forward(self, h):
        B, N, d = h.shape
        tok = h.unsqueeze(2) + self.field_emb.view(1, 1, self.C, d)
        tok = tok.reshape(B * N, self.C, d)
        for blk in self.blocks:
            tok = blk(tok)
        out = self.readout(self.norm(tok)).squeeze(-1)
        return out.reshape(B, N, self.C)


class MoEModel(nn.Module):
    """Vanilla-transformer frequency MoE. Uses only x_v (positions are its first two channels);
    all mesh arguments are accepted for trainer compatibility and ignored."""
    def __init__(self, embed, moe, decoder, d, dropout=0.1, pos_cols=(0, 1), analyzer=None):
        super().__init__()
        self.embed = embed
        self.moe = moe
        # v57_gated_hodge_vef: the LEARNED frequency analyzer. It is a submodule (so its stars are optimised and
        # checkpointed with everything else) but it is NOT in the prediction path — the trainer calls
        # `analyzer.bands(target, ...)` to build the supervision. The predictor stays the pure
        # point-transformer of v50-v53.
        self.analyzer = analyzer
        # v57_gated_hodge_vef: the atom -> expert assignment is LEARNED (column-stochastic, so completeness holds
        # for any value it takes). Set by the builder; the trainer reads it to form band targets.
        self.assign = None
        # v57: the spectral controller. Reads the full simplicial complex and emits the gate logits;
        # it never predicts the field.
        self.hodge_gate = None
        self.out_norm = nn.LayerNorm(d)
        self.decoder = decoder
        self.drop = nn.Dropout(dropout)
        self.pos_cols = list(pos_cols)
        self.aux = {}                       # {"bands": (B,N,C,K), "gate": (B,N,K)} — set per forward

    def forward(self, x_v, x_e=None, x_f=None, d_0=None, d_1=None,
                v_idx=None, e_idx=None, f_idx=None, mask=None):
        pos = x_v[..., self.pos_cols]                       # normalised coordinates
        h = self.embed(x_v)
        gl = (self.hodge_gate(x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx, mask)
              if self.hodge_gate is not None else None)
        h, bands, gate = self.moe(h, pos, gate_logits=gl)
        self.aux = {"bands": bands, "gate": gate}
        h = self.drop(self.out_norm(h))
        return self.decoder(h)


def build_v57_gated_hodge_vef_model(v_in, e_in=None, f_in=None, d_v=192, d_e=None, d_f=None, d_hidden=768,
                    n_layers=2, dropout=0.1, modes="v",
                    embed_kw=None, attn_kw=None, attn_basic_kw=None, decoder_kw=None,
                    n_transformer=4, moe_kw=None,
                    out_channels=4, head_hidden=(128, 128, 64), head_activation="gelu",
                    **ignored):
    """v57_gated_hodge_vef builder. `moe_kw` controls the mixture:

        n_experts        number of frequency experts (= number of target bands)
        d_expert         width of each expert (kept below d_v so the total matches v51's budget)
        blocks_per_expert / heads / d_hidden_expert
        sigmas           Fourier scale per expert (default geometric ladder 1,2,4,8)
        n_freq           Fourier features per expert
        gate_hidden      width of the per-node gate MLP
    """
    attn_kw = attn_kw or {}; decoder_kw = decoder_kw or {}; moe_kw = dict(moe_kw or {})
    K = int(moe_kw.get("n_experts", 4))
    d_expert = int(moe_kw.get("d_expert", 128))
    experts = build_experts(
        n_experts=K, d_model=d_v, d_expert=d_expert,
        n_blocks=int(moe_kw.get("blocks_per_expert", 2)),
        heads=int(moe_kw.get("heads", 8)),
        d_hidden=int(moe_kw.get("d_hidden_expert", 4 * d_expert)),
        sigmas=moe_kw.get("sigmas"), n_freq=int(moe_kw.get("n_freq", 8)),
        out_channels=out_channels, dropout=dropout, activation=head_activation)
    moe = FrequencyMoE(d_v, experts, gate_hidden=int(moe_kw.get("gate_hidden", 64)),
                       dropout=dropout, activation=head_activation)
    embed = nn.Sequential(nn.Linear(v_in, d_v), _ACT[head_activation](), nn.Linear(d_v, d_v))
    decoder = FieldAttnDecoder(
        d=d_v, out_channels=out_channels,
        n_dec_layers=decoder_kw.get("n_dec_layers", 2),
        dec_heads=decoder_kw.get("dec_heads", 4),
        d_hidden=decoder_kw.get("d_hidden", d_hidden),
        readout_hidden=decoder_kw.get("readout_hidden", 64),
        dropout=dropout, activation=head_activation)
    an_kw = dict(ignored.get("analyzer_kw", {}) or {})
    analyzer = LearnedHodgeAnalyzer(
        e_in=int(e_in or 21), v_in=v_in,
        hidden=int(an_kw.get("hidden", 64)),
        ladder=tuple(an_kw.get("ladder", (2, 8, 32))),
        learn_mass=bool(an_kw.get("learn_mass", False)),
        activation=head_activation)
    m = MoEModel(embed, moe, decoder, d=d_v, dropout=dropout, analyzer=analyzer)
    n_atoms = len(analyzer.ladder) + 1
    # initialise each expert on a contiguous slice of the atom ladder (the v53-style grouping) so
    # v57_gated_hodge_vef STARTS where a hand-picked split would and can move away from it
    per = max(1, n_atoms // K)
    groups = [list(range(k * per, min((k + 1) * per, n_atoms))) or [n_atoms - 1] for k in range(K)]
    m.assign = LearnableAssignment(K, n_atoms, init_groups=groups)
    hg_kw = dict(ignored.get("hodge_gate_kw", {}) or {})
    m.hodge_gate = HodgeGate(v_in=v_in, e_in=int(e_in or 21), f_in=int(f_in or 13), n_experts=K,
                             d=int(hg_kw.get("d", 64)), d_hidden=int(hg_kw.get("d_hidden", 256)),
                             h=int(hg_kw.get("heads", 4)), n_layers=int(hg_kw.get("n_layers", 2)),
                             dropout=dropout, embed_kw=embed_kw,
                             readout_hidden=int(hg_kw.get("readout_hidden", 64)),
                             window=int(hg_kw.get("window", 64)),
                             window_e=hg_kw.get("window_e"),
                             # `modes` selects which cochain streams the controller evolves with a
                             # Hodge-Laplacian, the same switch the original HodgeFormer uses.
                             # v57_gated_hodge_vef sets "vef": L₀ on vertices, L₁ on edges, L₂ on faces.
                             modes=str(hg_kw.get("modes", modes)),
                             activation=head_activation)
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return m
