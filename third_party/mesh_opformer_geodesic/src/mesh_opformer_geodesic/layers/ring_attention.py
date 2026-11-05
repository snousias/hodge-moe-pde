"""
ring_attention.py
-----------------
Sparse attention over geodesic-ring neighbourhoods, with an optional learned
**relative-geodesic positional bias**.

This is a drop-in replacement for ``mesh_opformer.layers.attention.sparse_attention``
(the kernel behind ``hodge_grouped``). The gather / mask / softmax path is
byte-for-byte the same — it reuses ``mesh_opformer``'s own helpers — so with
``ring_bias=None`` it is numerically identical to the stock kernel.

The only addition: a per-(head, ring) scalar added to the attention logits
before the softmax, indexed by the hop distance of each neighbour slot. That
tells the softmax *how far along the surface* each neighbour is — information the
stock sparse attention simply does not have. Think ALiBi / relative position,
but the "distance" is BFS hops on the mesh graph. Zero-initialised, so training
starts as an exact no-op and the bias is learned from there.
"""

import torch
import torch.nn as nn

# reuse the exact gather / index-prep primitives from the base package so the
# masking + reshape semantics stay identical.
from mesh_opformer.layers.attention.attention_sparse import (
    prepare_indices,
    batched_index_select,
    expand_dim,
)


class RingBias(nn.Module):
    """
    Learned additive attention bias per (head, hop-ring).

    Parameters
    ----------
    h : int
        Number of attention heads.
    max_ring : int
        Largest hop distance expected (rings ``0..max_ring``). Slots beyond it
        are clamped to ``max_ring``.
    """

    def __init__(self, h: int, max_ring: int = 8):
        super().__init__()
        self.h = h
        self.max_ring = max_ring
        # +1 so ring==max_ring is a valid index after clamping; zero-init -> no-op
        self.table = nn.Embedding(max_ring + 1, h)
        nn.init.zeros_(self.table.weight)

    def forward(self, ring: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ring : torch.Tensor
            Hop distance per slot, shape ``(b, n, j)`` (``-1`` on pad slots).

        Returns
        -------
        torch.Tensor
            Bias of shape ``(b, h, n, j)`` to add to the attention logits.
            Pad slots get bias 0 (they are ``-inf``-masked downstream anyway).
        """
        r = ring.clamp(min=0, max=self.max_ring).long()
        bias = self.table(r)               # (b, n, j, h)
        return bias.permute(0, 3, 1, 2)    # (b, h, n, j)


def ring_sparse_attention(x, q, k, idx, ring_bias=None, dropout=None):
    """
    Sparse attention with an optional precomputed ring bias.

    Parameters
    ----------
    x, q, k : torch.Tensor
        Value / query / key tensors, shape ``(b, h, n, d)`` — as passed by
        ``hodge_grouped``.
    idx : torch.Tensor
        Neighbour indices ``(b, n, j)``, ``-1`` padded.
    ring_bias : torch.Tensor | None
        Precomputed bias ``(b, h, n, j)`` from :class:`RingBias`, or ``None`` for
        the stock (unbiased) behaviour.
    dropout : float | None
        Attention dropout probability.

    Returns
    -------
    torch.Tensor
        Attended values, shape ``(b, h, n, d)``.
    """
    _q, _k = q, k
    b, h, n, d = _q.size()

    with torch.no_grad():
        _idx, mask = prepare_indices(idx.detach(), h, None)

    _k = batched_index_select(_k, _idx, d).reshape(b, h, n, -1, d)
    _v = batched_index_select(x, _idx, d).reshape(b, h, n, -1, d)

    qk = torch.einsum("bhnd,bhnjd->bhnj", _q, _k) * (d ** -0.5)

    if ring_bias is not None:
        qk = qk + ring_bias

    mask = expand_dim(mask, 1, h)
    qk = qk.masked_fill_(mask, float("-inf"))
    qk = torch.nn.functional.softmax(qk, dim=-1)

    if dropout is not None:
        qk = torch.nn.functional.dropout(qk, p=dropout)

    return torch.einsum("bhnj,bhnjd->bhnd", qk, _v)
