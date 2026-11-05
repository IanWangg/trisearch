"""Simplex-level SNN actor (lift → propagate → score).

1. **Lift**: max-pool encoder vertex embeddings over each top simplex.
2. **Propagate**: Chebyshev simplicial convolutions with the normalized
   top-degree down Laplacian (Ebli et al., 2020).
3. **Score**: max-pool the refined simplex features over the simplices of
   ``T+(Z)`` for every candidate flip ``Z``.

The batch must come from :func:`trisearch.data.featurize`, which precomputes the
Laplacian entries and the candidate/simplex memberships per graph.
"""

from __future__ import annotations

import torch
from torch import nn


class ChebyshevConv(nn.Module):
    """``y = Σ_k T_k(L) x Θ_k + b`` with ``T_0 = I, T_1 = L, T_k = 2 L T_{k-1} - T_{k-2}``."""

    def __init__(self, in_channels: int, out_channels: int, order: int = 3):
        super().__init__()
        self.order = order
        self.theta = nn.Parameter(torch.empty(out_channels, in_channels, order))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, laplacian: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        terms = [x]
        if self.order > 1:
            terms.append(torch.sparse.mm(laplacian, x))
        for k in range(2, self.order):
            terms.append(2.0 * torch.sparse.mm(laplacian, terms[k - 1]) - terms[k - 2])
        return torch.einsum("mik,oik->mo", torch.stack(terms, dim=-1), self.theta) + self.bias


class SNNSimplexActor(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, num_layers: int, order: int = 3):
        super().__init__()
        dims = [channels] + ([channels] if num_layers == 1 else [hidden_channels] * (num_layers - 1) + [channels])
        self.layers = nn.ModuleList(ChebyshevConv(dims[i], dims[i + 1], order) for i in range(len(dims) - 1))
        self.act = nn.SiLU()

    def forward(self, node_embeddings: torch.Tensor, batch):
        """Return ``(candidate_features [A, C], simplex_features [S, C], simplex_graph [S])``."""
        device = node_embeddings.device
        num_graphs = int(batch.num_actions.numel())
        graph_ids = torch.arange(num_graphs, device=device)
        num_simplices = batch.num_simplices_top.to(device)
        num_actions = batch.num_actions.to(device)
        simplex_start = torch.cumsum(num_simplices, 0) - num_simplices
        action_start = torch.cumsum(num_actions, 0) - num_actions

        # Lift.
        simplex_graph = torch.repeat_interleave(graph_ids, num_simplices)
        simplex_vertices = batch.simplex_vertices.to(device) + batch.ptr[:-1].to(device)[simplex_graph].unsqueeze(1)
        z = node_embeddings[simplex_vertices].amax(dim=1)

        # Propagate.
        entry_graph = torch.repeat_interleave(graph_ids, batch.num_laplacian_entries.to(device))
        offset = simplex_start[entry_graph]
        laplacian = torch.sparse_coo_tensor(
            torch.stack([batch.laplacian_row.to(device) + offset, batch.laplacian_col.to(device) + offset]),
            batch.laplacian_value.to(device=device, dtype=node_embeddings.dtype),
            size=(z.size(0), z.size(0)),
        ).coalesce()
        for i, layer in enumerate(self.layers):
            z = layer(laplacian, z)
            if i != len(self.layers) - 1:
                z = self.act(z)

        # Score.
        member_graph = torch.repeat_interleave(graph_ids, batch.num_memberships.to(device))
        member_action = batch.membership_candidate.to(device) + action_start[member_graph]
        member_simplex = batch.membership_simplex.to(device) + simplex_start[member_graph]
        channels = node_embeddings.size(-1)
        candidate = node_embeddings.new_full((int(num_actions.sum()), channels), float("-inf"))
        candidate.scatter_reduce_(
            0, member_action.unsqueeze(1).expand(-1, channels), z[member_simplex], reduce="amax", include_self=True
        )
        return candidate, z, simplex_graph
