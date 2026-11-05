"""E(n)-equivariant graph network (Satorras et al., 2021)."""

from __future__ import annotations

import torch
from torch import nn


def _segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    result = data.new_zeros((num_segments, data.size(1)))
    return result.scatter_add_(0, segment_ids.unsqueeze(-1).expand(-1, data.size(1)), data)


def _segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    index = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    total = data.new_zeros((num_segments, data.size(1))).scatter_add_(0, index, data)
    count = data.new_zeros((num_segments, data.size(1))).scatter_add_(0, index, torch.ones_like(data))
    return total / count.clamp(min=1)


class EGCL(nn.Module):
    """One EGNN layer: edge messages, mean coordinate update, residual feature update."""

    def __init__(self, hidden: int, act: nn.Module):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), act, nn.Linear(hidden, hidden), act)
        self.node_mlp = nn.Sequential(nn.Linear(2 * hidden, hidden), act, nn.Linear(hidden, hidden))
        last = nn.Linear(hidden, 1, bias=False)
        nn.init.xavier_uniform_(last.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(nn.Linear(hidden, hidden), act, last)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, coord: torch.Tensor):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = (coord_diff ** 2).sum(dim=1, keepdim=True)
        message = self.edge_mlp(torch.cat([h[row], h[col], radial], dim=1))
        coord = coord + _segment_mean(coord_diff * self.coord_mlp(message), row, coord.size(0))
        h = h + self.node_mlp(torch.cat([h, _segment_sum(message, row, h.size(0))], dim=1))
        return h, coord


class EGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int):
        super().__init__()
        act = nn.SiLU()
        self.num_layers = num_layers
        self.embedding_in = nn.Linear(in_channels, hidden_channels)
        self.embedding_out = nn.Linear(hidden_channels, out_channels)
        for i in range(num_layers):
            self.add_module(f"gcl_{i}", EGCL(hidden_channels, act))

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.embedding_in(h)
        for i in range(self.num_layers):
            h, x = self._modules[f"gcl_{i}"](h, edge_index, x)
        return self.embedding_out(h), x
