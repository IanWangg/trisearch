"""TriSearch actor-critic.

* encoder  : EGNN on the 1-skeleton; coordinates are both node features and positions.
* actor    : SNN over top simplices → one logit per available flip.
* critic   : max-pool of the SNN simplex features → MLP → scalar value.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import nn

from .egnn import EGNN
from .snn import SNNSimplexActor


class TriSearchPolicy(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 3,
        snn_layers: int = 2,
    ):
        super().__init__()
        self.config = dict(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            snn_layers=snn_layers,
        )
        self.encoder = EGNN(in_channels, hidden_channels, out_channels, num_layers)
        self.actor = SNNSimplexActor(out_channels, hidden_channels, snn_layers)
        self.actor_head = nn.Linear(out_channels, 1)
        self.value_head = nn.Sequential(nn.Linear(out_channels, out_channels), nn.SiLU(), nn.Linear(out_channels, 1))

    def get_value_and_logits(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Values ``[B]`` and logits ``[B, max_actions]`` padded with ``-inf``."""
        node_embeddings, _ = self.encoder(batch.x, batch.x, batch.edge_index)
        candidate, simplex_features, simplex_graph = self.actor(node_embeddings, batch)
        num_actions = batch.num_actions.to(candidate.device)
        num_graphs = int(num_actions.numel())
        value = self.value_head(gnn.pool.global_max_pool(simplex_features, simplex_graph, size=num_graphs))

        logits_flat = self.actor_head(candidate).squeeze(-1)
        padded = logits_flat.new_full((num_graphs, int(num_actions.max())), float("-inf"))
        graph = torch.repeat_interleave(torch.arange(num_graphs, device=candidate.device), num_actions)
        start = torch.cumsum(num_actions, 0) - num_actions
        position = torch.arange(logits_flat.size(0), device=candidate.device) - start[graph]
        padded[graph, position] = logits_flat
        return value.squeeze(-1), padded

    def get_value(self, batch) -> torch.Tensor:
        return self.get_value_and_logits(batch)[0]

    @staticmethod
    def log_prob_and_entropy(logits: torch.Tensor, action_index: torch.Tensor):
        dist = torch.distributions.Categorical(probs=F.softmax(logits, dim=1))
        return dist.log_prob(action_index.to(logits.device).long().view(-1)), dist.entropy()

    def forward(self, batch, deterministic: bool = False):
        """Sample (or argmax) one action per graph.

        Returns ``(action_index, value, log_prob)``; ``action_index`` indexes
        each state's :meth:`~trisearch.state.TriangulationState.subcomplex_actions`.
        """
        value, logits = self.get_value_and_logits(batch)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        action_index = torch.argmax(probs, dim=1) if deterministic else dist.sample()
        return action_index, value, dist.log_prob(action_index)


def load_policy(checkpoint_path: str, device="cpu") -> TriSearchPolicy:
    """Load a checkpoint written by ``trisearch-train`` (``{"config": ..., "state_dict": ...}``)."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy = TriSearchPolicy(**payload["config"])
    policy.load_state_dict(payload["state_dict"])
    return policy.to(device).eval()
