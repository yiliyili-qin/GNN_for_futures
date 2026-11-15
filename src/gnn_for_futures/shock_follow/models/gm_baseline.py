"""GraphMixer's parallel event/node encoders with the existing three-class task."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gnn_for_futures.shock_follow.models.head_v2 import (
    HierarchicalHeadOutput,
    HierarchicalShockFollowHead,
)


class FixedTimeEncode(nn.Module):
    """Original fixed cosine frequencies; elapsed time is supplied in 15m units."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("time dimension must be positive")
        self.register_buffer("frequencies", 10.0 ** -torch.linspace(0, 9, dimension))

    def forward(self, elapsed: Tensor) -> Tensor:
        return torch.cos(elapsed.unsqueeze(-1) * self.frequencies)


def _feed_forward(dimension: int, expansion: float, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dimension, max(1, int(dimension * expansion))),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(max(1, int(dimension * expansion)), dimension),
        nn.Dropout(dropout),
    )


class EventMixerBlock(nn.Module):
    """Residual token-axis mixing followed by residual channel-axis mixing."""

    def __init__(self, slots: int, dimension: int, dropout: float) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(dimension)
        self.token_mlp = _feed_forward(slots, .5, dropout)
        self.channel_norm = nn.LayerNorm(dimension)
        self.channel_mlp = _feed_forward(dimension, 4., dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        mixed = self.token_mlp(self.token_norm(tokens).transpose(-1, -2))
        tokens = tokens + mixed.transpose(-1, -2)
        return tokens + self.channel_mlp(self.channel_norm(tokens))


class GraphMixerBaseline(nn.Module):
    """Shared endpoint encoder and directed hierarchical decoder, without pair bypass.

    Inputs contain only already-realized historical edges. ``node_features`` may
    be static identities (ordinary GM) or current causal attributes (GM-TNF-style).
    Padding is zero before the Mixer; pooling uses its fixed slot count, as in GM.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        *,
        max_events: int = 50,
        time_dim: int = 100,
        hidden_dim: int = 100,
        num_layers: int = 2,
        head_dim: int = 100,
        dropout: float = .1,
        follow_only: bool = False,
    ) -> None:
        super().__init__()
        if min(node_feature_dim, edge_feature_dim, max_events, hidden_dim, num_layers, head_dim) < 1:
            raise ValueError("dimensions, event slots and layers must be positive")
        if not 0. <= dropout < 1.:
            raise ValueError("dropout must be in [0,1)")
        self.max_events = max_events
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.time_encoder = FixedTimeEncode(time_dim)
        self.event_projection = nn.Linear(edge_feature_dim + time_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            EventMixerBlock(max_events, hidden_dim, dropout) for _ in range(num_layers)
        )
        self.event_norm = nn.LayerNorm(hidden_dim)
        self.event_output = nn.Linear(hidden_dim, hidden_dim)
        self.head = HierarchicalShockFollowHead(
            hidden_dim + node_feature_dim, hidden_dim + node_feature_dim,
            pair_feature_dim=0, hidden_dim=head_dim, follow_only=follow_only,
        )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_ages: Tensor,
        edge_mask: Tensor,
        adjacency: Tensor,
    ) -> HierarchicalHeadOutput | Tensor:
        if node_features.ndim != 3 or node_features.shape[-1] != self.node_feature_dim:
            raise ValueError("node_features must be [batch,nodes,node_feature_dim]")
        batch, nodes, _ = node_features.shape
        expected = (batch, nodes, self.max_events)
        if edge_features.shape != expected + (self.edge_feature_dim,):
            raise ValueError("edge_features must be [batch,nodes,max_events,edge_feature_dim]")
        if edge_ages.shape != expected or edge_mask.shape != expected:
            raise ValueError("edge ages/mask must match the event slots")
        if edge_mask.dtype != torch.bool or adjacency.dtype != torch.bool:
            raise TypeError("edge_mask and adjacency must be boolean")
        if adjacency.shape != (batch, nodes, nodes):
            raise ValueError("adjacency must be [batch,nodes,nodes]")
        if not torch.isfinite(edge_ages[edge_mask]).all() or (edge_ages[edge_mask] < 0).any():
            raise ValueError("visible edge ages must be finite and nonnegative")
        # Padded attributes/ages, including projection bias and cos(0), carry no input.
        attributes = edge_features.masked_fill(~edge_mask.unsqueeze(-1), 0.)
        ages = edge_ages.masked_fill(~edge_mask, 0.)
        tokens = self.event_projection(torch.cat((attributes, self.time_encoder(ages)), -1))
        tokens = tokens.masked_fill(~edge_mask.unsqueeze(-1), 0.)
        for block in self.blocks:
            tokens = block(tokens)
        temporal = self.event_output(self.event_norm(tokens).mean(dim=-2))
        neighbors = adjacency.to(node_features.dtype)
        neighbors = neighbors / neighbors.sum(-1, keepdim=True).clamp_min(1.)
        structural = node_features + neighbors @ node_features
        representation = torch.cat((temporal, structural), dim=-1)
        return self.head(representation, representation)
