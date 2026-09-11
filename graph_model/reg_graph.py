"""Shallow GNN building block for regression-weighted economic edges."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RegressionGraphOutput:
    """Outputs left unaggregated for composition in a larger model."""

    node_embeddings: Tensor
    edge_messages: Tensor
    source_index: Tensor
    target_index: Tensor


class RegressionGraph(nn.Module):
    """Encode nodes and construct directed messages from two-channel edges.

    ``edge_attr[:, 0]`` is the economic-prior weight and
    ``edge_attr[:, 1]`` is the signed, confidence-shrunk regression weight.
    The module intentionally does not aggregate messages, stack graph layers,
    define a prediction head, or contain training behavior.
    """

    edge_feature_count = 2

    def __init__(
        self,
        node_feature_count: int,
        hidden_size: int,
        *,
        message_size: int | None = None,
    ) -> None:
        super().__init__()
        if node_feature_count <= 0 or hidden_size <= 0:
            raise ValueError("node_feature_count and hidden_size must be positive")

        output_size = hidden_size if message_size is None else message_size
        if output_size <= 0:
            raise ValueError("message_size must be positive")

        self.node_feature_count = node_feature_count
        self.hidden_size = hidden_size
        self.message_size = output_size
        self.node_encoder = nn.Linear(node_feature_count, hidden_size)
        self.edge_encoder = nn.Linear(self.edge_feature_count, hidden_size)
        self.message_projection = nn.Linear(hidden_size * 3, output_size)
        self.activation = nn.GELU()

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> RegressionGraphOutput:
        """Return encoded nodes and one message per directed edge."""
        self._validate_inputs(node_features, edge_index, edge_attr)
        source_index, target_index = edge_index

        node_embeddings = self.activation(self.node_encoder(node_features))
        edge_embeddings = self.activation(self.edge_encoder(edge_attr))
        message_inputs = torch.cat(
            (
                node_embeddings[source_index],
                node_embeddings[target_index],
                edge_embeddings,
            ),
            dim=-1,
        )
        edge_messages = self.message_projection(message_inputs)
        return RegressionGraphOutput(
            node_embeddings=node_embeddings,
            edge_messages=edge_messages,
            source_index=source_index,
            target_index=target_index,
        )

    def _validate_inputs(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> None:
        if node_features.ndim != 2:
            raise ValueError("node_features must have shape [num_nodes, node_feature_count]")
        if node_features.shape[1] != self.node_feature_count:
            raise ValueError(
                f"Expected {self.node_feature_count} node features, got {node_features.shape[1]}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long indices")
        if edge_attr.ndim != 2 or edge_attr.shape[1] != self.edge_feature_count:
            raise ValueError("edge_attr must have shape [num_edges, 2]")
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_attr and edge_index must contain the same number of edges")
        if edge_index.numel() and (
            edge_index.min().item() < 0
            or edge_index.max().item() >= node_features.shape[0]
        ):
            raise IndexError("edge_index contains a node index outside node_features")