"""End-to-end hierarchical shock-follow GraphMixer V2."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from gnn_for_futures.graph.temporal_attention import (
    CrossFuturesTemporalAttention,
    CrossFuturesTemporalAttentionOutput,
)
from gnn_for_futures.shock_follow.graph import (
    PAIR_FEATURE_SCHEMA,
    DirectedRoleEncoder,
    TorchAttentionGraph,
    TorchFusedGraph,
    TorchTopKDirectedViews,
    build_top_k_directed_views,
    fuse_graph_channels,
    restore_current_attention_source_target,
    scale_attention_strength,
)
from gnn_for_futures.shock_follow.models.head_v2 import (
    HierarchicalHeadOutput,
    HierarchicalShockFollowHead,
)


@dataclass(frozen=True)
class ShockFollowModelV2Output:
    """V2 predictions and auditable graph/representation intermediates."""

    follow_logits: Tensor
    direction_logits: Tensor
    joint_scores: Tensor
    source_representation: Tensor
    target_representation: Tensor
    attention: TorchAttentionGraph
    fused_graph: TorchFusedGraph
    directed_views: TorchTopKDirectedViews
    temporal: CrossFuturesTemporalAttentionOutput


class ShockFollowGraphMixerV2(nn.Module):
    """Retain the V1 encoder path and replace only the final prediction task."""

    def __init__(
        self,
        node_feature_dim: int,
        *,
        temporal_hidden_dim: int = 13,
        attention_history_length: int = 40,
        expected_num_futures: int | None = 44,
        role_dim: int = 32,
        neighbor_capacity: int = 5,
        head_hidden_dim: int = 100,
        fusion_weights: tuple[float, float, float] = (0.2, 0.4, 0.4),
        topology_threshold: float = 0.1,
        attention_robust_quantile: float = 0.75,
    ) -> None:
        super().__init__()
        if not 0.0 < attention_robust_quantile <= 1.0:
            raise ValueError("attention_robust_quantile must be in (0,1]")
        if not 0.0 <= topology_threshold <= 1.0:
            raise ValueError("topology_threshold must be in [0,1]")
        self.temporal_attention = CrossFuturesTemporalAttention(
            node_feature_dim,
            hidden_size=temporal_hidden_dim,
            history_length=attention_history_length,
            expected_num_futures=expected_num_futures,
        )
        pair_dim = len(PAIR_FEATURE_SCHEMA)
        self.source_role_encoder = DirectedRoleEncoder(
            temporal_hidden_dim,
            pair_dim,
            output_dim=role_dim,
            neighbor_capacity=neighbor_capacity,
        )
        self.target_role_encoder = DirectedRoleEncoder(
            temporal_hidden_dim,
            pair_dim,
            output_dim=role_dim,
            neighbor_capacity=neighbor_capacity,
        )
        self.prediction_head = HierarchicalShockFollowHead(
            role_dim,
            role_dim,
            pair_dim,
            hidden_dim=head_hidden_dim,
        )
        self.neighbor_capacity = neighbor_capacity
        self.fusion_weights = fusion_weights
        self.topology_threshold = topology_threshold
        self.attention_robust_quantile = attention_robust_quantile

    def forward(
        self,
        node_history: Tensor,
        economic_prior: Tensor,
        regression_weight: Tensor,
        *,
        prior_available: Tensor,
        regression_available: Tensor,
    ) -> ShockFollowModelV2Output:
        """Return raw two-task logits and normalized joint class scores."""
        if node_history.ndim != 4:
            raise ValueError(
                "node_history must have shape [batch,nodes,sequence,node_features]"
            )
        temporal = self.temporal_attention(node_history)
        current_raw, current_available = restore_current_attention_source_target(
            temporal.attention_scores,
            temporal.peer_indices,
        )
        attention = scale_attention_strength(
            current_raw,
            current_available,
            robust_quantile=self.attention_robust_quantile,
        )
        fused = fuse_graph_channels(
            economic_prior,
            regression_weight,
            attention,
            prior_available=prior_available,
            regression_available=regression_available,
            fusion_weights=self.fusion_weights,
            threshold=self.topology_threshold,
        )
        views = build_top_k_directed_views(
            fused.adjacency,
            fused.fused_scores,
            k=self.neighbor_capacity,
        )
        node_states = temporal.hidden_states[:, :, -1, :]
        source_representation = self.source_role_encoder(
            node_states,
            fused.pair_features,
            views.out_neighbors,
            views.out_valid,
        )
        target_representation = self.target_role_encoder(
            node_states,
            fused.pair_features,
            views.in_neighbors,
            views.in_valid,
            transpose_pairs=True,
        )
        head: HierarchicalHeadOutput = self.prediction_head(
            source_representation,
            target_representation,
            fused.pair_features,
        )
        return ShockFollowModelV2Output(
            follow_logits=head.follow_logits,
            direction_logits=head.direction_logits,
            joint_scores=head.joint_scores,
            source_representation=source_representation,
            target_representation=target_representation,
            attention=attention,
            fused_graph=fused,
            directed_views=views,
            temporal=temporal,
        )
