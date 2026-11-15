"""Causal-history graph variants with sample-dependent fusion gates."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from gnn_for_futures.shock_follow.graph import (
    TorchAttentionGraph, TorchFusedGraph, _broadcast_square,
    restore_current_attention_source_target, scale_attention_strength,
    fuse_graph_channels, build_top_k_directed_views,
)
from gnn_for_futures.shock_follow.models.mixer_v2 import (
    ShockFollowGraphMixerV2, ShockFollowModelV2Output,
)


VARIANTS = {
    "history-replace-prior": ("H", "R", "A"),
    "history-replace-regression": ("P", "H", "A"),
    "history-replace-attention": ("P", "R", "H"),
    "history-four": ("P", "R", "A", "H"),
    "history-prior": ("H", "P"),
    "history-regression": ("H", "R"),
    "history-attention": ("H", "A"),
    "history-attention-multiply": ("P", "R", "A"),
}


@dataclass(frozen=True)
class HistoryFusionOutput(ShockFollowModelV2Output):
    fusion_weights: Tensor
    history_multiplier: Tensor | None


class HistoryFusionGraphMixerV2(ShockFollowGraphMixerV2):
    def __init__(self, node_feature_dim: int, *, variant: str, **kwargs):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown history variant: {variant}")
        if kwargs.get("learnable_fusion", False):
            raise ValueError("History variants use dynamic fusion, not global fusion_logits")
        if kwargs.get("enabled_channels", (True, True, True)) != (True, True, True):
            raise ValueError("Select channels using variant")
        super().__init__(node_feature_dim, **kwargs)
        self.variant = variant
        self.channel_names = VARIANTS[variant]
        self.fusion_gate = nn.Linear(self.temporal_attention.hidden_size, len(self.channel_names))
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)
        self.register_parameter("raw_history_multiplier", nn.Parameter(torch.tensor(math.log(math.expm1(2.0))))
                                if variant == "history-attention-multiply" else None)

    @property
    def history_multiplier(self):
        return (None if self.raw_history_multiplier is None
                else functional.softplus(self.raw_history_multiplier) + torch.finfo(self.raw_history_multiplier.dtype).tiny)

    def modulate_attention(self, logits, history, available):
        links = history * available.to(logits.dtype)
        multiplier = 1 + (self.history_multiplier - 1) * links
        diagonal = torch.eye(logits.shape[-1], device=logits.device, dtype=torch.bool)
        return (logits * multiplier).masked_fill(diagonal, -torch.inf).softmax(dim=-2)

    def forward(self, node_history, economic_prior, regression_weight, *,
                prior_available, regression_available, historical_weight, historical_available):
        if node_history.ndim != 4:
            raise ValueError("node_history must have shape [batch,nodes,sequence,features]")
        batch_size, node_count = node_history.shape[:2]
        diagonal = torch.eye(node_count, dtype=torch.bool, device=node_history.device)
        values, masks = {}, {}
        for name, value, mask in (("P", economic_prior, prior_available),
                                  ("R", regression_weight, regression_available),
                                  ("H", historical_weight, historical_available)):
            value = _broadcast_square(value, batch_size, name)
            mask = _broadcast_square(mask, batch_size, name + "_available")
            if value.shape != (batch_size, node_count, node_count) or mask.shape != value.shape:
                raise ValueError("Graph shape must match node_history")
            if mask.dtype != torch.bool or not value.is_floating_point():
                raise TypeError("Graph values must be floating point and masks boolean")
            if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                raise ValueError("Graph values must be finite in [0,1]")
            if name == "H" and ((value != 0) & (value != 1)).any():
                raise ValueError("Historical links must be binary")
            masks[name] = mask & ~diagonal
            values[name] = value.masked_fill(~masks[name], 0)
        temporal = self.temporal_attention(node_history)
        raw, available = restore_current_attention_source_target(temporal.attention_scores, temporal.peer_indices)
        if self.history_multiplier is not None:
            raw = self.modulate_attention(raw, values["H"], masks["H"])
        attention = scale_attention_strength(raw, available,
            robust_quantile=self.attention_robust_quantile,
            fixed_scale=None if self.fixed_attention_scale is None else float(self.fixed_attention_scale))
        values["A"], masks["A"] = attention.current_scaled, attention.current_available
        states = temporal.hidden_states[:, :, -1, :]
        weights = self.fusion_gate(states.mean(dim=1)).softmax(dim=-1)
        channels = torch.stack([values[name] for name in self.channel_names], dim=-1)
        channel_masks = torch.stack([masks[name] for name in self.channel_names], dim=-1)
        scores = (channels * weights[:, None, None, :]).sum(-1)
        adjacency = scores.gt(self.topology_threshold) & ~diagonal
        slots = {name: name if name in self.channel_names else None for name in ("P", "R", "A")}
        if "H" in self.channel_names and len(self.channel_names) < 4:
            missing = next(name for name in ("R", "P", "A") if slots[name] is None)
            slots[missing] = "H"
        zero = torch.zeros_like(raw)
        false = torch.zeros_like(available)
        def slot(name):
            key = slots[name]
            return (zero, false) if key is None else (values[key], masks[key])
        prior, prior_mask = slot("P")
        regression, regression_mask = slot("R")
        attention_value, attention_mask = slot("A")
        feature_attention = TorchAttentionGraph(
            raw if slots["A"] == "A" else attention_value,
            attention_value, attention_mask, attention.scale, attention.saturation_rate)
        template = fuse_graph_channels(prior, regression, feature_attention,
            prior_available=prior_mask, regression_available=regression_mask,
            threshold=self.topology_threshold)
        pair_features = torch.cat((template.pair_features[..., :7], scores.unsqueeze(-1),
                                   adjacency.to(scores.dtype).unsqueeze(-1)), dim=-1)
        fused = TorchFusedGraph(channels, channel_masks, scores, adjacency, pair_features)
        views = build_top_k_directed_views(adjacency, scores, k=self.neighbor_capacity)
        source = self.source_role_encoder(states, pair_features, views.out_neighbors, views.out_valid)
        target = self.target_role_encoder(states, pair_features, views.in_neighbors, views.in_valid, transpose_pairs=True)
        head = self.prediction_head(source, target, pair_features)
        return HistoryFusionOutput(head.follow_logits, head.direction_logits, head.joint_scores,
            source, target, attention, fused, views, temporal, weights, self.history_multiplier)