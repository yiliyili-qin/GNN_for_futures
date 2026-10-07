"""Differentiable directed graph assembly for shock-follow GraphMixer V1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


PAIR_FEATURE_SCHEMA = (
    "economic_prior",
    "regression_weight",
    "attention_raw_signed",
    "attention_absolute_scaled",
    "prior_available",
    "regression_available",
    "attention_available",
    "fused_score",
    "topology",
)


@dataclass(frozen=True)
class TorchAttentionGraph:
    """Current attention restored to row-source, column-target orientation."""

    current_raw: Tensor
    current_scaled: Tensor
    current_available: Tensor
    scale: Tensor
    saturation_rate: Tensor


@dataclass(frozen=True)
class TorchFusedGraph:
    """Batched graph channels, pair features, scores, and directed topology."""

    channels: Tensor
    channel_available: Tensor
    fused_scores: Tensor
    adjacency: Tensor
    pair_features: Tensor


@dataclass(frozen=True)
class TorchTopKDirectedViews:
    """Safe padded outgoing and incoming neighbor slots."""

    out_neighbors: Tensor
    out_scores: Tensor
    out_valid: Tensor
    in_neighbors: Tensor
    in_scores: Tensor
    in_valid: Tensor


def _require_square_batch(values: Tensor, name: str) -> tuple[Tensor, bool]:
    if values.ndim == 2:
        values = values.unsqueeze(0)
        unbatched = True
    elif values.ndim == 3:
        unbatched = False
    else:
        raise ValueError(f"{name} must have shape [nodes,nodes] or [batch,nodes,nodes]")
    if values.shape[-2] != values.shape[-1]:
        raise ValueError(f"{name} must be square")
    return values, unbatched


def restore_current_attention_source_target(
    attention_scores: Tensor,
    peer_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Restore current raw scores from [target,peer,history] to [source,target]."""
    if attention_scores.ndim == 3:
        scores = attention_scores.unsqueeze(0)
        unbatched = True
    elif attention_scores.ndim == 4:
        scores = attention_scores
        unbatched = False
    else:
        raise ValueError(
            "attention_scores must have shape [nodes,nodes-1,history] or "
            "[batch,nodes,nodes-1,history]"
        )
    if not scores.is_floating_point():
        raise TypeError("attention_scores must use a floating-point dtype")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("attention_scores must be finite")
    batch_size, node_count, peer_count, history_length = scores.shape
    if node_count < 2 or peer_count != node_count - 1 or history_length < 1:
        raise ValueError("attention_scores has an invalid node or history dimension")
    if peer_indices.shape != (node_count, peer_count):
        raise ValueError("peer_indices must have shape [nodes,nodes-1]")
    if peer_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("peer_indices must contain integers")
    if peer_indices.device != scores.device:
        raise ValueError("peer_indices and attention_scores must use the same device")
    expected = torch.arange(node_count, device=scores.device)
    for target in range(node_count):
        observed = torch.sort(peer_indices[target].to(torch.int64)).values
        required = torch.cat((expected[:target], expected[target + 1 :]))
        if not torch.equal(observed, required):
            raise ValueError("Each target must list every other source exactly once")

    current_target_source = scores[..., -1].new_zeros(
        (batch_size, node_count, node_count)
    )
    scatter_indices = peer_indices.to(torch.int64).unsqueeze(0).expand(batch_size, -1, -1)
    current_target_source = current_target_source.scatter(
        2,
        scatter_indices,
        scores[..., -1],
    )
    current_raw = current_target_source.transpose(1, 2)
    off_diagonal = ~torch.eye(
        node_count,
        dtype=torch.bool,
        device=scores.device,
    ).unsqueeze(0)
    current_available = torch.isfinite(current_raw) & off_diagonal
    current_raw = torch.where(current_available, current_raw, torch.zeros_like(current_raw))
    if unbatched:
        return current_raw.squeeze(0), current_available.squeeze(0)
    return current_raw, current_available


def scale_attention_strength(
    current_raw: Tensor,
    current_available: Tensor,
    *,
    robust_quantile: float = 0.75,
    epsilon: float = 1e-8,
    clip_max: float = 2.0,
) -> TorchAttentionGraph:
    """Scale absolute attention per sample while retaining signed raw scores."""
    raw, unbatched = _require_square_batch(current_raw, "current_raw")
    available, availability_unbatched = _require_square_batch(
        current_available, "current_available"
    )
    if availability_unbatched != unbatched or available.shape != raw.shape:
        raise ValueError("current_raw and current_available must have matching shapes")
    if available.dtype != torch.bool:
        raise TypeError("current_available must be boolean")
    if not 0.0 < robust_quantile <= 1.0:
        raise ValueError("robust_quantile must be in (0,1]")
    if not torch.isfinite(raw.new_tensor(epsilon)) or not torch.isfinite(raw.new_tensor(clip_max)):
        raise ValueError("epsilon and clip_max must be finite")
    if epsilon <= 0.0 or clip_max <= 0.0:
        raise ValueError("epsilon and clip_max must be positive")
    if not raw.is_floating_point():
        raise TypeError("current_raw must use a floating-point dtype")

    absolute = raw.abs()
    scaled = torch.zeros_like(raw)
    scales: list[Tensor] = []
    saturation: list[Tensor] = []
    for batch_index in range(raw.shape[0]):
        eligible = available[batch_index] & torch.isfinite(raw[batch_index]) & absolute[
            batch_index
        ].gt(0.0)
        values = absolute[batch_index][eligible]
        if values.numel() == 0:
            scale = raw.new_zeros(())
            rate = raw.new_zeros(())
        else:
            scale = torch.quantile(values, robust_quantile).detach()
            normalized = torch.clamp(values / (scale + epsilon), 0.0, clip_max)
            sample_scaled = scaled[batch_index].masked_scatter(
                eligible,
                normalized / clip_max,
            )
            scaled[batch_index] = sample_scaled
            rate = torch.isclose(
                normalized,
                normalized.new_full((), clip_max),
            ).to(raw.dtype).mean()
        scales.append(scale)
        saturation.append(rate)
    scale_tensor = torch.stack(scales)
    saturation_tensor = torch.stack(saturation)
    if unbatched:
        return TorchAttentionGraph(
            raw.squeeze(0),
            scaled.squeeze(0),
            available.squeeze(0),
            scale_tensor.squeeze(0),
            saturation_tensor.squeeze(0),
        )
    return TorchAttentionGraph(
        raw,
        scaled,
        available,
        scale_tensor,
        saturation_tensor,
    )


def _broadcast_square(values: Tensor, batch_size: int, name: str) -> Tensor:
    batched, _ = _require_square_batch(values, name)
    if batched.shape[0] == 1 and batch_size != 1:
        batched = batched.expand(batch_size, -1, -1)
    if batched.shape[0] != batch_size:
        raise ValueError(f"{name} batch dimension does not match")
    return batched


def fuse_graph_channels(
    economic_prior: Tensor,
    regression_weight: Tensor,
    attention: TorchAttentionGraph,
    *,
    prior_available: Tensor,
    regression_available: Tensor,
    fusion_weights: tuple[float, float, float] = (0.2, 0.4, 0.4),
    threshold: float = 0.6,
) -> TorchFusedGraph:
    """Fuse three nonnegative channels and build the nine-field pair schema."""
    attention_scaled, _ = _require_square_batch(
        attention.current_scaled, "attention.current_scaled"
    )
    batch_size, node_count, _ = attention_scaled.shape
    prior = _broadcast_square(economic_prior, batch_size, "economic_prior")
    regression = _broadcast_square(regression_weight, batch_size, "regression_weight")
    prior_mask = _broadcast_square(prior_available, batch_size, "prior_available")
    regression_mask = _broadcast_square(
        regression_available, batch_size, "regression_available"
    )
    attention_mask = _broadcast_square(
        attention.current_available, batch_size, "attention.current_available"
    )
    attention_raw = _broadcast_square(
        attention.current_raw, batch_size, "attention.current_raw"
    )
    for name, values in (
        ("economic_prior", prior),
        ("regression_weight", regression),
        ("attention.current_scaled", attention_scaled),
    ):
        if not values.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if not torch.isfinite(values).all() or values.lt(0.0).any() or values.gt(1.0).any():
            raise ValueError(f"{name} must be finite and in [0,1]")
    for name, values in (
        ("prior_available", prior_mask),
        ("regression_available", regression_mask),
        ("attention.current_available", attention_mask),
    ):
        if values.dtype != torch.bool:
            raise TypeError(f"{name} must be boolean")
    if tuple(prior.shape[-2:]) != (node_count, node_count):
        raise ValueError("All graph matrices must have the same node dimensions")
    weights = prior.new_tensor(fusion_weights)
    if weights.shape != (3,) or not torch.isfinite(weights).all():
        raise ValueError("fusion_weights must contain three finite values")
    if weights.lt(0.0).any() or not torch.isclose(weights.sum(), weights.new_tensor(1.0)):
        raise ValueError("fusion_weights must be nonnegative and sum to one")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")

    diagonal = torch.eye(node_count, dtype=torch.bool, device=prior.device).unsqueeze(0)
    prior_mask = prior_mask & ~diagonal
    regression_mask = regression_mask & ~diagonal
    attention_mask = attention_mask & ~diagonal
    channels = torch.stack((prior, regression, attention_scaled), dim=-1)
    availability = torch.stack((prior_mask, regression_mask, attention_mask), dim=-1)
    effective = torch.where(availability, channels, torch.zeros_like(channels))
    fused_scores = (effective * weights).sum(dim=-1).masked_fill(diagonal, 0.0)
    adjacency = fused_scores.gt(threshold) & ~diagonal
    pair_features = torch.stack(
        (
            effective[..., 0],
            effective[..., 1],
            attention_raw,
            attention_scaled,
            prior_mask.to(prior.dtype),
            regression_mask.to(prior.dtype),
            attention_mask.to(prior.dtype),
            fused_scores,
            adjacency.to(prior.dtype),
        ),
        dim=-1,
    )
    return TorchFusedGraph(
        channels,
        availability,
        fused_scores,
        adjacency,
        pair_features,
    )


def _top_k_rows(adjacency: Tensor, scores: Tensor, k: int) -> tuple[Tensor, Tensor, Tensor]:
    batch_size, node_count, _ = adjacency.shape
    masked_scores = scores.masked_fill(~adjacency, float("-inf"))
    order = torch.argsort(masked_scores, dim=-1, descending=True, stable=True)
    kept = order[..., : min(k, node_count)]
    kept_scores = torch.gather(masked_scores, 2, kept)
    valid = torch.isfinite(kept_scores)
    if k > node_count:
        padding = k - node_count
        kept = torch.cat(
            (kept, torch.zeros((batch_size, node_count, padding), dtype=torch.long, device=kept.device)),
            dim=-1,
        )
        kept_scores = torch.cat(
            (kept_scores, torch.zeros((batch_size, node_count, padding), dtype=scores.dtype, device=scores.device)),
            dim=-1,
        )
        valid = torch.cat(
            (valid, torch.zeros((batch_size, node_count, padding), dtype=torch.bool, device=scores.device)),
            dim=-1,
        )
    safe_scores = torch.where(valid, kept_scores, torch.zeros_like(kept_scores))
    neighbors = torch.where(valid, kept, torch.full_like(kept, -1))
    return neighbors, safe_scores, valid


def build_top_k_directed_views(
    adjacency: Tensor,
    fused_scores: Tensor,
    *,
    k: int = 5,
) -> TorchTopKDirectedViews:
    """Build deterministic outgoing and incoming slots for a graph batch."""
    connected, adjacency_unbatched = _require_square_batch(adjacency, "adjacency")
    scores, scores_unbatched = _require_square_batch(fused_scores, "fused_scores")
    if adjacency_unbatched != scores_unbatched or connected.shape != scores.shape:
        raise ValueError("adjacency and fused_scores must have matching shapes")
    if connected.dtype != torch.bool:
        raise TypeError("adjacency must be boolean")
    if not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError("fused_scores must be finite floating-point values")
    if scores.lt(0.0).any() or scores.gt(1.0).any():
        raise ValueError("fused_scores must be in [0,1]")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    out = _top_k_rows(connected, scores, k)
    incoming = _top_k_rows(connected.transpose(1, 2), scores.transpose(1, 2), k)
    result = TorchTopKDirectedViews(
        out[0], out[1], out[2], incoming[0], incoming[1], incoming[2]
    )
    if adjacency_unbatched:
        return TorchTopKDirectedViews(*(value.squeeze(0) for value in result.__dict__.values()))
    return result


class DirectedRoleEncoder(nn.Module):
    """Encode one directed role with safe fixed-capacity neighbor pooling."""

    def __init__(
        self,
        node_dim: int,
        pair_feature_dim: int,
        *,
        output_dim: int,
        neighbor_capacity: int = 5,
    ) -> None:
        super().__init__()
        if min(node_dim, pair_feature_dim, output_dim, neighbor_capacity) < 1:
            raise ValueError("Encoder dimensions and neighbor_capacity must be positive")
        self.node_dim = node_dim
        self.pair_feature_dim = pair_feature_dim
        self.output_dim = output_dim
        self.neighbor_capacity = neighbor_capacity
        self.self_projection = nn.Linear(node_dim, output_dim)
        self.token_projection = nn.Linear(node_dim + pair_feature_dim, output_dim)

    def forward(
        self,
        node_states: Tensor,
        pair_features: Tensor,
        neighbors: Tensor,
        valid: Tensor,
        *,
        transpose_pairs: bool = False,
    ) -> Tensor:
        """Return role-specific node representations with padding re-masked."""
        if node_states.ndim != 3:
            raise ValueError("node_states must have shape [batch,nodes,node_dim]")
        batch_size, node_count, node_dim = node_states.shape
        expected_pairs = (batch_size, node_count, node_count, self.pair_feature_dim)
        if pair_features.shape != expected_pairs:
            raise ValueError(f"pair_features must have shape {expected_pairs}")
        expected_slots = (batch_size, node_count, self.neighbor_capacity)
        if neighbors.shape != expected_slots or valid.shape != expected_slots:
            raise ValueError(f"neighbors and valid must have shape {expected_slots}")
        if node_dim != self.node_dim:
            raise ValueError(f"Expected node_dim={self.node_dim}, got {node_dim}")
        if neighbors.dtype not in (torch.int32, torch.int64):
            raise TypeError("neighbors must contain integers")
        if valid.dtype != torch.bool:
            raise TypeError("valid must be boolean")
        if (neighbors[valid] < 0).any() or (neighbors[valid] >= node_count).any():
            raise ValueError("Valid neighbor indices are out of range")
        safe_neighbors = neighbors.clamp_min(0).to(torch.int64)
        flat_indices = safe_neighbors.reshape(batch_size, -1)
        gathered_nodes = torch.gather(
            node_states,
            1,
            flat_indices.unsqueeze(-1).expand(-1, -1, self.node_dim),
        ).reshape(batch_size, node_count, self.neighbor_capacity, self.node_dim)
        role_pairs = pair_features.transpose(1, 2) if transpose_pairs else pair_features
        gathered_pairs = torch.gather(
            role_pairs,
            2,
            safe_neighbors.unsqueeze(-1).expand(
                -1, -1, -1, self.pair_feature_dim
            ),
        )
        token_inputs = torch.cat((gathered_nodes, gathered_pairs), dim=-1)
        token_hidden = torch.relu(self.token_projection(token_inputs))
        token_hidden = token_hidden * valid.unsqueeze(-1).to(token_hidden.dtype)
        pooled = token_hidden.sum(dim=2) / self.neighbor_capacity
        return torch.relu(self.self_projection(node_states) + pooled)
