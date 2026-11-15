"""Pure V1 directed graph data contracts used before model approval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


DEFAULT_FUSION_WEIGHTS = (0.2, 0.4, 0.4)
DEFAULT_ATTENTION_HISTORY_LENGTH = 40


@dataclass(frozen=True)
class AttentionGraphData:
    """Attention history restored to source-by-target orientation."""

    history: np.ndarray
    current_raw: np.ndarray
    current_scaled: np.ndarray
    current_available: np.ndarray
    available: bool
    scale: float
    nonzero_rate: float
    saturation_rate: float


@dataclass(frozen=True)
class FusedDirectedGraphData:
    """Three channels, availability masks, continuous scores, and topology."""

    channels: np.ndarray
    channel_available: np.ndarray
    fusion_weights: np.ndarray
    fused_scores: np.ndarray
    adjacency: np.ndarray


@dataclass(frozen=True)
class TopKDirectedViews:
    """Current-cutoff source and target neighbor slots for two directed roles."""

    out_neighbors: np.ndarray
    out_scores: np.ndarray
    out_valid: np.ndarray
    in_neighbors: np.ndarray
    in_scores: np.ndarray
    in_valid: np.ndarray
    out_truncated_edges: int
    in_truncated_edges: int
    out_empty_slots: int
    in_empty_slots: int


def _validate_node_order(node_order: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in node_order)
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("node_order must contain at least two unique names")
    return names


def restore_attention_source_target(
    attention_scores: np.ndarray,
    peer_indices: np.ndarray,
    node_order: Sequence[str],
    *,
    robust_quantile: float = 0.75,
    epsilon: float = 1e-8,
    clip_max: float = 2.0,
    expected_history_length: int | None = DEFAULT_ATTENTION_HISTORY_LENGTH,
) -> AttentionGraphData:
    """Map [target, peer, history] scores into [source, target, history]."""
    names = _validate_node_order(node_order)
    if not 0.0 < robust_quantile <= 1.0:
        raise ValueError("robust_quantile must be in (0, 1]")
    if not np.isfinite(epsilon) or not np.isfinite(clip_max):
        raise ValueError("epsilon and clip_max must be finite")
    if epsilon <= 0.0 or clip_max <= 0.0:
        raise ValueError("epsilon and clip_max must be positive")
    scores = np.asarray(attention_scores, dtype=np.float64)
    peers = np.asarray(peer_indices)
    node_count = len(names)
    if scores.ndim != 3 or scores.shape[:2] != (node_count, node_count - 1):
        raise ValueError(
            "attention_scores must have shape [nodes, nodes-1, history]"
        )
    if scores.shape[2] < 1:
        raise ValueError("attention history must contain at least one position")
    if expected_history_length is not None:
        if (
            not isinstance(expected_history_length, int)
            or isinstance(expected_history_length, bool)
            or expected_history_length < 1
        ):
            raise ValueError("expected_history_length must be positive or None")
        if scores.shape[2] != expected_history_length:
            raise ValueError(
                f"attention history must contain exactly {expected_history_length} positions"
            )
    if peers.shape != (node_count, node_count - 1):
        raise ValueError("peer_indices must have shape [nodes, nodes-1]")
    if not np.issubdtype(peers.dtype, np.integer):
        raise ValueError("peer_indices must contain integers")
    expected_peers = set(range(node_count))
    history = np.full((node_count, node_count, scores.shape[2]), np.nan)
    for target in range(node_count):
        target_peers = peers[target].astype(np.int64, copy=False)
        if set(target_peers.tolist()) != expected_peers.difference({target}):
            raise ValueError(
                "Each attention target must list every other source exactly once"
            )
        history[target_peers, target, :] = scores[target, :, :]
    current_raw = history[:, :, -1].copy()
    np.fill_diagonal(current_raw, 0.0)
    absolute = np.abs(current_raw)
    off_diagonal = ~np.eye(node_count, dtype=bool)
    current_available = off_diagonal & np.isfinite(current_raw)
    eligible = current_available & (absolute > 0.0)
    current_scaled = np.zeros((node_count, node_count), dtype=np.float32)
    if not eligible.any():
        return AttentionGraphData(
            history,
            current_raw,
            current_scaled,
            current_available,
            False,
            0.0,
            0.0,
            0.0,
        )
    scale = float(np.quantile(absolute[eligible], robust_quantile))
    normalized = np.clip(absolute[eligible] / (scale + epsilon), 0.0, clip_max)
    current_scaled[eligible] = (normalized / clip_max).astype(np.float32)
    np.fill_diagonal(current_scaled, 0.0)
    eligible_count = int(eligible.sum())
    return AttentionGraphData(
        history,
        current_raw,
        current_scaled,
        current_available,
        True,
        scale,
        eligible_count / (node_count * (node_count - 1)),
        float(np.isclose(current_scaled[eligible], 1.0).mean()),
    )


def _availability_matrix(value: bool | np.ndarray, node_count: int) -> np.ndarray:
    if isinstance(value, (bool, np.bool_)):
        result = np.full((node_count, node_count), bool(value), dtype=bool)
    else:
        result = np.asarray(value)
        if result.dtype != np.bool_:
            raise TypeError("Channel availability arrays must be boolean")
        if result.shape != (node_count, node_count):
            raise ValueError("Channel availability must be scalar or [nodes, nodes]")
        result = result.copy()
    np.fill_diagonal(result, False)
    return result


def _channel_matrix(value: np.ndarray, node_count: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    if result.shape != (node_count, node_count):
        raise ValueError(f"{name} must have shape [nodes, nodes]")
    if not np.isfinite(result).all() or (result < 0.0).any() or (result > 1.0).any():
        raise ValueError(f"{name} must be finite and in [0, 1]")
    np.fill_diagonal(result, 0.0)
    return result


def simplex_weights(
    weights: Sequence[float] | np.ndarray = DEFAULT_FUSION_WEIGHTS,
    *,
    logits: bool = False,
) -> np.ndarray:
    """Validate fixed simplex weights or transform trainable logits with softmax."""
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Fusion weights must contain three finite values")
    if logits:
        shifted = values - values.max()
        result = np.exp(shifted)
        return result / result.sum()
    if (values < 0.0).any() or not np.isclose(values.sum(), 1.0):
        raise ValueError("Fixed fusion weights must be nonnegative and sum to one")
    return values


def fuse_directed_channels(
    prior: np.ndarray,
    regression: np.ndarray,
    attention: np.ndarray,
    *,
    prior_available: bool | np.ndarray = True,
    regression_available: bool | np.ndarray,
    attention_available: bool | np.ndarray,
    weights: Sequence[float] | np.ndarray = DEFAULT_FUSION_WEIGHTS,
    weights_are_logits: bool = False,
    threshold: float = 0.6,
) -> FusedDirectedGraphData:
    """Build g, channel masks, F, and strict F>L zero-diagonal topology."""
    prior_array = np.asarray(prior)
    if prior_array.ndim != 2 or prior_array.shape[0] != prior_array.shape[1]:
        raise ValueError("prior must be a square matrix")
    node_count = prior_array.shape[0]
    channels = np.stack(
        (
            _channel_matrix(prior, node_count, "prior"),
            _channel_matrix(regression, node_count, "regression"),
            _channel_matrix(attention, node_count, "attention"),
        ),
        axis=-1,
    )
    available = np.stack(
        (
            _availability_matrix(prior_available, node_count),
            _availability_matrix(regression_available, node_count),
            _availability_matrix(attention_available, node_count),
        ),
        axis=-1,
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    fusion_weights = simplex_weights(weights, logits=weights_are_logits)
    effective_channels = np.where(available, channels, 0.0).astype(np.float64)
    fused = np.sum(effective_channels * fusion_weights, axis=-1)
    fused[np.isclose(fused, threshold, rtol=0.0, atol=1e-12)] = threshold
    np.fill_diagonal(fused, 0.0)
    adjacency = fused > threshold
    np.fill_diagonal(adjacency, False)
    return FusedDirectedGraphData(
        channels,
        available,
        fusion_weights,
        fused.astype(np.float32),
        adjacency,
    )


def _top_k_rows(
    adjacency: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    node_count = adjacency.shape[0]
    neighbors = np.full((node_count, k), -1, dtype=np.int64)
    values = np.zeros((node_count, k), dtype=np.float32)
    valid = np.zeros((node_count, k), dtype=bool)
    truncated = 0
    for node in range(node_count):
        candidates = np.flatnonzero(adjacency[node])
        ordered = sorted(candidates.tolist(), key=lambda item: (-scores[node, item], item))
        truncated += max(0, len(ordered) - k)
        kept = ordered[:k]
        if kept:
            count = len(kept)
            neighbors[node, :count] = kept
            values[node, :count] = scores[node, kept]
            valid[node, :count] = True
    return neighbors, values, valid, truncated, int((~valid).sum())


def build_top_k_directed_views(
    adjacency: np.ndarray,
    fused_scores: np.ndarray,
    node_order: Sequence[str],
    *,
    k: int = 5,
) -> TopKDirectedViews:
    """Build deterministic current-cutoff out=C and in=C.T neighbor slots."""
    names = _validate_node_order(node_order)
    node_count = len(names)
    connected = np.asarray(adjacency, dtype=bool).copy()
    scores = np.asarray(fused_scores, dtype=np.float32).copy()
    if connected.shape != (node_count, node_count) or scores.shape != connected.shape:
        raise ValueError("adjacency and fused_scores must match the node order")
    if not np.isfinite(scores).all():
        raise ValueError("fused_scores must be finite")
    if (scores < 0.0).any() or (scores > 1.0).any():
        raise ValueError("fused_scores must be in [0, 1]")
    if (connected & (scores <= 0.0)).any():
        raise ValueError("Connected edges must have strictly positive fused scores")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    np.fill_diagonal(connected, False)
    np.fill_diagonal(scores, 0.0)
    out = _top_k_rows(connected, scores, k)
    incoming = _top_k_rows(connected.T, scores.T, k)
    return TopKDirectedViews(
        out[0],
        out[1],
        out[2],
        incoming[0],
        incoming[1],
        incoming[2],
        out[3],
        incoming[3],
        out[4],
        incoming[4],
    )
