"""Hierarchical prediction and losses for shock-follow GraphMixer V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from gnn_for_futures.shock_follow.contract import (
    CLASS_NO_FOLLOW,
    CLASS_OPPOSITE_DIRECTION,
    CLASS_ORDER,
    CLASS_SAME_DIRECTION,
)


FOLLOW_CLASS_ORDER = ("no_follow", "follow")
DIRECTION_CLASS_ORDER = ("opposite_direction", "same_direction")
FOLLOW_THRESHOLD = 0.5


@dataclass(frozen=True)
class HierarchicalHeadOutput:
    """Raw two-task logits and their normalized joint three-class scores."""

    follow_logits: Tensor
    direction_logits: Tensor
    joint_scores: Tensor


@dataclass(frozen=True)
class HierarchicalLossOutcome:
    """Source-event-reduced V2 losses and explicit empty-state metadata."""

    total_loss: Tensor | None
    follow_loss: Tensor | None
    direction_loss: Tensor | None
    is_empty: bool
    direction_is_empty: bool
    follow_event_count: int
    direction_event_count: int
    valid_pair_count: int
    direction_pair_count: int


def balanced_follow_class_weights(class_counts: Sequence[int]) -> tuple[float, float]:
    """Return training-only balanced weights for no-follow versus follow."""
    if len(class_counts) != len(CLASS_ORDER):
        raise ValueError("class_counts must contain the three shock-follow classes")
    normalized: list[int] = []
    for value in class_counts:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("class_counts must contain nonnegative integers")
        normalized.append(value)
    no_follow = normalized[CLASS_NO_FOLLOW]
    follow = normalized[CLASS_OPPOSITE_DIRECTION] + normalized[CLASS_SAME_DIRECTION]
    if no_follow == 0 or follow == 0:
        raise ValueError("balanced follow weights require both no-follow and follow cells")
    total = no_follow + follow
    return total / (2.0 * no_follow), total / (2.0 * follow)


def _validate_pair_logits(logits: Tensor, final_dim: int, name: str) -> None:
    if logits.ndim != 4 or logits.shape[-1] != final_dim:
        raise ValueError(
            f"{name} must have shape [batch,nodes,nodes,{final_dim}]"
        )
    if logits.shape[1] != logits.shape[2]:
        raise ValueError(f"{name} source and target dimensions must be equal")
    if not logits.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def hierarchical_joint_scores(
    follow_logits: Tensor,
    direction_logits: Tensor,
) -> Tensor:
    """Compose normalized [no, opposite, same] scores from the two tasks."""
    _validate_pair_logits(follow_logits, len(FOLLOW_CLASS_ORDER), "follow_logits")
    _validate_pair_logits(
        direction_logits,
        len(DIRECTION_CLASS_ORDER),
        "direction_logits",
    )
    if direction_logits.shape[:-1] != follow_logits.shape[:-1]:
        raise ValueError("follow_logits and direction_logits must address the same pairs")
    follow_scores = torch.softmax(follow_logits, dim=-1)
    direction_scores = torch.softmax(direction_logits, dim=-1)
    p_no = follow_scores[..., 0]
    p_follow = follow_scores[..., 1]
    return torch.stack(
        (
            p_no,
            p_follow * direction_scores[..., 0],
            p_follow * direction_scores[..., 1],
        ),
        dim=-1,
    )


def hierarchical_predictions(
    follow_logits: Tensor,
    direction_logits: Tensor,
) -> Tensor:
    """Apply the frozen 0.5 follow rule, with exact ties mapped to no-follow."""
    _validate_pair_logits(follow_logits, len(FOLLOW_CLASS_ORDER), "follow_logits")
    _validate_pair_logits(
        direction_logits,
        len(DIRECTION_CLASS_ORDER),
        "direction_logits",
    )
    if direction_logits.shape[:-1] != follow_logits.shape[:-1]:
        raise ValueError("follow_logits and direction_logits must address the same pairs")
    predicted_follow = follow_logits[..., 1] > follow_logits[..., 0]
    predicted_direction = direction_logits.argmax(dim=-1) + CLASS_OPPOSITE_DIRECTION
    return torch.where(
        predicted_follow,
        predicted_direction,
        torch.full_like(predicted_direction, CLASS_NO_FOLLOW),
    )


class HierarchicalShockFollowHead(nn.Module):
    """Score follow existence and conditional direction with independent outputs."""

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        pair_feature_dim: int = 0,
        *,
        hidden_dim: int = 100,
    ) -> None:
        super().__init__()
        if min(source_dim, target_dim, pair_feature_dim, hidden_dim) < 0:
            raise ValueError("dimensions must be nonnegative")
        if source_dim == 0 or target_dim == 0 or hidden_dim == 0:
            raise ValueError("source_dim, target_dim, and hidden_dim must be positive")
        self.source_projection = nn.Linear(source_dim, hidden_dim)
        self.target_projection = nn.Linear(target_dim, hidden_dim)
        self.pair_projection = (
            nn.Linear(pair_feature_dim, hidden_dim) if pair_feature_dim else None
        )
        self.follow_output = nn.Linear(hidden_dim, len(FOLLOW_CLASS_ORDER))
        self.direction_output = nn.Linear(hidden_dim, len(DIRECTION_CLASS_ORDER))

    def forward(
        self,
        source_representation: Tensor,
        target_representation: Tensor,
        pair_features: Tensor | None = None,
    ) -> HierarchicalHeadOutput:
        """Return two raw logit tensors and joint scores for every ordered pair."""
        if source_representation.ndim != 3 or target_representation.ndim != 3:
            raise ValueError(
                "source and target representations must be [batch,nodes,features]"
            )
        batch, nodes, _ = source_representation.shape
        if target_representation.shape[:2] != (batch, nodes):
            raise ValueError("source and target batch/node dimensions must match")
        hidden = self.source_projection(source_representation).unsqueeze(2)
        hidden = hidden + self.target_projection(target_representation).unsqueeze(1)
        if self.pair_projection is None:
            if pair_features is not None:
                raise ValueError("pair_features supplied but pair_feature_dim is zero")
        else:
            expected = (batch, nodes, nodes, self.pair_projection.in_features)
            if pair_features is None or pair_features.shape != expected:
                raise ValueError(f"pair_features must have shape {expected}")
            hidden = hidden + self.pair_projection(pair_features)
        hidden = torch.relu(hidden)
        follow_logits = self.follow_output(hidden)
        direction_logits = self.direction_output(hidden)
        diagonal = torch.eye(nodes, dtype=torch.bool, device=hidden.device)
        diagonal = diagonal.unsqueeze(0).unsqueeze(-1)
        follow_logits = follow_logits.masked_fill(diagonal, 0.0)
        direction_logits = direction_logits.masked_fill(diagonal, 0.0)
        return HierarchicalHeadOutput(
            follow_logits,
            direction_logits,
            hierarchical_joint_scores(follow_logits, direction_logits),
        )


def _source_event_mean(
    cell_losses: Tensor,
    mask: Tensor,
) -> tuple[Tensor | None, int]:
    counts = mask.sum(dim=-1)
    event_mask = counts > 0
    event_count = int(event_mask.sum().item())
    if event_count == 0:
        return None, 0
    per_cell = torch.zeros_like(mask, dtype=cell_losses.dtype)
    per_cell[mask] = cell_losses
    per_source = per_cell.sum(dim=-1) / counts.clamp_min(1)
    return per_source[event_mask].mean(), event_count


def hierarchical_source_event_loss(
    follow_logits: Tensor,
    direction_logits: Tensor,
    target_class: Tensor,
    loss_mask: Tensor,
    follow_class_weights: Tensor,
    *,
    lambda_follow: float = 1.0,
    lambda_direction: float = 1.0,
) -> HierarchicalLossOutcome:
    """Compute balanced follow loss and truth-masked conditional direction loss."""
    _validate_pair_logits(follow_logits, len(FOLLOW_CLASS_ORDER), "follow_logits")
    _validate_pair_logits(
        direction_logits,
        len(DIRECTION_CLASS_ORDER),
        "direction_logits",
    )
    if direction_logits.shape[:-1] != follow_logits.shape[:-1]:
        raise ValueError("follow_logits and direction_logits must address the same pairs")
    if target_class.shape != follow_logits.shape[:-1]:
        raise ValueError("target_class must match logits without their class dimension")
    if loss_mask.shape != target_class.shape:
        raise ValueError("loss_mask must have the same shape as target_class")
    if loss_mask.dtype != torch.bool:
        raise TypeError("loss_mask must be boolean")
    if follow_class_weights.shape != (len(FOLLOW_CLASS_ORDER),):
        raise ValueError("follow_class_weights must have shape [2]")
    if (
        not follow_class_weights.is_floating_point()
        or not torch.isfinite(follow_class_weights).all()
        or follow_class_weights.le(0.0).any()
    ):
        raise ValueError("follow_class_weights must be finite and strictly positive")
    if follow_class_weights.device != follow_logits.device:
        raise ValueError("follow_class_weights and logits must use the same device")
    for name, value in (
        ("lambda_follow", lambda_follow),
        ("lambda_direction", lambda_direction),
    ):
        scalar = follow_logits.new_tensor(value)
        if not torch.isfinite(scalar) or value <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")

    diagonal = torch.eye(
        target_class.shape[1],
        dtype=torch.bool,
        device=target_class.device,
    ).unsqueeze(0)
    mask = loss_mask & ~diagonal
    valid_targets = target_class[mask]
    if valid_targets.numel() == 0:
        return HierarchicalLossOutcome(None, None, None, True, True, 0, 0, 0, 0)
    if ((valid_targets < 0) | (valid_targets >= len(CLASS_ORDER))).any():
        raise ValueError("masked target classes must be in [0, 2]")

    follow_targets = valid_targets.ne(CLASS_NO_FOLLOW).long()
    follow_cell_losses = functional.cross_entropy(
        follow_logits[mask],
        follow_targets,
        weight=follow_class_weights.to(dtype=follow_logits.dtype),
        reduction="none",
    )
    follow_loss, follow_event_count = _source_event_mean(follow_cell_losses, mask)
    if follow_loss is None:
        raise AssertionError("a nonempty pair mask must contain a source event")

    direction_mask = mask & target_class.ne(CLASS_NO_FOLLOW)
    direction_targets = target_class[direction_mask] - CLASS_OPPOSITE_DIRECTION
    if direction_targets.numel():
        if ((direction_targets < 0) | (direction_targets >= 2)).any():
            raise ValueError("direction targets must map to [opposite, same]")
        direction_cell_losses = functional.cross_entropy(
            direction_logits[direction_mask],
            direction_targets.long(),
            reduction="none",
        )
        direction_loss, direction_event_count = _source_event_mean(
            direction_cell_losses,
            direction_mask,
        )
    else:
        direction_loss, direction_event_count = None, 0

    total_loss = follow_loss * lambda_follow
    if direction_loss is not None:
        total_loss = total_loss + direction_loss * lambda_direction
    return HierarchicalLossOutcome(
        total_loss=total_loss,
        follow_loss=follow_loss,
        direction_loss=direction_loss,
        is_empty=False,
        direction_is_empty=direction_loss is None,
        follow_event_count=follow_event_count,
        direction_event_count=direction_event_count,
        valid_pair_count=int(mask.sum().item()),
        direction_pair_count=int(direction_mask.sum().item()),
    )
