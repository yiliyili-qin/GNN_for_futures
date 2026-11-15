"""Asymmetric GraphMixer-style three-logit shock-follow prediction head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from gnn_for_futures.shock_follow.contract import CLASS_ORDER


@dataclass(frozen=True)
class MaskedCrossEntropyOutcome:
    """Loss outcome; an empty event batch intentionally has no scalar loss."""

    loss: Tensor | None
    is_empty: bool
    source_event_count: int


class AsymmetricShockFollowHead(nn.Module):
    """Score ordered source-target pairs with separate role projections."""

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
        self.output_projection = nn.Linear(hidden_dim, len(CLASS_ORDER))

    def forward(
        self,
        source_representation: Tensor,
        target_representation: Tensor,
        pair_features: Tensor | None = None,
    ) -> Tensor:
        """Return raw ``[batch, source, target, 3]`` logits with zero diagonal."""
        if source_representation.ndim != 3 or target_representation.ndim != 3:
            raise ValueError("source and target representations must be [batch,nodes,features]")
        batch, nodes, _ = source_representation.shape
        if target_representation.shape[:2] != (batch, nodes):
            raise ValueError("source and target batch/node dimensions must match")
        hidden = self.source_projection(source_representation).unsqueeze(2)
        hidden = hidden + self.target_projection(target_representation).unsqueeze(1)
        if self.pair_projection is None:
            if pair_features is not None:
                raise ValueError("pair_features supplied but pair_feature_dim is zero")
        else:
            if pair_features is None or pair_features.shape != (
                batch, nodes, nodes, self.pair_projection.in_features
            ):
                raise ValueError("pair_features must be [batch,nodes,nodes,pair_feature_dim]")
            hidden = hidden + self.pair_projection(pair_features)
        logits = self.output_projection(torch.relu(hidden))
        diagonal = torch.eye(nodes, dtype=torch.bool, device=logits.device)
        return logits.masked_fill(diagonal.unsqueeze(0).unsqueeze(-1), 0.0)

    @staticmethod
    def probabilities(logits: Tensor) -> Tensor:
        """Normalize mutually exclusive classes only, never across targets."""
        if logits.ndim != 4 or logits.shape[-1] != len(CLASS_ORDER):
            raise ValueError("logits must have shape [batch,nodes,nodes,3]")
        return torch.softmax(logits, dim=-1)


def source_event_balanced_masked_cross_entropy(
    logits: Tensor, target_class: Tensor, loss_mask: Tensor
) -> MaskedCrossEntropyOutcome:
    """Average targets within source events, then average source events.

    Returns an explicit empty outcome rather than a fake zero loss. Callers must
    skip their optimizer step when ``is_empty`` is true.
    """
    if logits.ndim != 4 or logits.shape[-1] != len(CLASS_ORDER):
        raise ValueError("logits must have shape [batch,nodes,nodes,3]")
    if target_class.shape != logits.shape[:-1] or loss_mask.shape != logits.shape[:-1]:
        raise ValueError("target_class and loss_mask must match logits without classes")
    mask = loss_mask.bool()
    diagonal = torch.eye(logits.shape[1], dtype=torch.bool, device=logits.device)
    mask = mask & ~diagonal.unsqueeze(0)
    counts = mask.sum(dim=-1)
    event_mask = counts > 0
    event_count = int(event_mask.sum().item())
    if event_count == 0:
        return MaskedCrossEntropyOutcome(None, True, 0)
    valid_targets = target_class[mask]
    if ((valid_targets < 0) | (valid_targets >= len(CLASS_ORDER))).any():
        raise ValueError("masked target classes must be in [0, 2]")
    cell_loss = functional.cross_entropy(logits[mask], valid_targets.long(), reduction="none")
    per_cell = torch.zeros_like(mask, dtype=cell_loss.dtype)
    per_cell[mask] = cell_loss
    per_source = per_cell.sum(dim=-1) / counts.clamp_min(1)
    return MaskedCrossEntropyOutcome(per_source[event_mask].mean(), False, event_count)
