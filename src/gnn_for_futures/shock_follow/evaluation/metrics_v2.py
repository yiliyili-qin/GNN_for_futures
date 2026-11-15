"""Mergeable metrics for hierarchical shock-follow GraphMixer V2.

All task metrics are computed only over mask-valid ordered pairs after removing
self-pairs.  Average precision uses whole equal-score groups: every item in a
tie is admitted before that threshold contributes precision.  This makes the
result deterministic and invariant to the input order within a tie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as functional

from gnn_for_futures.shock_follow.contract import (
    CLASS_NO_FOLLOW,
    CLASS_OPPOSITE_DIRECTION,
    CLASS_ORDER,
)
from gnn_for_futures.shock_follow.models.head_v2 import (
    DIRECTION_CLASS_ORDER,
    FOLLOW_CLASS_ORDER,
    hierarchical_joint_scores,
    hierarchical_predictions,
    hierarchical_source_event_loss,
)


AVERAGE_PRECISION_TIE_POLICY = "whole_equal_score_group"
SCORE_QUANTILE_LEVELS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
SCORE_QUANTILE_INTERPOLATION = "linear"


def _finite_positive_pair(values: Sequence[float] | Tensor) -> tuple[float, float]:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.shape != (len(FOLLOW_CLASS_ORDER),):
        raise ValueError("follow_class_weights must have shape [2]")
    if not torch.isfinite(tensor).all() or tensor.le(0.0).any():
        raise ValueError("follow_class_weights must be finite and strictly positive")
    return float(tensor[0].item()), float(tensor[1].item())


def _finite_positive_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return normalized


def _ratio_or_zero(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _per_class_metrics(
    confusion: Tensor,
    class_names: Sequence[str],
) -> tuple[list[dict[str, object]], float | None]:
    rows: list[dict[str, object]] = []
    f1_values: list[float] = []
    for class_id, class_name in enumerate(class_names):
        true_positive = int(confusion[class_id, class_id].item())
        predicted_count = int(confusion[:, class_id].sum().item())
        support = int(confusion[class_id, :].sum().item())
        false_positive = predicted_count - true_positive
        false_negative = support - true_positive
        precision = _ratio_or_zero(true_positive, predicted_count)
        recall = _ratio_or_zero(true_positive, support)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = _ratio_or_zero(2 * true_positive, denominator)
        f1_values.append(f1)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    macro_f1 = sum(f1_values) / len(f1_values) if int(confusion.sum()) else None
    return rows, macro_f1


def binary_average_precision(targets: Tensor, scores: Tensor) -> float | None:
    """Return tie-grouped binary AP, or ``None`` when there are no positives.

    Equal scores form one threshold group.  The group's cumulative precision is
    weighted by the group's share of all positives, rather than ordering tied
    positives and negatives arbitrarily.
    """

    targets = torch.as_tensor(targets).detach().to(device="cpu").reshape(-1)
    scores = torch.as_tensor(scores).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if targets.shape != scores.shape:
        raise ValueError("targets and scores must have the same number of elements")
    if targets.dtype == torch.bool:
        binary_targets = targets.to(dtype=torch.int64)
    elif targets.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        binary_targets = targets.to(dtype=torch.int64)
    else:
        raise TypeError("targets must be boolean or integer")
    if ((binary_targets != 0) & (binary_targets != 1)).any():
        raise ValueError("targets must contain only 0 and 1")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("average-precision scores contain non-finite values")
    positive_count = int(binary_targets.sum().item())
    if positive_count == 0:
        return None
    if scores.numel() == 0:
        raise AssertionError("a positive target requires a nonempty score tensor")

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = binary_targets[order]
    cumulative_positive = 0
    cumulative_count = 0
    average_precision = 0.0
    start = 0
    while start < sorted_scores.numel():
        end = start + 1
        while end < sorted_scores.numel() and bool(
            sorted_scores[end] == sorted_scores[start]
        ):
            end += 1
        group_positive = int(sorted_targets[start:end].sum().item())
        cumulative_positive += group_positive
        cumulative_count = end
        average_precision += (
            group_positive / positive_count
        ) * (cumulative_positive / cumulative_count)
        start = end
    return average_precision


def _confusion_matrix(targets: Tensor, predictions: Tensor, classes: int) -> Tensor:
    encoded = targets.to(torch.int64) * classes + predictions.to(torch.int64)
    return torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def _score_quantiles(values: Tensor) -> list[float]:
    levels = torch.tensor(SCORE_QUANTILE_LEVELS, dtype=torch.float64)
    observed = torch.quantile(
        values.to(dtype=torch.float64),
        levels,
        interpolation=SCORE_QUANTILE_INTERPOLATION,
    )
    return [float(value) for value in observed.tolist()]


@dataclass
class V2MetricsAccumulator:
    """Accumulate exact V2 metrics across arbitrary batches and merge shards."""

    follow_class_weights: Sequence[float] | Tensor
    lambda_follow: float = 1.0
    lambda_direction: float = 1.0
    _follow_targets: list[Tensor] = field(default_factory=list, init=False, repr=False)
    _follow_scores: list[Tensor] = field(default_factory=list, init=False, repr=False)
    _three_targets: list[Tensor] = field(default_factory=list, init=False, repr=False)
    _three_predictions: list[Tensor] = field(default_factory=list, init=False, repr=False)
    _direction_targets: list[Tensor] = field(default_factory=list, init=False, repr=False)
    _direction_predictions: list[Tensor] = field(default_factory=list, init=False, repr=False)
    _joint_scores: list[Tensor] = field(default_factory=list, init=False, repr=False)
    _joint_sum_errors: list[Tensor] = field(default_factory=list, init=False, repr=False)
    _joint_no_errors: list[Tensor] = field(default_factory=list, init=False, repr=False)
    _joint_follow_errors: list[Tensor] = field(default_factory=list, init=False, repr=False)
    _joint_recomposition_errors: list[Tensor] = field(default_factory=list, init=False, repr=False)
    _source_follow_loss_sum: float = field(default=0.0, init=False, repr=False)
    _source_unweighted_follow_loss_sum: float = field(default=0.0, init=False, repr=False)
    _source_direction_loss_sum: float = field(default=0.0, init=False, repr=False)
    _cell_follow_loss_sum: float = field(default=0.0, init=False, repr=False)
    _cell_unweighted_follow_loss_sum: float = field(default=0.0, init=False, repr=False)
    _cell_direction_loss_sum: float = field(default=0.0, init=False, repr=False)
    _source_event_count: int = field(default=0, init=False)
    _direction_source_event_count: int = field(default=0, init=False)
    _valid_pair_count: int = field(default=0, init=False)
    _direction_pair_count: int = field(default=0, init=False)
    _update_count: int = field(default=0, init=False)
    _empty_update_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.follow_class_weights = _finite_positive_pair(self.follow_class_weights)
        self.lambda_follow = _finite_positive_scalar(self.lambda_follow, "lambda_follow")
        self.lambda_direction = _finite_positive_scalar(
            self.lambda_direction, "lambda_direction"
        )

    @property
    def valid_pair_count(self) -> int:
        return self._valid_pair_count

    def update(
        self,
        follow_logits: Tensor,
        direction_logits: Tensor,
        target_class: Tensor,
        loss_mask: Tensor,
        *,
        joint_scores: Tensor | None = None,
    ) -> None:
        """Add one batch of raw V2 outputs and labels.

        Non-finite raw outputs are rejected even outside the evaluation mask.
        Diagonal entries are always removed from ``loss_mask``.
        """

        self._update_count += 1
        if not torch.isfinite(follow_logits).all():
            raise FloatingPointError("raw follow_logits contain non-finite values")
        if not torch.isfinite(direction_logits).all():
            raise FloatingPointError("raw direction_logits contain non-finite values")

        recomposed_joint = hierarchical_joint_scores(follow_logits, direction_logits)
        observed_joint = recomposed_joint if joint_scores is None else joint_scores
        if observed_joint.shape != (*follow_logits.shape[:-1], len(CLASS_ORDER)):
            raise ValueError("joint_scores must have shape [batch,nodes,nodes,3]")
        if not observed_joint.is_floating_point():
            raise TypeError("joint_scores must use a floating-point dtype")
        if observed_joint.device != follow_logits.device:
            raise ValueError("joint_scores and logits must use the same device")
        if not torch.isfinite(observed_joint).all():
            raise FloatingPointError("raw joint_scores contain non-finite values")

        expected_pair_shape = follow_logits.shape[:-1]
        if target_class.shape != expected_pair_shape:
            raise ValueError("target_class must match logits without their class dimension")
        if loss_mask.shape != expected_pair_shape:
            raise ValueError("loss_mask must have the same shape as target_class")
        if loss_mask.dtype != torch.bool:
            raise TypeError("loss_mask must be boolean")
        if target_class.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("target_class must use an integer dtype")
        if target_class.device != follow_logits.device or loss_mask.device != follow_logits.device:
            raise ValueError("targets, mask, and logits must use the same device")

        nodes = target_class.shape[1]
        diagonal = torch.eye(nodes, dtype=torch.bool, device=loss_mask.device).unsqueeze(0)
        mask = loss_mask & ~diagonal
        targets = target_class[mask].to(dtype=torch.int64)
        if targets.numel() and ((targets < 0) | (targets >= len(CLASS_ORDER))).any():
            raise ValueError("masked target classes must be in [0, 2]")

        weights = follow_logits.new_tensor(self.follow_class_weights)
        outcome = hierarchical_source_event_loss(
            follow_logits,
            direction_logits,
            target_class,
            mask,
            weights,
            lambda_follow=self.lambda_follow,
            lambda_direction=self.lambda_direction,
        )
        if outcome.is_empty:
            self._empty_update_count += 1
            return

        follow_probabilities = torch.softmax(follow_logits, dim=-1)[..., 1]
        if not torch.isfinite(follow_probabilities).all():
            raise FloatingPointError("derived follow probabilities contain non-finite values")
        predictions = hierarchical_predictions(follow_logits, direction_logits)

        # Force stage one to follow, then reuse the V2 prediction helper's frozen
        # direction argmax/tie behavior for conditional direction predictions.
        forced_follow_logits = torch.zeros_like(follow_logits)
        forced_follow_logits[..., 1] = 1.0
        conditional_predictions = hierarchical_predictions(
            forced_follow_logits, direction_logits
        )

        masked_follow_targets = targets.ne(CLASS_NO_FOLLOW).to(dtype=torch.int64)
        masked_follow_scores = follow_probabilities[mask]
        masked_predictions = predictions[mask].to(dtype=torch.int64)
        direction_mask = mask & target_class.ne(CLASS_NO_FOLLOW)
        masked_direction_targets = (
            target_class[direction_mask].to(dtype=torch.int64) - CLASS_OPPOSITE_DIRECTION
        )
        masked_direction_predictions = (
            conditional_predictions[direction_mask].to(dtype=torch.int64)
            - CLASS_OPPOSITE_DIRECTION
        )
        masked_joint = observed_joint[mask]
        masked_recomposed = recomposed_joint[mask]
        masked_follow_distribution = torch.softmax(follow_logits[mask], dim=-1)

        self._follow_targets.append(masked_follow_targets.detach().cpu())
        self._follow_scores.append(masked_follow_scores.detach().to("cpu", torch.float64))
        self._three_targets.append(targets.detach().cpu())
        self._three_predictions.append(masked_predictions.detach().cpu())
        self._joint_scores.append(masked_joint.detach().to("cpu", torch.float64))
        if masked_direction_targets.numel():
            self._direction_targets.append(masked_direction_targets.detach().cpu())
            self._direction_predictions.append(masked_direction_predictions.detach().cpu())

        self._joint_sum_errors.append(
            (masked_joint.sum(dim=-1) - 1.0).abs().detach().to("cpu", torch.float64)
        )
        self._joint_no_errors.append(
            (masked_joint[:, 0] - masked_follow_distribution[:, 0])
            .abs()
            .detach()
            .to("cpu", torch.float64)
        )
        self._joint_follow_errors.append(
            (masked_joint[:, 1:].sum(dim=-1) - masked_follow_distribution[:, 1])
            .abs()
            .detach()
            .to("cpu", torch.float64)
        )
        self._joint_recomposition_errors.append(
            (masked_joint - masked_recomposed)
            .abs()
            .amax(dim=-1)
            .detach()
            .to("cpu", torch.float64)
        )

        weighted_follow_cell_losses = functional.cross_entropy(
            follow_logits[mask],
            masked_follow_targets,
            weight=weights,
            reduction="none",
        )
        unweighted_follow_cell_losses = functional.cross_entropy(
            follow_logits[mask],
            masked_follow_targets,
            reduction="none",
        )
        follow_cell_losses = weighted_follow_cell_losses.sum()
        unweighted_follow_cell_loss_sum = unweighted_follow_cell_losses.sum()
        event_counts = mask.sum(dim=-1)
        event_mask = event_counts > 0
        unweighted_follow_grid = torch.zeros_like(
            mask, dtype=unweighted_follow_cell_losses.dtype
        )
        unweighted_follow_grid[mask] = unweighted_follow_cell_losses
        unweighted_follow_source_loss = (
            unweighted_follow_grid.sum(dim=-1) / event_counts.clamp_min(1)
        )[event_mask].mean()
        direction_cell_losses = None
        if masked_direction_targets.numel():
            direction_cell_losses = functional.cross_entropy(
                direction_logits[direction_mask],
                masked_direction_targets,
                reduction="sum",
            )
        finite_losses = [
            follow_cell_losses,
            unweighted_follow_cell_loss_sum,
            unweighted_follow_source_loss,
            outcome.follow_loss,
        ]
        if direction_cell_losses is not None:
            finite_losses.append(direction_cell_losses)
        if outcome.direction_loss is not None:
            finite_losses.append(outcome.direction_loss)
        if not all(bool(torch.isfinite(value)) for value in finite_losses):
            raise FloatingPointError("V2 metric loss contains a non-finite value")

        self._source_follow_loss_sum += (
            float(outcome.follow_loss.detach().cpu()) * outcome.follow_event_count
        )
        self._source_unweighted_follow_loss_sum += (
            float(unweighted_follow_source_loss.detach().cpu())
            * outcome.follow_event_count
        )
        if outcome.direction_loss is not None:
            self._source_direction_loss_sum += (
                float(outcome.direction_loss.detach().cpu())
                * outcome.direction_event_count
            )
        self._cell_follow_loss_sum += float(follow_cell_losses.detach().cpu())
        self._cell_unweighted_follow_loss_sum += float(
            unweighted_follow_cell_loss_sum.detach().cpu()
        )
        if direction_cell_losses is not None:
            self._cell_direction_loss_sum += float(direction_cell_losses.detach().cpu())
        self._source_event_count += outcome.follow_event_count
        self._direction_source_event_count += outcome.direction_event_count
        self._valid_pair_count += outcome.valid_pair_count
        self._direction_pair_count += outcome.direction_pair_count

    def merge(self, other: "V2MetricsAccumulator") -> "V2MetricsAccumulator":
        """Merge another compatible accumulator into this one and return self."""

        if not isinstance(other, V2MetricsAccumulator):
            raise TypeError("other must be a V2MetricsAccumulator")
        if other is self:
            raise ValueError("an accumulator cannot be merged with itself")
        if (
            self.follow_class_weights != other.follow_class_weights
            or self.lambda_follow != other.lambda_follow
            or self.lambda_direction != other.lambda_direction
        ):
            raise ValueError("cannot merge accumulators with different loss configuration")
        for name in (
            "_follow_targets",
            "_follow_scores",
            "_three_targets",
            "_three_predictions",
            "_direction_targets",
            "_direction_predictions",
            "_joint_scores",
            "_joint_sum_errors",
            "_joint_no_errors",
            "_joint_follow_errors",
            "_joint_recomposition_errors",
        ):
            getattr(self, name).extend(getattr(other, name))
        for name in (
            "_source_follow_loss_sum",
            "_source_unweighted_follow_loss_sum",
            "_source_direction_loss_sum",
            "_cell_follow_loss_sum",
            "_cell_unweighted_follow_loss_sum",
            "_cell_direction_loss_sum",
            "_source_event_count",
            "_direction_source_event_count",
            "_valid_pair_count",
            "_direction_pair_count",
            "_update_count",
            "_empty_update_count",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    def finalize(self) -> dict[str, object]:
        """Return JSON-compatible aggregate metrics without resetting state."""

        raw_finite_checks = {
            "follow_logits": True,
            "direction_logits": True,
            "joint_scores": True,
        }
        if self._valid_pair_count == 0:
            follow_confusion = torch.zeros((2, 2), dtype=torch.int64)
            direction_confusion = torch.zeros((2, 2), dtype=torch.int64)
            three_confusion = torch.zeros(
                (len(CLASS_ORDER), len(CLASS_ORDER)), dtype=torch.int64
            )
            direction_per_class, _ = _per_class_metrics(
                direction_confusion, DIRECTION_CLASS_ORDER
            )
            three_per_class, _ = _per_class_metrics(three_confusion, CLASS_ORDER)
            return {
                "valid_pair_count": 0,
                "source_event_count": 0,
                "direction_pair_count": 0,
                "direction_source_event_count": 0,
                "true_follow_count": 0,
                "predicted_follow_count": 0,
                "update_count": self._update_count,
                "empty_update_count": self._empty_update_count,
                "raw_finite_checks": raw_finite_checks,
                "average_precision_tie_policy": AVERAGE_PRECISION_TIE_POLICY,
                "score_quantile_levels": list(SCORE_QUANTILE_LEVELS),
                "score_quantile_interpolation": SCORE_QUANTILE_INTERPOLATION,
                "follow_average_precision": None,
                "follow_precision": None,
                "follow_recall": None,
                "follow_f1": None,
                "predicted_follow_rate": None,
                "follow_confusion_matrix": follow_confusion.tolist(),
                "follow_class_order": list(FOLLOW_CLASS_ORDER),
                "conditional_direction_accuracy": None,
                "conditional_direction_macro_f1": None,
                "conditional_direction_confusion_matrix": direction_confusion.tolist(),
                "conditional_direction_per_class": direction_per_class,
                "direction_class_order": list(DIRECTION_CLASS_ORDER),
                "three_class_accuracy": None,
                "three_class_macro_f1": None,
                "three_class_confusion_matrix": three_confusion.tolist(),
                "three_class_per_class": three_per_class,
                "predicted_opposite_count": 0,
                "predicted_same_count": 0,
                "no_follow_recall": None,
                "opposite_recall": None,
                "same_recall": None,
                "class_order": list(CLASS_ORDER),
                "confusion_matrix_rows": "true_class",
                "confusion_matrix_columns": "predicted_class",
                "source_event_weighted_follow_loss": None,
                "source_event_unweighted_follow_loss": None,
                "source_event_unweighted_direction_loss": None,
                "source_event_total_loss": None,
                "total_loss": None,
                "cell_weighted_follow_loss": None,
                "cell_unweighted_follow_loss": None,
                "cell_weighted_direction_loss": None,
                "cell_weighted_total_loss": None,
                "lambda_follow": self.lambda_follow,
                "lambda_direction": self.lambda_direction,
                "follow_class_weights": list(self.follow_class_weights),
                "score_quantiles": None,
                "joint_score_sum_max_abs_error": None,
                "joint_no_follow_identity_max_abs_error": None,
                "joint_follow_identity_max_abs_error": None,
                "joint_recomposition_max_abs_error": None,
            }

        follow_targets = torch.cat(self._follow_targets)
        follow_scores = torch.cat(self._follow_scores)
        three_targets = torch.cat(self._three_targets)
        three_predictions = torch.cat(self._three_predictions)
        joint_scores = torch.cat(self._joint_scores)
        follow_predictions = three_predictions.ne(CLASS_NO_FOLLOW).to(dtype=torch.int64)

        follow_confusion = _confusion_matrix(follow_targets, follow_predictions, 2)
        follow_true_positive = int(follow_confusion[1, 1].item())
        predicted_follow_count = int(follow_confusion[:, 1].sum().item())
        true_follow_count = int(follow_confusion[1, :].sum().item())
        follow_precision = _ratio_or_zero(follow_true_positive, predicted_follow_count)
        follow_recall = _ratio_or_zero(follow_true_positive, true_follow_count)
        follow_f1_denominator = predicted_follow_count + true_follow_count
        follow_f1 = _ratio_or_zero(2 * follow_true_positive, follow_f1_denominator)

        three_confusion = _confusion_matrix(three_targets, three_predictions, len(CLASS_ORDER))
        three_per_class, three_macro_f1 = _per_class_metrics(three_confusion, CLASS_ORDER)
        three_accuracy = float((three_targets == three_predictions).double().mean().item())

        if self._direction_pair_count:
            direction_targets = torch.cat(self._direction_targets)
            direction_predictions = torch.cat(self._direction_predictions)
            direction_confusion = _confusion_matrix(
                direction_targets, direction_predictions, len(DIRECTION_CLASS_ORDER)
            )
            direction_per_class, direction_macro_f1 = _per_class_metrics(
                direction_confusion, DIRECTION_CLASS_ORDER
            )
            direction_accuracy = float(
                (direction_targets == direction_predictions).double().mean().item()
            )
            source_direction_loss = (
                self._source_direction_loss_sum / self._direction_source_event_count
            )
            cell_direction_loss = self._cell_direction_loss_sum / self._direction_pair_count
        else:
            direction_confusion = torch.zeros((2, 2), dtype=torch.int64)
            direction_per_class = []
            direction_macro_f1 = None
            direction_accuracy = None
            source_direction_loss = None
            cell_direction_loss = None

        source_follow_loss = self._source_follow_loss_sum / self._source_event_count
        source_unweighted_follow_loss = (
            self._source_unweighted_follow_loss_sum / self._source_event_count
        )
        cell_follow_loss = self._cell_follow_loss_sum / self._valid_pair_count
        cell_unweighted_follow_loss = (
            self._cell_unweighted_follow_loss_sum / self._valid_pair_count
        )
        source_total_loss = self.lambda_follow * source_follow_loss
        cell_total_loss = self.lambda_follow * cell_follow_loss
        if source_direction_loss is not None:
            source_total_loss += self.lambda_direction * source_direction_loss
        if cell_direction_loss is not None:
            cell_total_loss += self.lambda_direction * cell_direction_loss

        no_follow_scores = follow_scores[follow_targets == 0]
        true_follow_scores = follow_scores[follow_targets == 1]
        score_quantiles = {
            "follow_score": _score_quantiles(follow_scores),
            "follow_score_true_no_follow": (
                _score_quantiles(no_follow_scores) if no_follow_scores.numel() else None
            ),
            "follow_score_true_follow": (
                _score_quantiles(true_follow_scores)
                if true_follow_scores.numel()
                else None
            ),
            "joint_no_follow": _score_quantiles(joint_scores[:, 0]),
            "joint_opposite_direction": _score_quantiles(joint_scores[:, 1]),
            "joint_same_direction": _score_quantiles(joint_scores[:, 2]),
        }
        result: dict[str, object] = {
            "valid_pair_count": self._valid_pair_count,
            "source_event_count": self._source_event_count,
            "direction_pair_count": self._direction_pair_count,
            "direction_source_event_count": self._direction_source_event_count,
            "true_follow_count": true_follow_count,
            "predicted_follow_count": predicted_follow_count,
            "update_count": self._update_count,
            "empty_update_count": self._empty_update_count,
            "raw_finite_checks": raw_finite_checks,
            "average_precision_tie_policy": AVERAGE_PRECISION_TIE_POLICY,
            "follow_average_precision": binary_average_precision(
                follow_targets, follow_scores
            ),
            "follow_precision": follow_precision,
            "follow_recall": follow_recall,
            "follow_f1": follow_f1,
            "predicted_follow_rate": predicted_follow_count / self._valid_pair_count,
            "follow_confusion_matrix": follow_confusion.tolist(),
            "follow_class_order": list(FOLLOW_CLASS_ORDER),
            "conditional_direction_accuracy": direction_accuracy,
            "conditional_direction_macro_f1": direction_macro_f1,
            "conditional_direction_confusion_matrix": direction_confusion.tolist(),
            "conditional_direction_per_class": direction_per_class,
            "direction_class_order": list(DIRECTION_CLASS_ORDER),
            "three_class_accuracy": three_accuracy,
            "three_class_macro_f1": three_macro_f1,
            "three_class_confusion_matrix": three_confusion.tolist(),
            "three_class_per_class": three_per_class,
            "predicted_opposite_count": three_per_class[CLASS_OPPOSITE_DIRECTION][
                "predicted_count"
            ],
            "predicted_same_count": three_per_class[2]["predicted_count"],
            "no_follow_recall": three_per_class[CLASS_NO_FOLLOW]["recall"],
            "opposite_recall": three_per_class[CLASS_OPPOSITE_DIRECTION]["recall"],
            "same_recall": three_per_class[2]["recall"],
            "class_order": list(CLASS_ORDER),
            "confusion_matrix_rows": "true_class",
            "confusion_matrix_columns": "predicted_class",
            "source_event_weighted_follow_loss": source_follow_loss,
            "source_event_unweighted_follow_loss": source_unweighted_follow_loss,
            "source_event_unweighted_direction_loss": source_direction_loss,
            "source_event_total_loss": source_total_loss,
            "total_loss": source_total_loss,
            "cell_weighted_follow_loss": cell_follow_loss,
            "cell_unweighted_follow_loss": cell_unweighted_follow_loss,
            "cell_weighted_direction_loss": cell_direction_loss,
            "cell_weighted_total_loss": cell_total_loss,
            "lambda_follow": self.lambda_follow,
            "lambda_direction": self.lambda_direction,
            "follow_class_weights": list(self.follow_class_weights),
            "score_quantile_levels": list(SCORE_QUANTILE_LEVELS),
            "score_quantile_interpolation": SCORE_QUANTILE_INTERPOLATION,
            "score_quantiles": score_quantiles,
            "joint_score_sum_max_abs_error": float(
                torch.cat(self._joint_sum_errors).max().item()
            ),
            "joint_no_follow_identity_max_abs_error": float(
                torch.cat(self._joint_no_errors).max().item()
            ),
            "joint_follow_identity_max_abs_error": float(
                torch.cat(self._joint_follow_errors).max().item()
            ),
            "joint_recomposition_max_abs_error": float(
                torch.cat(self._joint_recomposition_errors).max().item()
            ),
        }
        finite_values = (
            source_follow_loss,
            source_unweighted_follow_loss,
            cell_follow_loss,
            cell_unweighted_follow_loss,
            source_total_loss,
            cell_total_loss,
            three_accuracy,
            follow_precision,
            follow_recall,
            follow_f1,
            result["predicted_follow_rate"],
            result["joint_score_sum_max_abs_error"],
            result["joint_no_follow_identity_max_abs_error"],
            result["joint_follow_identity_max_abs_error"],
            result["joint_recomposition_max_abs_error"],
        )
        optional_finite_values = (
            result["follow_average_precision"],
            direction_accuracy,
            direction_macro_f1,
            source_direction_loss,
            cell_direction_loss,
        )
        if not all(math.isfinite(float(value)) for value in finite_values) or not all(
            value is None or math.isfinite(float(value)) for value in optional_finite_values
        ):
            raise FloatingPointError("V2 aggregate metrics contain a non-finite value")
        return result

    def compute(self) -> dict[str, object]:
        """Backward-compatible alias for :meth:`finalize`."""

        return self.finalize()


ShockFollowMetricsV2Accumulator = V2MetricsAccumulator


def compute_shock_follow_metrics_v2(
    follow_logits: Tensor,
    direction_logits: Tensor,
    target_class: Tensor,
    loss_mask: Tensor,
    follow_class_weights: Sequence[float] | Tensor,
    *,
    joint_scores: Tensor | None = None,
    lambda_follow: float = 1.0,
    lambda_direction: float = 1.0,
) -> dict[str, object]:
    """Compute V2 metrics for one tensor collection using the mergeable path."""

    accumulator = V2MetricsAccumulator(
        follow_class_weights,
        lambda_follow=lambda_follow,
        lambda_direction=lambda_direction,
    )
    accumulator.update(
        follow_logits,
        direction_logits,
        target_class,
        loss_mask,
        joint_scores=joint_scores,
    )
    return accumulator.finalize()
