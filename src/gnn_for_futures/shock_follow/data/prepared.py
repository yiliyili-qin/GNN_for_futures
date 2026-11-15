"""Cutoff samples, causal features, and chronological splits for shock-follow V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from gnn_for_futures.shock_follow.contract import HORIZON_MINUTES
from gnn_for_futures.shock_follow.data.labels import build_shock_follow_labels_for_cutoff
from gnn_for_futures.shock_follow.data.panel import ShockFollowBarPanel
from gnn_for_futures.shock_follow.data.thresholds import MonthlyThresholdTable


NODE_FEATURE_SCHEMA = (
    "signed_return",
    "absolute_return",
    "absolute_return_over_threshold",
    "log1p_volume",
    "open_interest_change",
    "return_available",
    "bar_available",
    "sin_minute_of_day",
    "cos_minute_of_day",
)
DEFAULT_SEQUENCE_LENGTH = 40
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class ChronologicalSplit:
    """Contiguous target-time intervals and exact sample memberships."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    purged_indices: np.ndarray
    train_end: np.datetime64
    validation_end: np.datetime64
    test_end: np.datetime64

    def indices(self, split: str) -> np.ndarray:
        mapping = {
            "train": self.train_indices,
            "validation": self.validation_indices,
            "test": self.test_indices,
        }
        try:
            return mapping[split].copy()
        except KeyError as error:
            raise ValueError(f"split must be one of {SPLIT_NAMES}") from error


@dataclass(frozen=True)
class ShockFollowPreparedData:
    """Immutable references to derived arrays shared across split datasets."""

    panel_materialization_sha256: str
    threshold_materialization_sha256: str
    sequence_length: int
    node_features: np.ndarray
    input_valid: np.ndarray
    threshold_matrix: np.ndarray
    common_cutoffs: np.ndarray
    common_histories: np.ndarray


@dataclass(frozen=True)
class ShockFollowSample:
    """One cutoff-level model sample with source-by-target supervision."""

    node_history: Tensor
    economic_prior: Tensor
    regression_weight: Tensor
    prior_available: Tensor
    regression_available: Tensor
    target_class: Tensor
    loss_mask: Tensor
    source_shock: Tensor
    source_sign: Tensor
    cutoff_at: np.datetime64
    target_at: np.datetime64
    trading_date: np.datetime64
    panel_index: int


def _datetime_scalar(value: object, name: str) -> np.datetime64:
    result = np.asarray(value).astype("datetime64[ns]")
    if result.ndim != 0 or np.isnat(result):
        raise ValueError(f"{name} must be one valid timestamp")
    return result


def build_monthly_threshold_matrix(
    panel: ShockFollowBarPanel,
    thresholds: MonthlyThresholdTable,
) -> np.ndarray:
    """Broadcast the frozen monthly threshold vector onto every panel timestamp."""
    if panel.node_order != thresholds.node_order:
        raise ValueError("bar panel and threshold table node orders differ")
    panel_months = panel.trading_dates.astype("datetime64[M]")
    positions = np.searchsorted(thresholds.months, panel_months)
    valid = positions < len(thresholds.months)
    matched = np.zeros(len(panel_months), dtype=bool)
    matched[valid] = thresholds.months[positions[valid]] == panel_months[valid]
    if not matched.all():
        missing = np.unique(panel_months[~matched]).astype(str).tolist()
        raise KeyError(f"No threshold table rows for panel months: {missing}")
    result = thresholds.thresholds[positions].copy()
    repeated_starts = thresholds.month_starts[positions]
    if np.any(panel.timestamps < repeated_starts):
        raise ValueError(
            "panel timestamp precedes the selected effective threshold month"
        )
    repeated_ends = thresholds.fit_ends[positions]
    available = np.isfinite(result)
    if available.any():
        starts = np.broadcast_to(repeated_starts[:, None], result.shape)
        if np.isnat(repeated_ends[available]).any() or not (
            repeated_ends[available] < starts[available]
        ).all():
            raise ValueError("threshold provenance violates fit_end < month_start")
    return result


def build_causal_node_features(
    panel: ShockFollowBarPanel,
    thresholds: MonthlyThresholdTable,
) -> tuple[np.ndarray, np.ndarray]:
    """Build finite node histories and a strict per-node input-valid mask."""
    panel.validate()
    thresholds.validate()
    threshold_matrix = build_monthly_threshold_matrix(panel, thresholds)
    returns = panel.signed_returns
    return_valid = panel.return_valid & np.isfinite(threshold_matrix) & (threshold_matrix > 0.0)
    signed_return = np.where(return_valid, returns, 0.0)
    absolute_return = np.abs(signed_return)
    relative_amplitude = np.divide(
        absolute_return,
        threshold_matrix,
        out=np.zeros_like(absolute_return),
        where=return_valid,
    )
    relative_amplitude = np.clip(relative_amplitude, 0.0, 10.0)

    volume_valid = panel.bar_available & np.isfinite(panel.volumes) & (panel.volumes > 0.0)
    log_volume = np.zeros_like(panel.volumes)
    log_volume[volume_valid] = np.log1p(panel.volumes[volume_valid])
    open_interest_change = np.where(
        np.isfinite(panel.open_interest_changes), panel.open_interest_changes, 0.0
    )
    open_interest_change = np.clip(open_interest_change, -1.0, 1.0)
    # Keep the V1 feature scale explicit and causal: no full-period fitted moments.
    log_volume = log_volume / 20.0

    timestamps_minutes = panel.timestamps.astype("datetime64[m]").astype(np.int64)
    minute_of_day = np.mod(timestamps_minutes, 24 * 60).astype(np.float64)
    phase = 2.0 * np.pi * minute_of_day / (24 * 60)
    sin_time = np.broadcast_to(np.sin(phase)[:, None], returns.shape)
    cos_time = np.broadcast_to(np.cos(phase)[:, None], returns.shape)

    features = np.stack(
        (
            signed_return,
            absolute_return,
            relative_amplitude,
            log_volume,
            open_interest_change,
            return_valid.astype(np.float64),
            panel.bar_available.astype(np.float64),
            sin_time,
            cos_time,
        ),
        axis=-1,
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("node feature construction produced non-finite values")
    return features, return_valid


def common_history_index_matrix(
    panel: ShockFollowBarPanel,
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Map every common-grid cutoff to its preceding observed history rows."""
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    common_grid = np.flatnonzero(panel.row_observed.all(axis=1))
    cutoffs: list[int] = []
    histories: list[np.ndarray] = []
    for common_position in range(sequence_length - 1, len(common_grid)):
        indices = common_grid[common_position - sequence_length + 1 : common_position + 1]
        cutoffs.append(int(indices[-1]))
        histories.append(indices.copy())
    if not histories:
        return np.empty(0, dtype=np.int64), np.empty((0, sequence_length), dtype=np.int64)
    return np.asarray(cutoffs, dtype=np.int64), np.stack(histories)


def prepare_shock_follow_data(
    panel: ShockFollowBarPanel,
    thresholds: MonthlyThresholdTable,
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> ShockFollowPreparedData:
    """Build split-independent arrays once for sharing by train/validation datasets."""
    panel.validate()
    thresholds.validate()
    if panel.node_order != thresholds.node_order:
        raise ValueError("bar panel and threshold table node orders differ")
    node_features, input_valid = build_causal_node_features(panel, thresholds)
    threshold_matrix = build_monthly_threshold_matrix(panel, thresholds)
    common_cutoffs, common_histories = common_history_index_matrix(
        panel, sequence_length=sequence_length
    )
    for values in (
        node_features,
        input_valid,
        threshold_matrix,
        common_cutoffs,
        common_histories,
    ):
        values.setflags(write=False)
    return ShockFollowPreparedData(
        panel.materialization_sha256,
        thresholds.materialization_sha256,
        sequence_length,
        node_features,
        input_valid,
        threshold_matrix,
        common_cutoffs,
        common_histories,
    )


def eligible_cutoff_indices(
    panel: ShockFollowBarPanel,
    input_valid: np.ndarray,
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    require_all_nodes_history: bool = False,
    common_cutoffs: np.ndarray | None = None,
) -> np.ndarray:
    """Return exact-horizon cutoffs with an aligned observed-row history.

    The history grid is the timestamp intersection where every node emitted a row.
    Endpoint validity remains a feature/mask rather than deleting zero-volume rows.
    """
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    expected = (len(panel.timestamps), len(panel.node_order))
    valid = np.asarray(input_valid, dtype=bool)
    if valid.shape != expected:
        raise ValueError(f"input_valid must have shape {expected}")
    timestamp_to_index = {timestamp: index for index, timestamp in enumerate(panel.timestamps)}
    if common_cutoffs is None:
        candidate_cutoffs, _ = common_history_index_matrix(
            panel, sequence_length=sequence_length
        )
    else:
        candidate_cutoffs = np.asarray(common_cutoffs)
        if candidate_cutoffs.ndim != 1 or candidate_cutoffs.dtype.kind not in "iu":
            raise ValueError("common_cutoffs must be a one-dimensional integer array")
    candidates: list[int] = []
    for index in candidate_cutoffs:
        index = int(index)
        target_at = panel.timestamps[index] + np.timedelta64(HORIZON_MINUTES, "m")
        target_index = timestamp_to_index.get(target_at)
        if target_index is None or panel.trading_dates[target_index] != panel.trading_dates[index]:
            continue
        current_valid = valid[index]
        if require_all_nodes_history:
            if current_valid.all():
                candidates.append(index)
        elif current_valid.any():
            candidates.append(index)
    return np.asarray(candidates, dtype=np.int64)


def chronological_split_by_target(
    timestamps: np.ndarray,
    cutoff_indices: np.ndarray,
    *,
    train_end: object,
    validation_end: object,
    test_end: object,
    horizon_minutes: int = HORIZON_MINUTES,
) -> ChronologicalSplit:
    """Assign by target timestamp and explicitly record out-of-range purges."""
    if not isinstance(horizon_minutes, int) or isinstance(horizon_minutes, bool) or horizon_minutes < 1:
        raise ValueError("horizon_minutes must be a positive integer")
    at = np.asarray(timestamps).astype("datetime64[ns]")
    raw_indices = np.asarray(cutoff_indices)
    if raw_indices.ndim != 1 or raw_indices.dtype.kind not in "iu":
        raise ValueError("cutoff_indices must be a one-dimensional integer array")
    indices = raw_indices.astype(np.int64, copy=False)
    if len(indices) and ((indices < 0).any() or (indices >= len(at)).any()):
        raise ValueError("cutoff_indices must be valid one-dimensional panel indices")
    if len(indices) and np.any(indices[1:] <= indices[:-1]):
        raise ValueError("cutoff_indices must be strictly ascending and unique")
    train_boundary = _datetime_scalar(train_end, "train_end")
    validation_boundary = _datetime_scalar(validation_end, "validation_end")
    test_boundary = _datetime_scalar(test_end, "test_end")
    if not train_boundary < validation_boundary < test_boundary:
        raise ValueError("split boundaries must satisfy train_end < validation_end < test_end")
    cutoffs = at[indices]
    targets = cutoffs + np.timedelta64(horizon_minutes, "m")
    train = (cutoffs <= train_boundary) & (targets <= train_boundary)
    validation = (
        (cutoffs > train_boundary)
        & (cutoffs <= validation_boundary)
        & (targets <= validation_boundary)
    )
    test = (
        (cutoffs > validation_boundary)
        & (cutoffs <= test_boundary)
        & (targets <= test_boundary)
    )
    # A source event and its exact target must never straddle split boundaries.
    purged = ~(train | validation | test)
    return ChronologicalSplit(
        indices[train],
        indices[validation],
        indices[test],
        indices[purged],
        train_boundary,
        validation_boundary,
        test_boundary,
    )


class ShockFollowCutoffDataset(Dataset[ShockFollowSample]):
    """Torch dataset that never precomputes trainable attention features."""

    def __init__(
        self,
        panel: ShockFollowBarPanel,
        thresholds: MonthlyThresholdTable,
        economic_prior: np.ndarray,
        cutoff_indices: Sequence[int] | np.ndarray,
        *,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        regression_weight: np.ndarray | None = None,
        regression_available: np.ndarray | None = None,
        prepared: ShockFollowPreparedData | None = None,
    ) -> None:
        panel.validate()
        thresholds.validate()
        self.panel = panel
        self.thresholds = thresholds
        self.sequence_length = sequence_length
        prepared_data = prepared or prepare_shock_follow_data(
            panel, thresholds, sequence_length=sequence_length
        )
        if (
            prepared_data.panel_materialization_sha256 != panel.materialization_sha256
            or prepared_data.threshold_materialization_sha256
            != thresholds.materialization_sha256
            or prepared_data.sequence_length != sequence_length
        ):
            raise ValueError("prepared data identity disagrees with dataset inputs")
        self.prepared = prepared_data
        self.node_features = prepared_data.node_features
        self.input_valid = prepared_data.input_valid
        self.threshold_matrix = prepared_data.threshold_matrix
        raw_cutoffs = np.asarray(cutoff_indices)
        if raw_cutoffs.ndim != 1 or raw_cutoffs.dtype.kind not in "iu":
            raise ValueError("cutoff_indices must be a one-dimensional integer array")
        self.cutoff_indices = raw_cutoffs.astype(np.int64, copy=False)
        if len(self.cutoff_indices) and (
            (self.cutoff_indices < 0).any()
            or (self.cutoff_indices >= len(panel.timestamps)).any()
            or np.any(self.cutoff_indices[1:] <= self.cutoff_indices[:-1])
        ):
            raise ValueError("cutoff_indices must be valid strictly ascending unique indices")
        common_cutoffs = prepared_data.common_cutoffs
        common_histories = prepared_data.common_histories
        history_by_cutoff = {
            int(cutoff): history for cutoff, history in zip(common_cutoffs, common_histories)
        }
        missing_histories = [
            int(cutoff) for cutoff in self.cutoff_indices if int(cutoff) not in history_by_cutoff
        ]
        if missing_histories:
            raise ValueError(
                "cutoff_indices do not have a common observed history: "
                f"{missing_histories[:5]}"
            )
        self.history_indices = np.stack(
            [history_by_cutoff[int(cutoff)] for cutoff in self.cutoff_indices]
        ) if len(self.cutoff_indices) else np.empty((0, sequence_length), dtype=np.int64)
        self.target_indices = np.searchsorted(
            panel.timestamps,
            panel.timestamps[self.cutoff_indices] + np.timedelta64(HORIZON_MINUTES, "m"),
        ).astype(np.int64)
        if len(self.target_indices) and (
            (self.target_indices >= len(panel.timestamps)).any()
            or not np.array_equal(
                panel.timestamps[self.target_indices],
                panel.timestamps[self.cutoff_indices] + np.timedelta64(HORIZON_MINUTES, "m"),
            )
        ):
            raise ValueError("cutoff_indices must have an exact cutoff + 15 minute target")
        node_count = len(panel.node_order)
        prior = np.asarray(economic_prior, dtype=np.float32)
        if prior.shape != (node_count, node_count):
            raise ValueError("economic_prior must have shape [nodes,nodes]")
        if not np.isfinite(prior).all() or (prior < 0.0).any() or (prior > 1.0).any():
            raise ValueError("economic_prior must be finite and in [0,1]")
        off_diagonal = ~np.eye(node_count, dtype=bool)
        self.economic_prior = prior
        self.prior_available = off_diagonal
        if regression_weight is None:
            self.regression_weight = np.zeros_like(prior)
            self.regression_available = np.zeros_like(off_diagonal)
        else:
            if regression_available is None:
                raise ValueError(
                    "regression_available is required when regression_weight is supplied"
                )
            regression = np.asarray(regression_weight, dtype=np.float32)
            available = np.asarray(regression_available)
            if available.dtype != np.bool_:
                raise TypeError("regression_available must be boolean")
            if regression.shape != prior.shape or available.shape != prior.shape:
                raise ValueError("regression arrays must have shape [nodes,nodes]")
            if not np.isfinite(regression).all() or (regression < 0.0).any() or (regression > 1.0).any():
                raise ValueError("regression_weight must be finite and in [0,1]")
            effective_available = available & off_diagonal
            if np.any(regression[~effective_available] != 0.0):
                raise ValueError("unavailable regression weights must be exactly zero")
            self.regression_weight = regression
            self.regression_available = effective_available

    def __len__(self) -> int:
        return len(self.cutoff_indices)

    def _labels(self, item: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        panel_index = int(self.cutoff_indices[item])
        labels = build_shock_follow_labels_for_cutoff(
            self.panel.timestamps,
            self.panel.trading_dates,
            self.panel.signed_returns,
            self.panel.node_order,
            self.threshold_matrix[panel_index],
            self.panel.timestamps[panel_index],
            source_input_valid=self.input_valid,
            target_input_valid=self.input_valid,
            validate_timestamps=False,
        )
        return (
            labels.target_class,
            labels.loss_mask,
            labels.source_shock,
            labels.source_sign,
        )

    def __getitem__(self, item: int) -> ShockFollowSample:
        panel_index = int(self.cutoff_indices[item])
        target_index = int(self.target_indices[item])
        history_indices = self.history_indices[item]
        target_class, loss_mask, source_shock, source_sign = self._labels(item)
        return ShockFollowSample(
            node_history=torch.from_numpy(
                self.node_features[history_indices].transpose(1, 0, 2).copy()
            ),
            economic_prior=torch.from_numpy(self.economic_prior.copy()),
            regression_weight=torch.from_numpy(self.regression_weight.copy()),
            prior_available=torch.from_numpy(self.prior_available.copy()),
            regression_available=torch.from_numpy(self.regression_available.copy()),
            target_class=torch.from_numpy(target_class.astype(np.int64, copy=True)),
            loss_mask=torch.from_numpy(loss_mask.copy()),
            source_shock=torch.from_numpy(source_shock.copy()),
            source_sign=torch.from_numpy(source_sign.astype(np.int64, copy=True)),
            cutoff_at=self.panel.timestamps[panel_index],
            target_at=self.panel.timestamps[target_index],
            trading_date=self.panel.trading_dates[panel_index],
            panel_index=panel_index,
        )


def collate_shock_follow_samples(
    samples: Sequence[ShockFollowSample],
) -> dict[str, object]:
    """Stack model tensors while retaining exact timestamp metadata."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    tensor_fields = (
        "node_history",
        "economic_prior",
        "regression_weight",
        "prior_available",
        "regression_available",
        "target_class",
        "loss_mask",
        "source_shock",
        "source_sign",
    )
    result: dict[str, object] = {
        field: torch.stack([getattr(sample, field) for sample in samples])
        for field in tensor_fields
    }
    result.update(
        {
            "cutoff_at": np.asarray([sample.cutoff_at for sample in samples]),
            "target_at": np.asarray([sample.target_at for sample in samples]),
            "trading_date": np.asarray([sample.trading_date for sample in samples]),
            "panel_index": np.asarray([sample.panel_index for sample in samples], dtype=np.int64),
        }
    )
    return result


def summarize_supervision(dataset: ShockFollowCutoffDataset) -> dict[str, object]:
    """Count events, classes, and masked cells without changing the dataset."""
    class_counts = np.zeros(3, dtype=np.int64)
    source_events = 0
    cutoff_with_events = 0
    valid_cells = 0
    for item in range(len(dataset)):
        sample = dataset[item]
        mask = sample.loss_mask.numpy()
        if mask.any():
            cutoff_with_events += 1
            source_events += int(mask.any(axis=-1).sum())
            targets = sample.target_class.numpy()[mask]
            class_counts += np.bincount(targets, minlength=3)
            valid_cells += len(targets)
    return {
        "cutoff_count": len(dataset),
        "cutoffs_with_source_events": cutoff_with_events,
        "source_event_count": source_events,
        "valid_pair_count": valid_cells,
        "class_counts": class_counts.tolist(),
        "class_fractions": (
            (class_counts / valid_cells).tolist() if valid_cells else [0.0, 0.0, 0.0]
        ),
    }
