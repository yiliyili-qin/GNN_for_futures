"""Exact event-label construction for shock-follow cutoffs and aligned panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gnn_for_futures.shock_follow.contract import (
    CLASS_NO_FOLLOW,
    CLASS_OPPOSITE_DIRECTION,
    CLASS_SAME_DIRECTION,
    HORIZON_MINUTES,
    LABEL_UNAVAILABLE,
    validate_node_order,
)
from gnn_for_futures.shock_follow.data.thresholds import MonthlyThresholdTable


@dataclass(frozen=True)
class ShockFollowLabels:
    """One cutoff's source-by-target labels, mask, and timing provenance."""

    cutoff_at: np.datetime64
    target_at: np.datetime64
    trading_date: np.datetime64
    target_trading_date: np.datetime64
    source_shock: np.ndarray
    source_sign: np.ndarray
    target_shock: np.ndarray
    target_sign: np.ndarray
    target_class: np.ndarray
    loss_mask: np.ndarray


def _timestamps(values: Sequence[object] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values).astype("datetime64[ns]")
    if result.ndim != 1 or np.isnat(result).any():
        raise ValueError(f"{name} must be one-dimensional valid timestamps")
    return result


def _thresholds_at(
    thresholds: np.ndarray | MonthlyThresholdTable,
    cutoff_at: np.datetime64,
    effective_trading_date: np.datetime64,
    node_count: int,
) -> np.ndarray:
    values = (
        thresholds.thresholds_for(
            cutoff_at, effective_trading_date=effective_trading_date
        )
        if isinstance(thresholds, MonthlyThresholdTable)
        else np.asarray(thresholds, dtype=float)
    )
    if values.shape != (node_count,):
        raise ValueError("thresholds must have one value per symbol")
    return values


def build_shock_follow_labels_for_cutoff(
    timestamps: Sequence[object] | np.ndarray,
    trading_dates: Sequence[object] | np.ndarray,
    signed_returns: np.ndarray,
    node_order: Sequence[str],
    thresholds: np.ndarray | MonthlyThresholdTable,
    cutoff_at: object,
    *,
    source_input_valid: np.ndarray | None = None,
    target_input_valid: np.ndarray | None = None,
    validate_timestamps: bool = True,
) -> ShockFollowLabels:
    """Build labels only when a real, same-date ``cutoff + 15m`` bar exists.

    Missing target returns are encoded as ``LABEL_UNAVAILABLE`` and excluded;
    they can never become ``no_follow``. Validity masks are aligned [time,node].
    """
    names = validate_node_order(node_order)
    if not isinstance(validate_timestamps, bool):
        raise TypeError("validate_timestamps must be boolean")
    if validate_timestamps:
        at = _timestamps(timestamps, "timestamps")
        dates = _timestamps(trading_dates, "trading_dates").astype("datetime64[D]")
    else:
        at = np.asarray(timestamps)
        dates = np.asarray(trading_dates)
        if at.ndim != 1 or at.dtype != np.dtype("datetime64[ns]"):
            raise TypeError(
                "prevalidated timestamps must be one-dimensional datetime64[ns]"
            )
        if dates.ndim != 1 or dates.dtype != np.dtype("datetime64[D]"):
            raise TypeError(
                "prevalidated trading_dates must be one-dimensional datetime64[D]"
            )
    returns = np.asarray(signed_returns, dtype=np.float64)
    if returns.shape != (len(at), len(names)) or len(dates) != len(at):
        raise ValueError("timestamps, trading_dates, and signed_returns must align")
    if validate_timestamps and len(at) and np.any(at[1:] <= at[:-1]):
        raise ValueError("timestamps must be strictly ascending and unique")
    cutoff = np.asarray(cutoff_at).astype("datetime64[ns]")
    if cutoff.ndim != 0 or np.isnat(cutoff):
        raise ValueError("cutoff_at must be one valid timestamp")
    current_index = int(np.searchsorted(at, cutoff))
    if current_index >= len(at) or at[current_index] != cutoff:
        raise KeyError("cutoff_at must be an observed aligned timestamp")
    target_at = cutoff + np.timedelta64(HORIZON_MINUTES, "m")
    candidate_target = int(np.searchsorted(at, target_at))
    target_index = (
        candidate_target
        if candidate_target < len(at) and at[candidate_target] == target_at
        else None
    )
    node_count = len(names)
    if source_input_valid is None:
        current_valid = np.isfinite(returns[current_index])
    else:
        source_validity = np.asarray(source_input_valid)
        if source_validity.dtype != np.dtype(bool):
            raise TypeError("source_input_valid must be boolean")
        if source_validity.shape != returns.shape:
            raise ValueError("source_input_valid must have shape [timestamps, node_order]")
        current_valid = source_validity[current_index]
    if target_input_valid is None:
        target_history_valid = np.isfinite(returns[current_index])
    else:
        target_validity = np.asarray(target_input_valid)
        if target_validity.dtype != np.dtype(bool):
            raise TypeError("target_input_valid must be boolean")
        if target_validity.shape != returns.shape:
            raise ValueError("target_input_valid must have shape [timestamps, node_order]")
        target_history_valid = target_validity[current_index]
    if current_valid.shape != (node_count,) or target_history_valid.shape != (node_count,):
        raise ValueError("input validity must have shape [timestamps, node_order]")
    threshold_values = _thresholds_at(
        thresholds, cutoff, dates[current_index], node_count
    )
    threshold_valid = np.isfinite(threshold_values) & (threshold_values > 0.0)
    current = returns[current_index]
    source_shock = current_valid & threshold_valid & (np.abs(current) >= threshold_values)
    source_sign = np.zeros(node_count, dtype=np.int8)
    finite_current = np.isfinite(current)
    source_sign[finite_current] = np.sign(current[finite_current]).astype(np.int8)

    target_class = np.full((node_count, node_count), LABEL_UNAVAILABLE, dtype=np.int8)
    loss_mask = np.zeros((node_count, node_count), dtype=bool)
    target_shock = np.zeros(node_count, dtype=bool)
    target_sign = np.zeros(node_count, dtype=np.int8)
    target_date = np.datetime64("NaT", "D")
    if target_index is not None:
        target_date = dates[target_index]
        next_returns = returns[target_index]
        label_valid = np.isfinite(next_returns) & threshold_valid
        target_shock = label_valid & (np.abs(next_returns) >= threshold_values)
        target_sign[label_valid] = np.sign(next_returns[label_valid]).astype(np.int8)
        same_date = dates[current_index] == target_date
        for source in np.flatnonzero(source_shock):
            valid_targets = target_history_valid & label_valid & same_date
            valid_targets[source] = False
            loss_mask[source, valid_targets] = True
            target_class[source, valid_targets] = CLASS_NO_FOLLOW
            following = valid_targets & target_shock
            same = following & (target_sign == source_sign[source])
            target_class[source, following & ~same] = CLASS_OPPOSITE_DIRECTION
            target_class[source, same] = CLASS_SAME_DIRECTION
    return ShockFollowLabels(
        cutoff, target_at, dates[current_index], target_date, source_shock,
        source_sign, target_shock, target_sign, target_class, loss_mask,
    )


def build_shock_follow_label_panel(
    timestamps: Sequence[object] | np.ndarray,
    trading_dates: Sequence[object] | np.ndarray,
    signed_returns: np.ndarray,
    node_order: Sequence[str],
    thresholds: np.ndarray | MonthlyThresholdTable,
    *,
    source_input_valid: np.ndarray | None = None,
    target_input_valid: np.ndarray | None = None,
) -> tuple[ShockFollowLabels, ...]:
    """Build all aligned-cutoff labels without substituting a later bar."""
    at = _timestamps(timestamps, "timestamps")
    return tuple(
        build_shock_follow_labels_for_cutoff(
            at, trading_dates, signed_returns, node_order, thresholds, cutoff,
            source_input_valid=source_input_valid,
            target_input_valid=target_input_valid,
        )
        for cutoff in at
    )
