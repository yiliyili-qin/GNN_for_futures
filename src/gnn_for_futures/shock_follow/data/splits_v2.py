"""Frozen target-time lifecycle splits for shock-follow GraphMixer V2."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from gnn_for_futures.shock_follow.contract import HORIZON_MINUTES


TRAIN_END = "2025-08-31T23:59:59"
DEVELOPMENT_START = "2025-09-01T00:00:00"
DEVELOPMENT_END = "2025-11-30T23:59:59"
CONFIRMATION_START = "2025-12-01T00:00:00"
CONFIRMATION_END = "2026-02-28T23:59:59"
PROHIBITED_TEST_START = "2026-03-01T00:00:00"
PROHIBITED_TEST_END = "2026-08-25T15:00:00"
V2_SPLIT_ORDER = (
    "training",
    "development",
    "confirmation",
    "prohibited_v1_test",
    "purged",
)


@dataclass(frozen=True)
class ShockFollowV2Split:
    """Exact cutoff memberships assigned exclusively by target timestamp."""

    training_indices: np.ndarray
    development_indices: np.ndarray
    confirmation_indices: np.ndarray
    prohibited_v1_test_indices: np.ndarray
    purged_indices: np.ndarray

    def indices(self, split: str) -> np.ndarray:
        mapping = {
            "training": self.training_indices,
            "development": self.development_indices,
            "confirmation": self.confirmation_indices,
            "prohibited_v1_test": self.prohibited_v1_test_indices,
            "purged": self.purged_indices,
        }
        try:
            return mapping[split].copy()
        except KeyError as error:
            raise ValueError(f"split must be one of {V2_SPLIT_ORDER}") from error


def _timestamp(value: object, name: str) -> np.datetime64:
    result = np.asarray(value).astype("datetime64[ns]")
    if result.ndim != 0 or np.isnat(result):
        raise ValueError(f"{name} must be one valid timestamp")
    return result


def split_index_sha256(indices: np.ndarray) -> str:
    """Hash one ordered int64 split membership."""
    values = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return sha256(values.tobytes()).hexdigest()


def shock_follow_v2_split_by_target(
    timestamps: np.ndarray,
    cutoff_indices: np.ndarray,
    *,
    train_end: object = TRAIN_END,
    development_start: object = DEVELOPMENT_START,
    confirmation_start: object = CONFIRMATION_START,
    confirmation_end: object = CONFIRMATION_END,
    prohibited_test_start: object = PROHIBITED_TEST_START,
    prohibited_test_end: object = PROHIBITED_TEST_END,
    horizon_minutes: int = HORIZON_MINUTES,
) -> ShockFollowV2Split:
    """Assign frozen V2 lifecycle memberships using exact target timestamps."""
    if (
        not isinstance(horizon_minutes, int)
        or isinstance(horizon_minutes, bool)
        or horizon_minutes < 1
    ):
        raise ValueError("horizon_minutes must be a positive integer")
    at = np.asarray(timestamps).astype("datetime64[ns]")
    if at.ndim != 1 or np.isnat(at).any():
        raise ValueError("timestamps must be a one-dimensional valid timestamp array")
    raw_indices = np.asarray(cutoff_indices)
    if raw_indices.ndim != 1 or raw_indices.dtype.kind not in "iu":
        raise ValueError("cutoff_indices must be a one-dimensional integer array")
    indices = raw_indices.astype(np.int64, copy=False)
    if len(indices) and (
        (indices < 0).any()
        or (indices >= len(at)).any()
        or np.any(indices[1:] <= indices[:-1])
    ):
        raise ValueError("cutoff_indices must be valid strictly ascending unique indices")

    boundaries = tuple(
        _timestamp(value, name)
        for value, name in (
            (train_end, "train_end"),
            (development_start, "development_start"),
            (confirmation_start, "confirmation_start"),
            (confirmation_end, "confirmation_end"),
            (prohibited_test_start, "prohibited_test_start"),
            (prohibited_test_end, "prohibited_test_end"),
        )
    )
    (
        train_boundary,
        development_boundary,
        confirmation_boundary,
        confirmation_end_boundary,
        prohibited_boundary,
        prohibited_end_boundary,
    ) = boundaries
    if not (
        train_boundary < development_boundary < confirmation_boundary
        <= confirmation_end_boundary < prohibited_boundary <= prohibited_end_boundary
    ):
        raise ValueError("V2 split boundaries are not strictly ordered")

    targets = at[indices] + np.timedelta64(horizon_minutes, "m")
    training = targets <= train_boundary
    development = (
        (targets >= development_boundary)
        & (targets < confirmation_boundary)
    )
    confirmation = (
        (targets >= confirmation_boundary)
        & (targets <= confirmation_end_boundary)
    )
    prohibited = (
        (targets >= prohibited_boundary)
        & (targets <= prohibited_end_boundary)
    )
    assigned = training | development | confirmation | prohibited
    return ShockFollowV2Split(
        training_indices=indices[training],
        development_indices=indices[development],
        confirmation_indices=indices[confirmation],
        prohibited_v1_test_indices=indices[prohibited],
        purged_indices=indices[~assigned],
    )


def split_manifest_payload(split: ShockFollowV2Split) -> dict[str, object]:
    """Return counts and hashes without reading any split labels."""
    return {
        "assignment": "target_timestamp_only",
        "horizon_minutes": HORIZON_MINUTES,
        "boundaries": {
            "training_target_end_inclusive": TRAIN_END,
            "development_target_start_inclusive": DEVELOPMENT_START,
            "development_target_end_exclusive": CONFIRMATION_START,
            "confirmation_target_start_inclusive": CONFIRMATION_START,
            "confirmation_target_end_inclusive": CONFIRMATION_END,
            "prohibited_v1_test_target_start_inclusive": PROHIBITED_TEST_START,
            "prohibited_v1_test_target_end_inclusive": PROHIBITED_TEST_END,
        },
        "counts": {
            name: len(split.indices(name))
            for name in V2_SPLIT_ORDER
        },
        "index_sha256": {
            name: split_index_sha256(split.indices(name))
            for name in V2_SPLIT_ORDER
        },
    }
