"""Chronological split and embargo helpers; no portfolio search code."""
import numpy as np


class InvalidNormalization(ValueError):
    """Finite utilities whose signed denominator cannot define a target."""


def utility_weights(utilities, *, leverage=1.):
    """Document mapping with x=u; L is net, not gross exposure."""
    utilities = np.asarray(utilities, dtype=float)
    if utilities.ndim != 1 or not utilities.size or not np.isfinite(utilities).all():
        raise ValueError("Utilities must be a nonempty finite vector")
    if not np.isfinite(leverage) or leverage < 0:
        raise ValueError("Leverage must be finite and nonnegative")
    raw = np.ones_like(utilities)
    negative = utilities < 0
    positive = utilities > 0
    raw[negative] = np.expm1(utilities[negative])
    raw[positive] = 3. - np.exp(-utilities[positive])
    denominator = raw.sum()
    if denominator <= 1e-8 * max(1., np.abs(raw).sum()):
        raise InvalidNormalization("Nonpositive or near-zero signed normalization denominator")
    return leverage * raw / denominator


SPLITS = {
    "train": ("2023-09-15", "2025-10-02"),
    "validation": ("2025-10-16", "2025-12-01"),
}


def chronological_split_bounds(trading_dates):
    days = np.unique(np.asarray(trading_dates, dtype="datetime64[D]"))
    if len(days) < 250 or np.isnat(days).any():
        raise ValueError("At least 250 valid trading days required")
    train_end = int(len(days) * .7)
    validation_end = int(len(days) * .8)
    validation_start = max(train_end + 21, int(np.searchsorted(days, days[train_end - 1] + np.timedelta64(16, "D"))))
    test_start = max(validation_end + 21, int(np.searchsorted(days, days[validation_end - 1] + np.timedelta64(16, "D"))))
    if validation_start >= validation_end or test_start >= len(days):
        raise ValueError("Insufficient days after rolling-window embargo")
    return {"train": (str(days[0]), str(days[train_end])),
            "validation": (str(days[validation_start]), str(days[validation_end])),
            "test": (str(days[test_start]), str(days[-1] + np.timedelta64(1, "D")))}


def split_masks(signal_times, label_end_times, trading_dates, test_end, *, bounds=None):
    """Require the full prediction interval and trading date inside one split.

    test_end is an exclusive, predeclared bound, not selected from test returns.
    """
    signals = np.asarray(signal_times, dtype="datetime64[ns]")
    ends = np.asarray(label_end_times, dtype="datetime64[ns]")
    dates = np.asarray(trading_dates, dtype="datetime64[D]")
    if signals.ndim != 1 or signals.shape != ends.shape or signals.shape != dates.shape:
        raise ValueError("Time arrays must be aligned one-dimensional arrays")
    if any(np.isnat(values).any() for values in (signals, ends, dates)):
        raise ValueError("Missing time metadata")
    if np.any(ends <= signals):
        raise ValueError("Labels must end strictly after the signal")
    upper = np.datetime64(test_end, "ns")
    if np.isnat(upper) or (bounds is None and upper <= np.datetime64("2025-12-15")):
        raise ValueError("A fixed exclusive test end after 2025-12-15 is required")
    bounds = bounds or dict(SPLITS, test=("2025-12-15", upper))
    return {
        name: ((signals >= np.datetime64(lower)) & (ends < np.datetime64(upper))
               & (dates >= np.datetime64(lower)) & (dates < np.datetime64(upper)))
        for name, (lower, upper) in bounds.items()
    }
