"""Metric helpers; legacy experiment replay is intentionally not distributed."""
import numpy as np

def macro_f1(counts):
    diagonal = np.diagonal(counts, axis1=-2, axis2=-1)
    denominator = counts.sum(axis=-1) + counts.sum(axis=-2)
    return np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator != 0).mean(axis=-1)


def summarize(draws, estimate):
    finite = np.isfinite(draws)
    valid = int(finite.sum())
    return dict(estimate=float(estimate) if np.isfinite(estimate) else None,
        percentile_95=np.quantile(draws[finite], [.025, .975]).tolist() if valid >= 9500 else None,
        valid_replicates=valid, invalid_replicates=int(len(draws) - valid),
        null_reason=None if valid >= 9500 else 'fewer_than_9500_valid_replicates')
