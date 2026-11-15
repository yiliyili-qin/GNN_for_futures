"""Frozen shifted-c0 allocation only; no historical search experiments."""
import numpy as np

def shifted_weights(utilities):
    utilities = np.asarray(utilities, dtype=float)
    if utilities.ndim != 1 or not utilities.size or not np.isfinite(utilities).all():
        raise ValueError('Utilities must be a nonempty finite vector')
    raw = np.zeros_like(utilities)
    negative, positive = utilities < 0., utilities > 0.
    raw[negative] = np.exp(utilities[negative]) - 2.
    raw[positive] = 2. - np.exp(-utilities[positive])
    denominator = np.abs(raw).sum()
    if not np.isfinite(denominator):
        raise ValueError('Nonfinite L1 denominator')
    return raw / denominator if denominator else raw
