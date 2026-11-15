"""Causal raw19 formula retained for Full+Dir119Linear-v1; no bypass experiments."""
import numpy as np

def features(returns, valid, timestamps, dates, thresholds, cutoffs):
    current = returns[cutoffs]
    scale = thresholds[cutoffs]
    usable = valid[cutoffs] & np.isfinite(current) & np.isfinite(scale) & (scale > 0)
    signed = np.divide(current, scale, out=np.zeros_like(current), where=usable)
    source_sign = np.sign(np.where(usable, current, 0))
    shape = (len(cutoffs), returns.shape[1], returns.shape[1])
    expand_source = lambda values: np.broadcast_to(values[:, :, None], shape)
    expand_target = lambda values: np.broadcast_to(values[:, None, :], shape)
    result = [expand_source(source_sign), expand_source(signed), expand_target(signed),
              expand_source(source_sign) * expand_target(signed)]
    momentum, available = [], []
    for window in (2, 4, 8):
        indices = cutoffs[:, None] - np.arange(window)[None]
        safe = np.maximum(indices, 0)
        eligible = np.all(valid[safe] & np.isfinite(returns[safe]), axis=1) & usable
        eligible &= (indices >= 0).all(1)[:, None]
        contiguous = timestamps[safe] == timestamps[cutoffs, None] - np.arange(window)[None] * np.timedelta64(15, 'm')
        same_day = dates[safe] == dates[cutoffs, None]
        eligible &= (contiguous & same_day).all(1)[:, None]
        compounded = np.prod(1 + np.where(np.isfinite(returns[safe]), returns[safe], 0), axis=1) - 1
        values = np.divide(compounded, scale, out=np.zeros_like(compounded), where=eligible)
        momentum.append(values)
        available.append(eligible.astype(float))
    result.extend(expand_source(values) for values in momentum)
    result.extend(expand_target(values) for values in momentum)
    result.extend(expand_source(source_sign) * expand_target(values) for values in momentum)
    result.extend(expand_source(values) for values in available)
    result.extend(expand_target(values) for values in available)
    output = np.stack(result, -1).astype(np.float32)
    enabled = usable[:, :, None] & usable[:, None, :] & (source_sign[:, :, None] != 0)
    enabled &= ~np.eye(returns.shape[1], dtype=bool)[None]
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite direction features')
    return output, enabled
