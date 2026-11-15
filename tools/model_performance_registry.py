"""Classification and ledger arithmetic shared by the released experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

def classification(labels, follow, direction):
    labels = np.asarray(labels, dtype=int).reshape(-1)
    follow, direction = [np.asarray(value, dtype=float).reshape(-1) for value in (follow, direction)]
    if not labels.size or labels.shape != follow.shape or labels.shape != direction.shape:
        raise ValueError('Nonempty aligned classification rows required')
    if not np.isin(labels, [0, 1, 2]).all() or any(not np.isfinite(value).all() or np.any((value < 0) | (value > 1)) for value in (follow, direction)):
        raise ValueError('Invalid labels or probabilities')
    probability = np.column_stack((1 - follow, follow * (1 - direction), follow * direction))
    predicted = np.where(follow <= .5, 0, np.where(direction > .5, 2, 1))
    counts = confusion_matrix(labels, predicted, labels=[0, 1, 2])
    support, predicted_support = counts.sum(axis=1), counts.sum(axis=0)
    recall = np.divide(counts.diagonal(), support, out=np.zeros(3), where=support != 0)
    precision = np.divide(counts.diagonal(), predicted_support, out=np.zeros(3), where=predicted_support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall != 0)

    def ap(target, score):
        return float(average_precision_score(target, score)) if np.unique(target).size == 2 else None

    ovr = [ap(labels == index, probability[:, index]) for index in range(3)]
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    clipped /= clipped.sum(axis=1, keepdims=True)
    confidence = probability[np.arange(len(labels)), predicted]
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(float((bins == index).mean()) * abs(float(confidence[bins == index].mean()) - float((predicted[bins == index] == labels[bins == index]).mean())) for index in range(10) if np.any(bins == index))
    return dict(macro_f1=float(f1.mean()), balanced_accuracy=float(recall[support > 0].mean()),
        supported_class_count=int((support > 0).sum()), precision=precision.tolist(), recall=recall.tolist(),
        f1=f1.tolist(), support=support.tolist(), confusion_counts=counts.tolist(),
        confusion_row_normalized=np.divide(counts, support[:, None], out=np.zeros((3, 3)), where=support[:, None] != 0).tolist(),
        follow_AP=ap(labels != 0, follow), OVR_AP=ovr, OVR_macro_AP=float(np.mean(ovr)) if all(value is not None for value in ovr) else None,
        undefined_AP_reason='no_positive_or_no_negative', log_loss=float(-np.log(clipped[np.arange(len(labels)), labels]).mean()),
        multiclass_Brier=float(((probability - np.eye(3)[labels]) ** 2).sum(axis=1).mean()), ECE=ece,
        accuracy=float((predicted == labels).mean()))


def marks_and_risk(inputs, result):
    opens, closes = inputs['opens'], inputs['closes']
    carried = np.zeros_like(result['units'])
    carried[1:] = result['units'][:-1] + result['close_trade_units'][:-1]
    changes = np.zeros_like(opens)
    changes[1:] = opens[1:] - closes[:-1]
    open_equity = result['equity'][:-1] + np.where(carried != 0, carried * changes, 0).sum(axis=1)
    close_equity = result['equity'][1:]
    marks = np.r_[1., np.column_stack((open_equity, close_equity)).reshape(-1)]
    if not np.isfinite(marks).all() or np.any(marks <= 0):
        raise ValueError('Invalid reconstructed equity')
    open_notional = np.where(result['trade_units'] != 0, np.abs(result['trade_units']) * opens, 0).sum(axis=1)
    close_notional = np.where(result['close_trade_units'] != 0, np.abs(result['close_trade_units']) * closes, 0).sum(axis=1)
    turnover = open_notional / open_equity + close_notional / close_equity
    np.testing.assert_allclose(turnover, result['turnover'], rtol=1e-12, atol=1e-12)
    drawdown = float(np.max(1 - marks / np.maximum.accumulate(marks)))
    daily = result['daily_returns']
    annual = float(close_equity[-1] ** (252 / len(daily)) - 1)
    deviation = float(daily.std(ddof=1)) if len(daily) > 1 else None
    risk = dict(final_equity=float(close_equity[-1]), total_return=float(close_equity[-1] - 1), annual_return=annual,
        annual_volatility=deviation * np.sqrt(252) if deviation is not None else None,
        sharpe=float(np.sqrt(252) * daily.mean() / deviation) if deviation else None,
        bar_max_drawdown=drawdown, calmar=annual / drawdown if drawdown else None,
        undefined_ratio_reason='zero_denominator_or_insufficient_days', turnover=float(turnover.sum()),
        average_daily_turnover=float(turnover.sum() / len(daily)), trading_days=len(daily))
    return open_equity, close_equity, open_notional, close_notional, risk


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_hash(*values):
    hasher = hashlib.sha256()
    for value in values:
        value = np.ascontiguousarray(value)
        hasher.update(str((value.shape, value.dtype.str)).encode())
        hasher.update(value.tobytes())
    return hasher.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
