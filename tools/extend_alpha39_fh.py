"""Frozen Full/History references and split helpers, without old research runners."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

FULL = ROOT / 'results/alpha39_intraday_20261020_v2_r3_download/alpha39_intraday_20261020_v2/selection/best.pt'


HISTORY = ROOT / 'results/alpha39_residual_20261022/historical_dual/best.pt'


HASHES = ('ced9d117acc16c6df56e39ae11e7a26eb3fda3d629371ee0aa667ca66ff697b6',
          '9937ffebb3370060685c1f24dd09372e75b0cde2dcb4286d25a808edf2cea5aa')


MULTIPLIER = FULL.with_name('multiplier_frozen.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def interval(timestamps, dates, bounds):
    start, end = map(np.datetime64, bounds)
    return (timestamps >= start) & (timestamps < end) & (dates >= start) & (dates < end)


def combine(follow, direction):
    if follow.shape != direction.shape or not np.isfinite([follow, direction]).all():
        raise ValueError('Invalid aligned probabilities')
    if np.any((follow < 0) | (follow > 1) | (direction < 0) | (direction > 1)):
        raise ValueError('Probability outside [0,1]')
    return dict(joint_scores=np.stack((1-follow, follow*(1-direction), follow*direction), -1),
                follow_probability=follow, predicted_follow=follow > .5)


def merged_hashes(audit, nodes, through):
    records = audit.get('records', [])
    if (audit.get('status') != 'completed' or set(audit.get('replaced', [])) != set(nodes)
            or len(records) != len(nodes) or {record['symbol'] for record in records} != set(nodes)):
        raise ValueError('Raw merge is not completed for all canonical nodes')
    if any(np.datetime64(record['merged_last_trading_date']) < np.datetime64(through) for record in records):
        raise ValueError('Merged coverage does not reach requested date')
    return {record['symbol'] + '.csv': record['after_sha256'] for record in records}


def verify_panel_sources(sources, raw_hashes):
    actual = {source.symbol + '.csv': source.sha256 for source in sources}
    if len(sources) != len(raw_hashes) or actual != raw_hashes:
        raise ValueError('Panel sources do not match completed raw merge')
