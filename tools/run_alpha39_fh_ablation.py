"""Isolated scratch FH channel ablations on the approved legacy 39-node split."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

BOUNDS = dict(train=['2021-02-01', '2024-12-24'],
              validation=['2025-01-23', '2025-07-17'], test=['2025-08-15', '2026-10-24'])
SEEDS = (20261002, 20261003, 20261004)
CHANNELS = dict(full=(True, True, True), no_economic=(False, True, True),
                no_regression=(True, False, True), no_attention=(True, True, False),
                history=(False, True, False))


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def split_mask(panel, cutoffs, bounds):
    from tools.extend_alpha39_fh import interval
    return interval(panel.timestamps[cutoffs], panel.trading_dates[cutoffs], bounds) & (
        panel.timestamps[cutoffs] + np.timedelta64(15, 'm') < np.datetime64(bounds[1]))


def forward(model, batch):
    return model(batch['node_history'], batch['economic_prior'], batch['regression_weight'],
                 prior_available=batch['prior_available'], regression_available=batch['regression_available'])


def classify(model, loader):
    import torch
    from tools.model_performance_registry import classification
    labels, follows, directions = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            prediction = forward(model, batch)
            mask = batch['loss_mask'] & ~torch.eye(39, dtype=torch.bool)[None]
            labels.append(batch['target_class'][mask].numpy())
            follows.append(prediction.follow_logits.softmax(-1)[..., 1][mask].numpy())
            directions.append(prediction.direction_logits.softmax(-1)[..., 1][mask].numpy())
    return classification(np.concatenate(labels), np.concatenate(follows), np.concatenate(directions))


def prepare(args):
    import pandas as pd
    from gnn_for_futures.shock_follow.data.panel import load_shock_follow_bar_panel
    from gnn_for_futures.shock_follow.data.prepared import (prepare_shock_follow_data,
        eligible_cutoff_indices, ShockFollowCutoffDataset)
    from graph_model.daily_graph_dataset import DailyGraphDataset
    from graph_model.daily_return_regression import trading_day_closes, daily_close_returns, fit_daily_graph
    from gnn_for_futures.shock_follow.data.thresholds import materialize_causal_monthly_thresholds
    from tools.extend_alpha39_fh import read, interval, verify_panel_sources, MULTIPLIER
    from tools.model_performance_registry import digest, write_json

    frozen = read(ROOT / 'frozen-base.json')
    artifact = read(MULTIPLIER)
    if artifact['protocol']['split_bounds']['train'] != BOUNDS['train'] or artifact['multiplier'] != 10:
        raise ValueError('Original training boundary or multiplier changed')
    provenance = read(args.panel / 'rebuild_sources.json')
    hashes = provenance['raw_sha256']
    if set(hashes) != {node + '.csv' for node in frozen['data']['node_order']}:
        raise ValueError('Raw hash universe mismatch')
    for filename, expected in hashes.items():
        if digest(args.raw / filename) != expected:
            raise ValueError('Raw hash mismatch: ' + filename)
    panel = load_shock_follow_bar_panel(args.panel / 'bar_panel.npz', args.panel / 'bar_panel_manifest.json')
    verify_panel_sources(panel.source_files, hashes)
    original = load_shock_follow_bar_panel(args.original / 'bar_panel.npz', args.original / 'bar_panel_manifest.json')
    if original.materialization_sha256 != artifact['panel_sha256']:
        raise ValueError('Original threshold panel identity mismatch')
    if list(panel.node_order) != frozen['data']['node_order'] or panel.node_order != original.node_order or len(panel.node_order) != 39:
        raise ValueError('Node order mismatch')
    training = interval(original.timestamps, original.trading_dates, BOUNDS['train'])
    positions = np.searchsorted(panel.timestamps, original.timestamps[training])
    np.testing.assert_array_equal(panel.timestamps[positions], original.timestamps[training])
    np.testing.assert_array_equal(panel.trading_dates[positions], original.trading_dates[training])
    np.testing.assert_allclose(panel.signed_returns[positions], original.signed_returns[training], rtol=0, atol=1e-12, equal_nan=True)
    fit = np.full_like(panel.signed_returns, np.nan)
    fit[positions] = original.signed_returns[training]
    thresholds = materialize_causal_monthly_thresholds(panel.timestamps, panel.trading_dates, fit, panel.node_order)
    prepared = prepare_shock_follow_data(panel, thresholds)
    cutoffs = eligible_cutoff_indices(panel, prepared.input_valid, common_cutoffs=prepared.common_cutoffs)
    masks = {name: split_mask(panel, cutoffs, bounds) for name, bounds in BOUNDS.items()}
    histories = dict(zip(prepared.common_cutoffs.tolist(), prepared.common_histories))
    for previous, current in [('train', 'validation'), ('validation', 'test')]:
        boundary = np.datetime64(BOUNDS[previous][1])
        masks[current] &= np.array([panel.timestamps[histories[int(cutoff)][0]] > boundary and
            panel.trading_dates[histories[int(cutoff)][0]] > boundary for cutoff in cutoffs])
    if any(not mask.any() for mask in masks.values()):
        raise ValueError('Empty split')
    prior = np.load(args.original / 'economic_prior.npy', allow_pickle=False)
    graph_dir = args.output / 'daily'
    graph_dir.mkdir()
    closes = {node: trading_day_closes(pd.read_csv(args.raw / (node + '.csv'),
        usecols=['datetime', 'trading_date', 'close'])) for node in panel.node_order}
    close_frame = pd.DataFrame(closes).sort_index()
    returns = daily_close_returns(close_frame, close_frame.index)
    required = np.unique(panel.trading_dates[cutoffs[np.logical_or.reduce(list(masks.values()))]])
    thresholds.to_frame().to_csv(args.output / 'thresholds.csv', index=False)
    audit = dict(bounds=BOUNDS, raw_sha256=hashes, original_panel_sha256=original.materialization_sha256,
        panel_sha256=panel.materialization_sha256, prior_sha256=digest(args.original / 'economic_prior.npy'),
        original_multiplier_sha256=digest(MULTIPLIER), thresholds_sha256=digest(args.output / 'thresholds.csv'),
        sample_hashes={name: array_hash(cutoffs[mask]) for name, mask in masks.items()},
        sample_counts={name: int(mask.sum()) for name, mask in masks.items()}, required_graph_days=len(required))
    write_json(args.output / 'input_audit.json', audit)
    for index, date in enumerate(required):
        fit_daily_graph(returns, pd.Timestamp(date)).to_csv(graph_dir / f'{date}.csv', index=False)
        if index % 20 == 0:
            print('graphs', index + 1, '/', len(required), str(date), flush=True)
    datasets = {name: DailyGraphDataset(ShockFollowCutoffDataset(panel, thresholds, prior,
        cutoffs[mask], prepared=prepared), graph_dir, panel.node_order, None) for name, mask in masks.items()}
    for date in required:
        datasets['train'].graph_for_date(date)
    write_json(args.output / 'graph_hashes.json', {path.name: digest(path) for path in graph_dir.glob('*.csv')})
    return dict(panel=panel, prepared=prepared, cutoffs=cutoffs, masks=masks, bounds=BOUNDS, datasets=datasets), hashes


def train(context, directory, seed, variant):
    import torch
    from torch.utils.data import DataLoader
    from graph_model.alpha_variants import HistoricalGraphDataset
    from gnn_for_futures.shock_follow.data.prepared import collate_shock_follow_samples, summarize_supervision
    from gnn_for_futures.shock_follow.models.head_v2 import balanced_follow_class_weights
    from graph_model.train_shock_follow_graphmixer_v2 import run_epoch
    from gnn_for_futures.shock_follow.models.mixer_v2 import ShockFollowGraphMixerV2
    from tools.model_performance_registry import digest, write_json
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    directory.mkdir()
    datasets = {name: HistoricalGraphDataset(daily.base) if variant == 'history' else daily
                for name, daily in context['datasets'].items()}
    generator = torch.Generator().manual_seed(seed)
    def loader(name, shuffle=False):
        return DataLoader(datasets[name], batch_size=16, shuffle=shuffle, generator=generator,
                          num_workers=0, collate_fn=collate_shock_follow_samples)
    model = ShockFollowGraphMixerV2(9, learnable_fusion=False, enabled_channels=CHANNELS[variant])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    weights = torch.tensor(balanced_follow_class_weights(summarize_supervision(context['datasets']['train'].base)['class_counts']))
    best, best_epoch, history = -1., 0, []
    for epoch in range(1, 41):
        training = run_epoch(model, loader('train', True), torch.device('cpu'), optimizer, weights)
        validation = classify(model, loader('validation'))
        score = validation['macro_f1']
        if not np.isfinite(score):
            raise ValueError('Undefined validation macro F1')
        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), directory / 'best.pt')
        history.append(dict(epoch=epoch, training=training, validation=validation))
        write_json(directory / 'history.json', history)
        print(seed, variant, 'epoch', epoch, 'macroF1', score, 'best', best_epoch, flush=True)
        if epoch - best_epoch >= 7:
            break
    model.load_state_dict(torch.load(directory / 'best.pt', weights_only=True, map_location='cpu'))
    model.eval()
    write_json(directory / 'frozen.json', dict(seed=seed, variant=variant, channels=CHANNELS[variant],
        best_epoch=best_epoch, validation_macro_f1=best, checkpoint_sha256=digest(directory / 'best.pt')))
    return model, datasets


def evaluate(context, full, historical, daily, history_data, directory, args):
    import torch
    import pandas as pd
    from gnn_for_futures.shock_follow.data.prepared import collate_shock_follow_samples
    from graph_model.run_alpha39_history_training import export_split
    from graph_model.intraday_ledger import execute_ledger
    from tools.extend_alpha39_fh import combine
    from tools.research_alpha39_shifted_mapping import shifted_weights
    from tools.model_performance_registry import classification, marks_and_risk, write_json
    from tools.run_alpha39_allocation_grid import KEYS
    summary = {}
    for split in ('validation', 'test'):
        labels, follows, directions = [], [], []
        indices = {int(cutoff): index for index, cutoff in enumerate(context['cutoffs'][context['masks'][split]])}
        def predict(cutoff):
            index = indices[cutoff]
            batch = collate_shock_follow_samples([daily[split][index]])
            history_batch = collate_shock_follow_samples([history_data[split][index]])
            with torch.inference_mode():
                follow = forward(full, batch).follow_logits.softmax(-1)[0, ..., 1].numpy()
                direction = forward(historical, history_batch).direction_logits.softmax(-1)[0, ..., 1].numpy()
            mask = batch['loss_mask'][0].numpy() & ~np.eye(39, dtype=bool)
            labels.append(batch['target_class'][0].numpy()[mask])
            follows.append(follow[mask])
            directions.append(direction[mask])
            return combine(follow, direction)
        counts = export_split(directory, split, context, predict, args.raw)
        with np.load(directory / f'{split}_ledger_inputs.npz', allow_pickle=True) as archive:
            inputs = {key: archive[key] for key in KEYS}
        ledger = execute_ledger(**inputs, allocation=shifted_weights, constrained=False)
        opened, closed, entry, exit_trade, risk = marks_and_risk(inputs, ledger)
        classified = classification(np.concatenate(labels), np.concatenate(follows), np.concatenate(directions))
        np.savez_compressed(directory / f'{split}_ledger.npz', **{key: value for key, value in ledger.items() if isinstance(value, np.ndarray)})
        curves = pd.DataFrame(dict(timestamp=inputs['timestamps'], trading_date=inputs['trading_dates'], open_equity=opened, equity=closed))
        stress = []
        for bps in (0, 1, 3, 5, 10):
            net = closed - np.cumsum(entry + exit_trade) * bps / 10000
            curves[f'fixed_path_fee_{bps}bps'] = net
            stress.append(dict(bps=bps, final_equity=float(net[-1]), total_cost=float(closed[-1] - net[-1]),
                               insolvent=bool(np.any(net <= 0))))
        curves.to_csv(directory / f'{split}_equity.csv', index=False)
        monthly = curves.groupby(pd.to_datetime(curves.trading_date).dt.to_period('M')).equity.last()
        pd.DataFrame(dict(equity=monthly, return_net_zero_fee=monthly / monthly.shift(1, fill_value=1.) - 1)).to_csv(directory / f'{split}_monthly.csv')
        market_hashes = {key: array_hash(inputs[key]) for key in KEYS if key != 'utilities'}
        summary[split] = dict(classification=classified, risk=risk, fee_stress=stress, export=counts, market_signal_hashes=market_hashes)
    write_json(directory / 'evaluation.json', summary)
    return summary


def run(args):
    import torch
    import pandas as pd
    from tools.model_performance_registry import digest, write_json
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    tracked = [Path(__file__), ROOT / 'frozen-base.json'] + sorted((ROOT / 'graph_model').glob('*.py')) + sorted((ROOT / 'src').rglob('*.py'))
    sources = {str(path.relative_to(ROOT)): digest(path) for path in tracked}
    write_json(args.output / 'sources.json', sources)
    write_json(args.output / 'protocol.json', dict(bounds=BOUNDS, seeds=SEEDS, channels=CHANNELS, epochs=40,
        patience=7, optimizer='Adam', lr=.001, weight_decay=1e-5, batch_size=16, device='cpu', threads=2,
        selection='native model validation hierarchical macro F1, earliest strict maximum',
        history_reuse='One scratch History per seed; identical frozen checkpoint and direction probabilities for all four FH cells',
        ablation='Full follow q only; History direction d unchanged; channels disabled throughout training',
        strict_pit=False, previously_inspected_test=True, allocation='shifted_c0', multiplier=10,
        fee_stress='Fixed executed unit path, cumulative one-way entry+exit notional costs; no fee-feedback rebalancing',
        pid=os.getpid()))
    context, hashes = prepare(args)
    rows, alignment = [], {}
    for seed in SEEDS:
        seed_dir = args.output / str(seed)
        seed_dir.mkdir()
        historical, history_data = train(context, seed_dir / 'history', seed, 'history')
        for variant in ('full', 'no_economic', 'no_regression', 'no_attention'):
            directory = seed_dir / variant
            full, daily = train(context, directory, seed, variant)
            metrics = evaluate(context, full, historical, daily, history_data, directory, args)
            for split, result in metrics.items():
                if split in alignment and alignment[split] != result['market_signal_hashes']:
                    raise ValueError('Cross-cell market/signal alignment changed')
                alignment[split] = result['market_signal_hashes']
                rows.append(dict(seed=seed, variant=variant, split=split,
                    macro_f1=result['classification']['macro_f1'], **result['risk']))
            write_json(directory / 'history_reuse.json', dict(path='../history/best.pt', sha256=digest(seed_dir / 'history/best.pt')))
            pd.DataFrame(rows).to_csv(args.output / 'metrics.csv', index=False)
    frame = pd.DataFrame(rows)
    numeric = ['macro_f1', 'final_equity', 'total_return', 'annual_return', 'sharpe', 'bar_max_drawdown', 'calmar']
    frame.groupby(['split', 'variant'])[numeric].agg(['mean', 'std', 'min', 'max']).to_csv(args.output / 'three_seed_comparison.csv')
    for filename, expected in hashes.items():
        if digest(args.raw / filename) != expected:
            raise ValueError('Raw changed during run')
    if any(digest(ROOT / path) != expected for path, expected in sources.items()):
        raise ValueError('Source changed during run')
    write_json(args.output / 'completed.json', dict(status='completed', models=15, fh_cells=12,
        artifacts={str(path.relative_to(args.output)): digest(path) for path in args.output.rglob('*') if path.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=ROOT / 'fh_retry/panel')
    parser.add_argument('--original', type=Path, default=ROOT / 'results/alpha39_expert_factor_v2_20261022')
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/raw_data/15min')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/alpha39_fh_ablation_20261025')
    options = parser.parse_args()
    try:
        run(options)
    except Exception as error:
        if options.output.is_dir():
            (options.output / 'failed.json').write_text(json.dumps(dict(error=repr(error), pid=os.getpid())), encoding='utf-8')
        raise