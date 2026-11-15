"""Independent ten-year training adapter; never loads legacy checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import random
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np

def classification(target, follow, direction):
    target, follow, direction = map(np.asarray, (target, follow, direction))
    if target.size == 0 or target.shape != follow.shape or target.shape != direction.shape:
        raise ValueError('Empty or mismatched prediction rows')
    if not np.isin(target, [0, 1, 2]).all():
        raise ValueError('Invalid class')
    for values in (follow, direction):
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Invalid probability')
    predicted = np.where(follow > 0.5, np.where(direction > 0.5, 2, 1), 0)
    confusion = np.bincount(3 * target.astype(int).ravel() + predicted.ravel(), minlength=9).reshape(3, 3)
    denominator = confusion.sum(0) + confusion.sum(1)
    scores = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(3), where=denominator > 0)
    return dict(macro_f1=float(scores.mean()), per_class_f1=scores.tolist(), confusion=confusion.tolist(), rows=int(target.size))

def frozen_fh(full, history):
    if full['seed'] != history['seed'] or not np.array_equal(full['row_keys'], history['row_keys']):
        raise ValueError('FH requires identical seed and ordered rows')
    if not np.array_equal(full['target'], history['target']):
        raise ValueError('FH target mismatch')
    return dict(full, d=history['d'].copy())
FROZEN = dict(seeds=[20261002, 20261003, 20261004], max_epochs=40, patience=7, batch_size=16, learning_rate=0.001, weight_decay=1e-05, metric='validation_macro_f1', optimizer='Adam')
FIELDS = ('node_history', 'economic_prior', 'regression_weight', 'prior_available', 'regression_available', 'historical_weight', 'historical_available', 'target_class', 'loss_mask', 'row_keys')

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def make_model(family, node_count, feature_count, sequence_length, kwargs):
    from gnn_for_futures.shock_follow.models.mixer_v2 import ShockFollowGraphMixerV2
    if family not in ('full', 'history'):
        raise ValueError('This release supports only full/history v2 models')
    return ShockFollowGraphMixerV2(feature_count, expected_num_futures=node_count, attention_history_length=sequence_length, enabled_channels=(True, True, True) if family == 'full' else (False, True, False), **kwargs)

def forward(model, family, batch):
    history = family == 'history'
    return model(batch['node_history'], batch['economic_prior'], batch['historical_weight' if history else 'regression_weight'], prior_available=batch['prior_available'], regression_available=batch['historical_available' if history else 'regression_available'])

def validate_arrays(arrays, nodes, sequence, families):
    for split in ('train', 'validation', 'test'):
        required = FIELDS
        missing = [f'{split}_{field}' for field in required if f'{split}_{field}' not in arrays]
        if missing:
            raise ValueError(f'Missing arrays: {missing}')
        history = arrays[f'{split}_node_history']
        if history.ndim != 4 or history.shape[1:3] != (len(nodes), sequence) or (not len(history)):
            raise ValueError(f'Invalid node history: {split}')
        pair_shape = (len(history), len(nodes), len(nodes))
        for field in required:
            values = arrays[f'{split}_{field}']
            expected = history.shape if field == 'node_history' else pair_shape + ((3,) if field == 'mlp_scores' else ())
            if values.shape != expected:
                raise ValueError(f'Invalid shape {split}_{field}: {values.shape} != {expected}')
            if field.endswith('available') or field == 'loss_mask':
                if values.dtype != np.bool_:
                    raise ValueError(f'Expected boolean {field}')
            elif field != 'row_keys' and (not np.isfinite(values).all()):
                raise ValueError(f'Nonfinite {field}')
        mask = arrays[f'{split}_loss_mask']
        if not mask.any() or np.diagonal(mask, axis1=1, axis2=2).any():
            raise ValueError('Empty or diagonal supervision')
        if not np.isin(arrays[f'{split}_target_class'][mask], [0, 1, 2]).all():
            raise ValueError('Invalid supervised labels')
        keys = arrays[f'{split}_row_keys'][mask]
        if keys.dtype.kind not in 'US' or len(np.unique(keys)) != len(keys):
            raise ValueError('Row keys must be unique strings')
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if np.intersect1d(arrays[f'{left}_row_keys'][arrays[f'{left}_loss_mask']], arrays[f'{right}_row_keys'][arrays[f'{right}_loss_mask']]).size:
            raise ValueError('Overlapping split row keys')

def split_boundaries(protocol):
    import pandas as pd
    data = protocol['data']
    boundaries = data.get('split_boundaries')
    if boundaries is None:
        splits = data['splits']
        boundaries = [splits[name]['nominal_start'] for name in ('train', 'validation', 'test')] + [splits['test'].get('end_exclusive', splits['test'].get('nominal_end_exclusive'))]
    edges = [pd.Timestamp(value) for value in boundaries]
    if len(edges) != 4 or any((lower >= upper for lower, upper in zip(edges, edges[1:]))):
        raise ValueError('Four increasing split boundaries required')
    if protocol.get('version', '').startswith('3.'):
        raise ValueError('This release distributes the frozen v2 adapter only')
    return edges

def verify_test_span(first, end_exclusive):
    import pandas as pd
    if pd.Timestamp(first) + pd.DateOffset(years=2) > pd.Timestamp(end_exclusive):
        raise ValueError('Post-embargo test coverage is less than two years')

def preflight(protocol_path, data_path, device):
    import torch
    report = dict(status='blocked', blockers=[], device=device, torch=torch.__version__, cuda_available=torch.cuda.is_available(), cpu_count=__import__('os').cpu_count(), pending=['residual', 'MOE'], legacy_checkpoints_allowed=False)
    try:
        if device == 'cuda' and (not torch.cuda.is_available()):
            raise ValueError('Requested CUDA unavailable')
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        if protocol.get('status') not in ('READY', 'TRAINING_READY'):
            raise ValueError(f"Independent protocol status: {protocol.get('status')}; no real training allowed")
        adapter = protocol['training_adapter']
        if adapter['schema'] != 'ten_year_npz_training_v1' or adapter['approved'] is not True:
            raise ValueError('Training adapter not approved')
        if adapter['eligibility_audit_passed'] is not True:
            raise ValueError('Independent ten-year eligibility audit required')
        split_boundaries(protocol)
        if not adapter.get('embargo_verified') or not adapter.get('audit_sha256'):
            raise ValueError('Audited embargo evidence required')
        audit_path = protocol_path.parent / adapter['audit_path']
        if sha(audit_path) != adapter['audit_sha256']:
            raise ValueError('Audit SHA256 mismatch')
        selection = protocol.get('selection', protocol.get('inherited', {}).get('selection', {}))
        for key, value in FROZEN.items():
            if selection.get(key) != value:
                raise ValueError(f'Frozen training selection mismatch: {key}')
        if sha(data_path) != adapter['data_sha256']:
            raise ValueError('Data SHA256 mismatch')
        if protocol.get('version', '').startswith('3.'):
            raise ValueError('This release distributes the frozen v2 adapter only')
        families = adapter['models']
        if not {'full', 'history'}.issubset(families) or not set(families) <= {'full', 'history'}:
            raise ValueError('Register full/history and optional mlp; residual/MOE pending')
        for family, spec in families.items():
            if spec['loss'] != 'hierarchical_source_event_balanced_train' or spec['initialization'] != 'scratch':
                raise ValueError(f'Unregistered loss/initialization: {family}')
            if family == 'mlp' and spec.get('feature_provenance') != 'same_seed_new_frozen_history_E_R_A':
                raise ValueError('MLP requires same-seed newly trained history E/R/A')
        nodes = adapter['node_order']
        if len(nodes) < 2 or len(set(nodes)) != len(nodes):
            raise ValueError('Invalid node universe')
        with np.load(data_path, allow_pickle=False) as archive:
            arrays = dict(archive)
        if arrays['node_order'].tolist() != nodes:
            raise ValueError('Node order mismatch')
        validate_arrays(arrays, nodes, adapter['sequence_bars'], families)
        for family, spec in families.items():
            make_model(family, len(nodes), arrays['train_node_history'].shape[-1], adapter['sequence_bars'], spec['kwargs'])
        report.update(status='ready', protocol_sha256=sha(protocol_path), data_sha256=sha(data_path), nodes=len(nodes), models=list(families), samples={split: len(arrays[f'{split}_node_history']) for split in ('train', 'validation', 'test')})
    except (OSError, ValueError, KeyError, TypeError) as error:
        report['blockers'].append(str(error))
    return report

def epoch(model, family, arrays, split, weights, device, optimizer=None, batch_size=16):
    import torch
    from gnn_for_futures.shock_follow.models.head_v2 import hierarchical_source_event_loss
    model.train(optimizer is not None)
    count = len(arrays[f'{split}_node_history'])
    order = torch.randperm(count).numpy() if optimizer is not None else np.arange(count)
    predictions = {key: [] for key in ('q', 'd', 'target', 'row_keys')}
    steps = 0
    for indices in (order[start:start + batch_size] for start in range(0, count, batch_size)):
        batch = {}
        for field in FIELDS + (('mlp_scores',) if family == 'mlp' else ()):
            if field != 'row_keys':
                tensor = torch.from_numpy(arrays[f'{split}_{field}'][indices]).to(device)
                batch[field] = tensor.long() if field == 'target_class' else tensor.float() if tensor.is_floating_point() else tensor
        mask = batch['loss_mask']
        with torch.set_grad_enabled(optimizer is not None):
            output = forward(model, family, batch)
            if optimizer is not None:
                outcome = hierarchical_source_event_loss(output.follow_logits, output.direction_logits, batch['target_class'], mask, weights)
                if outcome.total_loss is not None:
                    if not torch.isfinite(outcome.total_loss):
                        raise ValueError('Nonfinite training loss')
                    optimizer.zero_grad(set_to_none=True)
                    outcome.total_loss.backward()
                    optimizer.step()
                    steps += 1
        predictions['q'].append(output.follow_logits.softmax(-1)[..., 1][mask].detach().cpu().numpy())
        predictions['d'].append(output.direction_logits.softmax(-1)[..., 1][mask].detach().cpu().numpy())
        predictions['target'].append(batch['target_class'][mask].cpu().numpy())
        predictions['row_keys'].append(arrays[f'{split}_row_keys'][indices][mask.cpu().numpy()])
    predictions = {key: np.concatenate(value) for key, value in predictions.items()}
    return (classification(predictions['target'], predictions['q'], predictions['d']), predictions, steps)

def train(protocol_path, data_path, output, device, threads):
    import torch
    from gnn_for_futures.shock_follow.models.head_v2 import balanced_follow_class_weights
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    adapter = protocol['training_adapter']
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = dict(archive)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol_snapshot.json', protocol)
    write_json(output / 'environment.json', dict(torch=torch.__version__, python=sys.version, device=device, threads=threads))
    write_json(output / 'input_hashes.json', dict(protocol=sha(protocol_path), data=sha(data_path)))
    source_paths = [Path(__file__)] + sorted((ROOT / 'src/gnn_for_futures/shock_follow').rglob('*.py'))
    write_json(output / 'source_hashes.json', {str(path.relative_to(ROOT)): sha(path) for path in source_paths})
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    counts = np.bincount(arrays['train_target_class'][arrays['train_loss_mask']].astype(int), minlength=3)
    weights = torch.tensor(balanced_follow_class_weights(counts.tolist()), dtype=torch.float32, device=device)
    summaries = []
    for seed in FROZEN['seeds']:
        components = {}
        for family in (name for name in ('full', 'history', 'mlp') if name in adapter['models']):
            spec = adapter['models'][family]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device == 'cuda':
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
            model = make_model(family, len(adapter['node_order']), arrays['train_node_history'].shape[-1], adapter['sequence_bars'], spec['kwargs']).to(device)
            destination = output / f'{family}_{seed}'
            destination.mkdir()
            optimizer = torch.optim.Adam(model.parameters(), lr=FROZEN['learning_rate'], weight_decay=FROZEN['weight_decay'])
            best_score, best_epoch, steps, history = (-1.0, 0, 0, [])
            started = time.perf_counter()
            for number in range(1, FROZEN['max_epochs'] + 1):
                training, _, updates = epoch(model, family, arrays, 'train', weights, device, optimizer)
                validation, _, _ = epoch(model, family, arrays, 'validation', weights, device)
                steps += updates
                history.append(dict(epoch=number, train=training, validation=validation))
                write_json(destination / 'history.json', history)
                if validation['macro_f1'] > best_score:
                    best_score, best_epoch = (validation['macro_f1'], number)
                    torch.save(model.state_dict(), destination / 'best.pt')
                print(f"{family} seed={seed} epoch={number} macroF1={validation['macro_f1']:.8f} best={best_epoch}", flush=True)
                if number - best_epoch >= FROZEN['patience']:
                    break
            model.load_state_dict(torch.load(destination / 'best.pt', weights_only=True, map_location=device))
            model.requires_grad_(False)
            components[family] = {}
            for split in ('validation', 'test'):
                metrics, prediction, _ = epoch(model, family, arrays, split, weights, device)
                prediction['seed'] = seed
                components[family][split] = prediction
                np.savez_compressed(destination / f'{split}_predictions.npz', **prediction)
                write_json(destination / f'{split}_classification.json', metrics)
            summary = dict(family=family, seed=seed, best_epoch=best_epoch, validation_macro_f1=best_score, epochs=number, optimizer_steps=steps, parameters=sum((parameter.numel() for parameter in model.parameters())), wall_seconds=time.perf_counter() - started, checkpoint_sha256=sha(destination / 'best.pt'), peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None)
            summaries.append(summary)
            write_json(destination / 'training_completed.json', summary)
            del optimizer, model
        destination = output / f'FH_{seed}'
        destination.mkdir()
        for split in ('validation', 'test'):
            prediction = frozen_fh(components['full'][split], components['history'][split])
            np.savez_compressed(destination / f'{split}_predictions.npz', **prediction)
            write_json(destination / f'{split}_classification.json', classification(prediction['target'], prediction['q'], prediction['d']))
    write_json(output / 'training_completed.json', dict(status='training_and_classification_only', trials=summaries, pending=['residual', 'MOE', 'backtest_and_full_registry_admission']))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    report = preflight(args.protocol, args.data, args.device)
    print(json.dumps(report, indent=2), flush=True)
    if report['status'] != 'ready':
        return 2
    if args.smoke:
        import torch
        from gnn_for_futures.shock_follow.models.head_v2 import balanced_follow_class_weights
        torch.set_num_threads(args.threads)
        torch.manual_seed(20261002)
        protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
        with np.load(args.data, allow_pickle=False) as archive:
            arrays = {key: archive[key][:2].copy() for key in archive.files if key.startswith('train_')}
        counts = np.bincount(arrays['train_target_class'][arrays['train_loss_mask']], minlength=3)
        weights = torch.tensor(balanced_follow_class_weights(counts.tolist()), dtype=torch.float32, device=args.device)
        smoke = dict(status='real_two_sample_smoke_only', epochs=0, models={})
        for family, spec in protocol['training_adapter']['models'].items():
            if family == 'mlp':
                continue
            model = make_model(family, len(protocol['training_adapter']['node_order']), arrays['train_node_history'].shape[-1], protocol['training_adapter']['sequence_bars'], spec['kwargs']).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05)
            metrics, _, steps = epoch(model, family, arrays, 'train', weights, args.device, optimizer, batch_size=2)
            if steps != 1:
                raise ValueError('Smoke check did not execute one supervised optimizer step')
            smoke['models'][family] = dict(optimizer_steps=steps, rows=metrics['rows'])
        print(json.dumps(smoke), flush=True)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(args.output / 'smoke.json', smoke)
        return 0
    if args.execute:
        if args.output is None or args.threads < 1:
            parser.error('Execution requires --output and positive --threads')
        train(args.protocol, args.data, args.output, args.device, args.threads)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
