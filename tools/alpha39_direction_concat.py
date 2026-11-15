"""Training and evaluation of the frozen 119-D linear direction head only."""
import copy
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from tools.run_direction_bypass import features
from tools.ten_year_train import sha, write_json
SEEDS = (20261002, 20261003, 20261004)
GROUPS = {'concat119': ('tail', 'raw')}

def network(dimensions, kind):
    if kind != 'linear':
        raise ValueError('Only the frozen linear direction head is distributed')
    model = nn.Linear(dimensions, 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    return model

def matrix(cache, group):
    return np.concatenate([cache[key] for key in GROUPS[group]], axis=-1)

def event_weights(target, coordinates):
    selected = np.asarray(target) > 0
    if not selected.any():
        raise ValueError('Empty true-follow direction supervision')
    _, event, counts = np.unique(np.asarray(coordinates)[selected, :2], axis=0, return_inverse=True, return_counts=True)
    return (1.0 / counts[event]).astype(np.float32)

def scaler(values):
    values = np.asarray(values, dtype=np.float64)
    mean, scale = (values.mean(0), values.std(0))
    scale[scale < 1e-08] = 1.0
    return (mean, scale)

def standardize(values, mean, scale):
    result = ((values - mean) / scale).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite standardized features')
    return result

def direction_loss(labels, probability, weights):
    labels = np.asarray(labels, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    losses = -labels * np.log(probability) - (1 - labels) * np.log1p(-probability)
    return float(np.average(losses, weights=weights))

def fit(train_x, train_y, train_w, valid_x, valid_y, valid_w, seed, kind, destination, *, max_epochs=40, patience=7):
    if kind != 'linear':
        raise ValueError('Only the frozen linear direction head is distributed')
    torch.manual_seed(seed)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    mean, scale = scaler(train_x)
    np.savez_compressed(destination / 'scaler.npz', mean=mean, scale=scale)
    train_x, valid_x = (standardize(x, mean, scale) for x in (train_x, valid_x))
    started = time.perf_counter()
    model = network(train_x.shape[1], kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05)
    x, y, weights = map(torch.from_numpy, (train_x, train_y.astype(np.float32), train_w))
    vx, vy, vw = map(torch.from_numpy, (valid_x, valid_y.astype(np.float32), valid_w))

    def evaluate():
        model.eval()
        with torch.inference_mode():
            losses = nn.functional.binary_cross_entropy_with_logits(model(vx).squeeze(-1), vy, reduction='none')
            return float((losses * vw).sum() / vw.sum())
    best, best_epoch, state = (evaluate(), 0, copy.deepcopy(model.state_dict()))
    history = [dict(epoch=0, validation_direction_BCE=best)]
    steps = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for indices in torch.randperm(len(y)).split(2048):
            losses = nn.functional.binary_cross_entropy_with_logits(model(x[indices]).squeeze(-1), y[indices], reduction='none')
            loss = (losses * weights[indices]).mean() / weights.mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite direction training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
        current = evaluate()
        if not np.isfinite(current):
            raise ValueError('Nonfinite validation direction loss')
        history.append(dict(epoch=epoch, validation_direction_BCE=current))
        if current < best:
            best, best_epoch, state = (current, epoch, copy.deepcopy(model.state_dict()))
        print('DIRECTION_EPOCH', destination.name, epoch, current, 'best', best_epoch, flush=True)
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(state)
    if evaluate() != best:
        raise ValueError('Selected direction checkpoint does not reproduce')
    model.eval().requires_grad_(False)
    torch.save(state, destination / 'best.pt')
    write_json(destination / 'history.json', history)
    result = dict(kind=kind, best_epoch=best_epoch, epochs=epoch, optimizer_steps=steps, selection='validation_source_event_direction_BCE_earliest_strict_minimum_including_epoch_zero', validation_direction_BCE=best, parameters=sum((p.numel() for p in model.parameters())), checkpoint_sha256=sha(destination / 'best.pt'))
    with np.load(destination / 'scaler.npz', allow_pickle=False) as archive:
        mean, scale = (archive['mean'].copy(), archive['scale'].copy())
    model = network(train_x.shape[1], kind)
    model.load_state_dict(torch.load(destination / 'best.pt', map_location='cpu', weights_only=True))
    model.eval().requires_grad_(False)
    if evaluate() != best:
        raise ValueError('Disk checkpoint validation loss does not reproduce')
    result.update(seed=seed, features=train_x.shape[1], seconds=time.perf_counter() - started, disk_checkpoint_reloaded=True)
    write_json(destination / 'training_completed.json', result)
    return (dict(model=model, mean=mean, scale=scale, kind=kind), result)

def predict(fitted, values):
    if not len(values):
        return np.empty(0, dtype=np.float32)
    values = standardize(values, fitted['mean'], fitted['scale'])
    chunks = []
    for start in range(0, len(values), 65536):
        batch = values[start:start + 65536]
        with torch.inference_mode():
            chunks.append(fitted['model'](torch.from_numpy(batch)).squeeze(-1).sigmoid().numpy())
    return np.concatenate(chunks)

def metrics(cache, direction):
    from tools.run_follow_binary_experiment import binary_classification
    from tools.train_gm_baseline import classification_report
    target, q = (cache['target'], cache['q'])
    routed = np.where(q > 0.5, direction, 0.5)
    result = dict(three_class=classification_report(target, q, routed), follow=binary_classification(target, q), directions={}, probability_policy='q_negative_d=0.5; ungated_direction_only_for_conditional_diagnostics')
    for name, selected in (('true_follow', target > 0), ('alerted_true_follow', (target > 0) & (q > 0.5))):
        if not selected.any():
            result['directions'][name] = dict(rows=0, reason='empty_subset')
            continue
        report = binary_classification((target[selected] == 2).astype(np.int8), direction[selected])
        report['class_order'] = ['opposite', 'same']
        weights = event_weights(target[selected], cache['coordinates'][selected])
        report['source_event_BCE'] = direction_loss(target[selected] == 2, direction[selected], weights)
        report['source_events'] = int(np.unique(cache['coordinates'][selected, :2], axis=0).shape[0])
        sign = cache['raw'][selected, 0]
        up = np.where(sign > 0, direction[selected], 1 - direction[selected])
        labels_up = np.where(sign > 0, target[selected] == 2, target[selected] == 1).astype(np.int8)
        converted = np.where(sign > 0, direction[selected] > 0.5, direction[selected] <= 0.5)
        absolute = binary_classification(labels_up, up)
        hard = binary_classification(labels_up, converted.astype(float))
        probability_fields = {'follow_AP', 'ROC_AUC', 'log_loss', 'binary_Brier', 'ECE', 'calibration_bins'}
        absolute.update({key: value for key, value in hard.items() if key not in probability_fields})
        confidence = np.where(converted, up, 1 - up)
        bins = np.minimum((confidence * 10).astype(int), 9)
        calibration, ece = ([], 0.0)
        for index in range(10):
            included = bins == index
            count = int(included.sum())
            accuracy = float((converted[included] == labels_up[included]).mean()) if count else None
            average = float(confidence[included].mean()) if count else None
            if count:
                ece += count / len(up) * abs(accuracy - average)
            calibration.append(dict(bin=index, rows=count, accuracy=accuracy, confidence=average))
        absolute.update(class_order=['down', 'up'], ECE=float(ece), calibration_bins=calibration, decision_policy='convert_relative_S_if_d>0.5_else_O_using_source_sign; negative_source_tie_is_UP')
        report['absolute_up_down'] = absolute
        result['directions'][name] = report
    return result
