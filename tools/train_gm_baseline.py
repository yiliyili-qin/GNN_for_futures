"""GraphMixer task transfer for frozen alpha39 or audited ten-year prepared data.

Default: preflight only. --smoke runs two supervised train samples; --execute
trains from scratch with validation-only checkpoint selection. Never rewrites
raw data, existing panels, the frozen protocol or legacy experiment outputs.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tools.ten_year_train import FROZEN, classification, preflight, sha, split_boundaries, write_json
from graph_model.alpha_backtest import chronological_split_bounds, split_masks
from gnn_for_futures.shock_follow.data.gm_baseline import CausalEventHistory, GMBaselineDataset
from gnn_for_futures.shock_follow.data.labels import build_shock_follow_labels_for_cutoff
from gnn_for_futures.shock_follow.data.panel import load_shock_follow_bar_panel
from gnn_for_futures.shock_follow.data.prepared import (
    NODE_FEATURE_SCHEMA, eligible_cutoff_indices, prepare_shock_follow_data,
)
from gnn_for_futures.shock_follow.data.thresholds import materialize_causal_monthly_thresholds, save_monthly_threshold_table
from gnn_for_futures.shock_follow.models.gm_baseline import GraphMixerBaseline
from gnn_for_futures.shock_follow.models.head_v2 import (
    balanced_follow_class_weights, hierarchical_source_event_loss,
)


INPUT_FIELDS = ("node_features", "edge_features", "edge_ages", "edge_mask", "adjacency")


def load_alpha39_inputs(protocol_path, data_dir, node_input, max_events, structure_events, device):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema"] != "alpha39_unified_evaluation" or protocol["version"] != "1.0.0":
        raise ValueError("Expected frozen alpha39 protocol v1.0.0")
    for key, expected in dict(FROZEN, direction="maximize", min_delta=0.,
                              validation_frequency_epochs=1, scheduler="none", gradient_clip=None,
                              max_candidates_per_family=1, test_tuning=False).items():
        if protocol["selection"][key] != expected:
            raise ValueError(f"Unsupported frozen selection setting: {key}")
    decision = protocol["classification"]
    if (decision["follow_threshold"], decision["direction_threshold"], decision["follow_operator"],
        decision["direction_operator"], decision["joint_argmax"], decision["class_order"]) != (
            .5, .5, ">", ">", False, ["N", "O", "S"]):
        raise ValueError("Unsupported frozen classification decision")
    for name, expected in protocol["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen source SHA256 mismatch: {name}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    data = protocol["data"]
    panel_path = data_dir / "bar_panel.npz"
    manifest_path = data_dir / "bar_panel_manifest.json"
    panel = load_shock_follow_bar_panel(panel_path, manifest_path)
    if panel.materialization_sha256 != data["panel_identity"]:
        raise ValueError("Frozen alpha39 panel identity mismatch")
    if list(panel.node_order) != data["node_order"] or len(panel.node_order) != data["node_count"]:
        raise ValueError("Frozen alpha39 node order mismatch")
    bounds = chronological_split_bounds(panel.trading_dates[panel.row_observed.all(axis=1)])
    if {name: list(value) for name, value in bounds.items()} != data["splits"]:
        raise ValueError("Reconstructed alpha39 split boundaries differ from frozen protocol")
    if (data["sequence_bars"], data["bar_minutes"], data["label_horizon_minutes"]) != (40, 15, 15):
        raise ValueError("Unsupported alpha39 history or horizon")
    if data["thresholds"]["fit_returns"] != "train_only":
        raise ValueError("Expected 15m bars and train-only threshold fitting")
    lower, upper = map(np.datetime64, bounds["train"])
    fit = ((panel.timestamps >= lower) & (panel.timestamps < upper)
           & (panel.trading_dates >= lower) & (panel.trading_dates < upper))
    thresholds = materialize_causal_monthly_thresholds(
        panel.timestamps, panel.trading_dates,
        np.where(fit[:, None], panel.signed_returns, np.nan), panel.node_order,
    )
    prepared = prepare_shock_follow_data(panel, thresholds, sequence_length=data["sequence_bars"])
    cutoffs = eligible_cutoff_indices(panel, prepared.input_valid, common_cutoffs=prepared.common_cutoffs)
    histories = prepared.common_histories[np.searchsorted(prepared.common_cutoffs, cutoffs)]
    masks = split_masks(panel.timestamps[cutoffs], panel.timestamps[cutoffs] + np.timedelta64(15, "m"),
                        panel.trading_dates[cutoffs], bounds["test"][1], bounds=bounds)
    for previous, current in (("train", "validation"), ("validation", "test")):
        end = np.datetime64(bounds[previous][1])
        masks[current] &= ((panel.timestamps[histories[:, 0]] > end)
                           & (panel.trading_dates[histories[:, 0]] > end))
    events = CausalEventHistory(panel, prepared)
    datasets, row_keys, counts = {}, {}, {}
    for position, split in enumerate(("train", "validation", "test")):
        selected, history = cutoffs[masks[split]], histories[masks[split]]
        if len(selected) != data["candidate_samples_documented"][split]:
            raise ValueError(f"Frozen candidate clock mismatch: {split}")
        target = np.full((len(selected), len(panel.node_order), len(panel.node_order)), -1, dtype=np.int8)
        mask = np.zeros(target.shape, dtype=bool)
        for item, cutoff in enumerate(selected):
            labels = build_shock_follow_labels_for_cutoff(
                panel.timestamps, panel.trading_dates, panel.signed_returns, panel.node_order,
                prepared.threshold_matrix[cutoff], panel.timestamps[cutoff],
                source_input_valid=prepared.input_valid, target_input_valid=prepared.input_valid,
                validate_timestamps=False,
            )
            target[item], mask[item] = labels.target_class, labels.loss_mask
        previous = ("train", "train", "validation")[position]
        nominal_start = bounds["train"][0] if position == 0 else bounds[previous][1]
        datasets[split] = GMBaselineDataset(
            events, selected, history, target, mask, nominal_start=nominal_start,
            node_input=node_input, max_events=max_events, structure_events=structure_events,
        )
        row_keys[split] = np.asarray([
            f"{split}:{np.datetime_as_string(panel.timestamps[selected[item]], unit='s')}:{panel.node_order[source]}:{panel.node_order[target_node]}"
            for item, source, target_node in np.argwhere(mask)
        ])
        counts[split] = dict(candidates=len(selected), supervised_cutoffs=int(mask.any(axis=(1, 2)).sum()),
                             rows=int(mask.sum()), class_counts=np.bincount(target[mask], minlength=3).tolist())
    report = dict(
        status="ready", scope="GM training/classification only; not full backtest admission",
        protocol_sha256=sha(protocol_path), data_domain=protocol["schema"], models=["gm"],
        panel_path=str(panel_path.resolve()), panel_sha256=sha(panel_path),
        panel_manifest_sha256=sha(manifest_path), panel_materialization_sha256=panel.materialization_sha256,
        threshold_materialization_sha256=thresholds.materialization_sha256,
        nodes=list(panel.node_order), boundaries=bounds, samples=counts,
        realized_edges=len(events.sources), device=device,
        strict_pit=protocol["strict_pit"], fresh_oos=protocol["fresh_oos"],
        previously_inspected_test=protocol["previously_inspected_test"],
    )
    return datasets, row_keys, report


def load_inputs(protocol_path, data_path, node_input, max_events, structure_events, device, *, data_dir=None):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") == "alpha39_unified_evaluation":
        if data_path is not None:
            raise ValueError("Use --data-dir for alpha39, not ten-year --data arrays")
        directory = data_dir if data_dir is not None else ROOT / protocol["data"]["panel_path"]
        return load_alpha39_inputs(protocol_path, directory, node_input, max_events, structure_events, device)
    if data_dir is not None or data_path is None:
        raise ValueError("Prepared-data protocols require --data training_arrays.npz")
    return load_prepared_inputs(protocol_path, data_path, node_input, max_events, structure_events, device)


def load_prepared_inputs(protocol_path, data_path, node_input, max_events, structure_events, device):
    # Preserve existing admission, hash, embargo and (for v3) merge-audit gates.
    report = preflight(protocol_path, data_path, device)
    if report["status"] != "ready":
        raise ValueError("Prepared-data preflight blocked: " + "; ".join(report["blockers"]))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    adapter = protocol["training_adapter"]
    panel_path = protocol_path.parent / protocol["data"]["panel_path"]
    if sha(panel_path) != protocol["data"]["panel_sha256"]:
        raise ValueError("Panel SHA256 mismatch")
    panel = load_shock_follow_bar_panel(panel_path, panel_path.with_name("bar_panel_manifest.json"))
    if list(panel.node_order) != adapter["node_order"]:
        raise ValueError("Panel and prepared node orders differ")
    edges = [edge.to_datetime64() for edge in split_boundaries(protocol)]
    fit = ((panel.timestamps >= edges[0]) & (panel.timestamps < edges[1])
           & (panel.trading_dates >= edges[0]) & (panel.trading_dates < edges[1]))
    thresholds = materialize_causal_monthly_thresholds(
        panel.timestamps, panel.trading_dates,
        np.where(fit[:, None], panel.signed_returns, np.nan), panel.node_order,
    )
    with np.load(data_path.parent / "thresholds.npz", allow_pickle=False) as stored:
        if not np.array_equal(stored["months"], thresholds.months) or not np.allclose(
            stored["values"], thresholds.thresholds, rtol=0, atol=0, equal_nan=True
        ) or not np.array_equal(stored["fit_ends"].astype(str), thresholds.fit_ends.astype(str)):
            raise ValueError("Stored thresholds differ from train-only causal reconstruction")
    prepared = prepare_shock_follow_data(panel, thresholds, sequence_length=adapter["sequence_bars"])
    events = CausalEventHistory(panel, prepared)
    datasets, row_keys = {}, {}
    with np.load(data_path, allow_pickle=False) as arrays:
        for position, split in enumerate(("train", "validation", "test")):
            cutoffs, histories, target, mask = (
                arrays[f"{split}_{field}"] for field in
                ("cutoffs", "history_indices", "target_class", "loss_mask")
            )
            if ((panel.timestamps[cutoffs] < edges[position]).any()
                or (panel.timestamps[cutoffs] + np.timedelta64(15, "m") >= edges[position + 1]).any()
                or (panel.trading_dates[cutoffs] < edges[position]).any()
                or (panel.trading_dates[cutoffs] >= edges[position + 1]).any()):
                raise ValueError(f"Prepared cutoffs cross {split} boundaries")
            # Check the entire existing information window, not only its last bar.
            stored_history = arrays[f"{split}_node_history"]
            for start in range(0, len(cutoffs), 128):
                expected = prepared.node_features[histories[start:start + 128]].transpose(0, 2, 1, 3)
                if not np.array_equal(expected, stored_history[start:start + 128]):
                    raise ValueError(f"Node features/history identity mismatch: {split}")
            del stored_history
            datasets[split] = GMBaselineDataset(
                events, cutoffs, histories, target, mask,
                nominal_start=str(edges[position]), node_input=node_input,
                max_events=max_events, structure_events=structure_events,
            )
            # Keys are regenerated only after masks/labels were exactly replayed.
            row_keys[split] = np.asarray([
                f"{split}:{cutoffs[item]}:{panel.node_order[source]}:{panel.node_order[target_node]}"
                for item, source, target_node in np.argwhere(mask)
            ])
            if not np.array_equal(row_keys[split], arrays[f"{split}_row_keys"][mask]):
                raise ValueError(f"Existing row-key alignment mismatch: {split}")
    report.update(
        panel_sha256=sha(panel_path), panel_manifest_sha256=sha(panel_path.with_name("bar_panel_manifest.json")),
        thresholds_sha256=sha(data_path.parent / "thresholds.npz"),
        threshold_materialization_sha256=thresholds.materialization_sha256,
        data_domain=protocol.get("schema"),
        nominal_boundaries=[str(edge) for edge in edges],
        nodes=list(panel.node_order), realized_edges=len(events.sources),
        prepared_adapter_models=report["models"], models=["gm"],
        scope="GM training/classification only; not full backtest admission",
    )
    return datasets, row_keys, report


def run_epoch(model, dataset, device, weights, batch_size, optimizer=None, *, collect_all=False):
    model.train(optimizer is not None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=optimizer is not None)
    rows = {name: [] for name in ("q", "d", "target", "sample_index", "source", "target_node")}
    dense = {name: [] for name in ("dense_q", "dense_d", "dense_sample_index")} if collect_all else {}
    steps = 0
    for batch in loader:
        inputs = {field: batch[field].to(device) for field in INPUT_FIELDS}
        target, mask = batch["target_class"].to(device), batch["loss_mask"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(**inputs)
            if not torch.isfinite(output.follow_logits).all() or not torch.isfinite(output.direction_logits).all():
                raise ValueError("Nonfinite GM logits; no model-specific row dropping allowed")
            if optimizer is not None:
                loss = hierarchical_source_event_loss(
                    output.follow_logits, output.direction_logits, target, mask, weights,
                ).total_loss
                optimizer.zero_grad(set_to_none=True)
                if loss is not None:
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite GM training loss")
                    loss.backward()
                    optimizer.step()
                    steps += 1
        q = output.follow_logits.softmax(-1)[..., 1].detach().cpu().numpy()
        d = output.direction_logits.softmax(-1)[..., 1].detach().cpu().numpy()
        rows["q"].append(q[batch["loss_mask"].numpy()])
        rows["d"].append(d[batch["loss_mask"].numpy()])
        if collect_all:
            dense["dense_q"].append(q)
            dense["dense_d"].append(d)
            dense["dense_sample_index"].append(batch["sample_index"].numpy())
        rows["target"].append(target[mask].cpu().numpy())
        batch_item, source, target_node = np.nonzero(batch["loss_mask"].numpy())
        rows["sample_index"].append(batch["sample_index"].numpy()[batch_item])
        rows["source"].append(source)
        rows["target_node"].append(target_node)
    rows = {name: np.concatenate(values) for name, values in rows.items()}
    rows.update({name: np.concatenate(values) for name, values in dense.items()})
    return classification(rows["target"], rows["q"], rows["d"]), rows, steps


def classification_report(target, q, d):
    """Frozen classification diagnostics; no threshold fitting or negative sampling."""
    result = classification(target, q, d)
    target = np.asarray(target, dtype=np.int64)
    q, d = np.asarray(q, dtype=np.float64), np.asarray(d, dtype=np.float64)
    confusion = np.asarray(result["confusion"])
    support, predicted_support = confusion.sum(1), confusion.sum(0)
    precision = np.divide(confusion.diagonal(), predicted_support, out=np.zeros(3), where=predicted_support > 0)
    recall = np.divide(confusion.diagonal(), support, out=np.zeros(3), where=support > 0)
    joint = np.stack((1 - q, q * (1 - d), q * d), axis=-1)
    predicted = np.where(q > .5, np.where(d > .5, 2, 1), 0)

    def average_precision(binary, scores):
        positives = int(binary.sum())
        if positives == 0 or positives == len(binary):
            return dict(value=None, reason="no_positive" if positives == 0 else "no_negative")
        order = np.argsort(-scores, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(order) - 1]
        cumulative = np.cumsum(binary[order])[ends]
        value = np.sum(np.diff(np.r_[0, cumulative]) * cumulative / (ends + 1)) / positives
        return dict(value=float(value), reason=None)

    follow_ap = average_precision(target != 0, q)
    per_class_ap = [average_precision(target == index, joint[:, index]) for index in range(3)]
    clipped = np.clip(joint, 1e-12, 1 - 1e-12)
    clipped /= clipped.sum(1, keepdims=True)
    confidence = joint[np.arange(len(target)), predicted]
    bins = np.minimum((confidence * 10).astype(int), 9)
    correct = predicted == target
    calibration, ece = [], 0.
    for index in range(10):
        mask = bins == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(target) * abs(accuracy - mean_confidence)
        calibration.append(dict(bin=index, rows=count, accuracy=accuracy, confidence=mean_confidence))
    result.update(
        per_class_precision=precision.tolist(), per_class_recall=recall.tolist(), support=support.tolist(),
        balanced_accuracy=float(recall[support > 0].mean()), supported_class_count=int((support > 0).sum()),
        confusion_row_normalized=np.divide(confusion, support[:, None], out=np.zeros((3, 3)),
                                           where=support[:, None] > 0).tolist(),
        follow_AP=follow_ap, per_class_AP=per_class_ap,
        OVR_macro_AP=dict(value=float(np.mean([item["value"] for item in per_class_ap])), reason=None)
        if all(item["value"] is not None for item in per_class_ap)
        else dict(value=None, reason="undefined_class_AP"),
        log_loss=float(-np.log(clipped[np.arange(len(target)), target]).mean()),
        multiclass_Brier=float(np.square(joint - np.eye(3)[target]).sum(1).mean()),
        ECE=float(ece), calibration_bins=calibration, accuracy=float(correct.mean()),
    )
    return result


def train(args, datasets, keys, report, model_kwargs, weights):
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.joinpath("protocol_snapshot.json").write_bytes(args.protocol.read_bytes())
    args.output.joinpath("protocol_sha256").write_text(sha(args.protocol) + "\n", encoding="utf-8")
    if report.get("data_domain") == "alpha39_unified_evaluation":
        panel = datasets["train"].events.panel
        bounds = report["boundaries"]["train"]
        lower, upper = map(np.datetime64, bounds)
        fit = ((panel.timestamps >= lower) & (panel.timestamps < upper)
               & (panel.trading_dates >= lower) & (panel.trading_dates < upper))
        thresholds = materialize_causal_monthly_thresholds(
            panel.timestamps, panel.trading_dates,
            np.where(fit[:, None], panel.signed_returns, np.nan), panel.node_order,
        )
        if thresholds.materialization_sha256 != report["threshold_materialization_sha256"]:
            raise ValueError("Threshold identity changed before training")
        save_monthly_threshold_table(thresholds, args.output / "thresholds_with_fit_provenance.npz",
                                     args.output / "thresholds.csv", args.output / "thresholds_manifest.json")
    write_json(args.output / "gm_contract.json", dict(
        backbone="GraphMixer", task="hierarchical N/O/S task transfer; not paper experiment reproduction",
        node_input=args.node_input, model_kwargs=model_kwargs, selection=FROZEN,
        edge_rule="all realized O/S; source-time attributes; availability at target close",
        history_rule="only events wholly inside each existing node-history window",
        structure_rule=f"last {args.structure_events} global realized edges; undirected unique neighbors",
        time_unit="15 wall-clock minutes since realization", node_feature_schema=NODE_FEATURE_SCHEMA,
        preflight=report, strict_point_in_time=False,
    ))
    write_json(args.output / "data_manifest.json", report)
    write_json(args.output / "candidate_registry.json", dict(
        candidate_id="gm_identity" if args.node_input == "identity" else "gm_current",
        model_kwargs=model_kwargs, initialization="from_scratch_each_seed", seeds=FROZEN["seeds"],
        loss="train-balanced Follow CE plus unweighted true-follow Direction CE; source-event mean",
        follow_class_weights=weights.detach().cpu().tolist(),
        information_set="existing 40-bar span, realized O/S events, no future outcomes",
    ))
    write_json(args.output / "split_and_mask_hashes.json", {
        split: {field: hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
                for field, values in dict(cutoffs=dataset.cutoffs, history=dataset.histories,
                                          target=dataset.targets, mask=dataset.masks, row_keys=keys[split]).items()}
        for split, dataset in datasets.items()
    })
    source_paths = [Path(__file__), ROOT / "tools/ten_year_train.py", ROOT / "graph_model/alpha_backtest.py"] + [
        ROOT / f"src/gnn_for_futures/shock_follow/{name}.py" for name in (
            "data/gm_baseline", "data/panel", "data/prepared", "data/thresholds", "data/labels",
            "models/gm_baseline", "models/head_v2", "contract",
        )
    ]
    write_json(args.output / "source_hashes.json", {path.relative_to(ROOT).as_posix(): sha(path) for path in source_paths})
    write_json(args.output / "environment.json", dict(torch=torch.__version__, numpy=np.__version__,
               python=sys.version, device=args.device, threads=args.threads))
    summaries = []
    for seed in FROZEN["seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = GraphMixerBaseline(**model_kwargs).to(args.device)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.Adam(model.parameters(), lr=FROZEN["learning_rate"], weight_decay=FROZEN["weight_decay"])
        destination = args.output / f"gm_{seed}"
        destination.mkdir()
        best, best_epoch, history, steps, checkpoint_hash = -1., 0, [], 0, None
        started = time.perf_counter()
        for number in range(1, FROZEN["max_epochs"] + 1):
            training, _, updates = run_epoch(model, datasets["train"], args.device, weights, FROZEN["batch_size"], optimizer)
            validation, _, _ = run_epoch(model, datasets["validation"], args.device, weights, FROZEN["batch_size"])
            steps += updates
            history.append(dict(epoch=number, train=training, validation=validation))
            write_json(destination / "history.json", history)
            with (destination / "all_epoch_validation_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("epoch", "macro_f1", "rows"))
                writer.writeheader()
                writer.writerows(dict(epoch=item["epoch"], macro_f1=item["validation"]["macro_f1"],
                                      rows=item["validation"]["rows"]) for item in history)
            if validation["macro_f1"] > best:
                best, best_epoch = validation["macro_f1"], number
                torch.save(model.state_dict(), destination / "best.pt")
                checkpoint_hash = sha(destination / "best.pt")
                write_json(destination / "checkpoint_hashes.json", dict(best_epoch=best_epoch, sha256=checkpoint_hash))
            print(f"GM seed={seed} epoch={number} validation_macro_f1={validation['macro_f1']:.8f} best={best_epoch}", flush=True)
            if number - best_epoch >= FROZEN["patience"]:
                break
        if sha(destination / "best.pt") != checkpoint_hash:
            raise ValueError("Best checkpoint changed before inference")
        model.load_state_dict(torch.load(destination / "best.pt", weights_only=True, map_location=args.device))
        model.requires_grad_(False)
        prediction_hashes = {}
        for split in ("validation", "test"):
            metrics, rows, _ = run_epoch(model, datasets[split], args.device, weights, FROZEN["batch_size"], collect_all=True)
            dataset = datasets[split]
            coordinates = np.argwhere(dataset.masks)
            if not np.array_equal(rows["target"], dataset.targets[dataset.masks]) or not np.array_equal(
                np.column_stack((rows["sample_index"], rows["source"], rows["target_node"])), coordinates
            ) or len(keys[split]) != len(coordinates):
                raise ValueError("Export no longer aligns with frozen supervised row order")
            if not np.array_equal(rows["dense_sample_index"], np.arange(len(dataset))):
                raise ValueError("Dense inference cutoff clock changed")
            np.savez_compressed(destination / f"{split}_predictions.npz", **rows, row_keys=keys[split], seed=seed,
                                cutoff_timestamp=dataset.events.panel.timestamps[dataset.cutoffs[rows["sample_index"]]],
                                dense_cutoffs=dataset.cutoffs, dense_history_indices=dataset.histories,
                                node_order=np.asarray(dataset.events.panel.node_order))
            prediction_hashes[split] = sha(destination / f"{split}_predictions.npz")
            metrics = classification_report(rows["target"], rows["q"], rows["d"])
            write_json(destination / f"{split}_classification.json", metrics)
        summary = dict(seed=seed, best_epoch=best_epoch, epochs=number, optimizer_steps=steps,
                       validation_macro_f1=best, test_macro_f1=metrics["macro_f1"], seconds=time.perf_counter() - started,
                       parameters=sum(parameter.numel() for parameter in model.parameters()),
                       device=args.device, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated()
                       if args.device == "cuda" else None, checkpoint_sha256=sha(destination / "best.pt"),
                       prediction_sha256=prediction_hashes)
        summaries.append(summary)
        write_json(destination / "training_completed.json", summary)
    aggregate = {
        metric: dict(mean=float(np.mean([item[metric] for item in summaries])),
                     std=float(np.std([item[metric] for item in summaries], ddof=1)) if len(summaries) > 1 else None)
        for metric in ("validation_macro_f1", "test_macro_f1")
    }
    write_json(args.output / "training_completed.json", dict(
        status="classification_only", trials=summaries, across_seeds=aggregate,
        std_definition="sample, ddof=1; null for a single seed; no seed selection", backtest_performed=False,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "frozen-base.json",
                        help="Frozen alpha39 protocol (default), or audited prepared-data protocol")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-dir", type=Path, help="Existing alpha39 bar_panel.npz/manifest directory")
    source.add_argument("--data", type=Path, help="Matching training_arrays.npz for ten-year protocols only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--node-input", choices=("identity", "current"), default="identity")
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--structure-events", type=int, default=55)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--time-dim", type=int, default=100)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=.1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or min(args.max_events, args.structure_events, args.hidden_dim, args.time_dim, args.layers) < 1:
        parser.error("threads, dimensions and capacities must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0,1)")
    if args.execute and (args.output is None or args.output.exists()):
        parser.error("--execute requires a new --output directory")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(FROZEN["seeds"][0])
    try:
        datasets, keys, report = load_inputs(args.protocol, args.data, args.node_input, args.max_events,
                                             args.structure_events, args.device, data_dir=args.data_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="blocked", reason=str(error)), indent=2), flush=True)
        return 2
    print(json.dumps(report, indent=2), flush=True)
    node_dim = len(report["nodes"]) if args.node_input == "identity" else len(NODE_FEATURE_SCHEMA)
    model_kwargs = dict(node_feature_dim=node_dim, edge_feature_dim=2 * len(NODE_FEATURE_SCHEMA),
                        max_events=args.max_events, time_dim=args.time_dim, hidden_dim=args.hidden_dim,
                        num_layers=args.layers, dropout=args.dropout)
    counts = np.bincount(datasets["train"].targets[datasets["train"].masks], minlength=3)
    weights = torch.tensor(balanced_follow_class_weights(counts.tolist()), device=args.device)
    if args.smoke:
        model = GraphMixerBaseline(**model_kwargs).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=FROZEN["learning_rate"], weight_decay=FROZEN["weight_decay"])
        supervised = np.flatnonzero(datasets["train"].masks.any(axis=(1, 2)))[:2]
        metrics, _, steps = run_epoch(model, Subset(datasets["train"], supervised.tolist()),
                                      args.device, weights, batch_size=2, optimizer=optimizer)
        print(json.dumps(dict(status="real_train_sample_smoke_only", epochs=0,
                              optimizer_steps=steps, rows=metrics["rows"], model_kwargs=model_kwargs)), flush=True)
    elif args.execute:
        train(args, datasets, keys, report, model_kwargs, weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
