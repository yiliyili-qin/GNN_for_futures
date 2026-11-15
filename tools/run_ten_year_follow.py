"""Frozen ten-year binary-primary c=1/3 comparison; own models retain trained direction.

Default is audited preflight. New output directories only. --execute trains 18
scratch trials; no test/backtest selection, raw edits, or legacy checkpoints.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from graph_model.daily_graph_dataset import DailyGraphDataset
from graph_model.daily_return_regression import fit_daily_graph
from gnn_for_futures.shock_follow.models.head_v2 import balanced_follow_class_weights, hierarchical_source_event_loss
from tools import run_follow_binary_experiment as binary
from tools.backtest_follow_binary import evaluate_case
from tools.backtest_gm_baseline import check_hash, load_quotes, shared_schedule
from tools.ten_year_train import FROZEN, classification, sha, write_json
from tools.train_gm_baseline import INPUT_FIELDS, load_prepared_inputs

FAMILIES = ("full", "HH", "gm")
SPLITS = ("train", "validation", "test")


def metrics(target, q):
    result = binary.binary_classification(target, q)
    tn, fp, fn, tp = np.asarray(result["confusion"]).ravel().tolist()
    result.update(false_positives=fp, false_negatives=fn, true_positives=tp, true_negatives=tn,
                  false_positive_rate=fp / (tn + fp) if tn + fp else None,
                  false_discovery_rate=fp / (tp + fp) if tp + fp else None)
    return result


def model_kwargs(family, nodes):
    kwargs = binary.model_kwargs(family, nodes)
    kwargs["follow_only"] = family == "gm"
    return kwargs


def make_model(family, nodes):
    constructor = binary.GraphMixerBaseline if family == "gm" else binary.ShockFollowGraphMixerV2
    return constructor(**model_kwargs(family, nodes))


def run_epoch(model, dataset, family, device, weights, optimizer=None, *, collect_dense=False):
    model.train(optimizer is not None)
    names = ["q", "target_original", "sample_index", "source", "target_node"]
    if family != "gm":
        names.append("d")
    outputs, dense, steps = {key: [] for key in names}, {}, 0
    fields = INPUT_FIELDS if family == "gm" else binary.OWN_FIELDS
    for batch in DataLoader(dataset, batch_size=FROZEN["batch_size"], shuffle=optimizer is not None):
        inputs = {key: batch[key].to(device) for key in fields}
        target, mask = batch["target_class"].to(device), batch["loss_mask"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(**inputs)
            logits = output if family == "gm" else output.follow_logits
            direction = None if family == "gm" else output.direction_logits
            if not torch.isfinite(logits).all() or (direction is not None and not torch.isfinite(direction).all()):
                raise ValueError("Nonfinite logits, including unsupervised cells")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss = (binary.follow_loss(logits, target, mask, weights) if family == "gm" else
                        hierarchical_source_event_loss(logits, direction, target, mask, weights).total_loss)
                if loss is not None:
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite loss")
                    loss.backward()
                    optimizer.step()
                    steps += 1
        chosen = batch["loss_mask"].numpy()
        for key, values in (("q", logits), ("d", direction)):
            if values is not None:
                probability = values.softmax(-1)[..., 1].detach().cpu().numpy()
                outputs[key].append(probability[chosen])
                if collect_dense:
                    dense.setdefault("dense_" + key, []).append(probability)
        item, source, target_node = np.nonzero(chosen)
        for key, value in dict(target_original=batch["target_class"].numpy()[chosen],
                               sample_index=batch["sample_index"].numpy()[item], source=source,
                               target_node=target_node).items():
            outputs[key].append(value)
    rows = {key: np.concatenate(value) for key, value in outputs.items()}
    rows.update({key: np.concatenate(value) for key, value in dense.items()})
    return metrics(rows["target_original"], rows["q"]), rows, steps


def infer_schedule(model, dataset, family, schedule, device):
    model.eval()
    result = {}
    with torch.inference_mode():
        for start in range(0, len(schedule["cutoffs"]), 16):
            samples = []
            for cutoff, history in zip(schedule["cutoffs"][start:start + 16], schedule["histories"][start:start + 16]):
                if family == "gm":
                    sample = dataset.events.snapshot(int(cutoff), int(history[0]), 50, 55)
                    sample["node_features"] = dataset.identity
                else:
                    sample = dataset.inputs(int(cutoff), history)
                samples.append(sample)
            inputs = {key: torch.stack([torch.as_tensor(sample[key]) for sample in samples]).to(device) for key in samples[0]}
            output = model(**inputs)
            for key, logits in (("q", output if family == "gm" else output.follow_logits),
                                ("d", None if family == "gm" else output.direction_logits)):
                if logits is not None:
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite scheduled logits")
                    result.setdefault(key, []).append(logits.softmax(-1)[..., 1].cpu().numpy())
    return {key: np.concatenate(value) for key, value in result.items()}


def validate_auxiliary_inputs(args, protocol, parent, parent_path, panel):
    for name, directory in (("daily_returns.csv", args.data_dir), ("quote_manifest.json", args.quotes_dir)):
        check_hash(directory / name, protocol["auxiliary_input_sha256"][name])
    manifest = binary.read_json(args.quotes_dir / "quote_manifest.json")
    expected = dict(candidate_quote_sha256=parent["data"]["candidate_quote_panel"]["sha256"],
                    protocol_sha256=sha(parent_path), panel_sha256=parent["data"]["panel_sha256"],
                    panel_identity=panel.materialization_sha256, node_order=list(panel.node_order))
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Quote candidate/protocol/panel provenance mismatch")


def load_experiment(args):
    protocol = binary.read_json(args.protocol)
    if protocol != binary.read_json(ROOT / "frozen-ten-year-follow-20261101.json"):
        raise ValueError("Experiment differs from the preregistered contract")
    parent_path = ROOT / protocol["parent_protocol"]
    check_hash(parent_path, protocol["parent_sha256"])
    parent = binary.read_json(parent_path)
    if args.data_dir.resolve() != (parent_path.parent / parent["data"]["panel_path"]).parent.resolve():
        raise ValueError("Use the frozen parent's original data directory layout")
    arrays_path = args.data_dir / "training_arrays.npz"
    common, keys, report = load_prepared_inputs(parent_path, arrays_path, "identity", 50, 55, args.device)
    panel, prepared = common["train"].events.panel, common["train"].events.prepared
    validate_auxiliary_inputs(args, protocol, parent, parent_path, panel)
    verification = binary.read_json(args.data_dir / "verification.json")
    if (verification["status"] != "passed" or verification["protocol_sha256"] != sha(parent_path)
        or verification["arrays_sha256"] != sha(arrays_path)):
        raise ValueError("Independent original v2 verification required")
    sources = {str(path.resolve()): sha(path) for path in (
        args.protocol, parent_path, arrays_path, args.data_dir / "bar_panel.npz",
        args.data_dir / "bar_panel_manifest.json", args.data_dir / "thresholds.npz",
        args.data_dir / "audit.json", args.data_dir / "verification.json",
        args.data_dir / "economic_prior.npy", args.data_dir / "daily_returns.csv")}
    for relative, digest in verification["daily_graph_hashes"].items():
        path = args.data_dir / relative.replace("\\", "/")
        check_hash(path, digest)
        sources[str(path.resolve())] = digest
    # The ten-year policy uses common valid-return rows, not alpha39's observed-only grid.
    lower = np.datetime64(parent["data"]["window_start"])
    grid = np.flatnonzero(prepared.input_valid.all(1) & panel.row_observed.all(1)
                          & (panel.timestamps >= lower) & (panel.trading_dates >= lower))
    histories = np.lib.stride_tricks.sliding_window_view(grid, 40).copy()
    prepared = replace(prepared, common_cutoffs=histories[:, -1], common_histories=histories)
    common["train"].events.prepared = prepared
    for split, dataset in common.items():
        positions = np.searchsorted(prepared.common_cutoffs, dataset.cutoffs)
        if not np.array_equal(prepared.common_histories[positions], dataset.histories):
            raise ValueError("Original ten-year history policy changed")
    clock_protocol = dict(data=dict(splits={name: [spec["nominal_start"], spec["end_exclusive"]]
                                          for name, spec in parent["data"]["splits"].items()}))
    quotes, schedules = {}, {}
    for split in ("validation", "test"):
        quotes[split] = load_quotes(args.quotes_dir, split, panel, clock_protocol, sources)
        spec = parent["data"]["splits"][split]
        lower = np.datetime64(spec["nominal_start"])
        eligible_history = ((panel.timestamps[histories[:, 0]] > lower)
                            & (panel.trading_dates[histories[:, 0]] >= lower))
        split_prepared = replace(prepared, common_cutoffs=histories[eligible_history, -1],
                                 common_histories=histories[eligible_history])
        # shared_schedule uses strict date > prior-end; v2 permits history date == lower.
        # Precheck the exact v2 rule above, then relax only that helper's redundant gate.
        split_clock = dict(data=dict(splits={key: list(value) for key, value in clock_protocol["data"]["splits"].items()}))
        previous = "train" if split == "validation" else "validation"
        split_clock["data"]["splits"][previous][1] = str(lower.astype("datetime64[ns]") - np.timedelta64(1, "ns"))
        schedule = shared_schedule(panel, split_prepared, quotes[split], split_clock, split)
        # Preserve the ten-year effective embargo, including its union-date convention.
        chosen = ((schedule["signal_timestamp"] >= np.datetime64(spec["embargo_start"]))
                  & (schedule["trading_date"] >= np.datetime64(spec["embargo_start"]))
                  & (schedule["signal_timestamp"] + np.timedelta64(15, "m") < np.datetime64(spec["end_exclusive"])))
        for key in ("cutoffs", "histories", "execution_rows", "sigma", "signal_timestamp", "trading_date"):
            schedule[key] = schedule[key][chosen]
        if not len(schedule["cutoffs"]):
            raise ValueError("Empty auxiliary schedule")
        schedules[split] = schedule
    prior = np.load(args.data_dir / "economic_prior.npy", allow_pickle=False)
    original_reader = DailyGraphDataset(None, args.data_dir / "daily_valid_endpoints", panel.node_order, None)
    extra_reader = DailyGraphDataset(None, args.output / "schedule_graphs", panel.node_order, None)
    needed = np.unique(np.concatenate([panel.trading_dates[item.cutoffs] for item in common.values()]
                                     + [item["trading_date"] for item in schedules.values()]))
    # Materialize only missing dates from immutable daily returns; never read changed raw.
    daily = pd.read_csv(args.data_dir / "daily_returns.csv", index_col=0, parse_dates=True)
    if daily.columns.tolist() != list(panel.node_order):
        raise ValueError("Frozen daily-return node order mismatch")
    for day in needed:
        filename = f"{day}.csv"
        original = original_reader.graph_dir / filename
        if not original.is_file():
            extra_reader.graph_dir.mkdir(parents=True, exist_ok=True)
            path = extra_reader.graph_dir / filename
            fit_daily_graph(daily, str(day)).to_csv(path, index=False)
        else:
            sources[str(original.resolve())] = sha(original)
    reader = SimpleNamespace(graph_for_date=lambda day: (
        original_reader if (original_reader.graph_dir / f"{day}.csv").is_file() else extra_reader).graph_for_date(day))
    for day in needed:
        reader.graph_for_date(day)
    datasets = {family: {split: binary.FollowDataset(dataset, family, prior, reader)
                         for split, dataset in common.items()} for family in ("full", "HH")}
    datasets["gm"] = common
    # Exact feature/graph parity with the immutable prepared arrays, not just label parity.
    with np.load(arrays_path, allow_pickle=False) as archive:
        for split, dataset in common.items():
            arrays = {field: archive[f"{split}_{field}"] for field in (
                "node_history", "economic_prior", "prior_available", "regression_weight",
                "regression_available", "historical_weight", "historical_available")}
            for family in ("full", "HH"):
                own = datasets[family][split]
                graph_name = "regression" if family == "full" else "historical"
                for index in range(len(dataset)):
                    inputs = own.inputs(int(dataset.cutoffs[index]), dataset.histories[index])
                    for key, field in (("node_history", "node_history"), ("economic_prior", "economic_prior"),
                                       ("prior_available", "prior_available"),
                                       ("regression_weight", graph_name + "_weight"),
                                       ("regression_available", graph_name + "_available")):
                        if not np.array_equal(np.asarray(inputs[key]), arrays[field][index]):
                            raise ValueError(f"Frozen own input mismatch: {split}/{family}/{index}/{key}")
    report.update(scope="ten_year_binary_primary_own_dual_GM_binary", models=["full", "HH", "HF", "gm"],
                  binary_counts={split: np.bincount((item.targets[item.masks] > 0).astype(int), minlength=2).tolist()
                                 for split, item in common.items()},
                  schedules={split: len(item["cutoffs"]) for split, item in schedules.items()},
                  nominal_months=120, actual_supervision=parent["data"]["splits"], strict_pit=False, fresh_oos=False)
    return protocol, parent, datasets, keys, quotes, schedules, report, sources


def train_family(args, family, seed, coefficient, datasets, keys, weights):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model = make_model(family, len(datasets["train"].events.panel.node_order)).to(args.device)
    has_direction = any("direction_output" in name for name, _ in model.named_parameters())
    if has_direction != (family != "gm"):
        raise ValueError("Direction parameter contract violated")
    directory = args.output / f"c{int(coefficient)}" / "training" / f"{family}_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=FROZEN["learning_rate"], weight_decay=FROZEN["weight_decay"])
    best, best_epoch, history, steps = -1., 0, [], 0
    started = time.perf_counter()
    for epoch in range(1, FROZEN["max_epochs"] + 1):
        training, _, updates = run_epoch(model, datasets["train"], family, args.device, weights, optimizer)
        validation, _, _ = run_epoch(model, datasets["validation"], family, args.device, weights)
        steps += updates
        history.append(dict(epoch=epoch, train=training, validation=validation))
        write_json(directory / "history.json", history)
        if validation["macro_f1"] > best:
            best, best_epoch = validation["macro_f1"], epoch
            torch.save(model.state_dict(), directory / "best.pt")
        print(f"TEN_YEAR c={coefficient:g} family={family} seed={seed} epoch={epoch} binary_macro_f1={validation['macro_f1']:.8f} best={best_epoch}", flush=True)
        if epoch - best_epoch >= FROZEN["patience"]:
            break
    digest = sha(directory / "best.pt")
    write_json(directory / "checkpoint_hashes.json", dict(best_epoch=best_epoch, sha256=digest))
    model.load_state_dict(torch.load(directory / "best.pt", weights_only=True, map_location=args.device))
    prediction_hashes, scores = {}, {}
    for split in ("validation", "test"):
        scores[split], rows, _ = run_epoch(model, datasets[split], family, args.device, weights, collect_dense=True)
        path = directory / f"{split}_predictions.npz"
        binary.save_predictions(path, datasets[split], keys[split], rows, seed, family)
        prediction_hashes[split] = sha(path)
        write_json(directory / f"{split}_classification.json", scores[split])
        if family != "gm":
            write_json(directory / f"{split}_three_class.json", classification(rows["target_original"], rows["q"], rows["d"]))
    if scores["validation"]["macro_f1"] != best or max(history, key=lambda row: row["validation"]["macro_f1"])["epoch"] != best_epoch:
        raise ValueError("Reloaded checkpoint is not earliest validation binary maximum")
    trial = dict(family=family, seed=seed, negative_multiplier=coefficient, epochs=epoch, best_epoch=best_epoch,
                 optimizer_steps=steps, validation_macro_f1=best, test_macro_f1=scores["test"]["macro_f1"],
                 checkpoint_sha256=digest, prediction_sha256=prediction_hashes,
                 seconds=time.perf_counter() - started, parameters=sum(value.numel() for value in model.parameters()),
                 device=args.device, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                 peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
    write_json(directory / "training_completed.json", trial)
    print("TRIAL_COMPLETED", json.dumps(trial), flush=True)
    return trial


def replay(args, datasets, keys, quotes, schedules, trials):
    risk_rows = []
    for coefficient in (1, 3):
        base = args.output / f"c{coefficient}"
        for split, schedule in schedules.items():
            common = datasets["gm"][split]
            folder = base / "backtest" / split
            for seed in FROZEN["seeds"]:
                predictions = {}
                for family in FAMILIES:
                    source = base / "training" / f"{family}_{seed}"
                    trial = next(row for row in trials if row["family"] == family and row["seed"] == seed and row["negative_multiplier"] == coefficient)
                    check_hash(source / "best.pt", trial["checkpoint_sha256"])
                    path = source / f"{split}_predictions.npz"
                    check_hash(path, trial["prediction_sha256"][split])
                    with np.load(path, allow_pickle=False) as archive:
                        exported = dict(archive)
                    if (not np.array_equal(exported["row_keys"], keys[split])
                        or not np.array_equal(exported["target_original"], common.targets[common.masks])):
                        raise ValueError("Replay common-row identity mismatch")
                    model = make_model(family, len(common.events.panel.node_order)).to(args.device)
                    model.load_state_dict(torch.load(source / "best.pt", weights_only=True, map_location=args.device))
                    scheduled = infer_schedule(model, datasets[family][split], family, schedule, args.device)
                    overlap = np.isin(schedule["cutoffs"], common.cutoffs)
                    positions = np.searchsorted(common.cutoffs, schedule["cutoffs"][overlap])
                    if not overlap.any():
                        raise ValueError("No classification/schedule overlap")
                    for key, value in scheduled.items():
                        reference = exported["dense_" + key][positions]
                        np.testing.assert_allclose(value[overlap], reference, atol=1e-6, rtol=0)
                        if not np.array_equal(value[overlap] > .5, reference > .5):
                            raise ValueError("Scheduled decision differs from classification")
                    predictions[family] = exported, scheduled
                for family in (*FAMILIES, "HF"):
                    exported, scheduled = predictions["HH" if family == "HF" else family]
                    direction_export, direction_schedule = predictions["full"] if family == "HF" else (exported, scheduled)
                    case = evaluate_case(folder / f"{family}_{seed}", f"{family}_{seed}", quotes[split], schedule,
                                         common, keys[split], exported["q"], scheduled["q"],
                                         row_d=direction_export.get("d"), scheduled_d=direction_schedule.get("d"))
                    risk_rows.append(dict(coefficient=coefficient, split=split, family=family, seed=seed,
                                          binary_macro_f1=case["classification"]["macro_f1"], **case["risk"]))
            for name in ("cash", "EW"):
                case = evaluate_case(folder / name, name, quotes[split], schedule, common, keys[split], None, None)
                risk_rows.append(dict(coefficient=coefficient, split=split, family=name, seed=None,
                                      binary_macro_f1=None, **case["risk"]))
    write_json(args.output / "auxiliary_backtest_summary.json", risk_rows)


def classification_comparison(args, datasets):
    summary, paired = {}, {}
    for split in ("validation", "test"):
        dataset = datasets["gm"][split]
        panel = dataset.events.panel
        dates = panel.trading_dates[dataset.cutoffs[np.nonzero(dataset.masks)[0]]]
        days = np.unique(panel.trading_dates[(panel.trading_dates >= dates.min()) & (panel.trading_dates <= dates.max())])
        rng = np.random.default_rng(20261023)
        indices = (rng.integers(0, len(days) - 4, size=(10000, (len(days) + 4) // 5))[..., None] + np.arange(5)).reshape(10000, -1)[:, :len(days)]
        np.savez_compressed(args.output / f"{split}_paired_days.npz", trading_dates=days, indices=indices)
        for family in FAMILIES:
            for seed in FROZEN["seeds"]:
                counts = []
                for coefficient in (1, 3):
                    path = args.output / f"c{coefficient}" / "training" / f"{family}_{seed}"
                    report = binary.read_json(path / f"{split}_classification.json")
                    summary[f"{split}/c{coefficient}/{family}/{seed}"] = report
                    with np.load(path / f"{split}_predictions.npz", allow_pickle=False) as archive:
                        truth, pred = archive["target_original"] > 0, archive["q"] > .5
                    counts.append(np.stack([np.bincount(2 * truth[dates == day].astype(int) + pred[dates == day], minlength=4).reshape(2, 2) for day in days]))
                comparison = {}
                from tools.model_performance_registry_audit import macro_f1, summarize
                functions = dict(macro_f1=macro_f1,
                    false_positive_rate=lambda cm: np.divide(cm[..., 0, 1], cm[..., 0, :].sum(-1), out=np.full(cm.shape[:-2], np.nan), where=cm[..., 0, :].sum(-1) > 0),
                    positive_precision=lambda cm: np.divide(cm[..., 1, 1], cm[..., :, 1].sum(-1), out=np.zeros(cm.shape[:-2]), where=cm[..., :, 1].sum(-1) > 0),
                    positive_recall=lambda cm: np.divide(cm[..., 1, 1], cm[..., 1, :].sum(-1), out=np.full(cm.shape[:-2], np.nan), where=cm[..., 1, :].sum(-1) > 0))
                for name, function in functions.items():
                    draws = []
                    for count in counts:
                        draws.append(np.concatenate([function(count[indices[start:start + 200]].sum(1)) for start in range(0, 10000, 200)]))
                    observed = function(counts[1].sum(0)) - function(counts[0].sum(0))
                    comparison[name] = summarize(draws[1] - draws[0], float(observed))
                paired[f"{split}/{family}/{seed}/c3_minus_c1"] = comparison
    write_json(args.output / "binary_classification_summary.json", dict(metrics=summary, HF_binary_alias="HH", all_seeds_reported=True))
    write_json(args.output / "paired_coefficient_comparison.json", dict(comparisons=paired, block_days=5, repetitions=10000,
               seed=20261023, seed_pooling=False, multiplicity="unadjusted_descriptive", HF_not_independent=True))


def collect_workers(args, protocol, code, inputs):
    """Admit two completed coefficient workers before copying any training results."""
    trials, admitted, seen = [], [], set()
    for worker in args.merge_workers:
        marker = binary.read_json(worker / "completed.json")
        if marker["status"] != "training_only":
            raise ValueError("Merge requires completed training-only workers")
        manifest_path = worker / "artifact_hashes.json"
        check_hash(manifest_path, marker["artifact_hashes_sha256"])
        for relative, digest in binary.read_json(manifest_path).items():
            path = (worker / relative).resolve()
            if not path.is_relative_to(worker.resolve()):
                raise ValueError("Worker artifact escapes its directory")
            check_hash(path, digest)
        for name in ("protocol_snapshot.json", "parent_protocol_snapshot.json", "split_and_mask_hashes.json", "candidate_registry.json"):
            if binary.read_json(worker / name) != binary.read_json(args.output / name):
                raise ValueError(f"Worker metadata mismatch: {name}")
        if binary.read_json(worker / "source_hashes.json") != code or binary.read_json(worker / "input_hashes.json") != inputs:
            raise ValueError("Worker source/input hashes differ from this run")
        environment = binary.read_json(worker / "environment.json")
        current = binary.read_json(args.output / "environment.json")
        if any(environment[key] != current[key] for key in ("device", "python", "numpy", "torch", "threads", "deterministic_algorithms", "allow_tf32")):
            raise ValueError("Worker numerical environment differs from this run")
        for split in ("validation", "test"):
            with np.load(worker / f"{split}_signal_schedule.npz", allow_pickle=False) as left, np.load(args.output / f"{split}_signal_schedule.npz", allow_pickle=False) as right:
                if set(left.files) != set(right.files) or any(not np.array_equal(left[key], right[key]) for key in left.files):
                    raise ValueError("Worker auxiliary schedule mismatch")
        rows = binary.read_json(worker / "training_completed.json")["trials"]
        if marker["trained_trials"] != len(rows) or len(rows) != len(FAMILIES) * len(FROZEN["seeds"]):
            raise ValueError("Worker must contain one complete coefficient")
        coefficients = {row["negative_multiplier"] for row in rows}
        if len(coefficients) != 1 or not coefficients.issubset(set(protocol["negative_multipliers"])):
            raise ValueError("Worker coefficient mismatch")
        expected = {(family, seed) for family in FAMILIES for seed in FROZEN["seeds"]}
        if {(row["family"], row["seed"]) for row in rows} != expected:
            raise ValueError("Worker trial matrix is incomplete or duplicated")
        for row in rows:
            identity = (row["negative_multiplier"], row["family"], row["seed"])
            if identity in seen or row["device"] != args.device:
                raise ValueError("Duplicate trial or mixed training devices")
            seen.add(identity)
            relative = Path(f"c{int(identity[0])}/training/{identity[1]}_{identity[2]}")
            source = worker / relative
            check_hash(source / "best.pt", row["checkpoint_sha256"])
            for split, digest in row["prediction_sha256"].items():
                check_hash(source / f"{split}_predictions.npz", digest)
            admitted.append((source, args.output / relative))
            trials.append(row)
    if len(seen) != len(protocol["negative_multipliers"]) * len(FAMILIES) * len(FROZEN["seeds"]):
        raise ValueError("Merge requires both coefficient workers")
    for source, destination in admitted:
        shutil.copytree(source, destination)
    write_json(args.output / "worker_provenance.json", {str(path.resolve()): sha(path / "completed.json") for path in args.merge_workers})
    return trials


def run(args):
    if args.output.exists():
        raise FileExistsError("Output must be a new directory")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(args.output / "run_status.json", dict(status="preflight", trials=[]))
    protocol, parent, datasets, keys, quotes, schedules, report, inputs = load_experiment(args)
    code = binary.source_hashes()
    write_json(args.output / "source_hashes.json", code)
    write_json(args.output / "input_hashes.json", inputs)
    write_json(args.output / "data_manifest.json", report)
    write_json(args.output / "protocol_snapshot.json", protocol)
    write_json(args.output / "parent_protocol_snapshot.json", parent)
    write_json(args.output / "environment.json", dict(python=sys.version, numpy=np.__version__, torch=torch.__version__, device=args.device, threads=args.threads,
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), allow_tf32=False,
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
               gpu_name=torch.cuda.get_device_name() if args.device == "cuda" else None))
    write_json(args.output / "split_and_mask_hashes.json", {
        split: {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                for key, value in dict(cutoffs=item.cutoffs, histories=item.histories, targets=item.targets, masks=item.masks, row_keys=keys[split]).items()}
        for split, item in datasets["gm"].items()})
    for split, schedule in schedules.items():
        np.savez_compressed(args.output / f"{split}_signal_schedule.npz", **schedule)
    train = datasets["gm"]["train"]
    counts = np.bincount(train.targets[train.masks], minlength=3).tolist()
    weights = {coefficient: torch.tensor(balanced_follow_class_weights(counts, negative_multiplier=coefficient), dtype=torch.float32, device=args.device) for coefficient in protocol["negative_multipliers"]}
    write_json(args.output / "candidate_registry.json", dict(models={family: model_kwargs(family, len(report["nodes"])) for family in FAMILIES},
               weights={str(key): value.cpu().tolist() for key, value in weights.items()}, HF=protocol["HF"], selection=protocol["selection"]))
    state = dict(status="preflight_ready", started_utc=datetime.now(timezone.utc).isoformat(), trials=[], report=report)
    write_json(args.output / "run_status.json", state)
    print("PREFLIGHT_READY", json.dumps(report), flush=True)
    if args.smoke:
        for coefficient, weight in weights.items():
            for family in FAMILIES:
                torch.manual_seed(FROZEN["seeds"][0])
                model = make_model(family, len(report["nodes"])).to(args.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.00001)
                result, _, steps = run_epoch(model, Subset(datasets[family]["train"], [0, 1]), family, args.device, weight, optimizer)
                if steps != 1:
                    raise ValueError("Smoke must execute one update")
                print("SMOKE", coefficient, family, steps, result["rows"], flush=True)
        state["status"] = "smoke_only"
    elif args.execute:
        if getattr(args, "merge_workers", None):
            state["trials"] = collect_workers(args, protocol, code, inputs)
        else:
            for coefficient, weight in weights.items():
                if getattr(args, "coefficient", None) is not None and coefficient != args.coefficient:
                    continue
                for family in FAMILIES:
                    for seed in FROZEN["seeds"]:
                        state.update(status="training", active_trial=f"c{coefficient:g}/{family}_{seed}")
                        write_json(args.output / "run_status.json", state)
                        state["trials"].append(train_family(args, family, seed, coefficient, datasets[family], keys, weight))
                        write_json(args.output / "run_status.json", state)
        write_json(args.output / "training_completed.json", dict(trials=state["trials"], HF_binary_alias="HH", HF_backtest_alias=False))
        if getattr(args, "training_only", False):
            state["status"] = "training_only"
        else:
            classification_comparison(args, datasets)
            state.update(status="auxiliary_backtest", active_trial=None)
            write_json(args.output / "run_status.json", state)
            replay(args, datasets, keys, quotes, schedules, state["trials"])
            state["status"] = "completed"
    for path, digest in inputs.items():
        check_hash(Path(path), digest)
    for relative, digest in code.items():
        check_hash(ROOT / relative, digest)
    state.update(elapsed_seconds=time.perf_counter() - started, active_trial=None)
    write_json(args.output / "run_status.json", state)
    write_json(args.output / "artifact_hashes.json", {path.relative_to(args.output).as_posix(): sha(path) for path in sorted(args.output.rglob("*")) if path.is_file()})
    write_json(args.output / "completed.json", dict(status=state["status"], trained_trials=len(state["trials"]), artifact_hashes_sha256=sha(args.output / "artifact_hashes.json"), strict_pit=False, fresh_oos=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "frozen-ten-year-follow-20261101.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "models_performance/ten_year/data/v2_27node")
    parser.add_argument("--quotes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--coefficient", type=int, choices=(1, 3))
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--merge-workers", type=Path, nargs=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.coefficient is not None and not (args.execute and args.training_only):
        parser.error("A coefficient worker requires --execute --training-only")
    if args.training_only and (not args.execute or args.coefficient is None):
        parser.error("Training-only requires --execute and one coefficient")
    if args.merge_workers and (not args.execute or args.training_only or args.coefficient is not None):
        parser.error("Merge requires --execute without a coefficient/training-only")
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            parser.error("Requested CUDA is unavailable")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    existed = args.output.exists()
    try:
        run(args)
    except Exception as error:
        if not existed and (args.output / "run_status.json").is_file():
            state = binary.read_json(args.output / "run_status.json")
            state.update(status="failed", error=str(error))
            write_json(args.output / "run_status.json", state)
        print("FAILED", repr(error), flush=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
