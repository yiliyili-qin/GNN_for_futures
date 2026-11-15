"""Preregistered alpha39 Follow-only Full/HH/HF/GM comparison; HF is an HH alias.

Default is preflight. --smoke updates two train samples per independent family.
--execute trains all three seeds from scratch, then runs the shared-clock ledger.
The parent three-class protocol, raw data and old results are never rewritten.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset

from graph_model.alpha_variants import HistoricalGraphDataset
from graph_model.daily_graph_dataset import DailyGraphDataset
from gnn_for_futures.shock_follow.models.gm_baseline import GraphMixerBaseline
from gnn_for_futures.shock_follow.models.mixer_v2 import ShockFollowGraphMixerV2
from gnn_for_futures.shock_follow.models.head_v2 import balanced_follow_class_weights
from gnn_for_futures.shock_follow.data.thresholds import materialize_causal_monthly_thresholds, save_monthly_threshold_table
from tools.backtest_gm_baseline import load_quotes, shared_schedule, check_hash, write_csv
from tools.train_gm_baseline import INPUT_FIELDS, FROZEN, load_alpha39_inputs, sha, write_json

FAMILIES = ("full", "HH", "gm")
OWN_FIELDS = ("node_history", "economic_prior", "regression_weight", "prior_available", "regression_available")
BASELINES = ("cash", "no_follow", "train_prior", "EW")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binary_classification(target_original, q):
    target = np.asarray(target_original)
    q = np.asarray(q)
    if (target.ndim != 1 or q.shape != target.shape or not len(target)
        or not np.isin(target, [0, 1, 2]).all() or not np.isfinite(q).all()
        or ((q < 0) | (q > 1)).any()):
        raise ValueError("Expected nonempty valid original labels and finite aligned q in [0,1]")
    truth, prediction = (target > 0).astype(np.int64), (q > .5).astype(np.int64)
    confusion = np.bincount(2 * truth + prediction, minlength=4).reshape(2, 2)
    support, predicted = confusion.sum(1), confusion.sum(0)
    precision = np.divide(confusion.diagonal(), predicted, out=np.zeros(2), where=predicted > 0)
    recall = np.divide(confusion.diagonal(), support, out=np.zeros(2), where=support > 0)
    denominator = support + predicted
    f1 = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(2), where=denominator > 0)
    reason = "no_positive" if support[1] == 0 else "no_negative" if support[0] == 0 else None
    q64 = q.astype(np.float64)
    clipped = np.clip(q64, 1e-12, 1 - 1e-12)
    confidence = np.where(prediction == 1, q64, 1 - q64)
    correct = prediction == truth
    bins = np.minimum((confidence * 10).astype(int), 9)
    calibration, ece = [], 0.
    for index in range(10):
        selected = bins == index
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        average = float(confidence[selected].mean()) if count else None
        if count:
            ece += count / len(truth) * abs(accuracy - average)
        calibration.append(dict(bin=index, rows=count, accuracy=accuracy, confidence=average))
    return dict(
        macro_f1=float(f1.mean()), per_class_f1=f1.tolist(), per_class_precision=precision.tolist(),
        per_class_recall=recall.tolist(), support=support.tolist(), confusion=confusion.tolist(),
        confusion_row_normalized=np.divide(confusion, support[:, None], out=np.zeros((2, 2)), where=support[:, None] > 0).tolist(),
        rows=len(truth), class_order=["no_follow", "follow"], positive_precision=float(precision[1]),
        positive_recall=float(recall[1]), positive_f1=float(f1[1]), accuracy=float(correct.mean()),
        balanced_accuracy=float(recall[support > 0].mean()), supported_class_count=int((support > 0).sum()),
        true_follow_rate=float(truth.mean()), predicted_follow_rate=float(prediction.mean()),
        follow_AP=dict(value=float(average_precision_score(truth, q)) if reason is None else None, reason=reason),
        ROC_AUC=dict(value=float(roc_auc_score(truth, q)) if reason is None else None, reason=reason),
        log_loss=float(-(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped)).mean()),
        binary_Brier=float(np.square(q64 - truth).mean()), ECE=float(ece), calibration_bins=calibration,
    )


def follow_loss(logits, target_original, mask, weights):
    if logits.shape != target_original.shape + (2,) or mask.shape != target_original.shape or mask.dtype != torch.bool:
        raise ValueError("Binary logits/labels/mask must align")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite Follow logits, including outside supervision")
    if torch.diagonal(mask, dim1=1, dim2=2).any():
        raise ValueError("Self pairs must not be supervised")
    if not mask.any():
        return None
    targets = target_original[mask]
    if not torch.isin(targets, torch.tensor([0, 1, 2], device=targets.device)).all():
        raise ValueError("Unavailable labels must never enter binary supervision")
    losses = torch.nn.functional.cross_entropy(logits[mask], (targets > 0).long(), weight=weights.to(logits), reduction="none")
    counts = mask.sum(-1)
    cells = torch.zeros_like(mask, dtype=losses.dtype).masked_scatter(mask, losses)
    return (cells.sum(-1) / counts.clamp_min(1))[counts > 0].mean()


class FollowDataset(Dataset):
    """Own-model tensors on the exact same cutoff/history/label/mask objects as GM."""

    def __init__(self, common, family, prior, graph_reader):
        self.common, self.family, self.prior, self.graph_reader = common, family, prior, graph_reader
        self.events = common.events
        self.cutoffs, self.histories = common.cutoffs, common.histories
        self.targets, self.masks = common.targets, common.masks
        prepared = common.events.prepared
        self.history_reader = HistoricalGraphDataset(SimpleNamespace(
            panel=common.events.panel, input_valid=prepared.input_valid, threshold_matrix=prepared.threshold_matrix))

    def __len__(self):
        return len(self.cutoffs)

    def inputs(self, cutoff, history):
        prepared, panel = self.events.prepared, self.events.panel
        if self.family == "full":
            graph, available = self.graph_reader.graph_for_date(panel.trading_dates[cutoff])
        else:
            graph, available = self.history_reader.graph_for_cutoff(int(cutoff))
        return dict(node_history=prepared.node_features[history].transpose(1, 0, 2).copy(),
                    economic_prior=self.prior, regression_weight=graph, regression_available=available,
                    prior_available=~np.eye(len(panel.node_order), dtype=bool))

    def __getitem__(self, item):
        return dict(self.inputs(int(self.cutoffs[item]), self.histories[item]),
                    target_class=self.targets[item].astype(np.int64), loss_mask=self.masks[item], sample_index=item)


def model_kwargs(family, nodes):
    if family == "gm":
        return dict(node_feature_dim=nodes, edge_feature_dim=18, max_events=50, time_dim=100,
                    hidden_dim=100, num_layers=2, head_dim=100, dropout=.1, follow_only=True)
    return dict(node_feature_dim=9, expected_num_futures=nodes, enabled_channels=(True, True, True)
                if family == "full" else (False, True, False), follow_only=True)


def make_model(family, nodes):
    constructor = GraphMixerBaseline if family == "gm" else ShockFollowGraphMixerV2
    return constructor(**model_kwargs(family, nodes))


def run_epoch(model, dataset, family, device, weights, optimizer=None, *, collect_dense=False):
    model.train(optimizer is not None)
    outputs = {name: [] for name in ("q", "target_original", "sample_index", "source", "target_node")}
    dense, steps = [], 0
    fields = INPUT_FIELDS if family == "gm" else OWN_FIELDS
    for batch in DataLoader(dataset, batch_size=FROZEN["batch_size"], shuffle=optimizer is not None):
        inputs = {name: batch[name].to(device) for name in fields}
        labels, mask = batch["target_class"].to(device), batch["loss_mask"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(**inputs)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite binary model logits; no model-specific dropping")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss = follow_loss(logits, labels, mask, weights)
                if loss is not None:
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite Follow loss")
                    loss.backward()
                    optimizer.step()
                    steps += 1
        q = logits.softmax(-1)[..., 1].detach().cpu().numpy()
        chosen = batch["loss_mask"].numpy()
        outputs["q"].append(q[chosen])
        outputs["target_original"].append(batch["target_class"].numpy()[chosen])
        item, source, target = np.nonzero(chosen)
        outputs["sample_index"].append(batch["sample_index"].numpy()[item])
        outputs["source"].append(source)
        outputs["target_node"].append(target)
        if collect_dense:
            dense.append(q)
    outputs = {name: np.concatenate(values) for name, values in outputs.items()}
    if collect_dense:
        outputs["dense_q"] = np.concatenate(dense)
    return binary_classification(outputs["target_original"], outputs["q"]), outputs, steps


def infer_schedule(model, dataset, family, schedule, device):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(schedule["cutoffs"]), FROZEN["batch_size"]):
            samples = []
            for cutoff, history in zip(schedule["cutoffs"][start:start + 16], schedule["histories"][start:start + 16]):
                if family == "gm":
                    sample = dataset.events.snapshot(int(cutoff), int(history[0]), 50, 55)
                    sample["node_features"] = dataset.identity
                else:
                    sample = dataset.inputs(int(cutoff), history)
                samples.append(sample)
            inputs = {key: torch.stack([torch.as_tensor(sample[key]) for sample in samples]).to(device) for key in samples[0]}
            logits = model(**inputs)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite scheduled Follow logits")
            outputs.append(logits.softmax(-1)[..., 1].cpu().numpy())
    return np.concatenate(outputs)


def source_hashes():
    paths = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_file() and path.is_relative_to(ROOT):
                paths.add(path)
    paths.add(Path(__file__).resolve())
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}


def validate_binary_protocol(path):
    protocol = read_json(path)
    parent_path = ROOT / protocol["parent_protocol"]
    check_hash(parent_path, protocol["parent_protocol_sha256"])
    if ((protocol["schema"], protocol["version"]) != ("alpha39_follow_binary", "1.0.0")
        or protocol["task"]["mapping"] != dict(N=0, O=1, S=1)
        or protocol["task"]["required_outputs"] != ["q"]
        or protocol["models"]["HF"]["alias_of"] != "HH"
        or protocol["training"]["direction_loss"] is not False
        or protocol["backtest"]["direction_policy"] != "fixed_same_as_source_user_approved_20261031"):
        raise ValueError("Unsupported binary task/alias/trading protocol")
    # All preregistered fields must be byte-independent identical to the shipped contract.
    if protocol != read_json(ROOT / "frozen-follow-binary-20261031.json"):
        raise ValueError("Binary protocol differs from preregistered contract")
    return protocol, parent_path, read_json(parent_path)


def load_experiment(args):
    protocol, parent_path, parent = validate_binary_protocol(args.protocol)
    common, keys, report = load_alpha39_inputs(parent_path, args.data_dir, "identity", 50, 55, args.device)
    panel, prepared = common["train"].events.panel, common["train"].events.prepared
    sources = {str(args.protocol.resolve()): sha(args.protocol), str(parent_path.resolve()): sha(parent_path)}
    for name in ("bar_panel.npz", "bar_panel_manifest.json"):
        path = args.data_dir / name
        sources[str(path.resolve())] = sha(path)
    registry_path = ROOT / "data/graph_data/economic_prior_39_registry.json"
    registry = read_json(registry_path)
    if registry["node_order"] != list(panel.node_order):
        raise ValueError("Prior registry node order mismatch")
    prior = np.zeros((len(panel.node_order), len(panel.node_order)), dtype=np.float32)
    positions = {node: index for index, node in enumerate(panel.node_order)}
    for edge in registry["edges"]:
        source, target = positions[edge["source"]], positions[edge["target"]]
        prior[source, target] = prior[target, source] = 1
    prior_path = args.data_dir / "economic_prior.npy"
    if not np.array_equal(np.load(prior_path, allow_pickle=False), prior) or prior.sum() != 98:
        raise ValueError("Economic prior differs from canonical 49 symmetric relations")
    sources[str(prior_path.resolve())] = sha(prior_path)
    metadata_path = args.graph_dir.parent / "metadata.json"
    metadata = read_json(metadata_path)
    if (metadata["node_order"] != list(panel.node_order)
        or metadata["input_sha256"] != {source.symbol: source.sha256 for source in panel.source_files}
        or metadata["protocol"] != "daily20_bic12_common17_20261010"):
        raise ValueError("Daily regression provenance differs from original alpha39 panel")
    for name in ("build_daily_return_graphs.py", "daily_return_regression.py"):
        check_hash(ROOT / "graph_model" / name, metadata["source_sha256"][name])
    sources[str(metadata_path.resolve())] = sha(metadata_path)
    reader = DailyGraphDataset(None, args.graph_dir, panel.node_order, None)
    datasets = {family: {split: FollowDataset(dataset, family, prior, reader) for split, dataset in common.items()}
                for family in ("full", "HH")}
    datasets["gm"] = common
    quotes, schedules = {}, {}
    for split in ("validation", "test"):
        quotes[split] = load_quotes(args.quotes_dir, split, panel, parent, sources)
        schedules[split] = shared_schedule(panel, prepared, quotes[split], parent, split)
    needed = np.unique(np.concatenate([panel.trading_dates[dataset.cutoffs] for dataset in common.values()]
                                     + [value["trading_date"] for value in schedules.values()]))
    # Fail before any training if Full has a missing or future-fitted graph. Never drop rows.
    for day in needed:
        path = args.graph_dir / f"{day}.csv"
        sources[str(path.resolve())] = sha(path)
        reader.graph_for_date(day)
    report.update(binary_protocol_sha256=sha(args.protocol), models=["full", "HH", "HF", "gm"],
                  independent_families=list(FAMILIES), HF_alias_of="HH", scope="Follow-only comparison",
                  binary_counts={split: np.bincount((dataset.targets[dataset.masks] > 0).astype(int), minlength=2).tolist()
                                 for split, dataset in common.items()},
                  schedules={split: len(value["cutoffs"]) for split, value in schedules.items()},
                  regression_days=len(needed), direction_policy=protocol["backtest"]["direction_policy"])
    return protocol, parent, datasets, keys, report, sources, quotes, schedules


def save_predictions(path, dataset, keys, rows, seed, family):
    coordinates = np.argwhere(dataset.masks)
    if (not np.array_equal(np.column_stack([rows[name] for name in ("sample_index", "source", "target_node")]), coordinates)
        or not np.array_equal(rows["target_original"], dataset.targets[dataset.masks])
        or not np.array_equal(rows["q"], rows["dense_q"][dataset.masks]) or len(keys) != len(coordinates)):
        raise ValueError("Binary export changed common rows or masks")
    np.savez_compressed(path, **rows, target_binary=(rows["target_original"] > 0).astype(np.int8),
                        row_keys=keys, seed=seed, family=family, node_order=np.asarray(dataset.events.panel.node_order),
                        cutoff_timestamp=dataset.events.panel.timestamps[dataset.cutoffs[coordinates[:, 0]]],
                        dense_cutoffs=dataset.cutoffs, dense_history_indices=dataset.histories)


def train_family(args, family, seed, datasets, keys, weights):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model = make_model(family, len(datasets["train"].events.panel.node_order)).to(args.device)
    if any("direction_output" in name for name, _ in model.named_parameters()):
        raise AssertionError("Direction parameters must not exist in a binary model")
    optimizer = torch.optim.Adam(model.parameters(), lr=FROZEN["learning_rate"], weight_decay=FROZEN["weight_decay"])
    directory = args.output / "training" / f"{family}_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    best, best_epoch, history, steps, digest = -1., 0, [], 0, None
    started = time.perf_counter()
    for epoch in range(1, FROZEN["max_epochs"] + 1):
        train, _, updates = run_epoch(model, datasets["train"], family, args.device, weights, optimizer)
        validation, _, _ = run_epoch(model, datasets["validation"], family, args.device, weights)
        steps += updates
        history.append(dict(epoch=epoch, train=train, validation=validation))
        write_json(directory / "history.json", history)
        write_csv(directory / "all_epoch_validation_metrics.csv", dict(
            epoch=[item["epoch"] for item in history], macro_f1=[item["validation"]["macro_f1"] for item in history],
            follow_AP=[item["validation"]["follow_AP"]["value"] for item in history]))
        if validation["macro_f1"] > best:
            best, best_epoch = validation["macro_f1"], epoch
            torch.save(model.state_dict(), directory / "best.pt")
            digest = sha(directory / "best.pt")
            write_json(directory / "checkpoint_hashes.json", dict(best_epoch=best_epoch, sha256=digest))
        print(f"BINARY family={family} seed={seed} epoch={epoch} validation_macro_f1={validation['macro_f1']:.8f} best={best_epoch}", flush=True)
        if epoch - best_epoch >= FROZEN["patience"]:
            break
    check_hash(directory / "best.pt", digest)
    model.load_state_dict(torch.load(directory / "best.pt", weights_only=True, map_location=args.device))
    model.requires_grad_(False)
    predictions, metrics = {}, {}
    for split in ("validation", "test"):
        metrics[split], rows, _ = run_epoch(model, datasets[split], family, args.device, weights, collect_dense=True)
        path = directory / f"{split}_predictions.npz"
        save_predictions(path, datasets[split], keys[split], rows, seed, family)
        predictions[split] = sha(path)
        write_json(directory / f"{split}_classification.json", metrics[split])
    first_best = max(history, key=lambda item: item["validation"]["macro_f1"])
    if first_best["epoch"] != best_epoch or first_best["validation"]["macro_f1"] != metrics["validation"]["macro_f1"]:
        raise ValueError("Reloaded checkpoint is not earliest validation binary macro-F1 maximum")
    summary = dict(family=family, seed=seed, epochs=epoch, best_epoch=best_epoch, optimizer_steps=steps,
                   validation_macro_f1=best, test_macro_f1=metrics["test"]["macro_f1"],
                   checkpoint_sha256=digest, prediction_sha256=predictions, seconds=time.perf_counter() - started,
                   parameters=sum(value.numel() for value in model.parameters()), device=args.device,
                   peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
    write_json(directory / "training_completed.json", summary)
    print("TRIAL_COMPLETED", json.dumps(summary), flush=True)
    return summary


def replay(args, parent, datasets, keys, weights, quotes, schedules, trials):
    from tools.backtest_follow_binary import evaluate_case, paired_bootstrap
    directory = args.output / "backtest"
    directory.mkdir()
    comparison, bootstraps, summaries = [], {}, {}
    train = datasets["gm"]["train"]
    prior = float((train.targets[train.masks] > 0).mean())
    for split, schedule in schedules.items():
        folder, common = directory / split, datasets["gm"][split]
        folder.mkdir()
        np.savez_compressed(folder / "signal_schedule_and_boundaries.npz", **schedule)
        cases = {}
        shape = (len(schedule["cutoffs"]), len(common.events.panel.node_order), len(common.events.panel.node_order))
        for name in BASELINES:
            row_q = None if name in ("cash", "EW") else np.full(common.masks.sum(), prior if name == "train_prior" else 0.)
            scheduled_q = None if row_q is None else np.full(shape, prior if name == "train_prior" else 0.)
            cases[name] = evaluate_case(folder / name, name, quotes[split], schedule, common, keys[split], row_q, scheduled_q)
        for seed in parent["selection"]["seeds"]:
            seed_cases = {name: cases[name] for name in BASELINES}
            for family in FAMILIES:
                name = f"{family}_{seed}"
                trial = next(item for item in trials if item["family"] == family and item["seed"] == seed)
                source = args.output / "training" / name
                check_hash(source / "best.pt", trial["checkpoint_sha256"])
                path = source / f"{split}_predictions.npz"
                check_hash(path, trial["prediction_sha256"][split])
                with np.load(path, allow_pickle=False) as archive:
                    exported = {key: archive[key] for key in archive.files}
                dataset = datasets[family][split]
                if (not np.array_equal(exported["row_keys"], keys[split])
                    or not np.array_equal(exported["target_original"], common.targets[common.masks])
                    or not np.array_equal(exported["q"], exported["dense_q"][common.masks])):
                    raise ValueError("Replay prediction identity mismatch")
                model = make_model(family, len(common.events.panel.node_order)).to(args.device)
                model.load_state_dict(torch.load(source / "best.pt", weights_only=True, map_location=args.device))
                model.requires_grad_(False)
                q = infer_schedule(model, dataset, family, schedule, args.device)
                overlap = np.isin(schedule["cutoffs"], common.cutoffs)
                if not overlap.any():
                    raise ValueError("No scheduled/classification overlap")
                positions = np.searchsorted(common.cutoffs, schedule["cutoffs"][overlap])
                reference = exported["dense_q"][positions]
                np.testing.assert_allclose(q[overlap], reference, atol=1e-6, rtol=0)
                if not np.array_equal(q[overlap] > .5, reference > .5):
                    raise ValueError("Scheduled inference crosses binary decision threshold")
                case = evaluate_case(folder / name, name, quotes[split], schedule, common, keys[split], exported["q"], q)
                if case["classification"]["macro_f1"] != trial[f"{split}_macro_f1"]:
                    raise ValueError("Replay classification differs from trained export")
                cases[name] = seed_cases[name] = case
                if family == "HH":
                    alias = f"HF_{seed}"
                    cases[alias] = seed_cases[alias] = case
                    (folder / alias).mkdir()
                    write_json(folder / alias / "alias.json", dict(alias_of=name, independent_trial=False,
                               reason="HF=History q + Full d; d removed; identical classification and fixed-direction replay"))
            bootstraps[f"{split}/{seed}"] = paired_bootstrap(seed_cases, np.unique(quotes[split]["trading_dates"]), seed, folder)
        for name, case in cases.items():
            classified = case["classification"]
            comparison.append(dict(split=split, model=name, alias_of=name.replace("HF_", "HH_") if name.startswith("HF_") else "",
                                   macro_f1=classified["macro_f1"] if classified else None,
                                   positive_f1=classified["positive_f1"] if classified else None,
                                   follow_AP=classified["follow_AP"]["value"] if classified else None,
                                   ROC_AUC=classified["ROC_AUC"]["value"] if classified else None, **case["risk"]))
        summaries[split] = {family: {
            metric: dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)))
            for metric, values in (("macro_f1", [cases[f"{family}_{seed}"]["classification"]["macro_f1"] for seed in FROZEN["seeds"]]),
                                   ("total_return", [cases[f"{family}_{seed}"]["risk"]["total_return"] for seed in FROZEN["seeds"]]))}
            for family in (*FAMILIES, "HF")}
    write_csv(directory / "baseline_comparison.csv", {key: [item[key] for item in comparison] for key in comparison[0]})
    write_json(directory / "paired_daily_bootstrap.json", bootstraps)
    write_json(directory / "seed_summary.json", dict(by_split=summaries, std_definition="sample ddof=1", HF_alias_of="HH"))


def run(args):
    if args.output.exists():
        raise FileExistsError("Refusing to reuse experiment output")
    from tools import backtest_follow_binary  # Include replay code in the pre-training source snapshot.
    started = time.perf_counter()
    protocol, parent, datasets, keys, report, sources, quotes, schedules = load_experiment(args)
    weights = torch.tensor(balanced_follow_class_weights(np.bincount(
        datasets["gm"]["train"].targets[datasets["gm"]["train"].masks], minlength=3).tolist()), device=args.device)
    code = source_hashes()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.protocol, args.output / "protocol_snapshot.json")
    shutil.copyfile(ROOT / protocol["parent_protocol"], args.output / "parent_protocol_snapshot.json")
    write_json(args.output / "source_hashes.json", code)
    write_json(args.output / "input_hashes.json", sources)
    write_json(args.output / "data_manifest.json", report)
    write_json(args.output / "candidate_registry.json", dict(
        candidates={family: model_kwargs(family, len(report["nodes"])) for family in FAMILIES},
        aliases={"HF": "HH"}, selection=parent["selection"],
        metric="validation_binary_macro_f1", loss=protocol["training"]["loss"], follow_class_weights=weights.cpu().tolist(),
        information_sets=dict(full="40 common node rows + prior + causal daily regression + attention",
                              HH="40 common node rows + last realized exact15m history graph",
                              gm="identity + realized events inside same40row span; event times may fall between common rows")))
    write_json(args.output / "split_and_mask_hashes.json", {
        split: {field: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                for field, value in dict(cutoffs=dataset.cutoffs, histories=dataset.histories, mask=dataset.masks,
                                        original_target=dataset.targets, row_keys=keys[split]).items()}
        for split, dataset in datasets["gm"].items()})
    write_json(args.output / "environment.json", dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
               device=args.device, threads=args.threads, deterministic_algorithms=True))
    panel = datasets["gm"]["train"].events.panel
    lower, upper = map(np.datetime64, parent["data"]["splits"]["train"])
    fit = ((panel.timestamps >= lower) & (panel.timestamps < upper) & (panel.trading_dates >= lower) & (panel.trading_dates < upper))
    thresholds = materialize_causal_monthly_thresholds(panel.timestamps, panel.trading_dates,
                 np.where(fit[:, None], panel.signed_returns, np.nan), panel.node_order)
    if thresholds.materialization_sha256 != report["threshold_materialization_sha256"]:
        raise ValueError("Threshold identity changed")
    save_monthly_threshold_table(thresholds, args.output / "thresholds_with_fit_provenance.npz",
                                args.output / "thresholds.csv", args.output / "thresholds_manifest.json")
    for split, schedule in schedules.items():
        np.savez_compressed(args.output / f"{split}_signal_schedule.npz", **schedule)
    state = dict(status="preflight_ready", started_utc=datetime.now(timezone.utc).isoformat(),
                 protocol_sha256=sha(args.protocol), trials=[], report=report)
    write_json(args.output / "run_status.json", state)
    print("PREFLIGHT_READY", json.dumps(report), flush=True)
    if args.smoke:
        for family in FAMILIES:
            dataset = datasets[family]["train"]
            selected = np.flatnonzero(dataset.masks.any(axis=(1, 2)))[:2]
            torch.manual_seed(FROZEN["seeds"][0])
            model = make_model(family, len(panel.node_order)).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
            metrics, _, steps = run_epoch(model, Subset(dataset, selected.tolist()), family, args.device, weights, optimizer)
            print("SMOKE", family, "updates", steps, "rows", metrics["rows"], flush=True)
        state["status"] = "smoke_only"
    elif args.execute:
        state["status"] = "training"
        write_json(args.output / "run_status.json", state)
        for family in FAMILIES:
            for seed in FROZEN["seeds"]:
                state["active_trial"] = f"{family}_{seed}"
                write_json(args.output / "run_status.json", state)
                trial = train_family(args, family, seed, datasets[family], keys, weights)
                state["trials"].append(trial)
                write_json(args.output / "run_status.json", state)
        write_json(args.output / "training_completed.json", dict(trials=state["trials"], HF_alias_of="HH", no_seed_selection=True))
        state["status"] = "backtest"
        write_json(args.output / "run_status.json", state)
        replay(args, parent, datasets, keys, weights, quotes, schedules, state["trials"])
        state["status"] = "completed"
    for path, expected in sources.items():
        check_hash(Path(path), expected)
    for relative, expected in code.items():
        check_hash(ROOT / relative, expected)
    state.update(elapsed_seconds=time.perf_counter() - started, active_trial=None)
    write_json(args.output / "run_status.json", state)
    hashes = {path.relative_to(args.output).as_posix(): sha(path) for path in sorted(args.output.rglob("*")) if path.is_file()}
    write_json(args.output / "artifact_hashes.json", hashes)
    write_json(args.output / "completed.json", dict(status=state["status"], protocol_sha256=sha(args.protocol),
               artifact_hashes_sha256=sha(args.output / "artifact_hashes.json"), trained_trials=len(state["trials"]),
               HF_alias_of="HH", strict_pit=False, fresh_oos=False, direction_policy=protocol["backtest"]["direction_policy"]))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "frozen-follow-binary-20261031.json")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--quotes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    existed = args.output.exists()
    try:
        run(args)
    except Exception as error:
        # Never overwrite a pre-existing run merely to record this failed invocation.
        if not existed and (args.output / "run_status.json").is_file():
            state = read_json(args.output / "run_status.json")
            state.update(status="failed", error=str(error))
            write_json(args.output / "run_status.json", state)
            write_json(args.output / "failed.json", state)
        print(json.dumps(dict(status="failed", error=str(error))), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
