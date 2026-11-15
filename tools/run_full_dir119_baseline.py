"""Freeze the primary seven-year baseline and retrain/evaluate on immutable ten-year v2.

Single primary seed, classification only. No legacy weights enter ten-year training.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch

from tools import alpha39_direction_concat as heads
from tools import full_dir119_baseline as baseline
from tools.extend_alpha39_fh import FULL, HASHES
from tools.run_follow_binary_experiment import source_hashes
from tools.ten_year_train import FROZEN, FIELDS, epoch, make_model, preflight, sha, write_json
from tools.train_gm_baseline import classification_report

PARENT = ROOT / "models_performance/ten_year/protocol.v2.json"
SEVEN = ROOT / "results/alpha39_direction_concat_19d_20261103T124821Z_verified"
SPLITS = ("train", "validation", "test")
OWN_FIELDS = ("node_history", "economic_prior", "regression_weight", "prior_available", "regression_available")


def check_confusion(report, target, q, d):
    predicted = np.where(q > .5, np.where(d > .5, 2, 1), 0)
    confusion = np.bincount(3 * target.astype(int) + predicted, minlength=9).reshape(3, 3)
    np.testing.assert_array_equal(confusion, report["three_class"]["confusion"])
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(3), where=denominator > 0)
    np.testing.assert_allclose(f1.mean(), report["three_class"]["macro_f1"], rtol=0, atol=1e-12)


def freeze_seven(output):
    """Reuse the packaged, hash-checked primary baseline; do not retrain/reselect it."""
    packaged = ROOT / "models_performance/full_dir119linear_v1/20261103_primary"
    manifest = packaged / "seven_year_frozen.json"
    baseline.load(manifest)
    write_json(output / "seven_year_frozen.json", baseline.read(manifest))
    write_json(output / "seven_year_summary.json", baseline.read(packaged / "seven_year_summary.json"))
    print("SEVEN_YEAR_FROZEN", str(output / "seven_year_frozen.json"), flush=True)


def validate_data(protocol, arrays, panel, thresholds):
    nodes = protocol["data"]["node_order"]
    np.testing.assert_array_equal(arrays["node_order"], nodes)
    np.testing.assert_array_equal(panel.node_order, nodes)
    calendar = np.unique(panel.trading_dates)
    for split in SPLITS:
        spec = protocol["data"]["splits"][split]
        cutoffs, histories = arrays[f"{split}_cutoffs"], arrays[f"{split}_history_indices"]
        lower, upper = np.datetime64(spec["nominal_start"]), np.datetime64(spec["end_exclusive"])
        effective = lower if split == "train" else max(lower + np.timedelta64(15, "D"), calendar[calendar >= lower][21])
        if effective != np.datetime64(spec["embargo_start"]):
            raise ValueError("Ten-year embargo mismatch")
        if (not (panel.timestamps[histories[:, 0]] > lower).all()
            or not (panel.trading_dates[histories[:, 0]] >= lower).all()
            or not (panel.timestamps[cutoffs] >= effective).all()
            or not (panel.trading_dates[cutoffs] >= effective).all()
            or not (panel.timestamps[cutoffs] + np.timedelta64(15, "m") < upper).all()
            or not (panel.trading_dates[cutoffs] < upper).all()):
            raise ValueError("Ten-year split/history boundary violation")
        np.testing.assert_array_equal(histories[:, -1], cutoffs)
        target_rows = np.searchsorted(panel.timestamps, panel.timestamps[cutoffs] + np.timedelta64(15, "m"))
        np.testing.assert_array_equal(panel.timestamps[target_rows], panel.timestamps[cutoffs] + np.timedelta64(15, "m"))
        np.testing.assert_array_equal(panel.trading_dates[target_rows], panel.trading_dates[cutoffs])
        current, future, scale = panel.signed_returns[cutoffs], panel.signed_returns[target_rows], thresholds[cutoffs]
        mask = (np.abs(current) >= scale)[:, :, None] & panel.return_valid[target_rows, None, :] & ~np.eye(len(nodes), dtype=bool)
        np.testing.assert_array_equal(mask, arrays[f"{split}_loss_mask"])
        same = np.sign(current[:, :, None]) == np.sign(future[:, None, :])
        labels = np.where((np.abs(future) >= scale)[:, None, :], np.where(same, 2, 1), 0)
        np.testing.assert_array_equal(labels[mask], arrays[f"{split}_target_class"][mask])
        if len(cutoffs) != spec["samples"] or int(mask.sum()) != spec["supervised_pairs"]:
            raise ValueError("Frozen ten-year sample counts changed")


def load_data():
    directory = (PARENT.parent / baseline.read(PARENT)["data"]["panel_path"]).parent
    report = preflight(PARENT, directory / "training_arrays.npz", "cpu")
    if report["status"] != "ready":
        raise ValueError(report["blockers"])
    protocol = baseline.read(PARENT)
    verification = baseline.read(directory / "verification.json")
    if (verification["status"] != "passed" or verification["protocol_sha256"] != sha(PARENT)
        or verification["arrays_sha256"] != report["data_sha256"]):
        raise ValueError("Immutable ten-year verification mismatch")
    sources = {str(path.resolve()): sha(path) for path in (PARENT, directory / "training_arrays.npz",
        directory / "bar_panel.npz", directory / "bar_panel_manifest.json", directory / "thresholds.npz",
        directory / "audit.json", directory / "verification.json", directory / "economic_prior.npy")}
    if sha(directory / "bar_panel.npz") != protocol["data"]["panel_sha256"]:
        raise ValueError("Ten-year panel changed")
    for name, digest in verification["daily_graph_hashes"].items():
        path = directory / name.replace("\\", "/")
        if sha(path) != digest:
            raise ValueError("Verified ten-year graph changed")
        sources[str(path.resolve())] = digest
    with np.load(directory / "training_arrays.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    with np.load(directory / "bar_panel.npz", allow_pickle=False) as archive:
        panel = SimpleNamespace(**{name: archive[name] for name in
            ("node_order", "signed_returns", "return_valid", "timestamps", "trading_dates")})
    with np.load(directory / "thresholds.npz", allow_pickle=False) as archive:
        months, ends = archive["months"].astype("datetime64[M]"), archive["fit_ends"]
        positions = np.searchsorted(months, panel.trading_dates.astype("datetime64[M]"))
        np.testing.assert_array_equal(months[positions], panel.trading_dates.astype("datetime64[M]"))
        thresholds = archive["values"][positions]
        if not (ends[~np.isnat(ends)] < np.datetime64(protocol["data"]["splits"]["train"]["end_exclusive"])).all():
            raise ValueError("Threshold fitting reaches held-out data")
    validate_data(protocol, arrays, panel, thresholds)
    return protocol, arrays, panel, thresholds, sources


def train_full(output, arrays, nodes, threads):
    from gnn_for_futures.shock_follow.models.head_v2 import balanced_follow_class_weights
    random.seed(baseline.SEED)
    np.random.seed(baseline.SEED)
    torch.manual_seed(baseline.SEED)
    model = make_model("full", nodes, 9, 40, {"learnable_fusion": False})
    weights = torch.tensor(balanced_follow_class_weights(np.bincount(
        arrays["train_target_class"][arrays["train_loss_mask"]], minlength=3).tolist()), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    destination = output / "ten_year_full_20261002"
    destination.mkdir()
    best, best_epoch, history, steps = -1., 0, [], 0
    started = time.perf_counter()
    for number in range(1, 41):
        training, _, updates = epoch(model, "full", arrays, "train", weights, "cpu", optimizer)
        validation, _, _ = epoch(model, "full", arrays, "validation", weights, "cpu")
        steps += updates
        history.append(dict(epoch=number, train=training, validation=validation))
        write_json(destination / "history.json", history)
        if validation["macro_f1"] > best:
            best, best_epoch = validation["macro_f1"], number
            torch.save(model.state_dict(), destination / "best.pt")
        print("TEN_YEAR_FULL_EPOCH", number, "validation_macro_f1", validation["macro_f1"], "best_epoch", best_epoch,
              "elapsed_seconds", round(time.perf_counter() - started, 1), flush=True)
        if number - best_epoch >= 7:
            break
    model.load_state_dict(torch.load(destination / "best.pt", map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    if epoch(model, "full", arrays, "validation", weights, "cpu")[0]["macro_f1"] != best:
        raise ValueError("Reloaded Full validation score changed")
    info = dict(seed=baseline.SEED, initialization="scratch_no_seven_year_weights", best_epoch=best_epoch,
        epochs=number, optimizer_steps=steps, checkpoint_sha256=sha(destination / "best.pt"),
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-started,
        selection="parent_validation_three_class_macro_f1_earliest_strict_maximum", device="cpu", threads=threads)
    write_json(destination / "training_completed.json", info)
    return model


def collect(model, arrays, panel, thresholds, split):
    fields = {key: [] for key in ("tail", "raw", "q", "base_d", "target", "coordinates", "cutoffs", "row_keys")}
    cutoffs = arrays[f"{split}_cutoffs"]
    for start in range(0, len(cutoffs), 16):
        stop = start + 16
        inputs = {name: torch.from_numpy(arrays[f"{split}_{name}"][start:stop]) for name in OWN_FIELDS}
        q, tail, original_d = baseline.full_forward(model, inputs)
        raw, enabled = heads.features(panel.signed_returns, panel.return_valid, panel.timestamps,
            panel.trading_dates, thresholds, cutoffs[start:stop])
        mask = arrays[f"{split}_loss_mask"][start:stop]
        if not enabled[mask].all():
            raise ValueError("Would drop supervised ten-year direction rows")
        for name, value in dict(tail=tail, raw=raw, q=q, base_d=original_d).items():
            fields[name].append(value[mask])
        fields["target"].append(arrays[f"{split}_target_class"][start:stop][mask])
        coords = np.argwhere(mask)
        fields["cutoffs"].append(cutoffs[start:stop][coords[:, 0]])
        coords[:, 0] += start
        fields["coordinates"].append(coords)
        fields["row_keys"].append(arrays[f"{split}_row_keys"][start:stop][mask])
    result = {name: np.concatenate(value) for name, value in fields.items()}
    result["dates"] = panel.trading_dates[result["cutoffs"]]
    print("TEN_YEAR_EMBEDDINGS", split, len(result["q"]), flush=True)
    return result


def verify_saved(path, reports):
    from sklearn.metrics import confusion_matrix, f1_score
    with np.load(path, allow_pickle=False) as archive:
        target, q = archive["target"], archive["q"]
        for name, key in (("Full_reference", "full_d"), (baseline.MODEL, "d")):
            d = archive[key]
            check_confusion(reports[name], target, q, d)
            for subset, selected in (("true_follow", target > 0), ("alerted_true_follow", (target > 0) & (q > .5))):
                if not selected.any():
                    continue
                labels, predicted = target[selected] == 2, d[selected] > .5
                metric = reports[name]["directions"][subset]
                np.testing.assert_array_equal(confusion_matrix(labels, predicted, labels=[False, True]), metric["confusion"])
                np.testing.assert_allclose(f1_score(labels, predicted, labels=[False, True], average="macro", zero_division=0), metric["macro_f1"], rtol=0, atol=1e-12)
                event = archive["coordinates"][selected, :2]
                _, inverse, counts = np.unique(event, axis=0, return_inverse=True, return_counts=True)
                probability = np.clip(d[selected].astype(float), 1e-12, 1-1e-12)
                losses = np.where(labels, -np.log(probability), -np.log1p(-probability))
                np.testing.assert_allclose(np.average(losses, weights=1/counts[inverse]), metric["source_event_BCE"], rtol=0, atol=1e-6)
    return dict(status="passed", checks="saved_pair_confusions_macroF1_source_event_BCE", probability_metrics="computed_by_runner_not_independently_all_recomputed")


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(args.output / "run_status.json", dict(status="preparing", pid=os.getpid()))
    freeze_seven(args.output)
    protocol, arrays, panel, thresholds, inputs = load_data()
    write_json(args.output / "experiment_protocol.json", dict(schema="full_dir119_single_seed_ten_year", version="1.0.0",
        model=baseline.MODEL, parent=baseline.reference(PARENT), primary_seed=baseline.SEED, seed_replicates=1,
        scope="classification_only_single_seed_extension_not_complete_parent_suite", architecture=baseline.feature_contract(),
        backbone=dict(initialization="scratch", loss="parent_hierarchical_source_event_balanced_train",
            selection="validation_three_class_macro_f1_earliest_strict_maximum", max_epochs=40, patience=7,
            batch_size=16, optimizer="Adam", learning_rate=.001, weight_decay=1e-5),
        direction=dict(initialization="zero", training="all_true_OS_source_event_weighted_BCE",
            selection="validation_true_OS_source_event_BCE_earliest_strict_minimum_including_epoch0",
            max_epochs=40, patience=7, batch_size=2048, learning_rate=.001, weight_decay=1e-5),
        seven_year_weights_allowed=False, fit_scaler_on="train_true_OS_only", test_tuning=False,
        data_splits=protocol["data"]["splits"], backtest="not_run_not_requested",
        strict_pit=False, fresh_oos=False, previously_inspected_test=True))
    write_json(args.output / "input_hashes.json", inputs)
    write_json(args.output / "environment.json", dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        device="cpu", threads=args.threads, deterministic=True, started_utc=datetime.now(timezone.utc).isoformat()))
    # Import lazy training/evaluation dependencies before taking the source snapshot.
    from tools import run_direction_bypass
    from gnn_for_futures.shock_follow.models.head_v2 import balanced_follow_class_weights
    sources = source_hashes()
    write_json(args.output / "source_hashes.json", sources)
    write_json(args.output / "run_status.json", dict(status="training_full", pid=os.getpid()))
    model = train_full(args.output, arrays, len(panel.node_order), args.threads)
    frozen_state = {key: value.clone() for key, value in model.state_dict().items()}
    train, validation = (collect(model, arrays, panel, thresholds, split) for split in ("train", "validation"))
    selected, valid = train["target"] > 0, validation["target"] > 0
    write_json(args.output / "run_status.json", dict(status="training_direction", pid=os.getpid()))
    destination = args.output / "ten_year_direction_20261002"
    fitted, head_info = heads.fit(heads.matrix(train, "concat119")[selected], train["target"][selected] == 2,
        heads.event_weights(train["target"], train["coordinates"]),
        heads.matrix(validation, "concat119")[valid], validation["target"][valid] == 2,
        heads.event_weights(validation["target"], validation["coordinates"]), baseline.SEED, "linear", destination)
    mean, scale = heads.scaler(heads.matrix(train, "concat119")[selected])
    np.testing.assert_array_equal(mean, fitted["mean"])
    np.testing.assert_array_equal(scale, fitted["scale"])
    del train
    baseline.freeze(args.output / "ten_year_frozen.json", args.output / "ten_year_full_20261002/best.pt",
        destination, panel.node_order, "ten_year_v2_27node", args.output / "experiment_protocol.json")
    loaded_full, loaded_head, _ = baseline.load(args.output / "ten_year_frozen.json")
    for key, value in loaded_full.state_dict().items():
        torch.testing.assert_close(value, frozen_state[key], rtol=0, atol=0)
    model, fitted = loaded_full, loaded_head
    write_json(args.output / "run_status.json", dict(status="evaluating_locked_models", pid=os.getpid()))
    summary = dict(model=baseline.MODEL, domain="ten_year_v2_27node", backbone_seed=baseline.SEED,
        head_seed=baseline.SEED, seed_replicates=1, seed_aggregation=None, evaluation_scope="classification_only",
        splits={}, verification={}, risk=None, backtest_status="not_run_not_requested", new_protocol_admitted=False,
        strict_pit=False, fresh_oos=False, previously_inspected_test=True)
    for split in ("validation", "test"):
        cache = validation if split == "validation" else collect(model, arrays, panel, thresholds, split)
        direction = heads.predict(fitted, heads.matrix(cache, "concat119"))
        reports = {"Full_reference": heads.metrics(cache, cache["base_d"]), baseline.MODEL: heads.metrics(cache, direction)}
        if reports["Full_reference"]["follow"] != reports[baseline.MODEL]["follow"]:
            raise ValueError("Direction model changed Follow metrics")
        path = args.output / f"ten_year_{split}_predictions.npz"
        np.savez_compressed(path, target=cache["target"], q=cache["q"], d=direction, full_d=cache["base_d"],
            coordinates=cache["coordinates"], cutoffs=cache["cutoffs"], row_keys=cache["row_keys"], dates=cache["dates"])
        write_json(args.output / f"ten_year_{split}_classification.json", reports)
        summary["splits"][split] = reports
        summary["verification"][split] = verify_saved(path, reports)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, frozen_state[key], rtol=0, atol=0)
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Frozen backbone acquired gradients")
    for name, digest in inputs.items():
        if sha(name) != digest:
            raise ValueError("Input changed while running: " + name)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source changed while running: " + name)
    write_json(args.output / "ten_year_summary.json", summary)
    write_json(args.output / "verification.json", dict(status="passed", full_frozen_during_head_training=True,
        scaler_train_true_OS_exact=True, disk_components_reloaded=True, test_after_both_stages_locked=True,
        split_label_mask_exact=True, saved_prediction_checks=summary["verification"],
        input_files=len(inputs), source_files=len(sources), new_backtest_run=False))
    write_json(args.output / "run_status.json", dict(status="completed", pid=os.getpid(), seconds=time.perf_counter()-started))
    artifacts = {path.relative_to(args.output).as_posix(): sha(path) for path in sorted(args.output.rglob("*")) if path.is_file()}
    write_json(args.output / "artifact_hashes.json", artifacts)
    write_json(args.output / "completed.json", dict(status="classification_only_completed", model=baseline.MODEL,
        artifact_hashes_sha256=sha(args.output / "artifact_hashes.json"), seconds=time.perf_counter()-started,
        strict_pit=False, fresh_oos=False, single_primary_seed=baseline.SEED))
    print("BASELINE_CLASSIFICATION_COMPLETED", str(args.output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.threads < 1:
        parser.error("Positive threads required")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    try:
        run(args)
    except Exception as error:
        if args.output.exists() and not (args.output / "completed.json").exists():
            write_json(args.output / "run_status.json", dict(status="failed", pid=os.getpid(), error=repr(error)))
        raise


if __name__ == "__main__":
    main()
