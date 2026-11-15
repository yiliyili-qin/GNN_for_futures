"""Ten-year v2 Follow-only E/R/A ablation; FF and FH share the same Full q.

Default is audited preflight. --smoke updates two supervised training samples per
cell; --execute trains the requested frozen cells and runs auxiliary fixed-source-
direction replay unless --training-only. No deployment or network operations.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch
from torch.utils.data import Subset

from tools import run_follow_binary_experiment as binary
from tools import run_ten_year_follow as ten_year
from tools.backtest_follow_binary import evaluate_case
from tools.backtest_gm_baseline import check_hash, write_csv
from tools.ten_year_train import FROZEN, sha, write_json

SEEDS = (20261002, 20261003, 20261004)
CHANNELS = dict(full=(True, True, True), no_economic=(False, True, True),
                no_regression=(True, False, True), no_attention=(True, True, False),
                economic_only=(True, False, False), regression_only=(False, True, False),
                attention_only=(False, False, True), none=(False, False, False))
# Every one-channel deletion: all six Full -> pair -> single -> none chains.
DELETIONS = tuple((larger, smaller) for larger, left in CHANNELS.items()
                  for smaller, right in CHANNELS.items()
                  if sum(left) == sum(right) + 1 and all(not b or a for a, b in zip(left, right)))
ALIASES = {"FF": "full", "FH": "full"}
DIRECTION_POLICY = "fixed_same_as_source_user_approved_20261031"
EXPECTED_TRAINING = dict(seeds=list(SEEDS), max_epochs=40, patience=7, batch_size=16,
                         learning_rate=.001, weight_decay=.00001, optimizer="Adam")


class NodeOnlyGraphMixer(binary.ShockFollowGraphMixerV2):
    """Empty-E carrier avoids changing the shared model's nonempty-channel API.

    A/R are disabled at construction; E values AND availability are zero here.
    Existing graph fusion then produces zero pair features and no neighbors.
    Only the shared per-node LSTM, role self projections and Follow head remain
    active. The original parameterization is retained, including unused weights.
    """

    def forward(self, node_history, economic_prior, regression_weight, *, prior_available, regression_available):
        return super().forward(node_history, torch.zeros_like(economic_prior), torch.zeros_like(regression_weight),
                               prior_available=torch.zeros_like(prior_available),
                               regression_available=torch.zeros_like(regression_available))


def model_kwargs(variant, nodes):
    channels = CHANNELS[variant]
    return dict(node_feature_dim=9, expected_num_futures=nodes, follow_only=True,
                learnable_fusion=False, enabled_channels=channels if any(channels) else (True, False, False))


def make_model(variant, nodes):
    constructor = NodeOnlyGraphMixer if variant == "none" else binary.ShockFollowGraphMixerV2
    return constructor(**model_kwargs(variant, nodes))


def protocol_snapshot(parent_contract):
    if any(FROZEN[key] != value for key, value in EXPECTED_TRAINING.items()):
        raise ValueError("Frozen ten-year training settings changed")
    selection = parent_contract["selection"]
    expected = dict(EXPECTED_TRAINING, metric="validation_binary_macro_f1", ties="earliest_strict_maximum",
                    follow_threshold=.5, follow_operator=">")
    if any(selection[key] != value for key, value in expected.items()):
        raise ValueError("Ten-year selection settings changed")
    return dict(schema="ten_year_follow_only_graph_ablation", version="1.0.0",
                data_policy="immutable_v2_27node_original_masks_histories_thresholds_splits_embargo",
                mapping=dict(N=0, O=1, S=1), required_outputs=["q"], negative_multiplier=1.,
                initialization="all cells scratch Follow-only; Full is NOT a bitwise reproduction of old dual-head c1 Full",
                training=dict(EXPECTED_TRAINING, direction_loss=False, scheduler="none", gradient_clip=None,
                              loss="source_event_mean_weighted_binary_CE_valid_target_count_not_sum_weights"),
                selection=selection, channels=CHANNELS, channel_order=["economic", "causal_regression", "attention"],
                fusion=dict(base_weights=[.2, .4, .4], learnable=False,
                            normalization="renormalize_surviving_global_weights_as_alpha39_ablation",
                            topology_threshold=.1,
                            none="empty E carrier; zero graph values/availability/pair features/neighbors; node-only LSTM and same Follow head"),
                deletions=DELETIONS, chains=list(itertools.permutations("ERA")),
                aliases=ALIASES, independent_trials=24,
                alias_reason="FF=Full q+Full d and FH=Full q+History d; direction removed, identical Full q and fixed-direction replay",
                backtest=dict(parent_contract["backtest"], role="auxiliary_not_selection",
                              own_direction=DIRECTION_POLICY,
                              alpha="sum_i(source_shock_i*off_diagonal_ij*(q_ij>0.5)*q_ij*source_return_i)",
                              three_class_direction_claim=False),
                strict_pit=False, fresh_oos=False, previously_inspected_test=True,
                comparisons="all seeds and all 12 one-channel deletion edges; descriptive, no seed/variant/test selection")


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def train_cell(args, variant, seed, datasets, keys, weights):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    nodes = len(datasets["train"].events.panel.node_order)
    model = make_model(variant, nodes).to(args.device)
    if any("direction_output" in name for name, _ in model.named_parameters()):
        raise ValueError("Follow-only ablation must not contain direction parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=FROZEN["learning_rate"], weight_decay=FROZEN["weight_decay"])
    directory = args.output / "training" / f"{variant}_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    best, best_epoch, history, steps = -1., 0, [], 0
    started = time.perf_counter()
    for epoch in range(1, FROZEN["max_epochs"] + 1):
        training, _, updates = binary.run_epoch(model, datasets["train"], "full", args.device, weights, optimizer)
        validation, _, _ = binary.run_epoch(model, datasets["validation"], "full", args.device, weights)
        steps += updates
        if not np.isfinite(validation["macro_f1"]):
            raise ValueError("Undefined validation binary macro F1")
        history.append(dict(epoch=epoch, train=training, validation=validation))
        write_json(directory / "history.json", history)
        if validation["macro_f1"] > best:
            best, best_epoch = validation["macro_f1"], epoch
            torch.save(model.state_dict(), directory / "best.pt")
        print(f"GRAPH_ABLATION c=1 variant={variant} seed={seed} epoch={epoch} binary_macro_f1={validation['macro_f1']:.8f} best={best_epoch}", flush=True)
        if epoch - best_epoch >= FROZEN["patience"]:
            break
    checkpoint_hash = sha(directory / "best.pt")
    model.load_state_dict(torch.load(directory / "best.pt", weights_only=True, map_location=args.device))
    model.requires_grad_(False)
    scores, prediction_hashes = {}, {}
    for split in ("validation", "test"):
        scores[split], rows, _ = binary.run_epoch(model, datasets[split], "full", args.device, weights, collect_dense=True)
        path = directory / f"{split}_predictions.npz"
        binary.save_predictions(path, datasets[split], keys[split], rows, seed, variant)
        prediction_hashes[split] = sha(path)
        write_json(directory / f"{split}_classification.json", scores[split])
    if (scores["validation"]["macro_f1"] != best
        or max(history, key=lambda item: item["validation"]["macro_f1"])["epoch"] != best_epoch):
        raise ValueError("Reloaded checkpoint is not earliest validation binary maximum")
    trial = dict(variant=variant, seed=seed, negative_multiplier=1., channels=CHANNELS[variant],
                 epochs=epoch, best_epoch=best_epoch, optimizer_steps=steps,
                 validation_macro_f1=best, test_macro_f1=scores["test"]["macro_f1"],
                 checkpoint_sha256=checkpoint_hash, prediction_sha256=prediction_hashes,
                 seconds=time.perf_counter() - started, parameters=sum(p.numel() for p in model.parameters()),
                 device=args.device, direction_head=False,
                 peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
    write_json(directory / "checkpoint_hashes.json", dict(best_epoch=best_epoch, sha256=checkpoint_hash))
    write_json(directory / "training_completed.json", trial)
    return trial


def replay(args, datasets, keys, quotes, schedules, trials):
    rows = []
    for split, schedule in schedules.items():
        common = datasets[split]
        for trial in trials:
            variant, seed = trial["variant"], trial["seed"]
            source = args.output / "training" / f"{variant}_{seed}"
            check_hash(source / "best.pt", trial["checkpoint_sha256"])
            path = source / f"{split}_predictions.npz"
            check_hash(path, trial["prediction_sha256"][split])
            with np.load(path, allow_pickle=False) as archive:
                exported = dict(archive)
            if (not np.array_equal(exported["row_keys"], keys[split])
                or not np.array_equal(exported["target_original"], common.targets[common.masks])
                or not np.array_equal(exported["q"], exported["dense_q"][common.masks])
                or not np.array_equal(exported["dense_cutoffs"], common.cutoffs)
                or not np.array_equal(exported["dense_history_indices"], common.histories)
                or "d" in exported or "dense_d" in exported):
                raise ValueError("Replay frozen row/mask/history/binary identity mismatch")
            model = make_model(variant, len(common.events.panel.node_order)).to(args.device)
            model.load_state_dict(torch.load(source / "best.pt", weights_only=True, map_location=args.device))
            model.requires_grad_(False)
            q = binary.infer_schedule(model, common, "full", schedule, args.device)
            overlap = np.isin(schedule["cutoffs"], common.cutoffs)
            if not overlap.any():
                raise ValueError("No classification/schedule overlap")
            positions = np.searchsorted(common.cutoffs, schedule["cutoffs"][overlap])
            reference = exported["dense_q"][positions]
            np.testing.assert_allclose(q[overlap], reference, atol=1e-6, rtol=0)
            if not np.array_equal(q[overlap] > .5, reference > .5):
                raise ValueError("Scheduled inference changed binary decisions")
            name = f"{variant}_{seed}"
            # Deliberately omit row_d/scheduled_d: retain the old binary direction policy.
            case = evaluate_case(args.output / "backtest" / split / name, name, quotes[split], schedule,
                                 common, keys[split], exported["q"], q)
            if case["classification"]["macro_f1"] != trial[f"{split}_macro_f1"]:
                raise ValueError("Replay classification differs from trained export")
            rows.append(dict(split=split, variant=variant, seed=seed,
                             binary_macro_f1=case["classification"]["macro_f1"], **case["risk"]))
            if variant == "full":
                for alias in ALIASES:
                    directory = args.output / "backtest" / split / f"{alias}_{seed}"
                    directory.mkdir(parents=True, exist_ok=False)
                    write_json(directory / "alias.json", dict(alias_of=name, independent_trial=False,
                               direction_policy=DIRECTION_POLICY, three_class_direction_claim=False))
        for baseline in ("cash", "EW"):
            case = evaluate_case(args.output / "backtest" / split / baseline, baseline, quotes[split], schedule,
                                 common, keys[split], None, None)
            rows.append(dict(split=split, variant=baseline, seed=None, binary_macro_f1=None, **case["risk"]))
    write_json(args.output / "auxiliary_backtest_summary.json", dict(rows=rows, aliases=ALIASES,
               direction_policy=DIRECTION_POLICY, three_class_direction_claim=False))
    return rows


def comparison(args, trials):
    rows, paired, summary, complete_metrics = [], [], {}, {}
    scalar_fields = ("macro_f1", "positive_f1", "positive_precision", "positive_recall", "accuracy",
                     "balanced_accuracy", "predicted_follow_rate", "true_follow_rate", "log_loss", "binary_Brier", "ECE")
    for split in ("validation", "test"):
        for trial in trials:
            variant, seed = trial["variant"], trial["seed"]
            metric = binary.read_json(args.output / "training" / f"{variant}_{seed}" / f"{split}_classification.json")
            tn, fp, fn, tp = np.asarray(metric["confusion"]).ravel().tolist()
            errors = dict(true_negatives=tn, false_positives=fp, false_negatives=fn, true_positives=tp,
                          false_positive_rate=fp / (tn + fp) if tn + fp else None,
                          false_discovery_rate=fp / (tp + fp) if tp + fp else None)
            complete_metrics[f"{split}/{variant}/{seed}"] = dict(metric, **errors)
            rows.append(dict(split=split, variant=variant, seed=seed, **{key: metric[key] for key in scalar_fields},
                             follow_AP=metric["follow_AP"]["value"], ROC_AUC=metric["ROC_AUC"]["value"], **errors))
        index = {(row["variant"], row["seed"]): row for row in rows if row["split"] == split}
        for larger, smaller in DELETIONS:
            for seed in args.seeds:
                if (larger, seed) in index and (smaller, seed) in index:
                    paired.append(dict(split=split, seed=seed, larger=larger, smaller=smaller,
                                       macro_f1_larger_minus_smaller=index[larger, seed]["macro_f1"] - index[smaller, seed]["macro_f1"]))
        for variant in args.variants:
            selected = [row for row in rows if row["split"] == split and row["variant"] == variant]
            if selected:
                by_metric = {}
                for key in (*scalar_fields, "follow_AP", "ROC_AUC", "false_positive_rate", "false_discovery_rate"):
                    values = [row[key] for row in selected if row[key] is not None]
                    by_metric[key] = dict(defined_seeds=len(values), undefined_seeds=len(selected) - len(values),
                        mean=float(np.mean(values)) if values else None,
                        std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
                summary[f"{split}/{variant}"] = dict(seeds=len(selected), metrics=by_metric)
    write_csv(args.output / "binary_metrics.csv", {key: [row[key] for row in rows] for key in rows[0]})
    write_json(args.output / "binary_classification_summary.json", dict(metrics=complete_metrics, seed_summary=summary,
               aliases=ALIASES, aliases_not_independent=True, std_definition="sample ddof=1",
               undefined_policy="null preserved; aggregate defined seeds only with explicit defined/undefined counts",
               false_discovery_rate_policy="null when no positive predictions", seed_pooling=False, seed_selection=False))
    write_json(args.output / "ablation_comparison.json", dict(summary=summary, paired_deletions=paired, aliases=ALIASES,
               std_definition="sample ddof=1", seed_pooling=False, seed_selection=False,
               multiplicity="unadjusted_descriptive", requested_variants=args.variants, requested_seeds=args.seeds))


def run(args):
    if args.output.exists():
        raise FileExistsError("Output must be a new directory")
    contract = binary.read_json(args.protocol)
    protocol = protocol_snapshot(contract)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    state = dict(status="preflight", trials=[], active_trial=None, started_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / "run_status.json", state)
    _, parent, families, keys, quotes, schedules, report, inputs = ten_year.load_experiment(args)
    datasets = families["full"]  # Every cell sees the identical dataset; no History-graph substitution.
    nodes = len(datasets["train"].events.panel.node_order)
    if nodes != 27:
        raise ValueError("This protocol requires frozen v2_27node data")
    for path in [args.quotes_dir / "quote_manifest.json", *sorted((args.output / "schedule_graphs").glob("*.csv"))]:
        inputs[str(path.resolve())] = sha(path)
    code = binary.source_hashes()
    write_json(args.output / "source_hashes.json", code)
    write_json(args.output / "input_hashes.json", inputs)
    write_json(args.output / "protocol_snapshot.json", protocol)
    write_json(args.output / "ten_year_contract_snapshot.json", contract)
    write_json(args.output / "parent_protocol_snapshot.json", parent)
    report.update(scope="ten_year_FF_FH_follow_only_ERA_ablation", models=list(args.variants), aliases=ALIASES,
                  independent_families=list(args.variants), direction_policy=DIRECTION_POLICY)
    write_json(args.output / "data_manifest.json", report)
    write_json(args.output / "split_and_mask_hashes.json", {
        split: {key: array_hash(value) for key, value in dict(cutoffs=item.cutoffs, histories=item.histories,
                 targets=item.targets, masks=item.masks, row_keys=keys[split]).items()}
        for split, item in datasets.items()})
    train = datasets["train"]
    counts = np.bincount(train.targets[train.masks], minlength=3).tolist()
    weights = torch.tensor(binary.balanced_follow_class_weights(counts, negative_multiplier=1.), dtype=torch.float32, device=args.device)
    write_json(args.output / "candidate_registry.json", dict(
        models={variant: dict(constructor=model_kwargs(variant, nodes), effective_channels=CHANNELS[variant]) for variant in args.variants},
        seeds=args.seeds, negative_multiplier=1., weights=weights.cpu().tolist(), aliases=ALIASES,
        expected_trials=len(args.variants) * len(args.seeds), complete_matrix=list(args.variants) == list(CHANNELS) and tuple(args.seeds) == SEEDS))
    write_json(args.output / "environment.json", dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
        device=args.device, threads=args.threads, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        allow_tf32=False, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")))
    for split, schedule in schedules.items():
        np.savez_compressed(args.output / f"{split}_signal_schedule.npz", **schedule)
    state.update(status="preflight_ready", report=report)
    write_json(args.output / "run_status.json", state)
    print("PREFLIGHT_READY", json.dumps(report), flush=True)
    if args.smoke:
        selected = np.flatnonzero(train.masks.any(axis=(1, 2)))[:2].tolist()
        if not selected:
            raise ValueError("No supervised training samples for smoke")
        for variant in args.variants:
            torch.manual_seed(SEEDS[0])
            model = make_model(variant, nodes).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=FROZEN["learning_rate"], weight_decay=FROZEN["weight_decay"])
            metric, _, steps = binary.run_epoch(model, Subset(train, selected), "full", args.device, weights, optimizer)
            if steps != 1:
                raise ValueError("Smoke must execute exactly one update per cell")
            print("SMOKE", variant, steps, metric["rows"], flush=True)
        state["status"] = "smoke_only"
    elif args.execute:
        for variant in args.variants:
            for seed in args.seeds:
                state.update(status="training", active_trial=f"{variant}_{seed}")
                write_json(args.output / "run_status.json", state)
                state["trials"].append(train_cell(args, variant, seed, datasets, keys, weights))
                write_json(args.output / "run_status.json", state)
        write_json(args.output / "training_completed.json", dict(trials=state["trials"], aliases=ALIASES, alias_independent_training=False))
        comparison(args, state["trials"])
        if args.training_only:
            state["status"] = "training_only"
        else:
            state.update(status="auxiliary_backtest", active_trial=None)
            write_json(args.output / "run_status.json", state)
            replay(args, datasets, keys, quotes, schedules, state["trials"])
            state["status"] = "completed"
    for path, expected in inputs.items():
        check_hash(Path(path), expected)
    for relative, expected in code.items():
        check_hash(ROOT / relative, expected)
    state.update(active_trial=None, elapsed_seconds=time.perf_counter() - started)
    write_json(args.output / "run_status.json", state)
    write_json(args.output / "artifact_hashes.json", {path.relative_to(args.output).as_posix(): sha(path)
               for path in sorted(args.output.rglob("*")) if path.is_file()})
    write_json(args.output / "completed.json", dict(status=state["status"], trained_trials=len(state["trials"]), aliases=ALIASES,
               requested_trials=len(args.variants) * len(args.seeds), full_matrix=len(state["trials"]) == 24,
               artifact_hashes_sha256=sha(args.output / "artifact_hashes.json"), strict_pit=False, fresh_oos=False,
               direction_policy=DIRECTION_POLICY, three_class_direction_claim=False))
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "frozen-ten-year-follow-20261101.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "models_performance/ten_year/data/v2_27node")
    parser.add_argument("--quotes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--variants", choices=tuple(CHANNELS), nargs="+", default=list(CHANNELS))
    parser.add_argument("--seeds", choices=SEEDS, type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--training-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or len(set(args.variants)) != len(args.variants) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive threads and unique variants/seeds required")
    if args.training_only and not args.execute:
        parser.error("Training-only requires --execute")
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
            state.update(status="failed", active_trial=None, error=str(error))
            write_json(args.output / "run_status.json", state)
        print("FAILED", repr(error), flush=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
