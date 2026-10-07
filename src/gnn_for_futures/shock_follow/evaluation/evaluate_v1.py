"""Evaluate one frozen shock-follow GraphMixer checkpoint on its test split."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from gnn_for_futures.shock_follow.data.availability import load_retrospective_bar_data_contract
from gnn_for_futures.graph.regression import load_economic_prior_graph_snapshot
from gnn_for_futures.shock_follow.data.artifacts import load_and_validate_materialization_bundle
from gnn_for_futures.shock_follow.training.checkpoint_v1 import (
    file_sha256,
    load_checkpoint,
    model_from_checkpoint,
    read_run_config,
)
from gnn_for_futures.shock_follow.contract import (
    CLASS_ORDER,
    canonical_json_sha256,
    node_order_sha256,
    shock_follow_contract_sha256,
    threshold_contract_sha256,
)
from gnn_for_futures.shock_follow.data.panel import load_shock_follow_bar_panel
from gnn_for_futures.shock_follow.provenance.source import (
    SOURCE_LAYOUT_SCHEMA_VERSION,
    source_hashes,
)
from gnn_for_futures.shock_follow.graph import PAIR_FEATURE_SCHEMA
from gnn_for_futures.shock_follow.models.head_v1 import source_event_balanced_masked_cross_entropy
from gnn_for_futures.shock_follow.data.thresholds import load_monthly_threshold_table
from gnn_for_futures.shock_follow.data.prepared import (
    NODE_FEATURE_SCHEMA,
    ShockFollowCutoffDataset,
    chronological_split_by_target,
    collate_shock_follow_samples,
    eligible_cutoff_indices,
    prepare_shock_follow_data,
    summarize_supervision,
)


DEFAULT_DATA_DIR = Path("data/shock_follow_v1")
DEFAULT_TRAINING_DIR = Path("results/shock_follow_graphmixer_v1")
DEFAULT_OUTPUT_DIR = Path("results/shock_follow_graphmixer_v1_evaluation")
DEFAULT_RETROSPECTIVE_CONTRACT = Path(
    "data/contracts/bar_availability.retrospective.json"
)
DEFAULT_ECONOMIC_GRAPH = Path("data/graph_data/futures_graph_smm_simplified.html")


@dataclass(frozen=True)
class EvaluationCounts:
    source_events: int
    valid_pairs: int
    correct_pairs: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen retrospective shock-follow GraphMixer V1 checkpoint."
    )
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--retrospective-contract", type=Path, default=DEFAULT_RETROSPECTIVE_CONTRACT)
    parser.add_argument("--economic-graph", type=Path, default=DEFAULT_ECONOMIC_GRAPH)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("."),
        help="Root containing src/gnn_for_futures for source provenance",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _claim_test_evaluation(training_dir: Path, identity: dict[str, str]) -> Path:
    """Atomically allow one test evaluation per completed training directory."""
    claim_path = training_dir / "test_evaluation_claim.json"
    payload = json.dumps(
        {
            "status": "running",
            "claimed_at_utc": _utc_now(),
            **identity,
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(
            "this training run already has a claimed test evaluation"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as error:
        try:
            _atomic_json(
                claim_path,
                {
                    "status": "failed",
                    "claimed_at_utc": _utc_now(),
                    **identity,
                    "failed_at_utc": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        except BaseException as claim_error:
            error.add_note(
                "test evaluation access was consumed, but recording the initial "
                "claim failure also failed; the claim path was retained: "
                f"{type(claim_error).__name__}: {claim_error}"
            )
        raise
    return claim_path


def _read_running_test_evaluation_claim(claim_path: Path) -> dict[str, object]:
    if claim_path.is_symlink() or not claim_path.is_file():
        raise FileNotFoundError("test evaluation claim must be a regular nonsymlink file")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if not isinstance(claim, dict) or claim.get("status") != "running":
        raise ValueError("test evaluation claim is not in the running state")
    return claim


def _finish_test_evaluation_claim(claim_path: Path, result: dict[str, object]) -> None:
    claim = _read_running_test_evaluation_claim(claim_path)
    for key in ("training_run_config_sha256", "checkpoint_sha256"):
        if claim.get(key) != result.get(key):
            raise ValueError(f"test evaluation claim {key} disagrees with result")
    _atomic_json(
        claim_path,
        {
            **claim,
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "evaluation_config_sha256": result["evaluation_config_sha256"],
        },
    )


def _fail_test_evaluation_claim(claim_path: Path, error: BaseException) -> None:
    """Retain the spent one-look token with a truthful terminal failure state."""
    claim = _read_running_test_evaluation_claim(claim_path)
    _atomic_json(
        claim_path,
        {
            **claim,
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )


@contextmanager
def _consume_test_evaluation_claim(
    training_dir: Path,
    identity: dict[str, str],
):
    """Make every post-claim exit terminal while never restoring test access."""
    claim_path = _claim_test_evaluation(training_dir, identity)
    try:
        yield claim_path
    except BaseException as error:
        try:
            _fail_test_evaluation_claim(claim_path, error)
        except BaseException as claim_error:
            error.add_note(
                "test evaluation was claimed, but recording its terminal failure also "
                f"failed: {type(claim_error).__name__}: {claim_error}"
            )
        raise


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")


def _validate_cuda_determinism(device: torch.device) -> None:
    if device.type != "cuda":
        return
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "deterministic CUDA evaluation requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "or :16:8 before Python starts"
        )
    torch.use_deterministic_algorithms(True)


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _forward(model, batch: dict[str, object]):
    return model(
        batch["node_history"],
        batch["economic_prior"],
        batch["regression_weight"],
        prior_available=batch["prior_available"],
        regression_available=batch["regression_available"],
    )


def _split_hash(indices: np.ndarray) -> str:
    from hashlib import sha256

    return sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()


def _validate_run_identity(
    run_config: dict[str, object],
    *,
    panel,
    thresholds,
    retrospective,
    economic_graph_sha256: str,
    economic_prior_path: Path,
    materialization_manifest: dict[str, object],
    split,
    source_root: Path,
) -> np.ndarray:
    if run_config.get("schema_version") != 1:
        raise ValueError("unsupported run_config schema_version")
    if run_config.get("experiment") != "shock_follow_graphmixer_v1":
        raise ValueError("checkpoint is not a shock_follow_graphmixer_v1 run")
    if run_config.get("smoke") is not False:
        raise ValueError("formal test evaluation refuses smoke checkpoints")
    if run_config.get("bar_data_mode") != "retrospective-finalized":
        raise ValueError("run_config bar_data_mode is not retrospective-finalized")
    if run_config.get("research_only") is not True or run_config.get("strict_point_in_time_backtest") is not False:
        raise ValueError("run_config must retain retrospective research-only limitations")
    expected_exact = {
        "node_feature_schema": list(NODE_FEATURE_SCHEMA),
        "pair_feature_schema": list(PAIR_FEATURE_SCHEMA),
        "class_order": list(CLASS_ORDER),
        "node_order": list(panel.node_order),
        "node_order_sha256": node_order_sha256(panel.node_order),
        "bar_panel_materialization_sha256": panel.materialization_sha256,
        "threshold_materialization_sha256": thresholds.materialization_sha256,
        "shock_follow_contract_sha256": shock_follow_contract_sha256(),
        "threshold_contract_sha256": threshold_contract_sha256(),
        "retrospective_contract_sha256": retrospective.sha256,
        "materialization_manifest_sha256": canonical_json_sha256(materialization_manifest),
        "economic_graph_sha256": economic_graph_sha256,
        "economic_prior_sha256": file_sha256(economic_prior_path),
        "regression_channel": "unavailable_missing_measured_available_at_manifest",
    }
    for key, expected in expected_exact.items():
        if run_config.get(key) != expected:
            raise ValueError(f"run_config {key} disagrees with current authoritative inputs")
    if (
        run_config.get("source_layout_schema_version")
        != SOURCE_LAYOUT_SCHEMA_VERSION
    ):
        raise ValueError("run_config source layout schema is not supported")
    if run_config.get("source_sha256") != source_hashes(source_root):
        raise ValueError("current source files disagree with checkpoint source hashes")
    split_counts = run_config.get("split_counts")
    split_hashes = run_config.get("split_index_sha256")
    if not isinstance(split_counts, dict) or not isinstance(split_hashes, dict):
        raise ValueError("run_config must contain split counts and hashes")
    expected_splits = {
        "train": split.train_indices,
        "validation": split.validation_indices,
        "test": split.test_indices,
        "purged": split.purged_indices,
    }
    for name, indices in expected_splits.items():
        if split_counts.get(name) != len(indices) or split_hashes.get(name) != _split_hash(indices):
            raise ValueError(f"run_config {name} split identity disagrees with reconstructed split")
    return split.test_indices


def _per_class_metrics(confusion: np.ndarray) -> tuple[list[dict[str, object]], float | None]:
    rows: list[dict[str, object]] = []
    f1_values: list[float] = []
    for class_id, class_name in enumerate(CLASS_ORDER):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum() - true_positive)
        false_negative = int(confusion[class_id, :].sum() - true_positive)
        predicted = true_positive + false_positive
        support = true_positive + false_negative
        precision = true_positive / predicted if predicted else None
        recall = true_positive / support if support else None
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2.0 * true_positive / f1_denominator if f1_denominator else None
        if f1 is not None:
            f1_values.append(f1)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support": support,
                "predicted_count": predicted,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    macro_f1 = sum(f1_values) / len(f1_values) if len(f1_values) == len(CLASS_ORDER) else None
    return rows, macro_f1


def _validate_best_checkpoint_selection(
    checkpoint: dict[str, object],
    training_status: dict[str, object],
    validation_metrics: dict[str, object],
) -> None:
    checkpoint_loss = checkpoint.get("best_validation_loss")
    status_loss = training_status.get("best_validation_loss")
    checkpoint_best_epoch = checkpoint.get("best_epoch")
    status_best_epoch = training_status.get("best_epoch")
    checkpoint_epoch = checkpoint.get("epoch")
    observed_loss = validation_metrics.get("source_event_balanced_cross_entropy")
    for name, value in (
        ("checkpoint best_validation_loss", checkpoint_loss),
        ("training status best_validation_loss", status_loss),
        ("recomputed validation loss", observed_loss),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if (
        not isinstance(checkpoint_best_epoch, int)
        or isinstance(checkpoint_best_epoch, bool)
        or checkpoint_best_epoch < 1
        or checkpoint_epoch != checkpoint_best_epoch
        or status_best_epoch != checkpoint_best_epoch
    ):
        raise ValueError("best checkpoint epoch and training status best_epoch disagree")
    if not np.isclose(float(checkpoint_loss), float(status_loss), rtol=0.0, atol=0.0):
        raise ValueError("checkpoint and training status best_validation_loss disagree")
    if not np.isclose(
        float(observed_loss),
        float(checkpoint_loss),
        rtol=1e-7,
        atol=1e-8,
    ):
        raise ValueError("recomputed full validation loss disagrees with best checkpoint")


def evaluate_model(model, loader: DataLoader, device: torch.device) -> dict[str, object]:
    """Compute only mask-valid three-class test metrics."""
    model.eval()
    confusion = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    cell_loss_sum = 0.0
    event_loss_sum = 0.0
    source_events = 0
    valid_pairs = 0
    correct_pairs = 0
    probability_sum_max_error = 0.0
    derived_follow_sum = 0.0
    empty_batches = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            output = _forward(model, batch)
            raw_mask = batch["loss_mask"]
            if raw_mask.dtype != torch.bool:
                raise TypeError("loss_mask must be boolean")
            diagonal = torch.eye(
                raw_mask.shape[1], dtype=torch.bool, device=raw_mask.device
            ).unsqueeze(0)
            mask = raw_mask & ~diagonal
            outcome = source_event_balanced_masked_cross_entropy(
                output.edge_logits,
                batch["target_class"],
                mask,
            )
            if outcome.is_empty:
                empty_batches += 1
                continue
            logits = output.edge_logits[mask]
            targets = batch["target_class"][mask].long()
            if not torch.isfinite(logits).all() or not torch.isfinite(outcome.loss):
                raise FloatingPointError("evaluation produced non-finite logits or event loss")
            probabilities = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("evaluation produced non-finite probabilities")
            probability_error = float(
                (probabilities.sum(dim=-1) - 1.0).abs().max().item()
            )
            if not np.isfinite(probability_error):
                raise FloatingPointError("evaluation probability-sum check is non-finite")
            probability_sum_max_error = max(
                probability_sum_max_error,
                probability_error,
            )
            predictions = probabilities.argmax(dim=-1)
            encoded = (targets * len(CLASS_ORDER) + predictions).detach().cpu().numpy()
            confusion += np.bincount(encoded, minlength=9).reshape(3, 3)
            cell_losses = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
            if not torch.isfinite(cell_losses):
                raise FloatingPointError("evaluation produced a non-finite cell loss")
            cell_loss_sum += float(cell_losses.item())
            event_loss_sum += float(outcome.loss.item()) * outcome.source_event_count
            source_events += outcome.source_event_count
            valid_pairs += int(targets.numel())
            correct_pairs += int((predictions == targets).sum().item())
            derived_follow_sum += float(probabilities[:, 1:].sum(dim=-1).sum().item())
    if source_events == 0 or valid_pairs == 0:
        raise ValueError("test split contains no evaluable source events")
    per_class, macro_f1 = _per_class_metrics(confusion)
    source_event_loss = event_loss_sum / source_events
    cell_weighted_loss = cell_loss_sum / valid_pairs
    accuracy = correct_pairs / valid_pairs
    mean_follow_probability = derived_follow_sum / valid_pairs
    finite_metrics = (
        source_event_loss,
        cell_weighted_loss,
        accuracy,
        probability_sum_max_error,
        mean_follow_probability,
    )
    if not all(np.isfinite(value) for value in finite_metrics):
        raise FloatingPointError("evaluation produced non-finite aggregate metrics")
    return {
        "source_event_balanced_cross_entropy": source_event_loss,
        "cell_weighted_cross_entropy": cell_weighted_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_rows": "true_class",
        "confusion_matrix_columns": "predicted_class",
        "class_order": list(CLASS_ORDER),
        "source_event_count": source_events,
        "valid_pair_count": valid_pairs,
        "correct_pair_count": correct_pairs,
        "empty_batch_count": empty_batches,
        "probability_sum_max_abs_error": probability_sum_max_error,
        "mean_derived_follow_probability": mean_follow_probability,
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    device = _resolve_device(args.device)
    _validate_cuda_determinism(device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args._output_dir_claimed = True
    status_path = args.output_dir / "run_status.json"
    _atomic_json(status_path, {"status": "running", "started_at_utc": _utc_now()})

    run_config = read_run_config(args.training_dir / "run_config.json")
    training_status_path = args.training_dir / "run_status.json"
    if not training_status_path.is_file():
        raise FileNotFoundError(f"training run_status.json was not found: {training_status_path}")
    training_status = json.loads(training_status_path.read_text(encoding="utf-8"))
    if not isinstance(training_status, dict) or training_status.get("status") != "completed":
        raise ValueError("training run_status.json must be completed")
    run_config_hash = canonical_json_sha256(run_config)
    if training_status.get("run_config_sha256") != run_config_hash:
        raise ValueError("training status run_config hash disagrees with run_config.json")
    checkpoint_path = args.checkpoint or args.training_dir / "best.pt"
    if checkpoint_path.resolve() != (args.training_dir / "best.pt").resolve():
        raise ValueError("V1 evaluation only accepts the training run's best.pt checkpoint")
    expected_checkpoint_hash = training_status.get("best_checkpoint_sha256")
    if not isinstance(expected_checkpoint_hash, str):
        raise ValueError("training status does not contain best_checkpoint_sha256")
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_run_config=run_config,
        expected_sha256=expected_checkpoint_hash,
    )
    retrospective = load_retrospective_bar_data_contract(
        args.retrospective_contract,
        project_root=args.source_root,
    )
    panel = load_shock_follow_bar_panel(
        args.data_dir / "bar_panel.npz", args.data_dir / "bar_panel_manifest.json"
    )
    thresholds = load_monthly_threshold_table(
        args.data_dir / "monthly_thresholds.npz",
        args.data_dir / "monthly_thresholds_manifest.json",
    )
    if panel.node_order != thresholds.node_order:
        raise ValueError("panel and threshold node orders disagree")
    economic, economic_graph_sha256 = load_economic_prior_graph_snapshot(
        panel.node_order,
        args.economic_graph,
    )
    prior = economic.adjacency.astype(np.float32, copy=False)
    economic_prior_path = args.data_dir / "economic_prior.npy"
    persisted_prior = np.load(economic_prior_path, allow_pickle=False)
    if not np.array_equal(prior, persisted_prior):
        raise ValueError("persisted economic prior disagrees with authoritative HTML graph")
    materialization_manifest = load_and_validate_materialization_bundle(
        args.data_dir,
        panel=panel,
        thresholds=thresholds,
        retrospective=retrospective,
        economic_prior=persisted_prior,
        economic_graph_sha256=economic_graph_sha256,
        source_root=args.source_root,
    )

    training_config = run_config.get("training")
    if not isinstance(training_config, dict):
        raise ValueError("run_config.training must be an object")
    prepared_data = prepare_shock_follow_data(
        panel,
        thresholds,
        sequence_length=int(training_config["sequence_length"]),
    )
    cutoffs = eligible_cutoff_indices(
        panel,
        prepared_data.input_valid,
        sequence_length=int(training_config["sequence_length"]),
        require_all_nodes_history=False,
        common_cutoffs=prepared_data.common_cutoffs,
    )
    split = chronological_split_by_target(
        panel.timestamps,
        cutoffs,
        train_end=training_config["train_end"],
        validation_end=training_config["validation_end"],
        test_end=training_config["test_end"],
    )
    test_indices = _validate_run_identity(
        run_config,
        panel=panel,
        thresholds=thresholds,
        retrospective=retrospective,
        economic_graph_sha256=economic_graph_sha256,
        economic_prior_path=economic_prior_path,
        materialization_manifest=materialization_manifest,
        split=split,
        source_root=args.source_root,
    )
    validation_dataset = ShockFollowCutoffDataset(
        panel,
        thresholds,
        prior,
        split.validation_indices,
        sequence_length=int(training_config["sequence_length"]),
        prepared=prepared_data,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_shock_follow_samples,
        pin_memory=device.type == "cuda",
    )
    model = model_from_checkpoint(checkpoint, device=device)
    validation_metrics = evaluate_model(model, validation_loader, device)
    _validate_best_checkpoint_selection(
        checkpoint,
        training_status,
        validation_metrics,
    )
    claim_identity = {
        "training_run_config_sha256": run_config_hash,
        "checkpoint_sha256": expected_checkpoint_hash,
        "evaluation_output_directory": str(args.output_dir.resolve()),
    }
    with _consume_test_evaluation_claim(
        args.training_dir,
        claim_identity,
    ) as claim_path:
        dataset = ShockFollowCutoffDataset(
            panel,
            thresholds,
            prior,
            test_indices,
            sequence_length=int(training_config["sequence_length"]),
            prepared=prepared_data,
        )
        supervision = summarize_supervision(dataset)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_shock_follow_samples,
            pin_memory=device.type == "cuda",
        )
        metrics = evaluate_model(model, loader, device)
        if metrics["valid_pair_count"] != supervision["valid_pair_count"]:
            raise ValueError(
                "evaluation coverage disagrees with independently summarized supervision"
            )

        evaluation_config = {
            "schema_version": 1,
            "experiment": "shock_follow_graphmixer_v1_test_evaluation",
            "research_only": True,
            "strict_point_in_time_backtest": False,
            "training_run_config_sha256": run_config_hash,
            "training_directory": str(args.training_dir.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": expected_checkpoint_hash,
            "split": "test",
            "validation_loss_recomputed": validation_metrics[
                "source_event_balanced_cross_entropy"
            ],
            "validation_split_index_sha256": run_config["split_index_sha256"][
                "validation"
            ],
            "test_split_index_sha256": run_config["split_index_sha256"]["test"],
            "requested_device": args.device,
            "resolved_device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }
        _atomic_json(args.output_dir / "evaluation_config.json", evaluation_config)
        _atomic_json(args.output_dir / "test_supervision_summary.json", supervision)
        _atomic_json(args.output_dir / "test_metrics.json", metrics)
        result = {
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "research_only": True,
            "strict_point_in_time_backtest": False,
            "training_run_config_sha256": run_config_hash,
            "checkpoint_sha256": expected_checkpoint_hash,
            "evaluation_config_sha256": canonical_json_sha256(evaluation_config),
            "test_evaluated_once": True,
            "test_metrics_file": "test_metrics.json",
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "torch_version": torch.__version__,
        }
        _atomic_json(status_path, result)
        _finish_test_evaluation_claim(claim_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args._output_dir_claimed = False
    try:
        result = evaluate(args)
    except Exception as error:
        if args._output_dir_claimed and args.output_dir.is_dir():
            _atomic_json(
                args.output_dir / "run_status.json",
                {
                    "status": "failed",
                    "failed_at_utc": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    print(
        f"Evaluation completed: checkpoint_sha256={result['checkpoint_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
