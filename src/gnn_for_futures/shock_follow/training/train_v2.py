"""Train one approved hierarchical shock-follow GraphMixer V2 run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from gnn_for_futures.shock_follow.approval import (
    APPROVED_TRAINING_CLASS_COUNTS,
    APPROVED_TRAINING_VALID_PAIR_COUNT,
    V2_APPROVAL_DATE,
)
from gnn_for_futures.shock_follow.config import (
    LAMBDA_DIRECTION,
    LAMBDA_FOLLOW,
    ModelConfig,
    TrainingConfig,
)
from gnn_for_futures.shock_follow.data.availability import load_retrospective_bar_data_contract
from gnn_for_futures.graph.regression import load_economic_prior_graph_snapshot
from gnn_for_futures.shock_follow.data.artifacts import (
    load_and_validate_materialization_bundle,
    require_unchanged_materialization_identity,
)
from gnn_for_futures.shock_follow.training.checkpoint_v2 import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    model_from_checkpoint,
)
from gnn_for_futures.shock_follow.contract import (
    CLASS_ORDER,
    canonical_json_sha256,
    node_order_sha256,
    shock_follow_contract_sha256,
    threshold_contract_sha256,
)
from gnn_for_futures.shock_follow.data.panel import load_shock_follow_bar_panel
from gnn_for_futures.shock_follow.graph import PAIR_FEATURE_SCHEMA
from gnn_for_futures.shock_follow.models.head_v2 import (
    DIRECTION_CLASS_ORDER,
    FOLLOW_CLASS_ORDER,
    balanced_follow_class_weights,
    hierarchical_source_event_loss,
)
from gnn_for_futures.shock_follow.evaluation.metrics_v2 import V2MetricsAccumulator
from gnn_for_futures.shock_follow.models.mixer_v2 import ShockFollowGraphMixerV2
from gnn_for_futures.shock_follow.data.splits_v2 import (
    CONFIRMATION_END,
    CONFIRMATION_START,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    PROHIBITED_TEST_END,
    PROHIBITED_TEST_START,
    TRAIN_END,
    shock_follow_v2_split_by_target,
    split_manifest_payload,
)
from gnn_for_futures.shock_follow.data.thresholds import load_monthly_threshold_table
from gnn_for_futures.shock_follow.data.prepared import (
    DEFAULT_SEQUENCE_LENGTH,
    NODE_FEATURE_SCHEMA,
    ShockFollowCutoffDataset,
    collate_shock_follow_samples,
    eligible_cutoff_indices,
    prepare_shock_follow_data,
    summarize_supervision,
)
from gnn_for_futures.shock_follow.provenance.source import (
    DEFAULT_RELEASE_SOURCE_MANIFEST,
    build_source_manifest,
    file_sha256,
    require_unchanged_release_source_identity,
    require_unchanged_source_hashes,
    resolve_source_root,
    validate_release_source_identity,
)


DEFAULT_SOURCE_ROOT = Path(".")
DEFAULT_DATA_DIR = Path("data/shock_follow_v1")
DEFAULT_OUTPUT_DIR = Path("results/shock_follow_graphmixer_v2")
DEFAULT_RETROSPECTIVE_CONTRACT = Path(
    "data/contracts/bar_availability.retrospective.json"
)
DEFAULT_ECONOMIC_GRAPH = Path(
    "data/graph_data/futures_graph_smm_simplified.html"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the approved retrospective shock-follow GraphMixer V2."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root containing the released src/gnn_for_futures source tree.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--retrospective-contract",
        type=Path,
        default=DEFAULT_RETROSPECTIVE_CONTRACT,
    )
    parser.add_argument("--economic-graph", type=Path, default=DEFAULT_ECONOMIC_GRAPH)
    parser.add_argument(
        "--release-source-manifest",
        "--transfer-manifest",
        dest="release_source_manifest",
        type=Path,
        default=DEFAULT_RELEASE_SOURCE_MANIFEST,
        help=(
            "Schema-2 source-plus-packaging release manifest; --transfer-manifest is a"
            "deprecated compatibility alias."
        ),
    )
    parser.add_argument(
        "--expected-release-source-manifest-sha256",
        "--expected-transfer-manifest-sha256",
        dest="expected_release_source_manifest_sha256",
        help=(
            "Independently approved lowercase byte SHA-256 for the release-source "
            "manifest; the transfer-named flag is a deprecated alias."
        ),
    )
    parser.add_argument(
        "--expected-materialization-manifest-sha256",
        help=(
            "Independently approved lowercase byte SHA-256 for "
            "materialization_manifest.json."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-samples", type=int, default=64)
    parser.add_argument("--smoke-development-samples", type=int, default=32)
    args = parser.parse_args(argv)
    if args.max_epochs is None:
        args.max_epochs = 3 if args.smoke else 40
    return args


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(message.rstrip("\n") + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.max_epochs < 1 or args.patience < 1:
        raise ValueError("batch-size, max-epochs, and patience must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning-rate must be positive and weight-decay nonnegative")
    if args.sequence_length < 1 or args.num_workers < 0:
        raise ValueError("sequence-length must be positive and num-workers nonnegative")
    expected_release_hash = args.expected_release_source_manifest_sha256
    if (
        not isinstance(expected_release_hash, str)
        or len(expected_release_hash) != 64
        or expected_release_hash != expected_release_hash.lower()
    ):
        raise ValueError(
            "--expected-release-source-manifest-sha256 is required and must be "
            "one lowercase SHA-256"
        )
    try:
        int(expected_release_hash, 16)
    except ValueError as error:
        raise ValueError(
            "--expected-release-source-manifest-sha256 is required and must be "
            "one lowercase SHA-256"
        ) from error
    expected_materialization_hash = (
        args.expected_materialization_manifest_sha256
    )
    if (
        not isinstance(expected_materialization_hash, str)
        or len(expected_materialization_hash) != 64
        or expected_materialization_hash != expected_materialization_hash.lower()
    ):
        raise ValueError(
            "--expected-materialization-manifest-sha256 is required and must be "
            "one lowercase SHA-256"
        )
    try:
        int(expected_materialization_hash, 16)
    except ValueError as error:
        raise ValueError(
            "--expected-materialization-manifest-sha256 is required and must be "
            "one lowercase SHA-256"
        ) from error
    if args.smoke_train_samples < 1 or args.smoke_development_samples < 1:
        raise ValueError("smoke sample limits must be positive")
    if args.smoke and (
        args.max_epochs > 3
        or args.smoke_train_samples > 64
        or args.smoke_development_samples > 32
    ):
        raise ValueError(
            "smoke is limited to at most 3 epochs, 64 training samples, "
            "and 32 development samples"
        )
    if not args.smoke and (
        args.batch_size != 16
        or args.learning_rate != 1e-3
        or args.weight_decay != 1e-5
        or args.max_epochs != 40
        or args.patience != 7
        or args.seed != 20261002
        or args.sequence_length != DEFAULT_SEQUENCE_LENGTH
    ):
        raise ValueError("formal V2 training parameters are frozen by approval")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _validate_cuda_determinism(device: torch.device) -> None:
    if device.type != "cuda":
        return
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "deterministic CUDA training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "or :16:8 before Python starts"
        )


def _git_facts(source_root: str | Path | None = None) -> dict[str, object]:
    root = resolve_source_root(source_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        return {
            "commit": commit,
            "dirty": bool(status.strip()),
            "status_sha256": sha256(status.encode("utf-8")).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status_sha256": None}


def _limit_evenly(indices: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices.copy()
    positions = np.linspace(0, len(indices) - 1, num=limit, dtype=np.int64)
    return indices[positions]


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: (
            value.to(device, non_blocking=device.type == "cuda")
            if isinstance(value, Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _forward(model: ShockFollowGraphMixerV2, batch: dict[str, object]):
    return model(
        batch["node_history"],
        batch["economic_prior"],
        batch["regression_weight"],
        prior_available=batch["prior_available"],
        regression_available=batch["regression_available"],
    )


def _event_sums(
    cell_losses: Tensor,
    mask: Tensor,
) -> tuple[float, int]:
    counts = mask.sum(dim=-1)
    events = counts > 0
    event_count = int(events.sum().item())
    if not event_count:
        return 0.0, 0
    per_cell = torch.zeros_like(mask, dtype=cell_losses.dtype)
    per_cell[mask] = cell_losses
    per_source = per_cell.sum(dim=-1) / counts.clamp_min(1)
    return float(per_source[events].sum().item()), event_count


def run_epoch(
    model: ShockFollowGraphMixerV2,
    loader: DataLoader,
    device: torch.device,
    optimizer: Adam | None,
    follow_class_weights: Tensor,
) -> dict[str, object]:
    """Run one V2 epoch and aggregate losses by their defining events/cells."""
    training = optimizer is not None
    model.train(training)
    weights = follow_class_weights.to(device=device)
    weighted_follow_event_sum = 0.0
    unweighted_follow_event_sum = 0.0
    direction_event_sum = 0.0
    follow_events = 0
    direction_events = 0
    weighted_follow_cell_sum = 0.0
    unweighted_follow_cell_sum = 0.0
    direction_cell_sum = 0.0
    valid_pairs = 0
    direction_pairs = 0
    observed_batches = 0
    nonempty_batches = 0
    follow_empty_batches = 0
    direction_empty_batches = 0
    optimizer_steps = 0
    sample_count = 0
    edges = 0
    possible_edges = 0
    isolated_nodes = 0
    padded_out_slots = 0
    out_slots = 0
    saturation_sum = 0.0
    probability_sum_error = 0.0
    joint_follow_error = 0.0
    task_metrics = V2MetricsAccumulator(
        follow_class_weights,
        lambda_follow=LAMBDA_FOLLOW,
        lambda_direction=LAMBDA_DIRECTION,
    )

    for raw_batch in loader:
        observed_batches += 1
        batch = _move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = _forward(model, batch)
            raw_mask = batch["loss_mask"]
            if raw_mask.dtype != torch.bool:
                raise TypeError("loss_mask must be boolean")
            diagonal = torch.eye(
                raw_mask.shape[1], dtype=torch.bool, device=raw_mask.device
            ).unsqueeze(0)
            mask = raw_mask & ~diagonal
            outcome = hierarchical_source_event_loss(
                output.follow_logits,
                output.direction_logits,
                batch["target_class"],
                mask,
                weights,
                lambda_follow=LAMBDA_FOLLOW,
                lambda_direction=LAMBDA_DIRECTION,
            )
            task_metrics.update(
                output.follow_logits,
                output.direction_logits,
                batch["target_class"],
                mask,
                joint_scores=output.joint_scores,
            )
            adjacency = output.fused_graph.adjacency
            current_batch_size, node_count, _ = adjacency.shape
            sample_count += current_batch_size
            edges += int(adjacency.sum().item())
            possible_edges += current_batch_size * node_count * (node_count - 1)
            isolated_nodes += int((~adjacency.any(dim=-1)).sum().item())
            out_slots += output.directed_views.out_valid.numel()
            padded_out_slots += int((~output.directed_views.out_valid).sum().item())
            saturation_sum += float(output.attention.saturation_rate.sum().item())
            if not torch.isfinite(output.follow_logits).all() or not torch.isfinite(
                output.direction_logits
            ).all() or not torch.isfinite(output.joint_scores).all():
                raise FloatingPointError("encountered non-finite V2 model output")
            probability_sum_error = max(
                probability_sum_error,
                float((output.joint_scores.sum(dim=-1) - 1.0).abs().max().item()),
            )
            p_follow = torch.softmax(output.follow_logits, dim=-1)[..., 1]
            joint_follow_error = max(
                joint_follow_error,
                float((output.joint_scores[..., 1:].sum(dim=-1) - p_follow).abs().max().item()),
            )
            if outcome.is_empty:
                follow_empty_batches += 1
                direction_empty_batches += 1
                continue
            nonempty_batches += 1
            if outcome.direction_is_empty:
                direction_empty_batches += 1
            if outcome.total_loss is None or not torch.isfinite(outcome.total_loss):
                raise FloatingPointError("encountered non-finite V2 training loss")
            if training:
                outcome.total_loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("encountered non-finite V2 gradient")
                optimizer.step()
                optimizer_steps += 1

        targets = batch["target_class"]
        valid_targets = targets[mask]
        follow_targets = valid_targets.ne(0).long()
        weighted_cells = torch.nn.functional.cross_entropy(
            output.follow_logits[mask],
            follow_targets,
            weight=weights.to(dtype=output.follow_logits.dtype),
            reduction="none",
        )
        unweighted_cells = torch.nn.functional.cross_entropy(
            output.follow_logits[mask],
            follow_targets,
            reduction="none",
        )
        weighted_event_sum, event_count = _event_sums(weighted_cells, mask)
        unweighted_event_sum, unweighted_event_count = _event_sums(
            unweighted_cells, mask
        )
        if event_count != outcome.follow_event_count or event_count != unweighted_event_count:
            raise AssertionError("follow event aggregation count mismatch")
        weighted_follow_event_sum += weighted_event_sum
        unweighted_follow_event_sum += unweighted_event_sum
        follow_events += event_count
        weighted_follow_cell_sum += float(weighted_cells.sum().item())
        unweighted_follow_cell_sum += float(unweighted_cells.sum().item())
        valid_pairs += int(mask.sum().item())

        direction_mask = mask & targets.ne(0)
        if direction_mask.any():
            direction_targets = targets[direction_mask] - 1
            direction_cells = torch.nn.functional.cross_entropy(
                output.direction_logits[direction_mask],
                direction_targets.long(),
                reduction="none",
            )
            event_sum, event_count = _event_sums(direction_cells, direction_mask)
            if event_count != outcome.direction_event_count:
                raise AssertionError("direction event aggregation count mismatch")
            direction_event_sum += event_sum
            direction_events += event_count
            direction_cell_sum += float(direction_cells.sum().item())
            direction_pairs += int(direction_mask.sum().item())

    if follow_events:
        weighted_follow_loss = weighted_follow_event_sum / follow_events
        unweighted_follow_loss = unweighted_follow_event_sum / follow_events
        direction_loss = (
            direction_event_sum / direction_events if direction_events else None
        )
        total_loss = weighted_follow_loss + (direction_loss or 0.0)
    else:
        weighted_follow_loss = None
        unweighted_follow_loss = None
        direction_loss = None
        total_loss = None
    finite_values = [
        value
        for value in (
            weighted_follow_loss,
            unweighted_follow_loss,
            direction_loss,
            total_loss,
        )
        if value is not None
    ]
    if not all(np.isfinite(value) for value in finite_values):
        raise FloatingPointError("epoch produced non-finite aggregate V2 losses")
    classification_metrics = task_metrics.finalize()
    accumulator_loss = classification_metrics["total_loss"]
    if total_loss is None:
        if accumulator_loss is not None:
            raise AssertionError("empty epoch loss disagrees with V2 metrics accumulator")
    elif not isinstance(accumulator_loss, float) or not np.isclose(
        total_loss,
        accumulator_loss,
        rtol=1e-7,
        atol=1e-8,
    ):
        raise AssertionError("epoch loss disagrees with V2 metrics accumulator")
    return {
        "total_loss": total_loss,
        "classification": classification_metrics,
        "weighted_follow_source_event_loss": weighted_follow_loss,
        "unweighted_follow_source_event_loss": unweighted_follow_loss,
        "weighted_follow_cell_loss": (
            weighted_follow_cell_sum / valid_pairs if valid_pairs else None
        ),
        "unweighted_follow_cell_loss": (
            unweighted_follow_cell_sum / valid_pairs if valid_pairs else None
        ),
        "direction_source_event_loss": direction_loss,
        "direction_cell_loss": (
            direction_cell_sum / direction_pairs if direction_pairs else None
        ),
        "follow_source_event_count": follow_events,
        "direction_source_event_count": direction_events,
        "valid_pair_count": valid_pairs,
        "direction_pair_count": direction_pairs,
        "observed_batch_count": observed_batches,
        "nonempty_batch_count": nonempty_batches,
        "follow_empty_batch_count": follow_empty_batches,
        "direction_empty_batch_count": direction_empty_batches,
        "optimizer_step_count": optimizer_steps,
        "mean_edges_per_sample": edges / sample_count if sample_count else 0.0,
        "edge_density": edges / possible_edges if possible_edges else 0.0,
        "mean_out_isolated_nodes_per_sample": (
            isolated_nodes / sample_count if sample_count else 0.0
        ),
        "out_slot_padding_fraction": (
            padded_out_slots / out_slots if out_slots else 0.0
        ),
        "mean_attention_saturation_rate": (
            saturation_sum / sample_count if sample_count else 0.0
        ),
        "joint_score_sum_max_abs_error": probability_sum_error,
        "joint_follow_identity_max_abs_error": joint_follow_error,
    }


def _checkpoint_payload(
    *,
    model: ShockFollowGraphMixerV2,
    optimizer: Adam,
    data_loader_generator: torch.Generator,
    epoch: int,
    best_epoch: int,
    best_development_loss: float,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    run_config: dict[str, object],
    run_config_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_development_loss": best_development_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "run_config": run_config,
        "run_config_sha256": run_config_sha256,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "data_loader_generator": data_loader_generator.get_state(),
        },
    }


def _class_weight_audit(
    training_summary: dict[str, object],
) -> tuple[dict[str, object], Tensor]:
    raw_counts = training_summary.get("class_counts")
    if not isinstance(raw_counts, list) or len(raw_counts) != len(CLASS_ORDER):
        raise ValueError("training supervision must contain exactly three class counts")
    counts = tuple(int(value) for value in raw_counts)
    if list(counts) != raw_counts:
        raise ValueError("training class counts must be integers")
    weights = balanced_follow_class_weights(counts)
    payload = {
        "source_split": "training_only",
        "development_counts_accessed": False,
        "confirmation_counts_accessed": False,
        "prohibited_v1_test_counts_accessed": False,
        "three_class_counts": list(counts),
        "no_follow_count": counts[0],
        "follow_count": counts[1] + counts[2],
        "follow_class_order": list(FOLLOW_CLASS_ORDER),
        "follow_class_weights": list(weights),
        "formula": "N_total/(2*N_class)",
    }
    payload["audit_sha256"] = canonical_json_sha256(payload)
    return payload, torch.tensor(weights, dtype=torch.float32)


def train(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    _validate_cuda_determinism(device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args._output_dir_claimed = True
    status_path = args.output_dir / "run_status.json"
    log_path = args.output_dir / "training.log"
    started_at = _utc_now()
    _atomic_json(status_path, {"status": "running", "started_at_utc": started_at})
    _append_log(
        log_path,
        f"{started_at} status=running smoke={bool(args.smoke)} device={device}",
    )

    source_root = resolve_source_root(getattr(args, "source_root", None))
    source_manifest = build_source_manifest(source_root)
    source_hashes = source_manifest["source_sha256"]
    if not isinstance(source_hashes, dict):
        raise AssertionError("schema-2 source hash map changed type")
    release_source_identity = validate_release_source_identity(
        args.release_source_manifest,
        args.expected_release_source_manifest_sha256,
        source_root=source_root,
    )
    retrospective = load_retrospective_bar_data_contract(
        args.retrospective_contract, project_root=source_root
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
    expected_prior = np.load(args.data_dir / "economic_prior.npy", allow_pickle=False)
    if not np.array_equal(prior, expected_prior):
        raise ValueError("persisted economic prior disagrees with authoritative HTML graph")
    materialization_manifest = load_and_validate_materialization_bundle(
        args.data_dir,
        panel=panel,
        thresholds=thresholds,
        retrospective=retrospective,
        economic_prior=expected_prior,
        economic_graph_sha256=economic_graph_sha256,
        source_root=source_root,
        expected_materialization_manifest_sha256=(
            args.expected_materialization_manifest_sha256
        ),
    )

    prepared_data = prepare_shock_follow_data(
        panel, thresholds, sequence_length=args.sequence_length
    )
    cutoffs = eligible_cutoff_indices(
        panel,
        prepared_data.input_valid,
        sequence_length=args.sequence_length,
        require_all_nodes_history=False,
        common_cutoffs=prepared_data.common_cutoffs,
    )
    split = shock_follow_v2_split_by_target(panel.timestamps, cutoffs)
    full_split_manifest = split_manifest_payload(split)
    full_training_indices = split.training_indices
    full_development_indices = split.development_indices
    training_indices = full_training_indices
    development_indices = full_development_indices
    if args.smoke:
        training_indices = _limit_evenly(full_training_indices, args.smoke_train_samples)
        development_indices = _limit_evenly(
            full_development_indices, args.smoke_development_samples
        )
        max_epochs = min(args.max_epochs, 3)
        patience = min(args.patience, 3)
    else:
        max_epochs = args.max_epochs
        patience = args.patience
    if min(len(training_indices), len(development_indices)) < 1:
        raise ValueError("training and development must each contain at least one cutoff")

    split_manifest = {
        **full_split_manifest,
        "smoke_limited": bool(args.smoke),
        "effective_training_count": len(training_indices),
        "effective_training_index_sha256": sha256(
            np.ascontiguousarray(training_indices).tobytes()
        ).hexdigest(),
        "effective_development_count": len(development_indices),
        "effective_development_index_sha256": sha256(
            np.ascontiguousarray(development_indices).tobytes()
        ).hexdigest(),
        "training_entrypoint_dataset_construction": ["training", "development"],
        "confirmation_dataset_constructed": False,
        "prohibited_v1_test_dataset_constructed": False,
    }
    _atomic_json(args.output_dir / "split_manifest.json", split_manifest)

    full_training_dataset = ShockFollowCutoffDataset(
        panel,
        thresholds,
        prior,
        full_training_indices,
        sequence_length=args.sequence_length,
        prepared=prepared_data,
    )
    training_summary = summarize_supervision(full_training_dataset)
    if tuple(training_summary.get("class_counts", ())) != APPROVED_TRAINING_CLASS_COUNTS:
        raise ValueError(
            "full training class counts disagree with the approved V2 identity: "
            f"observed={training_summary.get('class_counts')!r}, "
            f"expected={list(APPROVED_TRAINING_CLASS_COUNTS)!r}"
        )
    if training_summary.get("valid_pair_count") != APPROVED_TRAINING_VALID_PAIR_COUNT:
        raise ValueError(
            "full training valid-pair count disagrees with the approved V2 identity: "
            f"observed={training_summary.get('valid_pair_count')!r}, "
            f"expected={APPROVED_TRAINING_VALID_PAIR_COUNT}"
        )
    weight_audit, follow_class_weights = _class_weight_audit(training_summary)
    training_dataset = (
        ShockFollowCutoffDataset(
            panel,
            thresholds,
            prior,
            training_indices,
            sequence_length=args.sequence_length,
            prepared=prepared_data,
        )
        if args.smoke
        else full_training_dataset
    )
    development_dataset = ShockFollowCutoffDataset(
        panel,
        thresholds,
        prior,
        development_indices,
        sequence_length=args.sequence_length,
        prepared=prepared_data,
    )
    development_summary = summarize_supervision(development_dataset)
    if training_summary["valid_pair_count"] < 1 or development_summary["valid_pair_count"] < 1:
        raise ValueError("training and development must each contain supervision")
    _atomic_json(
        args.output_dir / "training_supervision_summary.json", training_summary
    )
    _atomic_json(
        args.output_dir / "development_supervision_summary.json", development_summary
    )
    _atomic_json(args.output_dir / "class_weight_audit.json", weight_audit)

    model_config = ModelConfig(
        attention_history_length=args.sequence_length,
        expected_num_futures=len(panel.node_order),
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        seed=args.seed,
        sequence_length=args.sequence_length,
        train_end=TRAIN_END,
        development_start=DEVELOPMENT_START,
        development_end=DEVELOPMENT_END,
        confirmation_start=CONFIRMATION_START,
        confirmation_end=CONFIRMATION_END,
        prohibited_test_start=PROHIBITED_TEST_START,
        prohibited_test_end=PROHIBITED_TEST_END,
        lambda_follow=LAMBDA_FOLLOW,
        lambda_direction=LAMBDA_DIRECTION,
    )
    _atomic_json(
        args.output_dir / "source_manifest_verification.json", source_manifest
    )
    run_config = {
        "schema_version": 3,
        "experiment": "shock_follow_graphmixer_v2_hierarchical",
        "approval_date": V2_APPROVAL_DATE,
        "source_layout_schema_version": source_manifest[
            "source_layout_schema_version"
        ],
        "smoke": bool(args.smoke),
        "bar_data_mode": "retrospective-finalized",
        "research_only": True,
        "strict_point_in_time_backtest": False,
        "regression_channel": "unavailable_missing_measured_available_at_manifest",
        "model": asdict(model_config),
        "training": asdict(training_config),
        "node_feature_schema": list(NODE_FEATURE_SCHEMA),
        "pair_feature_schema": list(PAIR_FEATURE_SCHEMA),
        "class_order": list(CLASS_ORDER),
        "joint_class_order": list(CLASS_ORDER),
        "head_schema_version": "hierarchical_follow_direction_v2",
        "follow_class_order": list(FOLLOW_CLASS_ORDER),
        "direction_class_order": list(DIRECTION_CLASS_ORDER),
        "follow_threshold": 0.5,
        "tie_policy": "no_follow",
        "follow_class_counts": [
            weight_audit["no_follow_count"],
            weight_audit["follow_count"],
        ],
        "follow_class_weights": weight_audit["follow_class_weights"],
        "follow_class_weight_audit": weight_audit,
        "direction_mask_source": "true_target_class",
        "loss_reduction_contract": "cell_cross_entropy_then_source_event_mean",
        "warm_start": False,
        "old_frozen_test_prohibited": True,
        "node_order": list(panel.node_order),
        "node_order_sha256": node_order_sha256(panel.node_order),
        "bar_panel_materialization_sha256": panel.materialization_sha256,
        "threshold_materialization_sha256": thresholds.materialization_sha256,
        "shock_follow_contract_sha256": shock_follow_contract_sha256(),
        "threshold_contract_sha256": threshold_contract_sha256(),
        "retrospective_contract_sha256": retrospective.sha256,
        "materialization_manifest_sha256": canonical_json_sha256(
            materialization_manifest
        ),
        "materialization_manifest_file_sha256": (
            args.expected_materialization_manifest_sha256
        ),
        **release_source_identity,
        "economic_graph_sha256": economic_graph_sha256,
        "economic_prior_sha256": file_sha256(args.data_dir / "economic_prior.npy"),
        "split_manifest": split_manifest,
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "source_sha256": source_hashes,
        "confirmation_accessed_during_training": False,
        "prohibited_v1_test_accessed_during_training": False,
    }
    run_config_hash = canonical_json_sha256(run_config)
    _atomic_json(args.output_dir / "run_config.json", run_config)
    environment = {
        "created_at_utc": _utc_now(),
        "run_config_sha256": run_config_hash,
        "requested_device": args.device,
        "resolved_device": str(device),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "git": _git_facts(source_root),
    }
    _atomic_json(args.output_dir / "environment.json", environment)

    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_shock_follow_samples,
        "pin_memory": device.type == "cuda",
    }
    training_loader = DataLoader(
        training_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    development_loader = DataLoader(
        development_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    model = ShockFollowGraphMixerV2(**asdict(model_config)).to(device)
    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_development_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, object]] = []
    best_reference_state: dict[str, Tensor] | None = None
    for epoch in range(1, max_epochs + 1):
        training_metrics = run_epoch(
            model,
            training_loader,
            device,
            optimizer,
            follow_class_weights,
        )
        development_metrics = run_epoch(
            model,
            development_loader,
            device,
            None,
            follow_class_weights,
        )
        training_loss = training_metrics["total_loss"]
        development_loss = development_metrics["total_loss"]
        if not isinstance(training_loss, float) or not isinstance(
            development_loss, float
        ):
            raise ValueError("formal epoch must contain nonempty train/development losses")
        improved = development_loss < best_development_loss
        if improved:
            best_development_loss = development_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "training": training_metrics,
            "development": development_metrics,
            "improved": improved,
        }
        epoch_rows.append(row)
        _atomic_json(args.output_dir / "epoch_metrics.json", epoch_rows)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            data_loader_generator=generator,
            epoch=epoch,
            best_epoch=best_epoch,
            best_development_loss=best_development_loss,
            model_config=model_config,
            training_config=training_config,
            run_config=run_config,
            run_config_sha256=run_config_hash,
        )
        _atomic_torch_save(args.output_dir / "last.pt", payload)
        if improved:
            _atomic_torch_save(args.output_dir / "best.pt", payload)
            best_reference_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        epoch_message = (
            f"epoch={epoch:03d} training={training_loss:.6f} "
            f"development={development_loss:.6f} "
            f"follow_events={training_metrics['follow_source_event_count']} "
            f"direction_events={training_metrics['direction_source_event_count']} "
            f"edges/sample={training_metrics['mean_edges_per_sample']:.1f}"
        )
        print(epoch_message, flush=True)
        _append_log(log_path, f"{_utc_now()} {epoch_message} improved={improved}")
        if epochs_without_improvement >= patience:
            break

    best_checkpoint_path = args.output_dir / "best.pt"
    best_checkpoint_hash = file_sha256(best_checkpoint_path)
    reloaded_checkpoint = load_checkpoint(
        best_checkpoint_path,
        expected_run_config=run_config,
        expected_sha256=best_checkpoint_hash,
    )
    reloaded_model = model_from_checkpoint(reloaded_checkpoint, device=device)
    if best_reference_state is None:
        raise AssertionError("training completed without recording best-checkpoint state")
    reloaded_state = {
        name: value.detach().cpu()
        for name, value in reloaded_model.state_dict().items()
    }
    state_exact = reloaded_state.keys() == best_reference_state.keys() and all(
        torch.equal(best_reference_state[name], reloaded_state[name])
        for name in best_reference_state
    )
    if not state_exact:
        raise ValueError("best V2 checkpoint reload did not reproduce model state exactly")
    reload_metrics = run_epoch(
        reloaded_model,
        development_loader,
        device,
        None,
        follow_class_weights,
    )
    reload_loss = reload_metrics["total_loss"]
    if not isinstance(reload_loss, float) or not np.isclose(
        reload_loss,
        best_development_loss,
        rtol=1e-7,
        atol=1e-8,
    ):
        raise ValueError("best checkpoint reload did not reproduce development total loss")
    reload_validation = {
        "best_epoch": best_epoch,
        "checkpoint_best_development_loss": best_development_loss,
        "recomputed_development_total_loss": reload_loss,
        "rtol": 1e-7,
        "atol": 1e-8,
        "state_exact_match": True,
        "loss_match": True,
        "metrics": reload_metrics,
    }
    _atomic_json(
        args.output_dir / "best_checkpoint_reload_validation.json",
        reload_validation,
    )

    require_unchanged_source_hashes(
        source_root,
        source_hashes,
        operation="V2 training",
    )
    require_unchanged_release_source_identity(
        args.release_source_manifest,
        args.expected_release_source_manifest_sha256,
        release_source_identity,
        source_root=source_root,
        operation="V2 training",
    )
    require_unchanged_materialization_identity(
        args.data_dir,
        panel=panel,
        thresholds=thresholds,
        retrospective=retrospective,
        economic_prior=expected_prior,
        economic_graph_sha256=economic_graph_sha256,
        expected_materialization_manifest_sha256=(
            args.expected_materialization_manifest_sha256
        ),
        expected_manifest=materialization_manifest,
        source_root=source_root,
        operation="V2 training",
    )

    result = {
        "status": "completed",
        "completed_at_utc": _utc_now(),
        "run_config_sha256": run_config_hash,
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "best_epoch": best_epoch,
        "best_development_loss": best_development_loss,
        "epochs_completed": len(epoch_rows),
        "confirmation_evaluated_during_training": False,
        "prohibited_v1_test_accessed": False,
        "best_checkpoint_reload_verified": True,
        "best_checkpoint_state_exact_match": True,
        "best_checkpoint_development_loss_match": True,
        "best_checkpoint_sha256": best_checkpoint_hash,
        "last_checkpoint_sha256": file_sha256(args.output_dir / "last.pt"),
    }
    _atomic_json(status_path, result)
    _append_log(
        log_path,
        f"{result['completed_at_utc']} status=completed best_epoch={best_epoch} "
        f"best_development_loss={best_development_loss:.12g}",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args._output_dir_claimed = False
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    try:
        result = train(args)
    except BaseException as error:
        if args._output_dir_claimed and args.output_dir.is_dir():
            failed_at = _utc_now()
            try:
                _atomic_json(
                    args.output_dir / "run_status.json",
                    {
                        "status": "failed",
                        "failed_at_utc": failed_at,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                _append_log(
                    args.output_dir / "training.log",
                    f"{failed_at} status=failed error_type={type(error).__name__} "
                    f"error={str(error)!r}",
                )
            except BaseException as status_error:
                error.add_note(
                    "V2 training output was claimed, but recording terminal failure "
                    f"also failed: {type(status_error).__name__}: {status_error}"
                )
        raise
    print(
        f"V2 training completed: best_epoch={result['best_epoch']} "
        f"best_development_loss={result['best_development_loss']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
