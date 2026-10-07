"""Train the first retrospective shock-follow GraphMixer experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

from gnn_for_futures.shock_follow.data.availability import load_retrospective_bar_data_contract
from gnn_for_futures.graph.regression import load_economic_prior_graph_snapshot
from gnn_for_futures.shock_follow.data.artifacts import load_and_validate_materialization_bundle
from gnn_for_futures.shock_follow.training.checkpoint_v1 import (
    file_sha256,
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
from gnn_for_futures.shock_follow.provenance.source import (
    SOURCE_LAYOUT_SCHEMA_VERSION,
    require_unchanged_source_hashes,
    source_hashes,
)
from gnn_for_futures.shock_follow.graph import PAIR_FEATURE_SCHEMA
from gnn_for_futures.shock_follow.models.head_v1 import source_event_balanced_masked_cross_entropy
from gnn_for_futures.shock_follow.models.mixer_v1 import ShockFollowGraphMixer
from gnn_for_futures.shock_follow.data.thresholds import load_monthly_threshold_table
from gnn_for_futures.shock_follow.data.prepared import (
    DEFAULT_SEQUENCE_LENGTH,
    NODE_FEATURE_SCHEMA,
    ShockFollowCutoffDataset,
    chronological_split_by_target,
    collate_shock_follow_samples,
    eligible_cutoff_indices,
    prepare_shock_follow_data,
    summarize_supervision,
)


DEFAULT_DATA_DIR = Path("data/shock_follow_v1")
DEFAULT_OUTPUT_DIR = Path("results/shock_follow_graphmixer_v1")
DEFAULT_RETROSPECTIVE_CONTRACT = Path(
    "data/contracts/bar_availability.retrospective.json"
)
DEFAULT_ECONOMIC_GRAPH = Path("data/graph_data/futures_graph_smm_simplified.html")
TRAIN_END = "2025-08-31T23:59:59"
VALIDATION_END = "2026-02-28T23:59:59"
TEST_END = "2026-08-25T15:00:00"


@dataclass(frozen=True)
class ModelConfig:
    node_feature_dim: int = len(NODE_FEATURE_SCHEMA)
    temporal_hidden_dim: int = 13
    attention_history_length: int = DEFAULT_SEQUENCE_LENGTH
    expected_num_futures: int = 44
    role_dim: int = 32
    neighbor_capacity: int = 5
    head_hidden_dim: int = 100
    fusion_weights: tuple[float, float, float] = (0.2, 0.4, 0.4)
    topology_threshold: float = 0.1
    attention_robust_quantile: float = 0.75


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    seed: int
    sequence_length: int
    train_end: str
    validation_end: str
    test_end: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train retrospective shock-follow GraphMixer V1.")
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--validation-end", default=VALIDATION_END)
    parser.add_argument("--test-end", default=TEST_END)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-samples", type=int, default=64)
    parser.add_argument("--smoke-validation-samples", type=int, default=32)
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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
    if args.smoke_train_samples < 1 or args.smoke_validation_samples < 1:
        raise ValueError("smoke sample limits must be positive")


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


def _git_facts(project_root: str | Path | None = None) -> dict[str, object]:
    root = Path(project_root) if project_root is not None else Path.cwd()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=root, text=True,
            capture_output=True, check=True,
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
        key: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _forward(model: ShockFollowGraphMixer, batch: dict[str, object]):
    return model(
        batch["node_history"],
        batch["economic_prior"],
        batch["regression_weight"],
        prior_available=batch["prior_available"],
        regression_available=batch["regression_available"],
    )


def run_epoch(
    model: ShockFollowGraphMixer,
    loader: DataLoader,
    device: torch.device,
    optimizer: Adam | None,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    event_count = 0
    observed_batches = 0
    nonempty_batches = 0
    optimizer_steps = 0
    empty_batches = 0
    valid_cells = 0
    sample_count = 0
    edges = 0
    possible_edges = 0
    isolated_nodes = 0
    padded_out_slots = 0
    out_slots = 0
    saturation_sum = 0.0
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
            effective_mask = raw_mask & ~diagonal
            outcome = source_event_balanced_masked_cross_entropy(
                output.edge_logits, batch["target_class"], effective_mask
            )
            valid_cells += int(effective_mask.sum().item())
            adjacency = output.fused_graph.adjacency
            current_batch_size, node_count, _ = adjacency.shape
            sample_count += current_batch_size
            edges += int(adjacency.sum().item())
            possible_edges += current_batch_size * node_count * (node_count - 1)
            isolated_nodes += int((~adjacency.any(dim=-1)).sum().item())
            out_slots += output.directed_views.out_valid.numel()
            padded_out_slots += int((~output.directed_views.out_valid).sum().item())
            saturation_sum += float(output.attention.saturation_rate.sum().item())
            if outcome.is_empty:
                empty_batches += 1
                continue
            nonempty_batches += 1
            if not torch.isfinite(outcome.loss):
                raise FloatingPointError("encountered non-finite training loss")
            if training:
                outcome.loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("encountered non-finite gradient")
                optimizer.step()
                optimizer_steps += 1
        loss_sum += float(outcome.loss.detach()) * outcome.source_event_count
        event_count += outcome.source_event_count
    return {
        "loss": loss_sum / event_count if event_count else float("nan"),
        "source_event_count": event_count,
        "observed_batch_count": observed_batches,
        "nonempty_batch_count": nonempty_batches,
        "optimizer_step_count": optimizer_steps,
        "empty_batch_count": empty_batches,
        "valid_pair_count": valid_cells,
        "mean_edges_per_sample": edges / sample_count if sample_count else 0.0,
        "edge_density": edges / possible_edges if possible_edges else 0.0,
        "mean_out_isolated_nodes_per_sample": isolated_nodes / sample_count if sample_count else 0.0,
        "out_slot_padding_fraction": padded_out_slots / out_slots if out_slots else 0.0,
        "mean_attention_saturation_rate": saturation_sum / sample_count if sample_count else 0.0,
    }


def _checkpoint_payload(
    *,
    model: ShockFollowGraphMixer,
    optimizer: Adam,
    epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    run_config: dict[str, object],
    run_config_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
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
        },
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    _validate_cuda_determinism(device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args._output_dir_claimed = True
    status_path = args.output_dir / "run_status.json"
    _atomic_json(status_path, {"status": "running", "started_at_utc": _utc_now()})

    initial_source_hashes = source_hashes(args.source_root)
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
        source_root=args.source_root,
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
    split = chronological_split_by_target(
        panel.timestamps,
        cutoffs,
        train_end=args.train_end,
        validation_end=args.validation_end,
        test_end=args.test_end,
    )
    train_indices, validation_indices, test_indices = (
        split.train_indices,
        split.validation_indices,
        split.test_indices,
    )
    if args.smoke:
        train_indices = _limit_evenly(train_indices, args.smoke_train_samples)
        validation_indices = _limit_evenly(validation_indices, args.smoke_validation_samples)
        max_epochs = min(args.max_epochs, 3)
        patience = min(args.patience, 3)
    else:
        max_epochs, patience = args.max_epochs, args.patience
    if min(len(train_indices), len(validation_indices), len(test_indices)) < 1:
        raise ValueError("each chronological split must contain at least one sample")

    model_config = ModelConfig(
        attention_history_length=args.sequence_length,
        expected_num_futures=len(panel.node_order),
    )
    training_config = TrainingConfig(
        args.batch_size, args.learning_rate, args.weight_decay, max_epochs,
        patience, args.seed, args.sequence_length, args.train_end,
        args.validation_end, args.test_end,
    )
    split_hashes = {
        name: sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()
        for name, indices in (
            ("train", train_indices),
            ("validation", validation_indices),
            ("test", test_indices),
            ("purged", split.purged_indices),
        )
    }
    run_config = {
        "schema_version": 1,
        "experiment": "shock_follow_graphmixer_v1",
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
        "node_order": list(panel.node_order),
        "node_order_sha256": node_order_sha256(panel.node_order),
        "bar_panel_materialization_sha256": panel.materialization_sha256,
        "threshold_materialization_sha256": thresholds.materialization_sha256,
        "shock_follow_contract_sha256": shock_follow_contract_sha256(),
        "threshold_contract_sha256": threshold_contract_sha256(),
        "retrospective_contract_sha256": retrospective.sha256,
        "materialization_manifest_sha256": canonical_json_sha256(materialization_manifest),
        "economic_graph_sha256": economic_graph_sha256,
        "economic_prior_sha256": file_sha256(args.data_dir / "economic_prior.npy"),
        "split_counts": {
            "train": len(train_indices), "validation": len(validation_indices),
            "test": len(test_indices), "purged": len(split.purged_indices),
        },
        "split_index_sha256": split_hashes,
        "source_layout_schema_version": SOURCE_LAYOUT_SCHEMA_VERSION,
        "source_sha256": initial_source_hashes,
    }
    run_config_hash = canonical_json_sha256(run_config)
    _atomic_json(args.output_dir / "run_config.json", run_config)
    metadata = {
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
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "git": _git_facts(args.source_root),
    }
    _atomic_json(args.output_dir / "run_metadata.json", metadata)

    train_dataset = ShockFollowCutoffDataset(
        panel,
        thresholds,
        prior,
        train_indices,
        sequence_length=args.sequence_length,
        prepared=prepared_data,
    )
    validation_dataset = ShockFollowCutoffDataset(
        panel,
        thresholds,
        prior,
        validation_indices,
        sequence_length=args.sequence_length,
        prepared=prepared_data,
    )
    supervision = {
        "train": summarize_supervision(train_dataset),
        "validation": summarize_supervision(validation_dataset),
        "test": {
            "deferred": True,
            "reason": "training never materializes or summarizes test labels",
            "cutoff_count": len(test_indices),
            "split_index_sha256": split_hashes["test"],
        },
    }
    _atomic_json(args.output_dir / "supervision_summary.json", supervision)

    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_shock_follow_samples,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    model = ShockFollowGraphMixer(**asdict(model_config)).to(device)
    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, object]] = []
    best_reference_state: dict[str, Tensor] | None = None
    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, device, None)
        if not np.isfinite(train_metrics["loss"]) or not np.isfinite(validation_metrics["loss"]):
            raise FloatingPointError("epoch produced a non-finite aggregate loss")
        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "improved": improved,
        }
        epoch_rows.append(row)
        _atomic_json(args.output_dir / "epoch_metrics.json", epoch_rows)
        payload = _checkpoint_payload(
            model=model, optimizer=optimizer, epoch=epoch, best_epoch=best_epoch,
            best_validation_loss=best_validation_loss, model_config=model_config,
            training_config=training_config, run_config=run_config,
            run_config_sha256=run_config_hash,
        )
        _atomic_torch_save(args.output_dir / "last.pt", payload)
        if improved:
            _atomic_torch_save(args.output_dir / "best.pt", payload)
            best_reference_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f} "
            f"events={train_metrics['source_event_count']} "
            f"edges/sample={train_metrics['mean_edges_per_sample']:.1f} "
            f"density={train_metrics['edge_density']:.4f}",
            flush=True,
        )
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
    if reloaded_state.keys() != best_reference_state.keys() or any(
        not torch.equal(best_reference_state[name], reloaded_state[name])
        for name in best_reference_state
    ):
        raise ValueError("best checkpoint reload did not reproduce model state exactly")

    require_unchanged_source_hashes(
        args.source_root,
        initial_source_hashes,
        operation="V1 training",
    )

    result = {
        "status": "completed",
        "completed_at_utc": _utc_now(),
        "run_config_sha256": run_config_hash,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(epoch_rows),
        "test_evaluated_during_training": False,
        "best_checkpoint_reload_verified": True,
        "best_checkpoint_state_exact_match": True,
        "best_checkpoint_sha256": best_checkpoint_hash,
        "last_checkpoint_sha256": file_sha256(args.output_dir / "last.pt"),
    }
    _atomic_json(status_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args._output_dir_claimed = False
    try:
        result = train(args)
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
        f"Training completed: best_epoch={result['best_epoch']} "
        f"best_validation_loss={result['best_validation_loss']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
