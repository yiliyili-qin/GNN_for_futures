"""Run the approved one-shot confirmation for shock-follow GraphMixer V2.

The confirmation labels and V2 outputs are inaccessible until the completed
formal training run, its source/data identities, and a freshly recomputed
full-development loss have all been accepted.  Creating the claim spends the
single confirmation look even when a later step fails.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import heapq
import json
import math
import os
from pathlib import Path
import secrets
import sys
import tempfile
import traceback
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
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
)
from gnn_for_futures.shock_follow.data.availability import load_retrospective_bar_data_contract
from gnn_for_futures.graph.regression import load_economic_prior_graph_snapshot
from gnn_for_futures.shock_follow.data.artifacts import (
    load_and_validate_materialization_bundle,
    require_unchanged_materialization_identity,
)
from gnn_for_futures.shock_follow.training.checkpoint_v2 import (
    load_checkpoint,
    model_from_checkpoint,
    read_run_config,
)
from gnn_for_futures.shock_follow.evaluation.gate_v2 import evaluate_confirmation_gate
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
    hierarchical_predictions,
)
from gnn_for_futures.shock_follow.evaluation.metrics_v2 import V2MetricsAccumulator
from gnn_for_futures.shock_follow.data.splits_v2 import (
    CONFIRMATION_END,
    CONFIRMATION_START,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    PROHIBITED_TEST_END,
    PROHIBITED_TEST_START,
    TRAIN_END,
    V2_SPLIT_ORDER,
    ShockFollowV2Split,
    shock_follow_v2_split_by_target,
    split_index_sha256,
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
    LEGACY_SOURCE_LAYOUT_SCHEMA_VERSION,
    SOURCE_LAYOUT_SCHEMA_VERSION,
    file_sha256,
    require_unchanged_release_source_identity,
    resolve_source_root,
    validate_release_source_identity,
    validate_source_manifest,
)


DEFAULT_SOURCE_ROOT = Path(".")
DEFAULT_DATA_DIR = Path("data/shock_follow_v1")
DEFAULT_TRAINING_DIR = Path("results/shock_follow_graphmixer_v2")
DEFAULT_OUTPUT_DIR = Path("results/shock_follow_graphmixer_v2_confirmation")
DEFAULT_RETROSPECTIVE_CONTRACT = Path(
    "data/contracts/bar_availability.retrospective.json"
)
DEFAULT_ECONOMIC_GRAPH = Path(
    "data/graph_data/futures_graph_smm_simplified.html"
)
CONFIRMATION_SPLIT = "confirmation"
CLAIM_FILENAME = "confirmation_evaluation_claim.json"
APPROVED_OUTPUT_FILENAMES = (
    "confirmation_config.json",
    "confirmation_status.json",
    "confirmation_supervision_summary.json",
    "confirmation_metrics.json",
    "confirmation_gate.json",
    "confusion_matrix.json",
    "raw_output_summary.json",
)
DEVELOPMENT_LOSS_RTOL = 1e-7
DEVELOPMENT_LOSS_ATOL = 1e-8
RAW_SUMMARY_SCHEMA_VERSION = 2
RAW_SUMMARY_QUANTILE_LEVELS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
RAW_SUMMARY_RESERVOIR_CAPACITY = 8192
RAW_SUMMARY_RESERVOIR_SEED = 20261007
DEFAULT_MARGIN_AUDIT_COUNT = 20
_OLD_TEST_SPLIT_ALIASES = frozenset(
    {
        "test",
        "old_test",
        "v1_test",
        "old_v1_test",
        "old_frozen_test",
        "prohibited_test",
        "prohibited_v1_test",
    }
)
_CLAIM_IDENTITY_KEYS = (
    "training_run_config_sha256",
    "checkpoint_sha256",
    "split_index_sha256",
    "source_manifest_sha256",
    "source_sha256",
    "materialization_sha256",
    "materialization_artifact_sha256",
    "confirmation_output_directory",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one accepted formal shock-follow GraphMixer V2 best.pt "
            "on the one-shot confirmation split."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root containing the released src/gnn_for_futures source tree.",
    )
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--checkpoint", type=Path)
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
            "materialization_manifest.json; it must match the formal training run."
        ),
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        help=(
            "Independently approved lowercase byte SHA-256 for the formal run's "
            "best.pt; it must match the completed training status."
        ),
    )
    # Deliberately do not use argparse choices: old-test requests must reach the
    # explicit lifecycle refusal rather than looking like an unknown CLI value.
    parser.add_argument("--split", default=CONFIRMATION_SPLIT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--margin-audit-count",
        type=int,
        default=DEFAULT_MARGIN_AUDIT_COUNT,
    )
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    """Atomically replace one JSON artifact without permitting NaN/Infinity."""
    if path.is_symlink():
        raise ValueError(f"refusing to replace a symbolic-link JSON artifact: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise FileNotFoundError(
            f"JSON parent must be a regular nonsymlink directory: {path.parent}"
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            raise ValueError(f"refusing to replace a symbolic-link JSON artifact: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{description} must be a regular nonsymlink file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be one hexadecimal SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be one hexadecimal SHA-256 digest") from error
    return value.lower()


def _validate_approved_checkpoint_hash(
    approved_sha256: object,
    recorded_sha256: object,
) -> str:
    approved = _require_sha256(
        approved_sha256,
        "independently approved checkpoint SHA-256",
    )
    recorded = _require_sha256(
        recorded_sha256,
        "training status best_checkpoint_sha256",
    )
    if approved != recorded:
        raise ValueError(
            "independently approved checkpoint SHA-256 disagrees with the completed "
            "training status"
        )
    return approved


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _require_confirmation_split(split_name: object) -> str:
    """Accept only confirmation and explicitly refuse the consumed V1 test."""
    if not isinstance(split_name, str):
        raise ValueError("V2 confirmation split must be named explicitly")
    normalized = split_name.strip().lower().replace("-", "_")
    if normalized == CONFIRMATION_SPLIT:
        return CONFIRMATION_SPLIT
    if normalized in _OLD_TEST_SPLIT_ALIASES:
        raise ValueError(
            "V2 confirmation evaluator explicitly refuses the old consumed V1 test split"
        )
    raise ValueError("V2 confirmation evaluator only accepts split='confirmation'")


def _confirmation_indices(
    split: ShockFollowV2Split,
    split_name: object = CONFIRMATION_SPLIT,
) -> np.ndarray:
    _require_confirmation_split(split_name)
    return split.confirmation_indices.copy()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _validate_cuda_determinism(device: torch.device) -> None:
    if device.type == "cuda":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "deterministic CUDA confirmation requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before Python starts"
            )
    torch.use_deterministic_algorithms(True)


def _validate_args(args: argparse.Namespace) -> None:
    _require_confirmation_split(args.split)
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
    expected_checkpoint_hash = getattr(args, "expected_checkpoint_sha256", None)
    if (
        not isinstance(expected_checkpoint_hash, str)
        or len(expected_checkpoint_hash) != 64
        or expected_checkpoint_hash != expected_checkpoint_hash.lower()
    ):
        raise ValueError(
            "--expected-checkpoint-sha256 is required and must be one lowercase SHA-256"
        )
    try:
        int(expected_checkpoint_hash, 16)
    except ValueError as error:
        raise ValueError(
            "--expected-checkpoint-sha256 is required and must be one lowercase SHA-256"
        ) from error
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    if args.margin_audit_count < 1 or args.margin_audit_count > 100:
        raise ValueError("margin-audit-count must be in [1, 100]")
    if args.training_dir.is_symlink() or not args.training_dir.is_dir():
        raise FileNotFoundError(
            f"training-dir must be a regular nonsymlink directory: {args.training_dir}"
        )
    output_ancestor = args.output_dir.parent
    while not output_ancestor.exists() and not output_ancestor.is_symlink():
        if output_ancestor == output_ancestor.parent:
            break
        output_ancestor = output_ancestor.parent
    if output_ancestor.is_symlink() or not output_ancestor.is_dir():
        raise FileNotFoundError(
            "confirmation output parent must descend from a regular nonsymlink "
            f"directory: {output_ancestor}"
        )
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"confirmation output directory already exists: {args.output_dir}")


def _write_exclusive(path: Path, payload: bytes) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise FileNotFoundError(
            f"exclusive file parent must be a regular nonsymlink directory: {path.parent}"
        )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("exclusive claim write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _claim_confirmation_evaluation(
    training_dir: Path,
    identity: Mapping[str, object],
) -> Path:
    """Atomically spend this training run's single confirmation evaluation."""
    missing = [key for key in _CLAIM_IDENTITY_KEYS if key not in identity]
    if missing:
        raise ValueError(f"confirmation claim identity is missing keys: {missing}")
    claim_path = training_dir / CLAIM_FILENAME
    claim = {
        "schema_version": 1,
        "experiment": "shock_follow_graphmixer_v2_one_shot_confirmation",
        "status": "running",
        "claimed_at_utc": _utc_now(),
        "claim_nonce": secrets.token_hex(32),
        "split": CONFIRMATION_SPLIT,
        "old_consumed_v1_test_prohibited": True,
        **dict(identity),
    }
    payload = json.dumps(
        claim,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    try:
        _write_exclusive(claim_path, payload)
    except FileExistsError as error:
        raise FileExistsError(
            "this V2 training run already has a claimed confirmation evaluation"
        ) from error
    except BaseException as error:
        # O_EXCL may already have created the one-look token.  Never delete it;
        # make the spent token terminal if the initial durable write failed.
        if claim_path.exists() or claim_path.is_symlink():
            try:
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
            except BaseException as claim_error:
                error.add_note(
                    "exclusive confirmation claim was created, but recording its "
                    "terminal failure also failed: "
                    f"{type(claim_error).__name__}: {claim_error}"
                )
        raise
    return claim_path


def _read_running_confirmation_claim(
    claim_path: Path,
    expected_nonce: object,
) -> dict[str, object]:
    claim = _read_json_object(claim_path, "confirmation evaluation claim")
    if claim.get("status") != "running":
        raise ValueError("confirmation evaluation claim is not in the running state")
    if (
        not isinstance(expected_nonce, str)
        or not expected_nonce
        or claim.get("claim_nonce") != expected_nonce
    ):
        raise ValueError("confirmation evaluation claim ownership changed")
    return claim


def _assert_claim_identity(
    claim: Mapping[str, object],
    identity: Mapping[str, object],
) -> None:
    for key in _CLAIM_IDENTITY_KEYS:
        if claim.get(key) != identity.get(key):
            raise ValueError(f"confirmation evaluation claim {key} changed")


def _finish_confirmation_evaluation_claim(
    claim_path: Path,
    result: Mapping[str, object],
    *,
    claim_nonce: str | None = None,
) -> None:
    if claim_nonce is None:
        existing = _read_json_object(claim_path, "confirmation evaluation claim")
        claim_nonce = str(existing.get("claim_nonce", ""))
    claim = _read_running_confirmation_claim(claim_path, claim_nonce)
    _assert_claim_identity(claim, result)
    config_hash = _require_sha256(
        result.get("confirmation_config_sha256"),
        "confirmation_config_sha256",
    )
    output_hashes = result.get("output_artifact_sha256")
    if not isinstance(output_hashes, Mapping):
        raise ValueError("completed confirmation result lacks output artifact hashes")
    _atomic_json(
        claim_path,
        {
            **claim,
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "confirmation_config_sha256": config_hash,
            "confirmation_gate_all_passed": result.get(
                "confirmation_gate_all_passed"
            ),
            "output_artifact_sha256": dict(output_hashes),
        },
    )


def _fail_confirmation_evaluation_claim(
    claim_path: Path,
    error: BaseException,
    *,
    claim_nonce: str | None = None,
) -> None:
    """Retain the spent one-look token in a truthful terminal failure state."""
    if claim_nonce is None:
        existing = _read_json_object(claim_path, "confirmation evaluation claim")
        claim_nonce = str(existing.get("claim_nonce", ""))
    claim = _read_running_confirmation_claim(claim_path, claim_nonce)
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
def _consume_confirmation_evaluation_claim(
    training_dir: Path,
    identity: Mapping[str, object],
):
    """Make every exit after exclusive claim creation terminal and single-use."""
    claim_path = _claim_confirmation_evaluation(training_dir, identity)
    claim = _read_json_object(claim_path, "confirmation evaluation claim")
    claim_nonce = str(claim.get("claim_nonce", ""))
    try:
        yield claim_path, claim_nonce
    except BaseException as error:
        try:
            _fail_confirmation_evaluation_claim(
                claim_path,
                error,
                claim_nonce=claim_nonce,
            )
        except BaseException as claim_error:
            error.add_note(
                "confirmation was claimed, but recording its terminal failure also "
                f"failed: {type(claim_error).__name__}: {claim_error}"
            )
        raise


def _validate_supervision_summary(
    summary: Mapping[str, object],
    description: str,
) -> dict[str, object]:
    expected_keys = {
        "cutoff_count",
        "cutoffs_with_source_events",
        "source_event_count",
        "valid_pair_count",
        "class_counts",
        "class_fractions",
    }
    if set(summary) != expected_keys:
        raise ValueError(f"{description} schema disagrees with summarize_supervision")
    cutoff_count = _nonnegative_integer(summary["cutoff_count"], f"{description}.cutoff_count")
    cutoffs_with_events = _nonnegative_integer(
        summary["cutoffs_with_source_events"],
        f"{description}.cutoffs_with_source_events",
    )
    source_events = _nonnegative_integer(
        summary["source_event_count"], f"{description}.source_event_count"
    )
    valid_pairs = _nonnegative_integer(
        summary["valid_pair_count"], f"{description}.valid_pair_count"
    )
    raw_counts = summary["class_counts"]
    if not isinstance(raw_counts, list) or len(raw_counts) != len(CLASS_ORDER):
        raise ValueError(f"{description}.class_counts must contain three integers")
    counts = [
        _nonnegative_integer(value, f"{description}.class_counts[{index}]")
        for index, value in enumerate(raw_counts)
    ]
    raw_fractions = summary["class_fractions"]
    if not isinstance(raw_fractions, list) or len(raw_fractions) != len(CLASS_ORDER):
        raise ValueError(f"{description}.class_fractions must contain three values")
    fractions = [
        _finite_number(value, f"{description}.class_fractions[{index}]")
        for index, value in enumerate(raw_fractions)
    ]
    expected_fractions = (
        [count / valid_pairs for count in counts]
        if valid_pairs
        else [0.0, 0.0, 0.0]
    )
    if sum(counts) != valid_pairs or not np.allclose(
        fractions, expected_fractions, rtol=0.0, atol=1e-15
    ):
        raise ValueError(f"{description} class counts/fractions are inconsistent")
    if cutoffs_with_events > cutoff_count or source_events < cutoffs_with_events:
        raise ValueError(f"{description} event coverage is inconsistent")
    return dict(summary)


def _validate_weight_audit(
    training_summary: Mapping[str, object],
    audit: Mapping[str, object],
) -> tuple[dict[str, object], tuple[float, float]]:
    counts = training_summary["class_counts"]
    if not isinstance(counts, list):
        raise AssertionError("validated training class_counts changed type")
    normalized_counts = tuple(int(value) for value in counts)
    weights = balanced_follow_class_weights(normalized_counts)
    core = {
        "source_split": "training_only",
        "development_counts_accessed": False,
        "confirmation_counts_accessed": False,
        "prohibited_v1_test_counts_accessed": False,
        "three_class_counts": list(normalized_counts),
        "no_follow_count": normalized_counts[0],
        "follow_count": normalized_counts[1] + normalized_counts[2],
        "follow_class_order": list(FOLLOW_CLASS_ORDER),
        "follow_class_weights": list(weights),
        "formula": "N_total/(2*N_class)",
    }
    expected = {**core, "audit_sha256": canonical_json_sha256(core)}
    if dict(audit) != expected:
        raise ValueError("class_weight_audit.json disagrees with training-only counts")
    return expected, weights


def _validate_source_closure(
    training_dir: Path,
    run_config: Mapping[str, object],
    training_status: Mapping[str, object],
    *,
    source_root: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    source_manifest = _read_json_object(
        training_dir / "source_manifest_verification.json",
        "source_manifest_verification.json",
    )
    if source_manifest.get("schema_version") == LEGACY_SOURCE_LAYOUT_SCHEMA_VERSION:
        raise ValueError(
            "legacy source schema 1 is recognized as a historical graph_model "
            "identity but cannot be claimed as this package-layout formal run"
        )
    validation = validate_source_manifest(
        source_manifest,
        source_root=source_root,
        allow_legacy=False,
    )
    if validation["source_layout_schema_version"] != SOURCE_LAYOUT_SCHEMA_VERSION:
        raise ValueError("confirmation requires source-layout schema 2")
    current_hashes = validation["source_sha256"]
    if not isinstance(current_hashes, dict):
        raise AssertionError("validated source hash map changed type")
    expected_manifest_hash = validation["source_manifest_sha256"]
    if run_config.get("source_layout_schema_version") != SOURCE_LAYOUT_SCHEMA_VERSION:
        raise ValueError("run_config does not declare source-layout schema 2")
    if run_config.get("source_sha256") != current_hashes:
        raise ValueError("run_config source hashes disagree with source closure")
    if run_config.get("source_manifest_sha256") != expected_manifest_hash:
        raise ValueError("run_config source manifest hash disagrees with source closure")
    if training_status.get("source_manifest_sha256") != expected_manifest_hash:
        raise ValueError("training status source manifest hash disagrees with source closure")
    return source_manifest, current_hashes


def _formal_split_manifest(split: ShockFollowV2Split) -> dict[str, object]:
    full = split_manifest_payload(split)
    return {
        **full,
        "smoke_limited": False,
        "effective_training_count": len(split.training_indices),
        "effective_training_index_sha256": split_index_sha256(
            split.training_indices
        ),
        "effective_development_count": len(split.development_indices),
        "effective_development_index_sha256": split_index_sha256(
            split.development_indices
        ),
        "training_entrypoint_dataset_construction": ["training", "development"],
        "confirmation_dataset_constructed": False,
        "prohibited_v1_test_dataset_constructed": False,
    }


def _normalized_model_config(node_count: int) -> dict[str, object]:
    # JSON normalization changes the dataclass tuple to the persisted list.
    return json.loads(
        json.dumps(
            asdict(
                ModelConfig(
                    attention_history_length=DEFAULT_SEQUENCE_LENGTH,
                    expected_num_futures=node_count,
                )
            )
        )
    )


def _formal_training_config() -> dict[str, object]:
    return {
        "batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "max_epochs": 40,
        "patience": 7,
        "seed": 20261002,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "train_end": TRAIN_END,
        "development_start": DEVELOPMENT_START,
        "development_end": DEVELOPMENT_END,
        "confirmation_start": CONFIRMATION_START,
        "confirmation_end": CONFIRMATION_END,
        "prohibited_test_start": PROHIBITED_TEST_START,
        "prohibited_test_end": PROHIBITED_TEST_END,
        "lambda_follow": LAMBDA_FOLLOW,
        "lambda_direction": LAMBDA_DIRECTION,
    }


def _first_mapping_difference(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
) -> str:
    for key in sorted(set(observed) | set(expected)):
        if key not in observed:
            return f"missing key {key}"
        if key not in expected:
            return f"unexpected key {key}"
        if observed[key] != expected[key]:
            return f"field {key} disagrees"
    return "unknown difference"


def _validate_formal_run_config(
    run_config: Mapping[str, object],
    *,
    panel,
    retrospective,
    economic_graph_sha256: str,
    economic_prior_path: Path,
    materialization_manifest: Mapping[str, object],
    split_manifest: Mapping[str, object],
    weight_audit: Mapping[str, object],
    source_manifest_sha256: str,
    source_hashes: Mapping[str, str],
    release_source_identity: Mapping[str, object],
    materialization_manifest_file_sha256: str,
) -> None:
    if (
        run_config.get("materialization_manifest_file_sha256")
        != materialization_manifest_file_sha256
    ):
        raise ValueError(
            "confirmation materialization manifest hash disagrees with the "
            "formal training run"
        )
    counts = weight_audit["three_class_counts"]
    if not isinstance(counts, list) or len(counts) != 3:
        raise AssertionError("validated weight audit changed shape")
    expected = {
        "schema_version": 3,
        "experiment": "shock_follow_graphmixer_v2_hierarchical",
        "approval_date": V2_APPROVAL_DATE,
        "source_layout_schema_version": SOURCE_LAYOUT_SCHEMA_VERSION,
        "smoke": False,
        "bar_data_mode": "retrospective-finalized",
        "research_only": True,
        "strict_point_in_time_backtest": False,
        "regression_channel": "unavailable_missing_measured_available_at_manifest",
        "model": _normalized_model_config(len(panel.node_order)),
        "training": _formal_training_config(),
        "node_feature_schema": list(NODE_FEATURE_SCHEMA),
        "pair_feature_schema": list(PAIR_FEATURE_SCHEMA),
        "class_order": list(CLASS_ORDER),
        "joint_class_order": list(CLASS_ORDER),
        "head_schema_version": "hierarchical_follow_direction_v2",
        "follow_class_order": list(FOLLOW_CLASS_ORDER),
        "direction_class_order": list(DIRECTION_CLASS_ORDER),
        "follow_threshold": 0.5,
        "tie_policy": "no_follow",
        "follow_class_counts": [counts[0], counts[1] + counts[2]],
        "follow_class_weights": weight_audit["follow_class_weights"],
        "follow_class_weight_audit": dict(weight_audit),
        "direction_mask_source": "true_target_class",
        "loss_reduction_contract": "cell_cross_entropy_then_source_event_mean",
        "warm_start": False,
        "old_frozen_test_prohibited": True,
        "node_order": list(panel.node_order),
        "node_order_sha256": node_order_sha256(panel.node_order),
        "bar_panel_materialization_sha256": panel.materialization_sha256,
        "threshold_materialization_sha256": materialization_manifest[
            "threshold_materialization_sha256"
        ],
        "shock_follow_contract_sha256": shock_follow_contract_sha256(),
        "threshold_contract_sha256": threshold_contract_sha256(),
        "retrospective_contract_sha256": retrospective.sha256,
        "materialization_manifest_sha256": canonical_json_sha256(
            materialization_manifest
        ),
        "materialization_manifest_file_sha256": (
            materialization_manifest_file_sha256
        ),
        **dict(release_source_identity),
        "economic_graph_sha256": economic_graph_sha256,
        "economic_prior_sha256": file_sha256(economic_prior_path),
        "split_manifest": dict(split_manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "source_sha256": dict(source_hashes),
        "confirmation_accessed_during_training": False,
        "prohibited_v1_test_accessed_during_training": False,
    }
    if dict(run_config) != expected:
        difference = _first_mapping_difference(run_config, expected)
        raise ValueError(f"run_config is not the approved formal V2 identity: {difference}")


def _validate_training_status(
    status: Mapping[str, object],
    *,
    run_config_sha256: str,
    source_manifest_sha256: str,
) -> tuple[str, int, float]:
    if status.get("status") != "completed":
        raise ValueError("formal V2 training run_status.json must be completed")
    if status.get("run_config_sha256") != run_config_sha256:
        raise ValueError("training status run_config hash mismatch")
    if status.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("training status source manifest hash mismatch")
    for key in (
        "best_checkpoint_reload_verified",
        "best_checkpoint_state_exact_match",
        "best_checkpoint_development_loss_match",
    ):
        if status.get(key) is not True:
            raise ValueError(f"training status {key} must be true")
    if status.get("confirmation_evaluated_during_training") is not False:
        raise ValueError("training status reports confirmation access during training")
    if status.get("prohibited_v1_test_accessed") is not False:
        raise ValueError("training status reports prohibited old-test access")
    best_hash = _require_sha256(
        status.get("best_checkpoint_sha256"),
        "training status best_checkpoint_sha256",
    )
    best_epoch = _positive_integer(status.get("best_epoch"), "training status best_epoch")
    best_loss = _finite_number(
        status.get("best_development_loss"),
        "training status best_development_loss",
    )
    epochs_completed = _positive_integer(
        status.get("epochs_completed"), "training status epochs_completed"
    )
    if best_epoch > epochs_completed or epochs_completed > 40:
        raise ValueError("training status epoch counts are inconsistent")
    return best_hash, best_epoch, best_loss


def _validate_reload_artifact(
    reload_validation: Mapping[str, object],
    *,
    best_epoch: int,
) -> tuple[float, float]:
    if reload_validation.get("best_epoch") != best_epoch:
        raise ValueError("best checkpoint reload best_epoch mismatch")
    if reload_validation.get("state_exact_match") is not True:
        raise ValueError("best checkpoint reload state was not exact")
    if reload_validation.get("loss_match") is not True:
        raise ValueError("best checkpoint reload loss was not accepted")
    if reload_validation.get("rtol") != DEVELOPMENT_LOSS_RTOL or reload_validation.get(
        "atol"
    ) != DEVELOPMENT_LOSS_ATOL:
        raise ValueError("best checkpoint reload used unapproved loss tolerances")
    checkpoint_loss = _finite_number(
        reload_validation.get("checkpoint_best_development_loss"),
        "reload checkpoint_best_development_loss",
    )
    recomputed_loss = _finite_number(
        reload_validation.get("recomputed_development_total_loss"),
        "reload recomputed_development_total_loss",
    )
    metrics = reload_validation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("best checkpoint reload metrics must be an object")
    metric_loss = _finite_number(metrics.get("total_loss"), "reload metrics.total_loss")
    if not np.isclose(
        metric_loss,
        recomputed_loss,
        rtol=DEVELOPMENT_LOSS_RTOL,
        atol=DEVELOPMENT_LOSS_ATOL,
    ):
        raise ValueError("best checkpoint reload metrics loss mismatch")
    return checkpoint_loss, recomputed_loss


def _validate_best_checkpoint_selection(
    checkpoint: Mapping[str, object],
    training_status: Mapping[str, object],
    reload_validation: Mapping[str, object],
    development_metrics: Mapping[str, object],
) -> dict[str, object]:
    """Require a fresh full-development loss match before claim creation."""
    best_epoch = _positive_integer(
        training_status.get("best_epoch"), "training status best_epoch"
    )
    if checkpoint.get("epoch") != best_epoch or checkpoint.get("best_epoch") != best_epoch:
        raise ValueError("best.pt checkpoint epoch disagrees with training status best_epoch")
    observed = _finite_number(
        development_metrics.get("total_loss"),
        "recomputed development total_loss",
    )
    reload_checkpoint_loss, reload_recomputed_loss = _validate_reload_artifact(
        reload_validation,
        best_epoch=best_epoch,
    )
    recorded = {
        "checkpoint": _finite_number(
            checkpoint.get("best_development_loss"),
            "checkpoint best_development_loss",
        ),
        "training_status": _finite_number(
            training_status.get("best_development_loss"),
            "training status best_development_loss",
        ),
        "reload_checkpoint": reload_checkpoint_loss,
        "reload_recomputed": reload_recomputed_loss,
    }
    for source, value in recorded.items():
        if not np.isclose(
            observed,
            value,
            rtol=DEVELOPMENT_LOSS_RTOL,
            atol=DEVELOPMENT_LOSS_ATOL,
        ):
            raise ValueError(
                "fresh full-development total loss disagrees with "
                f"{source} at rtol={DEVELOPMENT_LOSS_RTOL}, "
                f"atol={DEVELOPMENT_LOSS_ATOL}"
            )
    return {
        "accepted": True,
        "recomputed_development_total_loss": observed,
        "checkpoint_best_development_loss": recorded["checkpoint"],
        "training_status_best_development_loss": recorded["training_status"],
        "reload_recorded_development_total_loss": recorded["reload_recomputed"],
        "rtol": DEVELOPMENT_LOSS_RTOL,
        "atol": DEVELOPMENT_LOSS_ATOL,
    }


def _move_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: (
            value.to(device, non_blocking=device.type == "cuda")
            if isinstance(value, Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _forward(model, batch: Mapping[str, object]):
    return model(
        batch["node_history"],
        batch["economic_prior"],
        batch["regression_weight"],
        prior_available=batch["prior_available"],
        regression_available=batch["regression_available"],
    )


_RAW_FEATURE_NAMES = (
    "follow_logit_no_follow",
    "follow_logit_follow",
    "direction_logit_opposite",
    "direction_logit_same",
    "joint_score_no_follow",
    "joint_score_opposite",
    "joint_score_same",
    "follow_probability",
    "joint_prediction_margin",
)


class _BoundedDistribution:
    """Stream exact moments and retain a deterministic bounded quantile sample."""

    def __init__(self, seed: int) -> None:
        self.count = 0
        self.minimum = np.full(len(_RAW_FEATURE_NAMES), np.inf, dtype=np.float64)
        self.maximum = np.full(len(_RAW_FEATURE_NAMES), -np.inf, dtype=np.float64)
        self.mean = np.zeros(len(_RAW_FEATURE_NAMES), dtype=np.float64)
        self.m2 = np.zeros(len(_RAW_FEATURE_NAMES), dtype=np.float64)
        self._rng = np.random.default_rng(seed)
        self._reservoir = np.empty(
            (RAW_SUMMARY_RESERVOIR_CAPACITY, len(_RAW_FEATURE_NAMES)),
            dtype=np.float64,
        )
        self._reservoir_count = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(_RAW_FEATURE_NAMES):
            raise ValueError("raw summary values have an unexpected shape")
        if not len(values):
            return
        if not np.isfinite(values).all():
            raise FloatingPointError("raw summary distribution contains invalid values")
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
        combined_count = self.count + batch_count
        if self.count:
            delta = batch_mean - self.mean
            self.mean += delta * batch_count / combined_count
            self.m2 += (
                batch_m2
                + delta**2 * self.count * batch_count / combined_count
            )
        else:
            self.mean = batch_mean
            self.m2 = batch_m2
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

        for offset, row in enumerate(values):
            sequence = self.count + offset
            if self._reservoir_count < RAW_SUMMARY_RESERVOIR_CAPACITY:
                self._reservoir[self._reservoir_count] = row
                self._reservoir_count += 1
            else:
                replacement = int(self._rng.integers(0, sequence + 1))
                if replacement < RAW_SUMMARY_RESERVOIR_CAPACITY:
                    self._reservoir[replacement] = row
        self.count = combined_count

    def finalize(self) -> dict[str, object]:
        if self.count == 0:
            return {
                "count": 0,
                "quantile_sample_count": 0,
                "quantiles_exact": True,
                "features": None,
            }
        reservoir = self._reservoir[: self._reservoir_count]
        feature_payload: dict[str, object] = {}
        for index, name in enumerate(_RAW_FEATURE_NAMES):
            feature_payload[name] = {
                "minimum": float(self.minimum[index]),
                "maximum": float(self.maximum[index]),
                "mean": float(self.mean[index]),
                "population_std": float(math.sqrt(self.m2[index] / self.count)),
                "quantiles": [
                    float(value)
                    for value in np.quantile(
                        reservoir[:, index],
                        RAW_SUMMARY_QUANTILE_LEVELS,
                        method="linear",
                    )
                ],
            }
        return {
            "count": self.count,
            "quantile_sample_count": self._reservoir_count,
            "quantiles_exact": self._reservoir_count == self.count,
            "features": feature_payload,
        }


class RawOutputSummaryAccumulator:
    """Aggregate raw-output distributions with bounded, deterministic storage."""

    def __init__(self, node_order: Sequence[str], audit_count: int) -> None:
        self.node_order = tuple(node_order)
        self.audit_count = audit_count
        self.total = 0
        self._sequence = 0
        seed_sequence = iter(
            np.random.SeedSequence(RAW_SUMMARY_RESERVOIR_SEED).spawn(10)
        )
        self._groups: dict[str, dict[str, _BoundedDistribution]] = {
            "true_class": {
                name: _BoundedDistribution(next(seed_sequence)) for name in CLASS_ORDER
            },
            "predicted_class": {
                name: _BoundedDistribution(next(seed_sequence)) for name in CLASS_ORDER
            },
            "source_sign": {
                name: _BoundedDistribution(next(seed_sequence))
                for name in ("negative", "positive")
            },
            "economic_prior_edge": {
                name: _BoundedDistribution(next(seed_sequence))
                for name in ("absent", "present")
            },
        }
        self._highest: list[tuple[float, int, dict[str, object]]] = []
        self._lowest: list[tuple[float, int, dict[str, object]]] = []

    def _retain_audit(self, row: dict[str, object]) -> None:
        margin = float(row["joint_prediction_margin"])
        sequence = self._sequence
        self._sequence += 1
        high_item = (margin, sequence, row)
        if len(self._highest) < self.audit_count:
            heapq.heappush(self._highest, high_item)
        elif margin > self._highest[0][0]:
            heapq.heapreplace(self._highest, high_item)

        low_item = (-margin, sequence, row)
        if len(self._lowest) < self.audit_count:
            heapq.heappush(self._lowest, low_item)
        elif margin < -self._lowest[0][0]:
            heapq.heapreplace(self._lowest, low_item)

    def update(
        self,
        raw_batch: Mapping[str, object],
        moved_batch: Mapping[str, object],
        output,
    ) -> None:
        target_class = moved_batch["target_class"]
        raw_mask = moved_batch["loss_mask"]
        if not isinstance(target_class, Tensor) or not isinstance(raw_mask, Tensor):
            raise TypeError("confirmation target_class and loss_mask must be tensors")
        diagonal = torch.eye(
            raw_mask.shape[1], dtype=torch.bool, device=raw_mask.device
        ).unsqueeze(0)
        mask = raw_mask & ~diagonal
        if not mask.any():
            return
        positions = mask.nonzero(as_tuple=False).detach().cpu().numpy()
        targets = target_class[mask].to(torch.int64)
        predictions = hierarchical_predictions(
            output.follow_logits, output.direction_logits
        )[mask].to(torch.int64)
        follow_probabilities = torch.softmax(output.follow_logits, dim=-1)[..., 1][mask]
        joint = output.joint_scores[mask]
        sorted_joint = torch.sort(joint, dim=-1, descending=True).values
        margins = sorted_joint[:, 0] - sorted_joint[:, 1]
        features = torch.cat(
            (
                output.follow_logits[mask],
                output.direction_logits[mask],
                joint,
                follow_probabilities.unsqueeze(-1),
                margins.unsqueeze(-1),
            ),
            dim=-1,
        )
        if not torch.isfinite(features).all():
            raise FloatingPointError("raw-output summary received non-finite values")
        source_sign_grid = moved_batch["source_sign"].unsqueeze(-1).expand_as(raw_mask)
        source_signs = source_sign_grid[mask].to(torch.int64)
        if ((source_signs != -1) & (source_signs != 1)).any():
            raise ValueError("mask-valid confirmation source signs must be -1 or +1")
        economic_values = moved_batch["economic_prior"][mask]
        economic_edges = economic_values > 0.0

        values = features.detach().to("cpu", torch.float64).numpy()
        true_values = targets.detach().cpu().numpy()
        predicted_values = predictions.detach().cpu().numpy()
        sign_values = source_signs.detach().cpu().numpy()
        edge_values = economic_edges.detach().cpu().numpy()
        economic_cpu = economic_values.detach().to("cpu", torch.float64).numpy()

        for class_id, class_name in enumerate(CLASS_ORDER):
            self._groups["true_class"][class_name].update(
                values[true_values == class_id]
            )
            self._groups["predicted_class"][class_name].update(
                values[predicted_values == class_id]
            )
        for sign, name in ((-1, "negative"), (1, "positive")):
            self._groups["source_sign"][name].update(values[sign_values == sign])
        for edge, name in ((False, "absent"), (True, "present")):
            self._groups["economic_prior_edge"][name].update(
                values[edge_values == edge]
            )

        panel_indices = np.asarray(raw_batch["panel_index"], dtype=np.int64)
        cutoff_at = np.asarray(raw_batch["cutoff_at"])
        target_at = np.asarray(raw_batch["target_at"])
        for row_index, (batch_index, source_index, target_index) in enumerate(positions):
            true_id = int(true_values[row_index])
            predicted_id = int(predicted_values[row_index])
            feature = values[row_index]
            audit = {
                "panel_index": int(panel_indices[batch_index]),
                "cutoff_at": np.datetime_as_string(cutoff_at[batch_index], unit="s"),
                "target_at": np.datetime_as_string(target_at[batch_index], unit="s"),
                "source_index": int(source_index),
                "source_symbol": self.node_order[source_index],
                "target_index": int(target_index),
                "target_symbol": self.node_order[target_index],
                "true_class_id": true_id,
                "true_class": CLASS_ORDER[true_id],
                "predicted_class_id": predicted_id,
                "predicted_class": CLASS_ORDER[predicted_id],
                "source_sign": int(sign_values[row_index]),
                "economic_prior_value": float(economic_cpu[row_index]),
                "economic_prior_edge": bool(edge_values[row_index]),
                "follow_logits": [float(value) for value in feature[0:2]],
                "direction_logits": [float(value) for value in feature[2:4]],
                "joint_scores": [float(value) for value in feature[4:7]],
                "follow_probability": float(feature[7]),
                "joint_prediction_margin": float(feature[8]),
            }
            self._retain_audit(audit)
        self.total += len(values)

    def finalize(self) -> dict[str, object]:
        groups = {
            dimension: {
                name: distribution.finalize()
                for name, distribution in dimension_groups.items()
            }
            for dimension, dimension_groups in self._groups.items()
        }
        for dimension, dimension_groups in groups.items():
            observed = sum(int(group["count"]) for group in dimension_groups.values())
            if observed != self.total:
                raise AssertionError(
                    f"raw summary {dimension} groups cover {observed}, expected {self.total}"
                )
        highest = [item[2] for item in self._highest]
        lowest = [item[2] for item in self._lowest]
        highest.sort(
            key=lambda row: (
                -float(row["joint_prediction_margin"]),
                int(row["panel_index"]),
                int(row["source_index"]),
                int(row["target_index"]),
            )
        )
        lowest.sort(
            key=lambda row: (
                float(row["joint_prediction_margin"]),
                int(row["panel_index"]),
                int(row["source_index"]),
                int(row["target_index"]),
            )
        )
        return {
            "schema_version": RAW_SUMMARY_SCHEMA_VERSION,
            "full_raw_rows_exported": False,
            "threshold_search_performed": False,
            "valid_pair_count": self.total,
            "feature_order": list(_RAW_FEATURE_NAMES),
            "quantile_levels": list(RAW_SUMMARY_QUANTILE_LEVELS),
            "quantile_method": "linear_deterministic_reservoir",
            "quantile_reservoir_capacity_per_group": RAW_SUMMARY_RESERVOIR_CAPACITY,
            "quantile_reservoir_seed": RAW_SUMMARY_RESERVOIR_SEED,
            "exact_statistics": ["count", "minimum", "maximum", "mean", "population_std"],
            "approximate_statistics": ["quantiles"],
            "margin_definition": "largest_joint_score_minus_second_largest_joint_score",
            "grouped_distributions": groups,
            "margin_audit": {
                "requested_count_per_tail": self.audit_count,
                "highest_margin": highest,
                "lowest_margin": lowest,
            },
        }


def _evaluate_loader(
    model,
    loader: DataLoader,
    device: torch.device,
    follow_class_weights: Sequence[float],
    *,
    node_order: Sequence[str] | None = None,
    margin_audit_count: int = DEFAULT_MARGIN_AUDIT_COUNT,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Evaluate a loader once using the registered V2 accumulator."""
    model.eval()
    accumulator = V2MetricsAccumulator(
        follow_class_weights,
        lambda_follow=LAMBDA_FOLLOW,
        lambda_direction=LAMBDA_DIRECTION,
    )
    raw_summary = (
        RawOutputSummaryAccumulator(node_order, margin_audit_count)
        if node_order is not None
        else None
    )
    sample_count = 0
    edges = 0
    possible_edges = 0
    isolated_nodes = 0
    padded_out_slots = 0
    out_slots = 0
    saturation_sum = 0.0
    observed_batches = 0
    with torch.no_grad():
        for raw_batch in loader:
            observed_batches += 1
            batch = _move_batch(raw_batch, device)
            output = _forward(model, batch)
            accumulator.update(
                output.follow_logits,
                output.direction_logits,
                batch["target_class"],
                batch["loss_mask"],
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
            if raw_summary is not None:
                raw_summary.update(raw_batch, batch, output)
    metrics = accumulator.finalize()
    if metrics["valid_pair_count"] == 0 or metrics["source_event_count"] == 0:
        raise ValueError("evaluation split contains no evaluable source events")
    diagnostics = {
        "observed_batch_count": observed_batches,
        "sample_count": sample_count,
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
    }
    if not all(math.isfinite(float(value)) for value in diagnostics.values()):
        raise FloatingPointError("graph diagnostics contain a non-finite value")
    metrics = {**metrics, "graph_diagnostics": diagnostics}
    return metrics, raw_summary.finalize() if raw_summary is not None else None


def _confirmation_gate_payload(metrics: Mapping[str, object]) -> dict[str, object]:
    gate = evaluate_confirmation_gate(metrics)
    if gate.get("threshold_search_performed") is not False:
        raise AssertionError("pre-registered confirmation gate attempted threshold search")
    return gate


def _confusion_payload(metrics: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "rows": "true_class",
        "columns": "predicted_class",
        "three_class": {
            "class_order": metrics["class_order"],
            "matrix": metrics["three_class_confusion_matrix"],
            "per_class": metrics["three_class_per_class"],
        },
        "follow": {
            "class_order": metrics["follow_class_order"],
            "matrix": metrics["follow_confusion_matrix"],
        },
        "conditional_direction_on_true_follow": {
            "class_order": metrics["direction_class_order"],
            "matrix": metrics["conditional_direction_confusion_matrix"],
            "per_class": metrics["conditional_direction_per_class"],
        },
    }


def _materialization_hash_identity(
    run_config: Mapping[str, object],
    materialization_manifest: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
    hashes = {
        "materialization_manifest": _require_sha256(
            run_config.get("materialization_manifest_sha256"),
            "run_config materialization_manifest_sha256",
        ),
        "materialization_manifest_file": _require_sha256(
            run_config.get("materialization_manifest_file_sha256"),
            "run_config materialization_manifest_file_sha256",
        ),
        "bar_panel": _require_sha256(
            run_config.get("bar_panel_materialization_sha256"),
            "run_config bar_panel_materialization_sha256",
        ),
        "thresholds": _require_sha256(
            run_config.get("threshold_materialization_sha256"),
            "run_config threshold_materialization_sha256",
        ),
        "economic_graph": _require_sha256(
            run_config.get("economic_graph_sha256"),
            "run_config economic_graph_sha256",
        ),
        "economic_prior": _require_sha256(
            run_config.get("economic_prior_sha256"),
            "run_config economic_prior_sha256",
        ),
        "retrospective_contract": _require_sha256(
            run_config.get("retrospective_contract_sha256"),
            "run_config retrospective_contract_sha256",
        ),
        "shock_follow_contract": _require_sha256(
            run_config.get("shock_follow_contract_sha256"),
            "run_config shock_follow_contract_sha256",
        ),
        "threshold_contract": _require_sha256(
            run_config.get("threshold_contract_sha256"),
            "run_config threshold_contract_sha256",
        ),
    }
    artifacts = materialization_manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict):
        raise ValueError("materialization manifest artifact_sha256 must be an object")
    artifact_hashes = {
        str(name): _require_sha256(value, f"materialization artifact {name}")
        for name, value in artifacts.items()
    }
    return hashes, artifact_hashes


def _accepted_run(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    """Validate all pre-claim identities and freshly reproduce development loss."""
    run_config = read_run_config(args.training_dir / "run_config.json")
    training_status = _read_json_object(
        args.training_dir / "run_status.json", "formal V2 training run_status.json"
    )
    run_config_hash = canonical_json_sha256(run_config)
    source_root = resolve_source_root(getattr(args, "source_root", None))
    release_source_identity = validate_release_source_identity(
        args.release_source_manifest,
        args.expected_release_source_manifest_sha256,
        source_root=source_root,
    )

    source_manifest, source_hashes = _validate_source_closure(
        args.training_dir,
        run_config,
        training_status,
        source_root=source_root,
    )
    source_manifest_hash = str(source_manifest["source_manifest_sha256"])
    recorded_checkpoint_hash, best_epoch, _ = _validate_training_status(
        training_status,
        run_config_sha256=run_config_hash,
        source_manifest_sha256=source_manifest_hash,
    )
    expected_checkpoint_hash = _validate_approved_checkpoint_hash(
        args.expected_checkpoint_sha256,
        recorded_checkpoint_hash,
    )

    checkpoint_path = args.checkpoint or args.training_dir / "best.pt"
    expected_checkpoint_path = args.training_dir / "best.pt"
    if checkpoint_path.resolve() != expected_checkpoint_path.resolve():
        raise ValueError("V2 confirmation only accepts the training run's best.pt")
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise FileNotFoundError("formal V2 best.pt must be a regular nonsymlink file")
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_run_config=run_config,
        expected_sha256=expected_checkpoint_hash,
    )

    retrospective = load_retrospective_bar_data_contract(
        args.retrospective_contract, project_root=source_root
    )
    panel = load_shock_follow_bar_panel(
        args.data_dir / "bar_panel.npz",
        args.data_dir / "bar_panel_manifest.json",
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
        source_root=source_root,
        expected_materialization_manifest_sha256=(
            args.expected_materialization_manifest_sha256
        ),
    )

    prepared_data = prepare_shock_follow_data(
        panel,
        thresholds,
        sequence_length=DEFAULT_SEQUENCE_LENGTH,
    )
    cutoffs = eligible_cutoff_indices(
        panel,
        prepared_data.input_valid,
        sequence_length=DEFAULT_SEQUENCE_LENGTH,
        require_all_nodes_history=False,
        common_cutoffs=prepared_data.common_cutoffs,
    )
    split = shock_follow_v2_split_by_target(panel.timestamps, cutoffs)
    expected_split_manifest = _formal_split_manifest(split)
    persisted_split_manifest = _read_json_object(
        args.training_dir / "split_manifest.json", "formal V2 split_manifest.json"
    )
    if persisted_split_manifest != expected_split_manifest:
        raise ValueError("persisted split manifest disagrees with reconstructed V2 splits")
    if run_config.get("split_manifest") != expected_split_manifest:
        raise ValueError("run_config split manifest disagrees with reconstructed V2 splits")

    training_summary = _validate_supervision_summary(
        _read_json_object(
            args.training_dir / "training_supervision_summary.json",
            "training_supervision_summary.json",
        ),
        "training supervision",
    )
    if tuple(training_summary["class_counts"]) != APPROVED_TRAINING_CLASS_COUNTS:
        raise ValueError("training class counts disagree with the approved V2 identity")
    if training_summary["valid_pair_count"] != APPROVED_TRAINING_VALID_PAIR_COUNT:
        raise ValueError(
            "training valid-pair count disagrees with the approved V2 identity"
        )
    weight_audit, follow_class_weights = _validate_weight_audit(
        training_summary,
        _read_json_object(
            args.training_dir / "class_weight_audit.json", "class_weight_audit.json"
        ),
    )
    _validate_formal_run_config(
        run_config,
        panel=panel,
        retrospective=retrospective,
        economic_graph_sha256=economic_graph_sha256,
        economic_prior_path=economic_prior_path,
        materialization_manifest=materialization_manifest,
        split_manifest=expected_split_manifest,
        weight_audit=weight_audit,
        source_manifest_sha256=source_manifest_hash,
        source_hashes=source_hashes,
        release_source_identity=release_source_identity,
        materialization_manifest_file_sha256=(
            args.expected_materialization_manifest_sha256
        ),
    )

    reload_validation = _read_json_object(
        args.training_dir / "best_checkpoint_reload_validation.json",
        "best_checkpoint_reload_validation.json",
    )
    if checkpoint.get("best_epoch") != best_epoch:
        raise ValueError("best.pt best_epoch disagrees with completed training status")
    development_dataset = ShockFollowCutoffDataset(
        panel,
        thresholds,
        prior,
        split.development_indices,
        sequence_length=DEFAULT_SEQUENCE_LENGTH,
        prepared=prepared_data,
    )
    observed_development_summary = summarize_supervision(development_dataset)
    persisted_development_summary = _validate_supervision_summary(
        _read_json_object(
            args.training_dir / "development_supervision_summary.json",
            "development_supervision_summary.json",
        ),
        "development supervision",
    )
    if observed_development_summary != persisted_development_summary:
        raise ValueError(
            "fresh development supervision disagrees with the completed training run"
        )
    loader = DataLoader(
        development_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_shock_follow_samples,
        pin_memory=device.type == "cuda",
    )
    model = model_from_checkpoint(checkpoint, device=device)
    development_metrics, development_raw = _evaluate_loader(
        model,
        loader,
        device,
        follow_class_weights,
    )
    if development_raw is not None:
        raise AssertionError("development acceptance unexpectedly retained raw outputs")
    if development_metrics["valid_pair_count"] != persisted_development_summary[
        "valid_pair_count"
    ]:
        raise ValueError("fresh development evaluation coverage mismatch")
    development_acceptance = _validate_best_checkpoint_selection(
        checkpoint,
        training_status,
        reload_validation,
        development_metrics,
    )

    materialization_hashes, materialization_artifact_hashes = (
        _materialization_hash_identity(run_config, materialization_manifest)
    )
    split_hashes = expected_split_manifest["index_sha256"]
    if not isinstance(split_hashes, dict) or set(split_hashes) != set(V2_SPLIT_ORDER):
        raise ValueError("reconstructed V2 split hashes are incomplete")
    normalized_split_hashes = {
        name: _require_sha256(split_hashes[name], f"split hash {name}")
        for name in V2_SPLIT_ORDER
    }
    identity = {
        "training_run_config_sha256": run_config_hash,
        "checkpoint_sha256": expected_checkpoint_hash,
        "split_index_sha256": normalized_split_hashes,
        "source_manifest_sha256": source_manifest_hash,
        "source_sha256": source_hashes,
        "materialization_sha256": materialization_hashes,
        "materialization_artifact_sha256": materialization_artifact_hashes,
        "confirmation_output_directory": str(args.output_dir.resolve()),
    }
    return {
        "identity": identity,
        "run_config": run_config,
        "training_status": training_status,
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "model": model,
        "panel": panel,
        "thresholds": thresholds,
        "prior": prior,
        "prepared_data": prepared_data,
        "split": split,
        "follow_class_weights": follow_class_weights,
        "development_metrics": development_metrics,
        "development_acceptance": development_acceptance,
        "source_root": source_root,
        "release_source_identity": release_source_identity,
        "materialization_manifest": materialization_manifest,
        "retrospective": retrospective,
        "economic_graph_sha256": economic_graph_sha256,
    }


def _failure_payload(error: BaseException, identity: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": "shock_follow_graphmixer_v2_one_shot_confirmation",
        "status": "failed",
        "failed_at_utc": _utc_now(),
        **dict(identity),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    device = _resolve_device(args.device)
    _validate_cuda_determinism(device)

    # No output directory, claim, confirmation Dataset, confirmation labels, or
    # confirmation forward outputs exist before this acceptance completes.
    accepted = _accepted_run(args, device)
    identity = accepted["identity"]
    if not isinstance(identity, Mapping):
        raise AssertionError("accepted run identity changed type")

    output_created = False
    with _consume_confirmation_evaluation_claim(
        args.training_dir,
        identity,
    ) as (claim_path, claim_nonce):
        try:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            output_created = True
            status_path = args.output_dir / "confirmation_status.json"
            _atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "experiment": "shock_follow_graphmixer_v2_one_shot_confirmation",
                    "status": "running",
                    "started_at_utc": _utc_now(),
                    **dict(identity),
                },
            )
            config = {
                "schema_version": 1,
                "experiment": "shock_follow_graphmixer_v2_one_shot_confirmation",
                "research_only": True,
                "strict_point_in_time_backtest": False,
                "split": CONFIRMATION_SPLIT,
                "old_consumed_v1_test_prohibited": True,
                "training_directory": str(args.training_dir.resolve()),
                "checkpoint": str(Path(accepted["checkpoint_path"]).resolve()),
                **dict(identity),
                "development_acceptance": accepted["development_acceptance"],
                "confirmation_model_evaluation_count": 1,
                "follow_threshold": 0.5,
                "tie_policy": "no_follow",
                "threshold_search_performed": False,
                "full_raw_rows_exported": False,
                "margin_audit_count_per_tail": args.margin_audit_count,
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
                "approved_output_filenames": list(APPROVED_OUTPUT_FILENAMES),
            }
            _atomic_json(args.output_dir / "confirmation_config.json", config)

            # This is intentionally the first construction or label access for
            # confirmation.  There is no API branch that can construct old test.
            split = accepted["split"]
            if not isinstance(split, ShockFollowV2Split):
                raise AssertionError("accepted split changed type")
            confirmation_dataset = ShockFollowCutoffDataset(
                accepted["panel"],
                accepted["thresholds"],
                accepted["prior"],
                _confirmation_indices(split, args.split),
                sequence_length=DEFAULT_SEQUENCE_LENGTH,
                prepared=accepted["prepared_data"],
            )
            supervision = summarize_supervision(confirmation_dataset)
            _validate_supervision_summary(supervision, "confirmation supervision")
            loader = DataLoader(
                confirmation_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                collate_fn=collate_shock_follow_samples,
                pin_memory=device.type == "cuda",
            )
            metrics, raw_summary = _evaluate_loader(
                accepted["model"],
                loader,
                device,
                accepted["follow_class_weights"],
                node_order=accepted["panel"].node_order,
                margin_audit_count=args.margin_audit_count,
            )
            if raw_summary is None:
                raise AssertionError("confirmation raw-output summary was not collected")
            if metrics["valid_pair_count"] != supervision["valid_pair_count"]:
                raise ValueError(
                    "confirmation evaluation coverage disagrees with supervision summary"
                )
            if metrics["source_event_count"] != supervision["source_event_count"]:
                raise ValueError(
                    "confirmation source-event coverage disagrees with supervision summary"
                )
            if raw_summary["valid_pair_count"] != supervision["valid_pair_count"]:
                raise ValueError("raw-output summary coverage mismatch")
            confusion = metrics["three_class_confusion_matrix"]
            if [sum(row) for row in confusion] != supervision["class_counts"]:
                raise ValueError("confirmation confusion support disagrees with labels")

            release_source_identity = accepted.get("release_source_identity")
            if not isinstance(release_source_identity, Mapping):
                raise AssertionError("accepted release source identity changed type")
            require_unchanged_release_source_identity(
                args.release_source_manifest,
                args.expected_release_source_manifest_sha256,
                release_source_identity,
                source_root=accepted.get("source_root"),
                operation="V2 confirmation",
            )
            materialization_manifest = accepted.get("materialization_manifest")
            if not isinstance(materialization_manifest, Mapping):
                raise AssertionError("accepted materialization manifest changed type")
            require_unchanged_materialization_identity(
                args.data_dir,
                panel=accepted["panel"],
                thresholds=accepted["thresholds"],
                retrospective=accepted["retrospective"],
                economic_prior=accepted["prior"],
                economic_graph_sha256=accepted["economic_graph_sha256"],
                expected_materialization_manifest_sha256=(
                    args.expected_materialization_manifest_sha256
                ),
                expected_manifest=materialization_manifest,
                source_root=accepted.get("source_root"),
                operation="V2 confirmation",
            )

            gate = _confirmation_gate_payload(metrics)
            confusion_payload = _confusion_payload(metrics)
            artifacts = {
                "confirmation_supervision_summary.json": supervision,
                "confirmation_metrics.json": metrics,
                "confirmation_gate.json": gate,
                "confusion_matrix.json": confusion_payload,
                "raw_output_summary.json": raw_summary,
            }
            for filename, payload in artifacts.items():
                _atomic_json(args.output_dir / filename, payload)

            config_hash = canonical_json_sha256(config)
            result = {
                "schema_version": 1,
                "experiment": "shock_follow_graphmixer_v2_one_shot_confirmation",
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "research_only": True,
                "strict_point_in_time_backtest": False,
                **dict(identity),
                "confirmation_config_sha256": config_hash,
                "confirmation_evaluated_once": True,
                "old_consumed_v1_test_accessed": False,
                "threshold_search_performed": False,
                "confirmation_gate_all_passed": gate["all_passed"],
                "conclusion": gate["conclusion"],
                "approved_output_filenames": list(APPROVED_OUTPUT_FILENAMES),
                "python_version": sys.version,
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "torch_version": torch.__version__,
            }
            _atomic_json(status_path, result)
            output_hashes = {
                filename: file_sha256(args.output_dir / filename)
                for filename in APPROVED_OUTPUT_FILENAMES
            }
            result_with_hashes = {
                **result,
                "output_artifact_sha256": output_hashes,
            }
            _finish_confirmation_evaluation_claim(
                claim_path,
                result_with_hashes,
                claim_nonce=claim_nonce,
            )
            return result
        except BaseException as error:
            if output_created:
                try:
                    _atomic_json(
                        args.output_dir / "confirmation_status.json",
                        _failure_payload(error, identity),
                    )
                except BaseException as status_error:
                    error.add_note(
                        "recording terminal confirmation_status.json also failed: "
                        f"{type(status_error).__name__}: {status_error}"
                    )
            raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate(args)
    print(
        "V2 confirmation completed: "
        f"checkpoint_sha256={result['checkpoint_sha256']} "
        f"gate_passed={result['confirmation_gate_all_passed']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
