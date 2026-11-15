"""Fail-closed checkpoint loading for shock-follow GraphMixer V1."""

from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Mapping

import torch

from gnn_for_futures.shock_follow.contract import canonical_json_sha256
from gnn_for_futures.shock_follow.models.mixer_v1 import ShockFollowGraphMixer
from gnn_for_futures.shock_follow.provenance.source import file_sha256


CHECKPOINT_SCHEMA_VERSION = 1
REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "epoch",
        "best_epoch",
        "best_validation_loss",
        "model_state_dict",
        "optimizer_state_dict",
        "model_config",
        "training_config",
        "run_config",
        "run_config_sha256",
        "rng_state",
    }
)


def read_run_config(path: str | Path) -> dict[str, object]:
    """Read one JSON run identity, requiring an object payload."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"run_config.json was not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run_config.json must contain a JSON object")
    return payload


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _expected_model_config_keys() -> set[str]:
    return {
        "node_feature_dim",
        "temporal_hidden_dim",
        "attention_history_length",
        "expected_num_futures",
        "role_dim",
        "neighbor_capacity",
        "head_hidden_dim",
        "fusion_weights",
        "topology_threshold",
        "attention_robust_quantile",
    }


def validate_checkpoint_payload(
    payload: object,
    *,
    expected_run_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate payload identity before constructing or evaluating a model."""
    checkpoint = dict(_require_mapping(payload, "checkpoint payload"))
    missing = sorted(REQUIRED_CHECKPOINT_KEYS.difference(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {missing}")
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema_version={checkpoint['schema_version']!r}"
        )
    run_config = dict(_require_mapping(checkpoint["run_config"], "checkpoint run_config"))
    stored_hash = checkpoint["run_config_sha256"]
    observed_hash = canonical_json_sha256(run_config)
    if not isinstance(stored_hash, str) or stored_hash != observed_hash:
        raise ValueError("checkpoint run_config_sha256 does not match its run_config")
    if expected_run_config is not None:
        expected = dict(expected_run_config)
        if canonical_json_sha256(expected) != stored_hash:
            raise ValueError("checkpoint run_config disagrees with expected run identity")
    model_config = dict(
        _require_mapping(checkpoint["model_config"], "checkpoint model_config")
    )
    expected_keys = _expected_model_config_keys()
    if set(model_config) != expected_keys:
        raise ValueError("checkpoint model_config schema does not match ModelConfig")
    if run_config.get("model") != model_config:
        raise ValueError("checkpoint model_config disagrees with run_config.model")
    training_config = dict(
        _require_mapping(checkpoint["training_config"], "checkpoint training_config")
    )
    if run_config.get("training") != training_config:
        raise ValueError("checkpoint training_config disagrees with run_config.training")
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    return checkpoint


def load_checkpoint(
    path: str | Path,
    *,
    expected_run_config: Mapping[str, object] | None = None,
    expected_sha256: str,
) -> dict[str, object]:
    """Load a trusted local checkpoint onto CPU and validate its run identity.

    PyTorch checkpoints use pickle internally. Call this only for files whose
    expected SHA-256 has been established by this experiment's provenance.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file was not found: {checkpoint_path}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected_sha256 must be one hexadecimal SHA-256 digest")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError("expected_sha256 must be one hexadecimal SHA-256 digest") from error
    checkpoint_bytes = checkpoint_path.read_bytes()
    observed_sha256 = sha256(checkpoint_bytes).hexdigest()
    if observed_sha256 != expected_sha256.lower():
        raise ValueError("checkpoint file SHA-256 does not match expected_sha256")
    payload = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location="cpu",
        weights_only=False,
    )
    return validate_checkpoint_payload(
        payload,
        expected_run_config=expected_run_config,
    )


def model_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    device: torch.device | str = "cpu",
) -> ShockFollowGraphMixer:
    """Construct an eval-mode model after strict payload validation."""
    validated = validate_checkpoint_payload(checkpoint)
    model_config = dict(validated["model_config"])
    model = ShockFollowGraphMixer(**model_config)
    model.load_state_dict(validated["model_state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model
