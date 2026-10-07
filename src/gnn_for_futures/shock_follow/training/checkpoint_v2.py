"""Fail-closed checkpoint loading for hierarchical shock-follow GraphMixer V2."""

from __future__ import annotations

from hashlib import sha256
import io
import json
import math
from pathlib import Path
from typing import Mapping

import torch

from gnn_for_futures.shock_follow.contract import canonical_json_sha256
from gnn_for_futures.shock_follow.models.mixer_v2 import ShockFollowGraphMixerV2
from gnn_for_futures.shock_follow.provenance.source import file_sha256


CHECKPOINT_SCHEMA_VERSION = 2
REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "epoch",
        "best_epoch",
        "best_development_loss",
        "model_state_dict",
        "optimizer_state_dict",
        "model_config",
        "training_config",
        "run_config",
        "run_config_sha256",
        "rng_state",
    }
)
MODEL_CONFIG_KEYS = frozenset(
    {
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
)
RNG_STATE_KEYS = frozenset(
    {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "data_loader_generator",
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


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    name: str,
) -> None:
    observed = set(value)
    missing = sorted(expected.difference(observed))
    unexpected = sorted(observed.difference(expected))
    if missing or unexpected:
        raise ValueError(
            f"{name} schema does not match V2: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _validate_expected_sha256(expected_sha256: object) -> str:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected_sha256 must be one hexadecimal SHA-256 digest")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError(
            "expected_sha256 must be one hexadecimal SHA-256 digest"
        ) from error
    return expected_sha256.lower()


def validate_checkpoint_payload(
    payload: object,
    *,
    expected_run_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate the exact V2 payload schema and its canonical run identity."""
    checkpoint = dict(_require_mapping(payload, "checkpoint payload"))
    _require_exact_keys(
        checkpoint,
        REQUIRED_CHECKPOINT_KEYS,
        "checkpoint payload",
    )
    schema_version = checkpoint["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported V2 checkpoint schema_version={schema_version!r}"
        )

    run_config = dict(
        _require_mapping(checkpoint["run_config"], "checkpoint run_config")
    )
    stored_hash = checkpoint["run_config_sha256"]
    observed_hash = canonical_json_sha256(run_config)
    if not isinstance(stored_hash, str) or stored_hash.lower() != observed_hash:
        raise ValueError("checkpoint run_config_sha256 does not match its run_config")
    if expected_run_config is not None:
        expected = dict(
            _require_mapping(expected_run_config, "expected_run_config")
        )
        if canonical_json_sha256(expected) != stored_hash.lower():
            raise ValueError("checkpoint run_config disagrees with expected run identity")

    model_config = dict(
        _require_mapping(checkpoint["model_config"], "checkpoint model_config")
    )
    _require_exact_keys(model_config, MODEL_CONFIG_KEYS, "checkpoint model_config")
    run_model_config = dict(
        _require_mapping(run_config.get("model"), "checkpoint run_config.model")
    )
    if run_model_config != model_config:
        raise ValueError("checkpoint model_config disagrees with run_config.model")

    training_config = dict(
        _require_mapping(
            checkpoint["training_config"],
            "checkpoint training_config",
        )
    )
    run_training_config = dict(
        _require_mapping(
            run_config.get("training"),
            "checkpoint run_config.training",
        )
    )
    if run_training_config != training_config:
        raise ValueError("checkpoint training_config disagrees with run_config.training")

    epoch = checkpoint["epoch"]
    best_epoch = checkpoint["best_epoch"]
    if (
        not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch < 1
        or not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or best_epoch < 1
        or best_epoch > epoch
    ):
        raise ValueError(
            "checkpoint epochs must be integers satisfying 1 <= best_epoch <= epoch"
        )
    best_loss = checkpoint["best_development_loss"]
    if (
        not isinstance(best_loss, (int, float))
        or isinstance(best_loss, bool)
        or not math.isfinite(float(best_loss))
        or float(best_loss) < 0.0
    ):
        raise ValueError(
            "checkpoint best_development_loss must be one finite nonnegative number"
        )
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    if not isinstance(checkpoint["optimizer_state_dict"], Mapping):
        raise ValueError("checkpoint optimizer_state_dict must be a mapping")
    rng_state = _require_mapping(checkpoint["rng_state"], "checkpoint rng_state")
    _require_exact_keys(rng_state, RNG_STATE_KEYS, "checkpoint rng_state")
    for key in ("torch_cpu", "data_loader_generator"):
        value = rng_state[key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError(
                f"checkpoint rng_state.{key} must be a one-dimensional uint8 tensor"
            )
    torch_cuda = rng_state["torch_cuda"]
    if not isinstance(torch_cuda, (list, tuple)) or any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
        for value in torch_cuda
    ):
        raise ValueError(
            "checkpoint rng_state.torch_cuda must contain one-dimensional uint8 tensors"
        )
    if not isinstance(rng_state["python"], tuple):
        raise ValueError("checkpoint rng_state.python must be a tuple")
    if not isinstance(rng_state["numpy"], tuple):
        raise ValueError("checkpoint rng_state.numpy must be a tuple")
    return checkpoint


def load_checkpoint(
    path: str | Path,
    *,
    expected_run_config: Mapping[str, object] | None = None,
    expected_sha256: str,
) -> dict[str, object]:
    """Verify trusted bytes before deserializing a V2 checkpoint onto CPU.

    PyTorch checkpoints use pickle internally. Call this only with an expected
    SHA-256 established independently by this experiment's provenance.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file was not found: {checkpoint_path}")
    normalized_expected_sha256 = _validate_expected_sha256(expected_sha256)
    checkpoint_bytes = checkpoint_path.read_bytes()
    observed_sha256 = sha256(checkpoint_bytes).hexdigest()
    if observed_sha256 != normalized_expected_sha256:
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
) -> ShockFollowGraphMixerV2:
    """Construct an eval-mode V2 model using strict state-dict loading."""
    validated = validate_checkpoint_payload(checkpoint)
    model_config = dict(validated["model_config"])
    model = ShockFollowGraphMixerV2(**model_config)
    model.load_state_dict(validated["model_state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model
