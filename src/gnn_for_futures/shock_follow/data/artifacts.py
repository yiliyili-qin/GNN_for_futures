"""Cross-artifact identity checks for one shock-follow materialization bundle."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from gnn_for_futures.shock_follow.provenance.source import file_sha256, resolve_source_root

from gnn_for_futures.shock_follow.data.availability import RetrospectiveBarDataContract
from gnn_for_futures.shock_follow.contract import (
    MATERIALIZATION_SCHEMA_VERSION,
    MATERIALIZATION_SOURCE_PATHS,
    canonical_json_sha256,
    node_order_sha256,
    shock_follow_contract_sha256,
    threshold_contract_sha256,
)
from gnn_for_futures.shock_follow.data.panel import ShockFollowBarPanel
from gnn_for_futures.shock_follow.data.thresholds import (
    MonthlyThresholdTable,
    threshold_input_sha256,
)


def _json_bytes_without_duplicate_keys(data: bytes, *, name: str) -> object:
    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=object_from_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error


def _json_without_duplicate_keys(path: Path, *, name: str) -> object:
    return _json_bytes_without_duplicate_keys(path.read_bytes(), name=name)


def _lowercase_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
    ):
        raise ValueError(f"{name} must be one lowercase hexadecimal SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{name} must be one lowercase hexadecimal SHA-256"
        ) from error
    return value


def load_and_validate_materialization_bundle(
    data_dir: str | Path,
    *,
    panel: ShockFollowBarPanel,
    thresholds: MonthlyThresholdTable,
    retrospective: RetrospectiveBarDataContract,
    economic_prior: np.ndarray,
    economic_graph_sha256: str,
    source_root: str | Path | None = None,
    expected_materialization_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Bind independently valid artifacts to one completed materialization.

    V2 callers supply an independently approved manifest byte hash.  The optional
    function parameter preserves the read-only V1 validation contract.
    """
    if (
        not isinstance(economic_graph_sha256, str)
        or len(economic_graph_sha256) != 64
        or economic_graph_sha256 != economic_graph_sha256.lower()
    ):
        raise ValueError("economic_graph_sha256 must be a lowercase SHA-256")
    try:
        int(economic_graph_sha256, 16)
    except ValueError as error:
        raise ValueError(
            "economic_graph_sha256 must be a lowercase SHA-256"
        ) from error
    directory = Path(data_dir)
    status_path = directory / "run_status.json"
    manifest_path = directory / "materialization_manifest.json"
    if (
        status_path.is_symlink()
        or manifest_path.is_symlink()
        or not status_path.is_file()
        or not manifest_path.is_file()
    ):
        raise FileNotFoundError(
            "regular nonsymlink materialization status and manifest are required"
        )
    status = _json_without_duplicate_keys(status_path, name="materialization status")
    manifest_bytes = manifest_path.read_bytes()
    manifest_file_sha256 = sha256(manifest_bytes).hexdigest()
    if expected_materialization_manifest_sha256 is not None:
        approved_sha256 = _lowercase_sha256(
            expected_materialization_manifest_sha256,
            name="expected materialization manifest SHA-256",
        )
        if manifest_file_sha256 != approved_sha256:
            raise ValueError(
                "materialization manifest byte SHA-256 disagrees with the "
                "independently approved hash"
            )
    manifest = _json_bytes_without_duplicate_keys(
        manifest_bytes,
        name="materialization manifest",
    )
    if not isinstance(status, dict) or status.get("status") != "completed":
        raise ValueError("materialization run_status.json must be completed")
    if status.get("materialization_manifest") != manifest_path.name:
        raise ValueError("materialization status does not name materialization_manifest.json")
    if status.get("materialization_manifest_sha256") != canonical_json_sha256(manifest):
        raise ValueError("materialization status canonical manifest hash mismatch")
    if status.get("materialization_manifest_file_sha256") != manifest_file_sha256:
        raise ValueError("materialization status manifest file hash mismatch")
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported materialization manifest")
    if manifest.get("bar_data_mode") != "retrospective-finalized":
        raise ValueError("materialization is not retrospective-finalized")
    if manifest.get("research_only") is not True or manifest.get("strict_point_in_time_backtest") is not False:
        raise ValueError("materialization must preserve research-only limitations")
    observed_threshold_input = threshold_input_sha256(
        panel.timestamps,
        panel.trading_dates,
        panel.signed_returns,
    )
    expected = {
        "bar_availability_contract_sha256": retrospective.sha256,
        "bar_availability_contract_identity_sha256": retrospective.identity_sha256,
        "bar_availability_evidence_sha256": {
            "support_response_document": (
                retrospective.support_response_document_sha256
            ),
            "evidence_document": retrospective.evidence_document_sha256,
        },
        "economic_graph_sha256": economic_graph_sha256,
        "bar_panel_materialization_sha256": panel.materialization_sha256,
        "threshold_materialization_sha256": thresholds.materialization_sha256,
        "threshold_input_sha256": thresholds.input_sha256,
        "shock_follow_contract_sha256": shock_follow_contract_sha256(),
        "threshold_contract_sha256": threshold_contract_sha256(),
        "node_order": list(panel.node_order),
        "node_order_sha256": node_order_sha256(panel.node_order),
        "timestamp_count": len(panel.timestamps),
        "first_timestamp": np.datetime_as_string(panel.timestamps[0], unit="s"),
        "last_timestamp": np.datetime_as_string(panel.timestamps[-1], unit="s"),
        "effective_month_count": len(thresholds.months),
        "economic_directed_edge_count": int((economic_prior > 0.0).sum()),
        "regression_channel": "unavailable_missing_measured_available_at_manifest",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"materialization manifest {key} mismatch")
    root = resolve_source_root(source_root)
    expected_source_hashes = {
        relative: file_sha256(root / relative)
        for relative in MATERIALIZATION_SOURCE_PATHS
    }
    if manifest.get("source_sha256") != expected_source_hashes:
        raise ValueError("materialization manifest source_sha256 mismatch")
    if thresholds.input_sha256 != observed_threshold_input:
        raise ValueError("threshold input hash does not match the loaded bar panel")
    artifacts = manifest.get("artifacts")
    expected_artifacts = {
        "bar_panel": "bar_panel.npz",
        "bar_manifest": "bar_panel_manifest.json",
        "thresholds": "monthly_thresholds.npz",
        "threshold_rows": "monthly_thresholds.csv",
        "threshold_manifest": "monthly_thresholds_manifest.json",
        "economic_prior": "economic_prior.npy",
    }
    if artifacts != expected_artifacts:
        raise ValueError("materialization manifest artifact names mismatch")
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(expected_artifacts):
        raise ValueError("materialization manifest artifact hashes mismatch")
    for name, filename in expected_artifacts.items():
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"materialization artifact is missing or nonregular: {filename}")
        expected_hash = artifact_hashes.get(name)
        if not isinstance(expected_hash, str) or file_sha256(path) != expected_hash:
            raise ValueError(f"materialization artifact hash mismatch: {filename}")
    return manifest


def require_unchanged_materialization_identity(
    data_dir: str | Path,
    *,
    panel: ShockFollowBarPanel,
    thresholds: MonthlyThresholdTable,
    retrospective: RetrospectiveBarDataContract,
    economic_prior: np.ndarray,
    economic_graph_sha256: str,
    expected_materialization_manifest_sha256: str,
    expected_manifest: Mapping[str, object],
    source_root: str | Path | None = None,
    operation: str,
) -> None:
    """Fail if an approved materialization bundle changed during an operation."""
    try:
        observed = load_and_validate_materialization_bundle(
            data_dir,
            panel=panel,
            thresholds=thresholds,
            retrospective=retrospective,
            economic_prior=economic_prior,
            economic_graph_sha256=economic_graph_sha256,
            source_root=source_root,
            expected_materialization_manifest_sha256=(
                expected_materialization_manifest_sha256
            ),
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"materialization bundle changed during {operation}"
        ) from error
    if observed != dict(expected_manifest):
        raise ValueError(f"materialization bundle changed during {operation}")
