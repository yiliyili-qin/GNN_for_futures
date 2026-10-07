"""Verify V2 release-source manifests; retain legacy validation explicitly."""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

from gnn_for_futures.shock_follow.provenance.source import (
    SOURCE_LAYOUT_SCHEMA_VERSION,
    read_regular_file_bytes,
    regular_file_snapshot,
)
from gnn_for_futures.shock_follow.contract import (
    CANONICAL_NODE_ORDER,
    LEGACY_MATERIALIZATION_SCHEMA_VERSION,
    LEGACY_MATERIALIZATION_SOURCE_PATHS,
    canonical_json_sha256,
    node_order_sha256,
    shock_follow_contract_sha256,
    threshold_contract_sha256,
)


DEFAULT_MANIFEST = Path("shock_follow_v2_release_source_manifest.json")
TRANSFER_MANIFEST_SCHEMA_VERSION = 3
TRANSFER_BUNDLE_NAME = "gnn_for_futures_v2_release_source"
SOURCE_KIND = "release-source-and-packaging"

LEGACY_DEFAULT_MANIFEST = Path("shock_follow_v2_remote_manifest.json")
LEGACY_TRANSFER_MANIFEST_SCHEMA_VERSION = 1
LEGACY_TRANSFER_BUNDLE_NAME = "shock_follow_graphmixer_v2_remote_cuda"
LEGACY_TRANSFER_DESTINATION_SHA256 = (
    "8adb33d6f06e52d04c5bf65ed44ead02c229b36ae4038b28ef6ed466ffea8bd7"
)

LEGACY_REQUIREMENTS_FILE = "requirements-shock-follow-v1.txt"
LEGACY_RETROSPECTIVE_CONTRACT = "data/contracts/bar_availability.retrospective.json"
LEGACY_ECONOMIC_GRAPH = "data/graph_data/futures_graph_smm_simplified.html"

LEGACY_REQUIRED_V2_RUNTIME_FILES = (
    "graph_model/evaluate_shock_follow_graphmixer_v2_confirmation.py",
    "graph_model/shock_follow_checkpoint_v2.py",
    "graph_model/shock_follow_confirmation_gate_v2.py",
    "graph_model/shock_follow_head_v2.py",
    "graph_model/shock_follow_metrics_v2.py",
    "graph_model/shock_follow_model_v2.py",
    "graph_model/shock_follow_split_v2.py",
    "graph_model/train_shock_follow_graphmixer_v2.py",
)
LEGACY_REQUIRED_SHARED_RUNTIME_FILES = (
    LEGACY_REQUIREMENTS_FILE,
    LEGACY_RETROSPECTIVE_CONTRACT,
    "data/contracts/ricequant_bar_availability_evidence.md",
    "data/contracts/ricequant_support_response_2026-09-21.md",
    LEGACY_ECONOMIC_GRAPH,
    "graph_model/availability.py",
    "graph_model/cross_futures_temporal_attention.py",
    "graph_model/materialize_shock_follow_data.py",
    "graph_model/reg_graph_utils.py",
    "graph_model/shock_follow_artifacts.py",
    "graph_model/shock_follow_checkpoint.py",
    "graph_model/shock_follow_contract.py",
    "graph_model/shock_follow_data.py",
    "graph_model/shock_follow_dataset.py",
    "graph_model/shock_follow_graph.py",
    "graph_model/shock_follow_head.py",
    "graph_model/shock_follow_model.py",
    "graph_model/shock_follow_thresholds.py",
    "graph_model/shock_follow_training_data.py",
)
LEGACY_REQUIRED_OPERATIONAL_FILES = (
    "graph_model/build_shock_follow_v2_remote_manifest.py",
    "graph_model/shock_follow_graphmixer_v2_hierarchical_approval_proposal.md",
    "graph_model/verify_shock_follow_v2_remote_manifest.py",
)
LEGACY_REQUIRED_V2_TEST_FILES = (
    "graph_model/test_shock_follow_checkpoint_v2.py",
    "graph_model/test_shock_follow_confirmation_gate_v2.py",
    "graph_model/test_shock_follow_confirmation_v2.py",
    "graph_model/test_shock_follow_head_v2.py",
    "graph_model/test_shock_follow_metrics_v2.py",
    "graph_model/test_shock_follow_model_v2.py",
    "graph_model/test_shock_follow_split_v2.py",
    "graph_model/test_shock_follow_training_v2.py",
    "graph_model/test_shock_follow_v2_remote_manifest.py",
)
LEGACY_MATERIALIZATION_ARTIFACTS = {
    "bar_panel": "bar_panel.npz",
    "bar_manifest": "bar_panel_manifest.json",
    "thresholds": "monthly_thresholds.npz",
    "threshold_rows": "monthly_thresholds.csv",
    "threshold_manifest": "monthly_thresholds_manifest.json",
    "economic_prior": "economic_prior.npy",
}
LEGACY_MATERIALIZATION_FILENAMES = (
    *LEGACY_MATERIALIZATION_ARTIFACTS.values(),
    "materialization_manifest.json",
    "run_status.json",
)
LEGACY_EXCLUDED_CATEGORIES = (
    ".git",
    ".venv",
    ".env and .env*",
    ".vscode",
    "TOP and TOP_faliure archives/results",
    "secrets",
    "raw 15-minute CSVs",
    "eligible-cutoff caches",
    "unrelated results/logs/checkpoints",
)

_LEGACY_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "bundle",
        "created_at_utc",
        "destination",
        "materialization_schema_version",
        "materialization_data_directory",
        "materialization_manifest_sha256",
        "file_count",
        "total_size_bytes",
        "node_order",
        "excluded_categories",
        "files",
        "manifest_payload_sha256",
    }
)
_LEGACY_FILE_ENTRY_KEYS = frozenset({"path", "size_bytes", "sha256"})
_LEGACY_FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".venv",
        ".vscode",
        "top",
        "top_faliure",
        "top_failure",
        "secrets",
        "secret",
        "raw_data",
        "results",
    }
)
_LEGACY_FORBIDDEN_NAMES = frozenset(
    {
        "dynamic_forecasts.csv",
        "eligible_cutoff_indices_all_valid.npy",
        "eligible_cutoff_indices_any_valid.npy",
    }
)


def _payload_sha256(payload: dict[str, object]) -> str:
    identity = dict(payload)
    identity.pop("manifest_payload_sha256", None)
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _json_bytes_without_duplicate_keys(data: bytes, *, name: str) -> object:
    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        data.decode("utf-8"),
        object_pairs_hook=object_from_pairs,
    )


def _json_without_duplicate_keys(path: Path, *, name: str) -> object:
    data, _ = read_regular_file_bytes(path)
    return _json_bytes_without_duplicate_keys(data, name=name)


def _canonical_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise ValueError(f"unsafe or noncanonical {field}: {value!r}")
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or bool(windows_path.drive)
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe or noncanonical {field}: {value!r}")
    return value


def _project_root(project_root: Path) -> Path:
    supplied = Path(project_root).absolute()
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValueError("project_root must be a regular nonsymlink directory")
    return supplied.resolve()


def _local_path(root: Path, relative: str) -> Path:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"manifest source traverses a symlink: {relative}")
    return current


def _regular_manifest_path(manifest_path: Path, root: Path) -> Path:
    supplied = Path(manifest_path).absolute()
    try:
        relative = supplied.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("transfer manifest must be inside project_root") from error
    canonical = _canonical_relative_path(relative, field="manifest location")
    local = _local_path(root, canonical)
    if not local.is_file():
        raise FileNotFoundError("transfer manifest must be a regular nonsymlink file")
    return local


def _valid_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256") from error
    return value


def _validate_created_at(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("created_at_utc must be a canonical ISO-8601 UTC timestamp")
    if not value.endswith("+00:00") or "T" not in value or value.count("T") != 1:
        raise ValueError("created_at_utc must be a canonical ISO-8601 UTC timestamp")
    date_text, time_with_offset = value.split("T")
    time_text = time_with_offset[: -len("+00:00")]
    if len(date_text) != 10 or len(time_text) not in (8, 15):
        raise ValueError("created_at_utc must be a canonical ISO-8601 UTC timestamp")
    if len(time_text) == 15 and time_text[8] != ".":
        raise ValueError("created_at_utc must be a canonical ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            "created_at_utc must be a canonical ISO-8601 UTC timestamp"
        ) from error
    if parsed.isoformat(timespec="microseconds") != (
        value if "." in time_text else value.replace("+00:00", ".000000+00:00")
    ):
        raise ValueError("created_at_utc must be a canonical ISO-8601 UTC timestamp")


def expected_legacy_transfer_paths(data_directory: str) -> frozenset[str]:
    """Return the frozen V2 closure for one canonical materialization directory."""
    data_relative = _canonical_relative_path(
        data_directory,
        field="materialization_data_directory",
    )
    parts = PurePosixPath(data_relative).parts
    if len(parts) < 2 or parts[0] != "data":
        raise ValueError("materialization_data_directory must be below data/")
    materialization_paths = {
        f"{data_relative}/{filename}" for filename in LEGACY_MATERIALIZATION_FILENAMES
    }
    paths = (
        *LEGACY_REQUIRED_V2_RUNTIME_FILES,
        *LEGACY_REQUIRED_SHARED_RUNTIME_FILES,
        *LEGACY_REQUIRED_OPERATIONAL_FILES,
        *LEGACY_REQUIRED_V2_TEST_FILES,
        *materialization_paths,
    )
    if len(paths) != len(set(paths)):
        raise AssertionError("V2 transfer closure contains duplicate paths")
    return frozenset(paths)


def _validate_materialization_binding(
    manifest: dict[str, object],
    *,
    entries: dict[str, dict[str, object]],
    authenticated_json_bytes: dict[str, bytes],
) -> None:
    data_directory = _canonical_relative_path(
        manifest.get("materialization_data_directory"),
        field="materialization_data_directory",
    )
    expected_paths = expected_legacy_transfer_paths(data_directory)
    observed_paths = set(entries)
    if observed_paths != expected_paths:
        missing = sorted(expected_paths.difference(observed_paths))
        unexpected = sorted(observed_paths.difference(expected_paths))
        raise ValueError(
            f"V2 transfer file closure mismatch: missing={missing}, unexpected={unexpected}"
        )

    materialization_relative = f"{data_directory}/materialization_manifest.json"
    status_relative = f"{data_directory}/run_status.json"
    try:
        materialization_bytes = authenticated_json_bytes[materialization_relative]
        status_bytes = authenticated_json_bytes[status_relative]
    except KeyError as error:
        raise ValueError("authenticated materialization JSON bytes are unavailable") from error
    materialization = _json_bytes_without_duplicate_keys(
        materialization_bytes,
        name="materialization manifest",
    )
    status = _json_bytes_without_duplicate_keys(
        status_bytes,
        name="materialization status",
    )
    if not isinstance(materialization, dict):
        raise ValueError("materialization manifest must contain an object")
    if (
        type(materialization.get("schema_version")) is not int
        or materialization.get("schema_version")
        != LEGACY_MATERIALIZATION_SCHEMA_VERSION
    ):
        raise ValueError("V2 transfer requires the frozen legacy materialization schema")
    if materialization.get("bar_data_mode") != "retrospective-finalized":
        raise ValueError("materialization must be retrospective-finalized")
    if (
        materialization.get("research_only") is not True
        or materialization.get("strict_point_in_time_backtest") is not False
    ):
        raise ValueError("materialization must preserve research-only limitations")
    if materialization.get("node_order") != list(CANONICAL_NODE_ORDER):
        raise ValueError("materialization node order is not canonical")
    if materialization.get("node_order_sha256") != node_order_sha256(
        CANONICAL_NODE_ORDER
    ):
        raise ValueError("materialization node-order hash mismatch")
    if materialization.get("shock_follow_contract_sha256") != shock_follow_contract_sha256():
        raise ValueError("materialization shock-follow contract hash mismatch")
    if materialization.get("threshold_contract_sha256") != threshold_contract_sha256():
        raise ValueError("materialization threshold contract hash mismatch")
    if materialization.get("bar_availability_contract_sha256") != entries[
        LEGACY_RETROSPECTIVE_CONTRACT
    ]["sha256"]:
        raise ValueError("materialization availability contract hash mismatch")

    expected_source_hashes = {
        relative: entries[relative]["sha256"]
        for relative in LEGACY_MATERIALIZATION_SOURCE_PATHS
    }
    if materialization.get("source_sha256") != expected_source_hashes:
        raise ValueError("materialization source hash binding mismatch")
    if materialization.get("artifacts") != LEGACY_MATERIALIZATION_ARTIFACTS:
        raise ValueError("materialization artifact names mismatch")
    artifact_hashes = materialization.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(
        LEGACY_MATERIALIZATION_ARTIFACTS
    ):
        raise ValueError("materialization artifact hash map mismatch")
    for name, filename in LEGACY_MATERIALIZATION_ARTIFACTS.items():
        expected_hash = entries[f"{data_directory}/{filename}"]["sha256"]
        if artifact_hashes.get(name) != expected_hash:
            raise ValueError(f"materialization artifact hash binding mismatch: {filename}")

    bar_manifest_relative = f"{data_directory}/bar_panel_manifest.json"
    threshold_manifest_relative = f"{data_directory}/monthly_thresholds_manifest.json"
    try:
        bar_manifest_bytes = authenticated_json_bytes[bar_manifest_relative]
        threshold_manifest_bytes = authenticated_json_bytes[
            threshold_manifest_relative
        ]
    except KeyError as error:
        raise ValueError("authenticated materialization JSON bytes are unavailable") from error
    bar_manifest = _json_bytes_without_duplicate_keys(
        bar_manifest_bytes,
        name="bar-panel manifest",
    )
    threshold_manifest = _json_bytes_without_duplicate_keys(
        threshold_manifest_bytes,
        name="threshold manifest",
    )
    if not isinstance(bar_manifest, dict) or bar_manifest.get("node_order") != list(
        CANONICAL_NODE_ORDER
    ):
        raise ValueError("bar-panel materialization node order is not canonical")
    if not isinstance(threshold_manifest, dict) or threshold_manifest.get(
        "node_order"
    ) != list(CANONICAL_NODE_ORDER):
        raise ValueError("threshold materialization node order is not canonical")
    if materialization.get("bar_panel_materialization_sha256") != bar_manifest.get(
        "materialization_sha256"
    ):
        raise ValueError("bar-panel materialization identity mismatch")
    if materialization.get("threshold_materialization_sha256") != threshold_manifest.get(
        "materialization_sha256"
    ):
        raise ValueError("threshold materialization identity mismatch")
    if materialization.get("threshold_input_sha256") != threshold_manifest.get(
        "input_sha256"
    ):
        raise ValueError("threshold input identity mismatch")

    materialization_hash = entries[materialization_relative]["sha256"]
    if manifest.get("materialization_manifest_sha256") != materialization_hash:
        raise ValueError("transfer materialization manifest hash mismatch")
    if not isinstance(status, dict) or status.get("status") != "completed":
        raise ValueError("materialization run status must be completed")
    if status.get("materialization_manifest") != "materialization_manifest.json":
        raise ValueError("materialization status names the wrong manifest")
    if status.get("materialization_manifest_sha256") != canonical_json_sha256(
        materialization
    ):
        raise ValueError("materialization canonical manifest hash mismatch")
    if status.get("materialization_manifest_file_sha256") != materialization_hash:
        raise ValueError("materialization file-byte manifest hash mismatch")


def verify_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    source_root: str | Path | None = None,
    project_root: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Verify a V2 schema-2 source closure plus packaging identity."""
    from gnn_for_futures.shock_follow.provenance.transfer_v1 import (
        verify_release_manifest,
    )

    return verify_release_manifest(
        manifest_path,
        expected_schema_version=TRANSFER_MANIFEST_SCHEMA_VERSION,
        expected_bundle=TRANSFER_BUNDLE_NAME,
        source_root=source_root,
        project_root=project_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )


def verify_legacy_manifest(
    manifest_path: Path = LEGACY_DEFAULT_MANIFEST,
    *,
    project_root: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Fail closed on changed bytes, unsafe paths, or an incomplete V2 closure."""
    root = _project_root(
        Path(project_root) if project_root is not None else Path.cwd()
    )
    local_manifest = _regular_manifest_path(manifest_path, root)
    manifest_bytes, manifest_snapshot = read_regular_file_bytes(local_manifest)
    if expected_manifest_sha256 is not None:
        approved_sha256 = _valid_sha256(
            expected_manifest_sha256,
            field="expected legacy transfer manifest sha256",
        )
        if manifest_snapshot.sha256 != approved_sha256:
            raise ValueError(
                "legacy transfer manifest byte SHA-256 disagrees with the "
                "independently approved hash"
            )
    manifest = _json_bytes_without_duplicate_keys(
        manifest_bytes,
        name="transfer manifest",
    )
    if not isinstance(manifest, dict):
        raise ValueError("transfer manifest must contain an object")
    missing_keys = sorted(_LEGACY_MANIFEST_KEYS.difference(manifest))
    unexpected_keys = sorted(set(manifest).difference(_LEGACY_MANIFEST_KEYS))
    if missing_keys or unexpected_keys:
        raise ValueError(
            f"transfer manifest schema mismatch: missing={missing_keys}, "
            f"unexpected={unexpected_keys}"
        )
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != LEGACY_TRANSFER_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("unsupported V2 transfer manifest")
    if manifest.get("bundle") != LEGACY_TRANSFER_BUNDLE_NAME:
        raise ValueError("unexpected V2 transfer bundle name")
    destination = manifest.get("destination")
    if (
        not isinstance(destination, str)
        or sha256(destination.encode("utf-8")).hexdigest()
        != LEGACY_TRANSFER_DESTINATION_SHA256
    ):
        raise ValueError("unexpected V2 transfer destination")
    _validate_created_at(manifest.get("created_at_utc"))
    if (
        type(manifest.get("materialization_schema_version")) is not int
        or manifest.get("materialization_schema_version")
        != LEGACY_MATERIALIZATION_SCHEMA_VERSION
    ):
        raise ValueError("unexpected materialization schema version")
    if manifest.get("node_order") != list(CANONICAL_NODE_ORDER):
        raise ValueError("transfer node order is not canonical")
    if manifest.get("excluded_categories") != list(LEGACY_EXCLUDED_CATEGORIES):
        raise ValueError("excluded_categories does not match the frozen exclusion policy")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest files must be a nonempty list")
    entries: dict[str, dict[str, object]] = {}
    authenticated_json_bytes: dict[str, bytes] = {}
    ordered_paths: list[str] = []
    observed_total = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("each manifest file entry must be an object")
        if set(entry) != _LEGACY_FILE_ENTRY_KEYS:
            raise ValueError("each manifest file entry must have the exact V2 schema")
        relative = _canonical_relative_path(entry.get("path"), field="manifest path")
        if relative in entries:
            raise ValueError(f"duplicate manifest path: {relative}")
        parts = tuple(part.casefold() for part in PurePosixPath(relative).parts)
        if _LEGACY_FORBIDDEN_PARTS.intersection(parts) or PurePosixPath(relative).name.casefold() in _LEGACY_FORBIDDEN_NAMES:
            raise ValueError(f"forbidden manifest path: {relative}")
        if any(part.startswith(".env") for part in parts):
            raise ValueError(f"environment file is forbidden: {relative}")
        local = _local_path(root, relative)
        if PurePosixPath(relative).name in {
            "materialization_manifest.json",
            "run_status.json",
            "bar_panel_manifest.json",
            "monthly_thresholds_manifest.json",
        }:
            data, snapshot = read_regular_file_bytes(local)
            authenticated_json_bytes[relative] = data
        else:
            snapshot = regular_file_snapshot(local)
        size = entry.get("size_bytes")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size != snapshot.size_bytes
        ):
            raise ValueError(f"size mismatch: {relative}")
        expected_hash = _valid_sha256(entry.get("sha256"), field=f"sha256 for {relative}")
        if expected_hash != snapshot.sha256:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        entries[relative] = entry
        ordered_paths.append(relative)
        observed_total += snapshot.size_bytes

    if ordered_paths != sorted(ordered_paths):
        raise ValueError("manifest file entries must be sorted by path")
    file_count = manifest.get("file_count")
    if not isinstance(file_count, int) or isinstance(file_count, bool) or file_count != len(files):
        raise ValueError("manifest file_count mismatch")
    total_size = manifest.get("total_size_bytes")
    if (
        not isinstance(total_size, int)
        or isinstance(total_size, bool)
        or total_size != observed_total
    ):
        raise ValueError("manifest total_size_bytes mismatch")

    _validate_materialization_binding(
        manifest,
        entries=entries,
        authenticated_json_bytes=authenticated_json_bytes,
    )
    expected_payload_hash = _valid_sha256(
        manifest.get("manifest_payload_sha256"),
        field="manifest_payload_sha256",
    )
    observed_payload_hash = _payload_sha256(manifest)
    if expected_payload_hash != observed_payload_hash:
        raise ValueError("manifest payload SHA-256 mismatch")
    return {
        "status": "verified",
        "file_count": len(files),
        "total_size_bytes": observed_total,
        "manifest_payload_sha256": observed_payload_hash,
        "materialization_manifest_sha256": manifest[
            "materialization_manifest_sha256"
        ],
        "materialization_data_directory": manifest[
            "materialization_data_directory"
        ],
        "destination_sha256": LEGACY_TRANSFER_DESTINATION_SHA256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Validate a frozen historical graph_model transfer manifest",
    )
    args = parser.parse_args(argv)
    manifest_path = args.manifest or (
        LEGACY_DEFAULT_MANIFEST if args.legacy else DEFAULT_MANIFEST
    )
    if args.legacy:
        result = verify_legacy_manifest(
            manifest_path,
            project_root=args.source_root,
        )
    else:
        result = verify_manifest(manifest_path, source_root=args.source_root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
