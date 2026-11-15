"""Verify V1 release-source manifests; retain legacy validation explicitly."""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

from gnn_for_futures.shock_follow.provenance.source import (
    PROHIBITED_RELEASE_PACKAGING_PATHS,
    RELEASE_PACKAGING_PATH,
    SOURCE_LAYOUT_SCHEMA_VERSION,
    _validate_source_manifest_against_hashes,
    canonical_source_paths,
    read_regular_file_bytes,
    regular_file_snapshot,
    resolve_source_root,
)
from gnn_for_futures.shock_follow.contract import (
    CANONICAL_NODE_ORDER,
    LEGACY_MATERIALIZATION_SCHEMA_VERSION,
    LEGACY_MATERIALIZATION_SOURCE_PATHS,
    canonical_json_sha256,
)


DEFAULT_MANIFEST = Path("shock_follow_v1_release_source_manifest.json")
TRANSFER_MANIFEST_SCHEMA_VERSION = 3
TRANSFER_BUNDLE_NAME = "gnn_for_futures_v1_release_source"
SOURCE_KIND = "release-source-and-packaging"

LEGACY_DEFAULT_MANIFEST = Path("shock_follow_v1_remote_manifest.json")
LEGACY_TRANSFER_MANIFEST_SCHEMA_VERSION = 2
LEGACY_TRANSFER_BUNDLE_NAME = "shock_follow_graphmixer_v1_remote_cuda"
LEGACY_TRANSFER_DESTINATION_SHA256 = (
    "a12d0528deafbc3d6efb4ea0c05863a27df903990f5d56b6c2f1631b3323973f"
)

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "source_layout_schema_version",
        "bundle",
        "source_kind",
        "created_at_utc",
        "source_identity",
        "packaging_identity",
        "file_count",
        "total_size_bytes",
        "files",
        "manifest_payload_sha256",
    }
)
_FILE_ENTRY_KEYS = frozenset({"path", "size_bytes", "sha256"})
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
LEGACY_EXCLUDED_CATEGORIES = (
    ".git",
    ".venv",
    ".env*",
    ".vscode",
    "TOP_faliure",
    "raw 15-minute CSVs because a validated derived bundle is transferred",
    "legacy schema-1 shock-follow bundle",
    "unrelated results/logs/checkpoints",
    "dynamic_forecasts.csv",
    "secrets",
)
LEGACY_FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    ".vscode",
    "TOP_faliure",
    "raw_data",
    "results",
}
LEGACY_FORBIDDEN_NAMES = {
    "dynamic_forecasts.csv",
    "eligible_cutoff_indices_all_valid.npy",
    "eligible_cutoff_indices_any_valid.npy",
}
LEGACY_REQUIRED_RUNTIME_FILES = (
    "requirements-shock-follow-v1.txt",
    "data/contracts/bar_availability.retrospective.json",
    "data/contracts/ricequant_bar_availability_evidence.md",
    "data/contracts/ricequant_support_response_2026-09-21.md",
    "data/graph_data/futures_graph_smm_simplified.html",
    "graph_model/availability.py",
    "graph_model/cross_futures_temporal_attention.py",
    "graph_model/evaluate_shock_follow_graphmixer.py",
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
    "graph_model/train_shock_follow_graphmixer.py",
)
LEGACY_REQUIRED_OPERATIONAL_FILES = (
    "graph_model/build_shock_follow_remote_manifest.py",
    "graph_model/shock_follow_graphmixer_v1_approval_proposal.md",
    "graph_model/shock_follow_graphmixer_v1_remote_runbook.md",
    "graph_model/verify_shock_follow_remote_manifest.py",
)
LEGACY_REQUIRED_TEST_FILES = (
    "graph_model/test_availability.py",
    "graph_model/test_cross_futures_temporal_attention.py",
    "graph_model/test_reg_graph_utils.py",
    "graph_model/test_shock_follow_artifacts.py",
    "graph_model/test_shock_follow_contract.py",
    "graph_model/test_shock_follow_data.py",
    "graph_model/test_shock_follow_dataset.py",
    "graph_model/test_shock_follow_graph.py",
    "graph_model/test_shock_follow_head.py",
    "graph_model/test_shock_follow_lifecycle.py",
    "graph_model/test_shock_follow_model.py",
    "graph_model/test_shock_follow_remote_manifest.py",
    "graph_model/test_shock_follow_thresholds.py",
    "graph_model/test_shock_follow_training.py",
    "graph_model/test_shock_follow_training_data.py",
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
    if not Path(data_directory).parts or Path(data_directory).parts[0] != "data":
        raise ValueError("materialization_data_directory must be below data/")
    expected_bundle_paths = {
        f"{data_directory}/{filename}" for filename in LEGACY_MATERIALIZATION_FILENAMES
    }
    expected_paths = {
        *LEGACY_REQUIRED_RUNTIME_FILES,
        *LEGACY_REQUIRED_OPERATIONAL_FILES,
        *LEGACY_REQUIRED_TEST_FILES,
        *expected_bundle_paths,
    }
    if set(entries) != expected_paths:
        missing = sorted(expected_paths.difference(entries))
        unexpected = sorted(set(entries).difference(expected_paths))
        raise ValueError(
            f"transfer file closure mismatch: missing={missing}, unexpected={unexpected}"
        )

    materialization_relative = f"{data_directory}/materialization_manifest.json"
    status_relative = f"{data_directory}/run_status.json"
    try:
        materialization_bytes = authenticated_json_bytes[materialization_relative]
        status_bytes = authenticated_json_bytes[status_relative]
    except KeyError as error:
        raise ValueError("authenticated materialization JSON bytes are unavailable") from error
    materialization = _json_bytes_without_duplicate_keys(
        materialization_bytes, name="materialization manifest"
    )
    status = _json_bytes_without_duplicate_keys(
        status_bytes, name="materialization status"
    )
    if (
        not isinstance(materialization, dict)
        or type(materialization.get("schema_version")) is not int
        or materialization.get("schema_version") != LEGACY_MATERIALIZATION_SCHEMA_VERSION
    ):
        raise ValueError("remote transfer requires a current schema materialization")
    if materialization.get("node_order") != list(CANONICAL_NODE_ORDER):
        raise ValueError("materialization node order is not canonical")
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
        relative = f"{data_directory}/{filename}"
        if artifact_hashes.get(name) != entries[relative].get("sha256"):
            raise ValueError(f"materialization artifact hash binding mismatch: {filename}")
    materialization_hash = entries[
        f"{data_directory}/materialization_manifest.json"
    ].get("sha256")
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


def _valid_release_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256") from error
    return value


def _canonical_release_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("release manifest path must be a nonempty string")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        "\\" in value
        or posix.is_absolute()
        or bool(windows.drive)
        or value != posix.as_posix()
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"unsafe or noncanonical release manifest path: {value!r}")
    if posix.parts[:1] == ("graph_model",):
        raise ValueError(
            "source-layout schema 2 refuses legacy graph_model paths; "
            "use verify_legacy_manifest for historical manifests"
        )
    return value


def _regular_release_source(root: Path, relative: str) -> Path:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"release source path traverses a symlink: {relative}")
    if not current.is_file():
        raise FileNotFoundError(
            f"release source is missing or not a regular file: {relative}"
        )
    return current


def verify_release_manifest(
    manifest_path: str | Path,
    *,
    expected_schema_version: int,
    expected_bundle: str,
    source_root: str | Path | None = None,
    project_root: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Verify one exact source-layout schema-2 release closure.

    ``project_root`` remains a compatibility alias for ``source_root``.  Passing
    both is rejected so path authority is never ambiguous.  When supplied, the
    independently approved byte hash and JSON content are checked from the same
    immutable manifest snapshot.
    """
    if source_root is not None and project_root is not None:
        raise ValueError("pass only one of source_root or project_root")
    root = resolve_source_root(
        source_root if source_root is not None else project_root
    )
    path = Path(manifest_path).absolute()
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("release manifest path must not traverse a symlink")
    manifest_bytes, manifest_snapshot = read_regular_file_bytes(path)
    if expected_manifest_sha256 is not None:
        approved_sha256 = _valid_release_sha256(
            expected_manifest_sha256,
            field="expected release manifest sha256",
        )
        if manifest_snapshot.sha256 != approved_sha256:
            raise ValueError(
                "release-source manifest byte SHA-256 disagrees with the "
                "independently approved hash"
            )
    manifest = _json_bytes_without_duplicate_keys(
        manifest_bytes,
        name="release manifest",
    )
    if not isinstance(manifest, dict):
        raise ValueError("release manifest must contain an object")
    if "source_layout_schema_version" not in manifest:
        raise ValueError(
            "legacy transfer manifest is not a schema-2 release source manifest; "
            "use verify_legacy_manifest explicitly"
        )
    missing = sorted(_MANIFEST_KEYS.difference(manifest))
    unexpected = sorted(set(manifest).difference(_MANIFEST_KEYS))
    if missing or unexpected:
        raise ValueError(
            f"release manifest schema mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != expected_schema_version:
        raise ValueError("unsupported release manifest schema")
    layout_version = manifest.get("source_layout_schema_version")
    if (
        type(layout_version) is not int
        or layout_version != SOURCE_LAYOUT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported release source_layout_schema_version")
    if manifest.get("bundle") != expected_bundle:
        raise ValueError("unexpected release bundle name")
    if manifest.get("source_kind") != SOURCE_KIND:
        raise ValueError("release manifest must describe source and packaging")
    _validate_created_at(manifest.get("created_at_utc"))
    source_identity = manifest.get("source_identity")
    if not isinstance(source_identity, dict):
        raise ValueError("release manifest source_identity must be an object")
    if (
        type(source_identity.get("source_layout_schema_version")) is not int
        or type(source_identity.get("source_count")) is not int
    ):
        raise ValueError("release source_identity versions and counts must be integers")
    packaging_identity = manifest.get("packaging_identity")
    if (
        not isinstance(packaging_identity, dict)
        or set(packaging_identity) != _FILE_ENTRY_KEYS
    ):
        raise ValueError("release manifest packaging_identity must have the exact schema")
    prohibited_packaging = [
        relative
        for relative in PROHIBITED_RELEASE_PACKAGING_PATHS
        if (root / PurePosixPath(relative)).is_symlink()
        or (root / PurePosixPath(relative)).exists()
    ]
    if prohibited_packaging:
        raise ValueError(
            "release source contains prohibited alternate packaging controls: "
            f"{prohibited_packaging}"
        )
    packaging_path = _canonical_release_path(packaging_identity.get("path"))
    if packaging_path != RELEASE_PACKAGING_PATH:
        raise ValueError("release packaging path mismatch")
    packaging_snapshot = regular_file_snapshot(
        _regular_release_source(root, packaging_path)
    )
    packaging_size = packaging_identity.get("size_bytes")
    if (
        type(packaging_size) is not int
        or packaging_size < 0
        or packaging_size != packaging_snapshot.size_bytes
    ):
        raise ValueError("release packaging size mismatch")
    packaging_sha256 = _valid_release_sha256(
        packaging_identity.get("sha256"),
        field="release packaging sha256",
    )
    if packaging_sha256 != packaging_snapshot.sha256:
        raise ValueError("release packaging SHA-256 mismatch")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("release manifest files must be a nonempty list")
    expected_paths = canonical_source_paths(root)
    observed_paths: list[str] = []
    observed_total = 0
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != _FILE_ENTRY_KEYS:
            raise ValueError("release file entries must have the exact schema")
        relative = _canonical_release_path(entry.get("path"))
        if relative in observed_paths:
            raise ValueError(f"duplicate release manifest path: {relative}")
        local = _regular_release_source(root, relative)
        snapshot = regular_file_snapshot(local)
        size = entry.get("size_bytes")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size != snapshot.size_bytes
        ):
            raise ValueError(f"size mismatch: {relative}")
        digest = _valid_release_sha256(
            entry.get("sha256"), field=f"sha256 for {relative}"
        )
        if digest != snapshot.sha256:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        observed_paths.append(relative)
        observed_total += snapshot.size_bytes
    if observed_paths != sorted(observed_paths):
        raise ValueError("release manifest file entries must be sorted by path")
    if tuple(observed_paths) != expected_paths:
        missing_paths = sorted(set(expected_paths).difference(observed_paths))
        unexpected_paths = sorted(set(observed_paths).difference(expected_paths))
        raise ValueError(
            "release source closure mismatch: "
            f"missing={missing_paths}, unexpected={unexpected_paths}"
        )
    file_count = manifest.get("file_count")
    if type(file_count) is not int or file_count != len(files):
        raise ValueError("release manifest file_count mismatch")
    total_size = manifest.get("total_size_bytes")
    if type(total_size) is not int or total_size != observed_total:
        raise ValueError("release manifest total_size_bytes mismatch")
    observed_hashes = {
        str(entry["path"]): str(entry["sha256"])
        for entry in files
    }
    validated_identity = _validate_source_manifest_against_hashes(
        source_identity,
        observed_hashes,
    )
    if validated_identity["source_sha256"] != observed_hashes:
        raise ValueError("release file hashes disagree with source_identity")
    expected_payload_hash = _valid_release_sha256(
        manifest.get("manifest_payload_sha256"),
        field="manifest_payload_sha256",
    )
    observed_payload_hash = _payload_sha256(manifest)
    if expected_payload_hash != observed_payload_hash:
        raise ValueError("release manifest payload SHA-256 mismatch")
    return {
        "status": "verified",
        "source_layout_schema_version": SOURCE_LAYOUT_SCHEMA_VERSION,
        "bundle": expected_bundle,
        "file_count": len(files),
        "total_size_bytes": observed_total,
        "packaging_path": packaging_path,
        "packaging_size_bytes": packaging_size,
        "packaging_sha256": packaging_sha256,
        "manifest_payload_sha256": observed_payload_hash,
    }


def verify_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    source_root: str | Path | None = None,
    project_root: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Verify a V1 schema-2 source closure plus packaging identity."""
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
    """Fail closed on unsafe paths, changed bytes, or an incomplete bundle."""
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
        raise ValueError("unsupported transfer manifest")
    if manifest.get("bundle") != LEGACY_TRANSFER_BUNDLE_NAME:
        raise ValueError("unexpected transfer bundle name")
    destination = manifest.get("destination")
    if (
        not isinstance(destination, str)
        or sha256(destination.encode("utf-8")).hexdigest()
        != LEGACY_TRANSFER_DESTINATION_SHA256
    ):
        raise ValueError("unexpected transfer destination")
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
    observed_total = 0
    ordered_paths: list[str] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("each manifest file entry must be an object")
        if set(entry) != _LEGACY_FILE_ENTRY_KEYS:
            raise ValueError("each manifest file entry must have the exact V1 schema")
        relative = _canonical_relative_path(entry.get("path"), field="manifest path")
        path = PurePosixPath(relative)
        if relative in entries:
            raise ValueError(f"duplicate manifest path: {relative}")
        entries[relative] = entry
        ordered_paths.append(relative)
        folded_parts = tuple(part.casefold() for part in path.parts)
        if (
            {part.casefold() for part in LEGACY_FORBIDDEN_PARTS}.intersection(
                folded_parts
            )
            or path.name.casefold()
            in {name.casefold() for name in LEGACY_FORBIDDEN_NAMES}
        ):
            raise ValueError(f"forbidden manifest path: {relative}")
        if any(part.startswith(".env") for part in folded_parts):
            raise ValueError(f"environment file is forbidden: {relative}")
        local = _local_path(root, relative)
        if PurePosixPath(relative).name in {
            "materialization_manifest.json",
            "run_status.json",
        }:
            data, snapshot = read_regular_file_bytes(local)
            authenticated_json_bytes[relative] = data
        else:
            snapshot = regular_file_snapshot(local)
        size = entry.get("size_bytes")
        if type(size) is not int or size < 0 or size != snapshot.size_bytes:
            raise ValueError(f"size mismatch: {relative}")
        expected_hash = _valid_sha256(
            entry.get("sha256"), field=f"sha256 for {relative}"
        )
        if expected_hash != snapshot.sha256:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        observed_total += snapshot.size_bytes
    if ordered_paths != sorted(ordered_paths):
        raise ValueError("manifest file entries must be sorted by path")
    file_count = manifest.get("file_count")
    if type(file_count) is not int or file_count != len(files):
        raise ValueError("manifest file_count mismatch")
    total_size = manifest.get("total_size_bytes")
    if type(total_size) is not int or total_size != observed_total:
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
