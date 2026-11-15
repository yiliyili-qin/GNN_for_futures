"""Source-layout provenance and frozen transfer-identity validation.

Source-layout schema 2 identifies the released ``src/gnn_for_futures`` Python
closure.  It is intentionally distinct from legacy formal-run manifests whose
paths begin with ``graph_model/``; legacy records are recognized only through an
explicit read-only validation branch and are never translated to schema 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Mapping


SOURCE_LAYOUT_SCHEMA_VERSION = 2
LEGACY_SOURCE_LAYOUT_SCHEMA_VERSION = 1
SOURCE_LAYOUT_PACKAGE_ROOT = "src/gnn_for_futures"
RELEASE_PACKAGING_PATH = "pyproject.toml"
PROHIBITED_RELEASE_PACKAGING_PATHS = (
    "setup.py",
    "setup.cfg",
    "MANIFEST.in",
)
DEFAULT_SOURCE_ROOT = Path(".")
DEFAULT_RELEASE_SOURCE_MANIFEST = Path(
    "shock_follow_v2_release_source_manifest.json"
)
DEFAULT_LEGACY_TRANSFER_MANIFEST = Path("shock_follow_v2_remote_manifest.json")

# Exact historical closure written by the approved graph_model V2 trainer.  This
# remains read-only provenance and is not a path migration map.
LEGACY_FORMAL_V2_SOURCE_PATHS = (
    "graph_model/availability.py",
    "graph_model/cross_futures_temporal_attention.py",
    "graph_model/reg_graph_utils.py",
    "graph_model/shock_follow_artifacts.py",
    "graph_model/shock_follow_checkpoint.py",
    "graph_model/shock_follow_checkpoint_v2.py",
    "graph_model/shock_follow_confirmation_gate_v2.py",
    "graph_model/shock_follow_contract.py",
    "graph_model/shock_follow_data.py",
    "graph_model/shock_follow_dataset.py",
    "graph_model/shock_follow_graph.py",
    "graph_model/shock_follow_graphmixer_v2_hierarchical_approval_proposal.md",
    "graph_model/shock_follow_head.py",
    "graph_model/shock_follow_head_v2.py",
    "graph_model/shock_follow_metrics_v2.py",
    "graph_model/shock_follow_model.py",
    "graph_model/shock_follow_model_v2.py",
    "graph_model/shock_follow_split_v2.py",
    "graph_model/shock_follow_thresholds.py",
    "graph_model/shock_follow_training_data.py",
    "graph_model/train_shock_follow_graphmixer_v2.py",
    "graph_model/evaluate_shock_follow_graphmixer_v2_confirmation.py",
    "graph_model/verify_shock_follow_v2_remote_manifest.py",
)

_SCHEMA_2_KEYS = frozenset(
    {
        "source_layout_schema_version",
        "source_sha256",
        "source_count",
        "source_manifest_sha256",
    }
)
_LEGACY_SCHEMA_1_KEYS = frozenset(
    {"schema_version", "source_sha256", "source_count", "source_manifest_sha256"}
)
_IGNORED_SOURCE_DIRECTORY_NAMES = frozenset({"__pycache__"})
_PROHIBITED_PACKAGE_DIRECTORY_NAMES = frozenset({"tests", "docs", "results"})


@dataclass(frozen=True)
class RegularFileSnapshot:
    """Size and digest read from one descriptor-stable regular-file snapshot."""

    size_bytes: int
    sha256: str


def _read_regular_file(
    path: str | Path,
    chunk_size: int,
    *,
    retain_bytes: bool,
) -> tuple[RegularFileSnapshot, bytes | None]:
    file_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    try:
        descriptor = os.open(file_path, flags)
    except OSError as error:
        raise FileNotFoundError(
            f"regular nonsymlink file was not found: {file_path}"
        ) from error
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError(f"path is not a regular file: {file_path}")
        digest = sha256()
        chunks: list[bytes] | None = [] if retain_bytes else None
        size = 0
        while True:
            chunk = os.read(descriptor, chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        completed_stat = os.fstat(descriptor)
        unchanged_metadata = (
            os.path.samestat(opened_stat, completed_stat)
            and completed_stat.st_size == opened_stat.st_size
            and completed_stat.st_mtime_ns == opened_stat.st_mtime_ns
            and completed_stat.st_ctime_ns == opened_stat.st_ctime_ns
        )
        if size != completed_stat.st_size or not unchanged_metadata:
            raise ValueError(f"file changed while reading: {file_path}")
        try:
            path_stat = os.stat(file_path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"file path changed while reading: {file_path}") from error
        if not stat.S_ISREG(path_stat.st_mode) or not os.path.samestat(
            completed_stat, path_stat
        ):
            raise ValueError(f"file path changed while reading: {file_path}")
        snapshot = RegularFileSnapshot(size, digest.hexdigest())
        return snapshot, b"".join(chunks) if chunks is not None else None
    finally:
        os.close(descriptor)


def regular_file_snapshot(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> RegularFileSnapshot:
    """Open once, validate with ``fstat``, and hash that same file object.

    ``O_NOFOLLOW`` is used where the platform exposes it.  On every platform the
    opened descriptor is compared with a fresh no-follow path stat before its
    snapshot is accepted, rejecting an atomic path replacement during the read.
    """
    snapshot, _ = _read_regular_file(path, chunk_size, retain_bytes=False)
    return snapshot


def read_regular_file_bytes(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> tuple[bytes, RegularFileSnapshot]:
    """Return bytes and identity from one descriptor-stable file read."""
    snapshot, data = _read_regular_file(path, chunk_size, retain_bytes=True)
    if data is None:
        raise AssertionError("retained regular-file bytes are unavailable")
    return data, snapshot


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 from one descriptor-stable regular-file snapshot."""
    return regular_file_snapshot(path, chunk_size).sha256


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def resolve_source_root(source_root: str | Path | None = None) -> Path:
    """Resolve an explicit source root, defaulting only to the caller's CWD."""
    supplied = Path.cwd() if source_root is None else Path(source_root)
    absolute = supplied.absolute()
    if absolute.is_symlink() or not absolute.is_dir():
        raise ValueError("source_root must be a regular nonsymlink directory")
    return absolute.resolve()


def _regular_source_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or path.is_absolute()
        or relative != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"source path is unsafe or noncanonical: {relative!r}")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"source path traverses a symlink: {relative}")
    if not current.is_file():
        raise FileNotFoundError(f"source closure file is missing: {current}")
    return current


def canonical_source_paths(
    source_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return the canonical schema-2 production Python closure.

    The closure is restricted to ``src/gnn_for_futures`` and therefore cannot
    include repository tests, docs, data, or results.  Python files in package
    test/docs/results paths, and test-named Python files anywhere in the package,
    are rejected rather than silently omitted from install provenance.
    """
    root = resolve_source_root(source_root)
    package_root = root / PurePosixPath(SOURCE_LAYOUT_PACKAGE_ROOT)
    if package_root.is_symlink() or not package_root.is_dir():
        raise FileNotFoundError(
            f"package source directory was not found: {package_root}"
        )

    paths: list[str] = []
    for current_name, directory_names, file_names in os.walk(
        package_root, followlinks=False
    ):
        current = Path(current_name)
        retained_directories: list[str] = []
        for name in directory_names:
            candidate = current / name
            if candidate.is_symlink():
                raise ValueError(
                    "schema-2 source closure must not traverse a symlink: "
                    f"{candidate}"
                )
            folded = name.casefold()
            if folded in _PROHIBITED_PACKAGE_DIRECTORY_NAMES:
                relative = candidate.relative_to(root).as_posix()
                raise ValueError(
                    "schema-2 package source contains a prohibited directory: "
                    f"{relative}"
                )
            if folded not in _IGNORED_SOURCE_DIRECTORY_NAMES:
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in file_names:
            if not name.endswith(".py"):
                continue
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            if name.startswith("test_") or name == "conftest.py":
                raise ValueError(
                    "schema-2 package source contains a prohibited test-like "
                    f"Python file: {relative}"
                )
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(
                    f"schema-2 source must be a regular nonsymlink file: {candidate}"
                )
            if not relative.startswith(SOURCE_LAYOUT_PACKAGE_ROOT + "/"):
                raise AssertionError("schema-2 source escaped its package root")
            paths.append(relative)
    paths.sort()
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("schema-2 source closure must be nonempty and unique")
    return tuple(paths)


def source_hashes(source_root: str | Path | None = None) -> dict[str, str]:
    """Hash the canonical package-layout schema-2 source closure."""
    root = resolve_source_root(source_root)
    return {
        relative: file_sha256(_regular_source_path(root, relative))
        for relative in canonical_source_paths(root)
    }


def require_unchanged_source_hashes(
    source_root: str | Path | None,
    expected_hashes: Mapping[str, str],
    *,
    operation: str,
) -> None:
    """Fail if the package source closure changed during one operation."""
    if source_hashes(source_root) != dict(expected_hashes):
        raise ValueError(f"source files changed during {operation}")


def release_packaging_identity(
    source_root: str | Path | None = None,
) -> dict[str, object]:
    """Bind the sole permitted packaging control file for this release."""
    root = resolve_source_root(source_root)
    prohibited = []
    for relative in PROHIBITED_RELEASE_PACKAGING_PATHS:
        candidate = root / PurePosixPath(relative)
        if candidate.is_symlink() or candidate.exists():
            prohibited.append(relative)
    if prohibited:
        raise ValueError(
            "release source contains prohibited alternate packaging controls: "
            f"{prohibited}"
        )
    path = _regular_source_path(root, RELEASE_PACKAGING_PATH)
    snapshot = regular_file_snapshot(path)
    return {
        "path": RELEASE_PACKAGING_PATH,
        "size_bytes": snapshot.size_bytes,
        "sha256": snapshot.sha256,
    }


def legacy_formal_v2_source_hashes(
    source_root: str | Path | None = None,
) -> dict[str, str]:
    """Hash the frozen legacy V2 closure without translating any path."""
    root = resolve_source_root(source_root)
    return {
        relative: file_sha256(_regular_source_path(root, relative))
        for relative in LEGACY_FORMAL_V2_SOURCE_PATHS
    }


def build_source_manifest_from_hashes(
    hashes: Mapping[str, str],
) -> dict[str, object]:
    """Build a schema-2 identity from caller-authenticated file snapshots."""
    source_sha256 = dict(hashes)
    core: dict[str, object] = {
        "source_layout_schema_version": SOURCE_LAYOUT_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "source_count": len(source_sha256),
    }
    return {**core, "source_manifest_sha256": _canonical_json_sha256(core)}


def build_source_manifest(
    source_root: str | Path | None = None,
) -> dict[str, object]:
    """Build an unapproved schema-2 package source identity.

    This only records current package bytes.  It does not create experiment
    approval and does not assert identity with any historical formal run.
    """
    return build_source_manifest_from_hashes(source_hashes(source_root))


def source_layout_schema_version(manifest: Mapping[str, object]) -> int:
    """Recognize schema 2 or the exact legacy schema-1 record shape."""
    if "source_layout_schema_version" in manifest:
        version = manifest.get("source_layout_schema_version")
        if type(version) is not int or version != SOURCE_LAYOUT_SCHEMA_VERSION:
            raise ValueError(f"unsupported source_layout_schema_version={version!r}")
        if "schema_version" in manifest:
            raise ValueError("source manifest must not mix layout and legacy schema fields")
        return SOURCE_LAYOUT_SCHEMA_VERSION
    version = manifest.get("schema_version")
    if type(version) is int and version == LEGACY_SOURCE_LAYOUT_SCHEMA_VERSION:
        return LEGACY_SOURCE_LAYOUT_SCHEMA_VERSION
    raise ValueError("source manifest has no recognized source-layout schema")


def _normalized_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("source manifest must contain a nonempty source_sha256 map")
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("source manifest paths must be nonempty strings")
        if (
            not isinstance(raw_digest, str)
            or len(raw_digest) != 64
            or raw_digest != raw_digest.lower()
        ):
            raise ValueError(f"source_sha256[{raw_name!r}] is not a lowercase SHA-256")
        try:
            int(raw_digest, 16)
        except ValueError as error:
            raise ValueError(
                f"source_sha256[{raw_name!r}] is not a lowercase SHA-256"
            ) from error
        normalized[raw_name] = raw_digest
    return normalized


def _validate_source_manifest_against_hashes(
    manifest: Mapping[str, object],
    expected_hashes: Mapping[str, str],
    *,
    allow_legacy: bool = False,
) -> dict[str, object]:
    """Validate manifest structure against already authenticated file snapshots."""
    payload = dict(manifest)
    version = source_layout_schema_version(payload)
    expected_keys = (
        _SCHEMA_2_KEYS
        if version == SOURCE_LAYOUT_SCHEMA_VERSION
        else _LEGACY_SCHEMA_1_KEYS
    )
    if set(payload) != expected_keys:
        raise ValueError(
            f"source manifest schema {version} has unexpected keys: "
            f"{sorted(set(payload).symmetric_difference(expected_keys))}"
        )
    hashes = _normalized_hashes(payload.get("source_sha256"))

    if version == SOURCE_LAYOUT_SCHEMA_VERSION:
        legacy_paths = sorted(
            path for path in hashes if PurePosixPath(path).parts[:1] == ("graph_model",)
        )
        if legacy_paths:
            raise ValueError(
                "source-layout schema 2 explicitly refuses legacy graph_model paths: "
                f"{legacy_paths}"
            )
        core_version_key = "source_layout_schema_version"
    else:
        if not allow_legacy:
            raise ValueError(
                "legacy source schema 1 is read-only and requires explicit allow_legacy=True"
            )
        if set(hashes) != set(LEGACY_FORMAL_V2_SOURCE_PATHS):
            raise ValueError("legacy source manifest closure is not the frozen formal V2 closure")
        core_version_key = "schema_version"

    if hashes != dict(expected_hashes):
        raise ValueError("current source bytes disagree with the source manifest")
    source_count = payload.get("source_count")
    if type(source_count) is not int or source_count != len(hashes):
        raise ValueError("source manifest count mismatch")
    core = {
        core_version_key: version,
        "source_sha256": hashes,
        "source_count": len(hashes),
    }
    manifest_hash = _canonical_json_sha256(core)
    if payload.get("source_manifest_sha256") != manifest_hash:
        raise ValueError("source manifest canonical hash mismatch")
    return {
        "source_layout_schema_version": version,
        "legacy": version == LEGACY_SOURCE_LAYOUT_SCHEMA_VERSION,
        "source_sha256": hashes,
        "source_manifest_sha256": manifest_hash,
    }


def validate_source_manifest(
    manifest: Mapping[str, object],
    *,
    source_root: str | Path | None = None,
    allow_legacy: bool = False,
) -> dict[str, object]:
    """Validate one source manifest against current descriptor snapshots.

    Legacy schema 1 is rejected unless ``allow_legacy=True``.  Even then it is
    checked only against its frozen ``graph_model`` closure and remains a legacy
    identity; no path is mapped into the package layout.
    """
    payload = dict(manifest)
    version = source_layout_schema_version(payload)
    expected_hashes = (
        source_hashes(source_root)
        if version == SOURCE_LAYOUT_SCHEMA_VERSION
        else legacy_formal_v2_source_hashes(source_root)
    )
    return _validate_source_manifest_against_hashes(
        payload,
        expected_hashes,
        allow_legacy=allow_legacy,
    )


def _expected_manifest_sha256(expected_sha256: object, *, option: str) -> str:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or expected_sha256 != expected_sha256.lower()
    ):
        raise ValueError(f"{option} must be one lowercase hexadecimal SHA-256")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError(
            f"{option} must be one lowercase hexadecimal SHA-256"
        ) from error
    return expected_sha256


def validate_release_source_identity(
    manifest_path: str | Path,
    expected_sha256: object,
    *,
    source_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate the current package-layout V2 release identity.

    The manifest binds the schema-2 Python source closure and packaging metadata.
    The independent schema-3 materialization validator binds data artifacts later
    in the training/confirmation preflight.
    """
    expected = _expected_manifest_sha256(
        expected_sha256,
        option="--expected-release-source-manifest-sha256",
    )
    root = resolve_source_root(source_root)
    path = Path(manifest_path)

    # Local import keeps this neutral source module dependency-light and avoids a
    # source/transfer import cycle.  The verifier reads the path once, then hashes
    # and parses that same immutable bytes snapshot.
    from gnn_for_futures.shock_follow.provenance.transfer_v2 import verify_manifest

    verification = verify_manifest(
        path,
        source_root=root,
        expected_manifest_sha256=expected,
    )
    observed_manifest_sha256 = expected
    return {
        "release_source_manifest_payload_sha256": verification[
            "manifest_payload_sha256"
        ],
        "release_source_manifest_file_sha256": observed_manifest_sha256,
        "release_source_manifest_file_count": verification["file_count"],
        "release_source_manifest_total_size_bytes": verification[
            "total_size_bytes"
        ],
        "release_packaging_path": verification["packaging_path"],
        "release_packaging_size_bytes": verification["packaging_size_bytes"],
        "release_packaging_sha256": verification["packaging_sha256"],
        "release_source_bundle": verification["bundle"],
        "release_source_layout_schema_version": verification[
            "source_layout_schema_version"
        ],
    }


def require_unchanged_release_source_identity(
    manifest_path: str | Path,
    expected_sha256: object,
    expected_identity: Mapping[str, object],
    *,
    source_root: str | Path | None = None,
    operation: str,
) -> None:
    """Fail if an approved release manifest, source, or packaging changed."""
    try:
        observed = validate_release_source_identity(
            manifest_path,
            expected_sha256,
            source_root=source_root,
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"release source or packaging changed during {operation}"
        ) from error
    if observed != dict(expected_identity):
        raise ValueError(
            f"release source or packaging changed during {operation}"
        )


def validate_legacy_transfer_identity(
    manifest_path: str | Path,
    data_dir: str | Path,
    expected_sha256: object,
    *,
    source_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate the frozen legacy V2 transfer identity without path translation."""
    expected = _expected_manifest_sha256(
        expected_sha256,
        option="expected legacy transfer manifest SHA-256",
    )
    root = resolve_source_root(source_root)
    path = Path(manifest_path)

    from gnn_for_futures.shock_follow.provenance.transfer_v2 import (
        verify_legacy_manifest,
    )

    verification = verify_legacy_manifest(
        path,
        project_root=root,
        expected_manifest_sha256=expected,
    )
    observed_manifest_sha256 = expected
    try:
        expected_data_dir = (
            root / str(verification["materialization_data_directory"])
        ).resolve()
    except (KeyError, TypeError) as error:
        raise ValueError(
            "legacy transfer manifest lacks materialization_data_directory"
        ) from error
    if Path(data_dir).resolve() != expected_data_dir:
        raise ValueError(
            "data_dir disagrees with the independently verified legacy transfer manifest"
        )
    return {
        "transfer_manifest_payload_sha256": verification[
            "manifest_payload_sha256"
        ],
        "transfer_manifest_file_sha256": observed_manifest_sha256,
        "transfer_manifest_file_count": verification["file_count"],
        "transfer_manifest_total_size_bytes": verification["total_size_bytes"],
        "transfer_destination_sha256": verification["destination_sha256"],
        "materialization_manifest_file_sha256": verification[
            "materialization_manifest_sha256"
        ],
    }
