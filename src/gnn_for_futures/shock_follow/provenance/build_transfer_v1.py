"""Build a V1 source-layout schema-2 release manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
from typing import Sequence

from gnn_for_futures.shock_follow.provenance.source import (
    SOURCE_LAYOUT_SCHEMA_VERSION,
    build_source_manifest_from_hashes,
    canonical_source_paths,
    regular_file_snapshot,
    release_packaging_identity,
    resolve_source_root,
)
from gnn_for_futures.shock_follow.provenance.transfer_v1 import (
    DEFAULT_MANIFEST,
    SOURCE_KIND,
    TRANSFER_BUNDLE_NAME,
    TRANSFER_MANIFEST_SCHEMA_VERSION,
    verify_manifest,
)


DEFAULT_OUTPUT = DEFAULT_MANIFEST


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


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


def _safe_output_path(output: str | Path) -> Path:
    path = Path(output).absolute()
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("output manifest must be a regular file or not yet exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("output manifest path must not traverse a symlink")
    return path


def build_release_manifest(
    *,
    output: str | Path,
    expected_schema_version: int,
    expected_bundle: str,
    verifier,
    source_root: str | Path | None = None,
) -> dict[str, object]:
    root = resolve_source_root(source_root)
    paths = canonical_source_paths(root)
    entries: list[dict[str, object]] = []
    source_hashes: dict[str, str] = {}
    for relative in paths:
        local = root / relative
        snapshot = regular_file_snapshot(local)
        entries.append(
            {
                "path": relative,
                "size_bytes": snapshot.size_bytes,
                "sha256": snapshot.sha256,
            }
        )
        source_hashes[relative] = snapshot.sha256
    source_identity = build_source_manifest_from_hashes(source_hashes)
    packaging_identity = release_packaging_identity(root)
    output_path = _safe_output_path(output)
    protected_inputs = {
        (root / PurePosixPath(relative)).resolve()
        for relative in (*paths, str(packaging_identity["path"]))
    }
    if output_path.resolve() in protected_inputs:
        raise ValueError("output manifest must not overwrite a release input")
    payload: dict[str, object] = {
        "schema_version": expected_schema_version,
        "source_layout_schema_version": SOURCE_LAYOUT_SCHEMA_VERSION,
        "bundle": expected_bundle,
        "source_kind": SOURCE_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity": source_identity,
        "packaging_identity": packaging_identity,
        "file_count": len(entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "files": entries,
    }
    payload["manifest_payload_sha256"] = _payload_sha256(payload)

    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.is_symlink():
        raise ValueError("temporary output manifest must not be a symlink")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        verifier(temporary, source_root=root)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return payload


def build_manifest(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    source_root: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Build and verify one V1 source closure plus packaging identity."""
    if source_root is not None and project_root is not None:
        raise ValueError("pass only one of source_root or project_root")
    root_argument = source_root if source_root is not None else project_root
    payload = build_release_manifest(
        output=output,
        expected_schema_version=TRANSFER_MANIFEST_SCHEMA_VERSION,
        expected_bundle=TRANSFER_BUNDLE_NAME,
        verifier=verify_manifest,
        source_root=root_argument,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_manifest(output=args.output, source_root=args.source_root)
    print(
        json.dumps(
            {
                "status": "built_and_verified",
                "source_layout_schema_version": payload[
                    "source_layout_schema_version"
                ],
                "file_count": payload["file_count"],
                "total_size_bytes": payload["total_size_bytes"],
                "manifest_payload_sha256": payload["manifest_payload_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
