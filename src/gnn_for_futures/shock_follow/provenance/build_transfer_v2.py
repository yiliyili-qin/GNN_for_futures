"""Build a V2 source-layout schema-2 release manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gnn_for_futures.shock_follow.provenance.build_transfer_v1 import (
    build_release_manifest,
)
from gnn_for_futures.shock_follow.provenance.transfer_v2 import (
    DEFAULT_MANIFEST,
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


def build_manifest(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    source_root: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Build and verify one V2 source closure plus packaging identity."""
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
