"""CLI for audited shock-follow bar panels and monthly causal thresholds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Sequence

import numpy as np
import pandas as pd

from gnn_for_futures.shock_follow.data.availability import load_retrospective_bar_data_contract
from gnn_for_futures.shock_follow.provenance.source import file_sha256, resolve_source_root
from gnn_for_futures.graph.regression import load_economic_prior_graph_snapshot
from gnn_for_futures.shock_follow.contract import (
    CANONICAL_NODE_ORDER,
    MATERIALIZATION_SCHEMA_VERSION,
    MATERIALIZATION_SOURCE_PATHS,
    canonical_json_sha256,
    node_order_sha256,
    shock_follow_contract_sha256,
    threshold_contract_sha256,
)
from gnn_for_futures.shock_follow.data.panel import (
    load_shock_follow_bar_panel,
    materialize_shock_follow_bar_panel,
    save_shock_follow_bar_panel,
)
from gnn_for_futures.shock_follow.data.thresholds import (
    load_monthly_threshold_table,
    materialize_causal_monthly_thresholds,
    save_monthly_threshold_table,
)


DEFAULT_SOURCE_ROOT = Path(".")
DEFAULT_RAW_DIR = Path("data/raw_data/15min")
DEFAULT_OUTPUT_DIR = Path("data/shock_follow_v1")
DEFAULT_CONTRACT = Path("data/contracts/bar_availability.retrospective.json")
DEFAULT_ECONOMIC_GRAPH = Path("data/graph_data/futures_graph_smm_simplified.html")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize exact 15-minute returns and causal monthly shock thresholds."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root used for source hashes and repository-relative evidence paths.",
    )
    parser.add_argument("--raw-15min-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--retrospective-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--economic-graph", type=Path, default=DEFAULT_ECONOMIC_GRAPH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deprecated and rejected; materialize into a new staging directory.",
    )
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "bar_panel": output_dir / "bar_panel.npz",
        "bar_manifest": output_dir / "bar_panel_manifest.json",
        "thresholds": output_dir / "monthly_thresholds.npz",
        "threshold_rows": output_dir / "monthly_thresholds.csv",
        "threshold_manifest": output_dir / "monthly_thresholds_manifest.json",
        "economic_prior": output_dir / "economic_prior.npy",
        "materialization_manifest": output_dir / "materialization_manifest.json",
        "status": output_dir / "run_status.json",
    }


def _materialization_source_hashes(source_root: Path) -> dict[str, str]:
    return {
        relative: file_sha256(source_root / relative)
        for relative in MATERIALIZATION_SOURCE_PATHS
    }


def _require_unchanged_materialization_source(
    source_root: Path,
    expected_hashes: dict[str, str],
) -> None:
    if _materialization_source_hashes(source_root) != expected_hashes:
        raise ValueError("materializer source files changed during materialization")


def materialize(args: argparse.Namespace) -> dict[str, object]:
    source_root = resolve_source_root(getattr(args, "source_root", None))
    paths = _artifact_paths(args.output_dir)
    if args.force:
        raise ValueError(
            "--force is disabled because a multi-file bundle cannot be replaced atomically; "
            "use a new staging output directory"
        )
    initial_source_hashes = _materialization_source_hashes(source_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args._output_dir_claimed = True
    _atomic_json(
        paths["status"],
        {"status": "running", "started_at_utc": _utc_now()},
    )

    bar_contract = load_retrospective_bar_data_contract(
        args.retrospective_contract, project_root=source_root
    )
    node_order = CANONICAL_NODE_ORDER
    economic, economic_graph_sha256 = load_economic_prior_graph_snapshot(
        node_order,
        args.economic_graph,
    )
    panel = materialize_shock_follow_bar_panel(
        args.raw_15min_dir, node_order, project_root=source_root
    )
    thresholds = materialize_causal_monthly_thresholds(
        panel.timestamps,
        panel.trading_dates,
        panel.signed_returns,
        node_order,
    )

    save_shock_follow_bar_panel(panel, paths["bar_panel"], paths["bar_manifest"])
    save_monthly_threshold_table(
        thresholds,
        paths["thresholds"],
        paths["threshold_rows"],
        paths["threshold_manifest"],
    )
    temporary_prior = paths["economic_prior"].with_name(paths["economic_prior"].name + ".tmp")
    with temporary_prior.open("wb") as stream:
        np.save(stream, economic.adjacency.astype(np.float32, copy=False), allow_pickle=False)
    os.replace(temporary_prior, paths["economic_prior"])

    # Independent disk reload ensures the success marker never blesses a partial artifact set.
    restored_panel = load_shock_follow_bar_panel(paths["bar_panel"], paths["bar_manifest"])
    restored_thresholds = load_monthly_threshold_table(
        paths["thresholds"], paths["threshold_manifest"]
    )
    restored_prior = np.load(paths["economic_prior"], allow_pickle=False)
    if not np.array_equal(restored_prior, economic.adjacency.astype(np.float32)):
        raise ValueError("economic prior failed persisted round-trip validation")
    if file_sha256(args.economic_graph) != economic_graph_sha256:
        raise ValueError("economic graph changed during materialization")

    manifest = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "bar_data_mode": "retrospective-finalized",
        "research_only": True,
        "strict_point_in_time_backtest": False,
        "bar_availability_contract_sha256": bar_contract.sha256,
        "bar_availability_contract_identity_sha256": bar_contract.identity_sha256,
        "bar_availability_evidence_sha256": {
            "support_response_document": (
                bar_contract.support_response_document_sha256
            ),
            "evidence_document": bar_contract.evidence_document_sha256,
        },
        "economic_graph_sha256": economic_graph_sha256,
        "bar_panel_materialization_sha256": restored_panel.materialization_sha256,
        "threshold_materialization_sha256": restored_thresholds.materialization_sha256,
        "threshold_input_sha256": restored_thresholds.input_sha256,
        "shock_follow_contract_sha256": shock_follow_contract_sha256(),
        "threshold_contract_sha256": threshold_contract_sha256(),
        "node_order": list(node_order),
        "node_order_sha256": node_order_sha256(node_order),
        "timestamp_count": len(restored_panel.timestamps),
        "first_timestamp": np.datetime_as_string(restored_panel.timestamps[0], unit="s"),
        "last_timestamp": np.datetime_as_string(restored_panel.timestamps[-1], unit="s"),
        "effective_month_count": len(restored_thresholds.months),
        "economic_directed_edge_count": int((restored_prior > 0.0).sum()),
        "regression_channel": "unavailable_missing_measured_available_at_manifest",
        "source_sha256": initial_source_hashes,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": sys.version,
        "artifacts": {
            name: path.name
            for name, path in paths.items()
            if name not in {"status", "materialization_manifest"}
        },
        "artifact_sha256": {
            name: file_sha256(path)
            for name, path in paths.items()
            if name not in {"status", "materialization_manifest"}
        },
    }
    _atomic_json(paths["materialization_manifest"], manifest)
    manifest_sha256 = canonical_json_sha256(manifest)
    _require_unchanged_materialization_source(source_root, initial_source_hashes)
    _atomic_json(
        paths["status"],
        {
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "materialization_manifest": paths["materialization_manifest"].name,
            "materialization_manifest_sha256": manifest_sha256,
            "materialization_manifest_file_sha256": file_sha256(
                paths["materialization_manifest"]
            ),
        },
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _artifact_paths(args.output_dir)
    args._output_dir_claimed = False
    try:
        manifest = materialize(args)
    except Exception as error:
        if not args._output_dir_claimed:
            raise
        _atomic_json(
            paths["status"],
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
        "Materialized "
        f"{manifest['timestamp_count']:,} timestamps, "
        f"{manifest['effective_month_count']} months, and "
        f"{len(manifest['node_order'])} nodes in {args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
