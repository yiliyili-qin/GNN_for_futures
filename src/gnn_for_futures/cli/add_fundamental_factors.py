"""Append approved bar-frequency and lagged inventory factors to graph futures."""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from gnn_for_futures.features.technical import (
    FUNDAMENTAL_FACTOR_COLUMNS,
    INVENTORY_FACTOR_COLUMNS,
    calculate_fundamental_factors,
    merge_daily_inventory_factors,
)


OUTPUT_COLUMNS = FUNDAMENTAL_FACTOR_COLUMNS + INVENTORY_FACTOR_COLUMNS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append 15-minute price factors and lagged daily inventory factors."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw_data/15min"))
    parser.add_argument(
        "--inventory-dir", type=Path, default=Path("data/raw_data/warehouse_stocks")
    )
    parser.add_argument(
        "--graph-path", type=Path, default=Path("data/graph_data/economic_graph.csv")
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("data/raw_data/15min/quality_reports")
    )
    parser.add_argument("--symbols", nargs="+", help="Optional subset of graph futures.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_graph_futures(graph_path: Path) -> list[str]:
    graph = pd.read_csv(graph_path, dtype=str, keep_default_na=False)
    required_columns = {"code", "is_future"}
    missing_columns = sorted(required_columns - set(graph.columns))
    if missing_columns:
        raise ValueError(f"graph is missing required columns: {missing_columns}")
    symbols = graph.loc[graph["is_future"].str.lower().eq("true"), "code"].str.upper()
    if symbols.duplicated().any():
        raise ValueError("graph contains duplicate futures codes.")
    return sorted(symbols.tolist())


def validate_bars(frame: pd.DataFrame, path: Path) -> None:
    required_columns = {
        "underlying_symbol",
        "datetime",
        "trading_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
        "return_15m",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")
    timestamps = pd.to_datetime(frame["datetime"], errors="coerce")
    trading_dates = pd.to_datetime(frame["trading_date"], errors="coerce")
    if timestamps.isna().any() or trading_dates.isna().any():
        raise ValueError(f"{path} contains invalid datetime or trading_date values.")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError(f"{path} datetime values must be strictly increasing and unique.")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_symbol(
    bars_path: Path,
    inventory_path: Path,
    *,
    dry_run: bool,
) -> dict[str, object]:
    original = pd.read_csv(bars_path, low_memory=False)
    validate_bars(original, bars_path)
    symbol = bars_path.stem.upper()
    symbol_values = original["underlying_symbol"].astype(str).str.upper()
    if not symbol_values.eq(symbol).all():
        raise ValueError(f"{bars_path} contains rows for another underlying symbol.")
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing inventory file: {inventory_path}")

    source_columns = [column for column in original.columns if column not in OUTPUT_COLUMNS]
    source = original.loc[:, source_columns].copy()
    inventory = pd.read_csv(inventory_path, low_memory=False)
    features = calculate_fundamental_factors(source)
    features = merge_daily_inventory_factors(features, inventory, lag_trading_days=1)
    features["trading_date"] = source["trading_date"].to_numpy()
    features = features.loc[:, source_columns + list(OUTPUT_COLUMNS)]

    assert_frame_equal(
        features.loc[:, source_columns],
        source,
        check_dtype=True,
        check_exact=True,
    )
    if len(features) != len(original):
        raise AssertionError(f"{symbol}: row count changed during factor calculation.")
    numeric_outputs = features.loc[:, OUTPUT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric_outputs.to_numpy(dtype=float)).any():
        raise AssertionError(f"{symbol}: output factors contain infinite values.")
    valid_rsi = features["rsi_14"].dropna()
    if not valid_rsi.between(0, 100).all():
        raise AssertionError(f"{symbol}: rsi_14 is outside [0, 100].")
    if not features["atr_14"].dropna().ge(0).all():
        raise AssertionError(f"{symbol}: atr_14 contains negative values.")
    deliverable_expected = features["on_warrant"] + features["effective_forecast"]
    if not features["deliverable"].equals(deliverable_expected):
        raise AssertionError(f"{symbol}: deliverable does not equal its available components.")

    if not dry_run:
        atomic_write_csv(features, bars_path)

    summary: dict[str, object] = {
        "symbol": symbol,
        "rows": len(features),
        "first_datetime": features["datetime"].iloc[0],
        "last_datetime": features["datetime"].iloc[-1],
        "source_columns": len(source_columns),
        "output_columns": len(OUTPUT_COLUMNS),
    }
    for column in OUTPUT_COLUMNS:
        values = pd.to_numeric(features[column], errors="coerce")
        summary[f"{column}_non_null"] = int(values.notna().sum())
        summary[f"{column}_missing_rate"] = float(values.isna().mean())
        summary[f"{column}_min"] = float(values.min()) if values.notna().any() else None
        summary[f"{column}_max"] = float(values.max()) if values.notna().any() else None
    return summary


def write_reports(
    summaries: list[dict[str, object]],
    report_dir: Path,
    *,
    dry_run: bool,
    graph_path: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries).sort_values("symbol")
    mode = "dry_run" if dry_run else "write"
    summary.to_csv(report_dir / f"fundamental_factor_{mode}_summary.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "frequency_mode": "15min_bar",
        "inventory_lag_trading_days": 1,
        "rolling_std_ddof": 0,
        "missing_inventory_policy": "preserve_nan",
        "return_policy": "retain_existing_return_15m_without_adding_return",
        "graph_path": str(graph_path),
        "symbol_count": len(summary),
        "symbols": summary["symbol"].tolist(),
        "factor_columns": list(OUTPUT_COLUMNS),
    }
    (report_dir / f"fundamental_factor_{mode}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    graph_symbols = load_graph_futures(args.graph_path)
    if args.symbols:
        requested = {symbol.upper() for symbol in args.symbols}
        unknown = sorted(requested - set(graph_symbols))
        if unknown:
            raise ValueError(f"Requested symbols are not graph futures: {unknown}")
        graph_symbols = [symbol for symbol in graph_symbols if symbol in requested]
    if not graph_symbols:
        raise ValueError("No graph futures selected.")

    summaries: list[dict[str, object]] = []
    for symbol in graph_symbols:
        bars_path = args.data_dir / f"{symbol}.csv"
        if not bars_path.exists():
            raise FileNotFoundError(f"Missing 15-minute file for graph future: {bars_path}")
        summary = process_symbol(
            bars_path,
            args.inventory_dir / f"{symbol}.csv",
            dry_run=args.dry_run,
        )
        summaries.append(summary)
        action = "Validated" if args.dry_run else "Updated"
        print(f"{action} {symbol}: {summary['rows']:,} rows")
        gc.collect()

    write_reports(
        summaries,
        args.report_dir,
        dry_run=args.dry_run,
        graph_path=args.graph_path,
    )
    print(f"Processed {len(summaries)} graph futures with frequency_mode=15min_bar")


if __name__ == "__main__":
    main()