"""Append technical indicators to the retained raw 15-minute futures CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import calculate_technical_indicators


INDICATOR_COLUMNS = (
    "return_15m",
    "return_30m",
    "return_1h",
    "return_2h",
    "return_4h",
    "return_1d",
    "geometric_mean_return_30m",
    "geometric_mean_return_1h",
    "geometric_mean_return_2h",
    "geometric_mean_return_4h",
    "geometric_mean_return_1d",
    "rolling_volatility_30m",
    "rolling_volatility_1h",
    "rolling_volatility_2h",
    "rolling_volatility_4h",
    "rolling_volatility_1d",
    "high_low_range",
    "r_OC",
    "delta_open_interest",
    "open_interest_pct_change",
    "turnover_to_open_interest",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append 15-minute technical indicators to raw futures CSVs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw_data/15min"))
    parser.add_argument("--bars-per-trading-day", type=int, default=16)
    parser.add_argument("--symbols", nargs="+", help="Optional underlying symbols to process.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def add_indicators(path: Path, bars_per_trading_day: int, dry_run: bool) -> tuple[int, int]:
    frame = pd.read_csv(path, low_memory=False)
    if "datetime" not in frame.columns:
        raise ValueError(f"{path} is missing datetime.")
    timestamps = pd.to_datetime(frame["datetime"], errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"{path} has invalid or unsorted datetime values.")

    raw_columns = [column for column in frame.columns if column not in INDICATOR_COLUMNS]
    features = calculate_technical_indicators(
        frame.loc[:, raw_columns], bars_per_trading_day=bars_per_trading_day
    )
    if not dry_run:
        features.to_csv(path, index=False)
    return len(features), len(features.columns) - len(raw_columns)


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir.resolve()}")

    files = sorted(args.data_dir.glob("*.csv"))
    if args.symbols:
        requested_symbols = {symbol.upper() for symbol in args.symbols}
        files = [path for path in files if path.stem.upper() in requested_symbols]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.data_dir.resolve()}")
    for path in files:
        rows, indicator_count = add_indicators(path, args.bars_per_trading_day, args.dry_run)
        action = "Validated" if args.dry_run else "Updated"
        print(f"{action} {path.name}: {rows:,} rows, {indicator_count} indicator columns")


if __name__ == "__main__":
    main()