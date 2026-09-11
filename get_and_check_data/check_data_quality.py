"""Profile the raw commodity futures CSV files stored in ``raw_data``.

Run in PkuFinLab or locally:
    python check_data_quality.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "underlying_symbol",
    "datetime",
    "trading_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "total_turnover",
    "open_interest",
)
NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "total_turnover",
    "open_interest",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check raw futures CSV data quality.")
    parser.add_argument("--data-dir", type=Path, default=Path("raw_data"))
    parser.add_argument(
        "--report-dir", type=Path, default=None,
        help="Defaults to <data-dir>/quality_reports.",
    )
    args, _ = parser.parse_known_args(argv)
    return args


def get_interval_counts(datetimes: pd.Series) -> dict[str, int]:
    unique_times = datetimes.dropna().drop_duplicates().sort_values()
    if len(unique_times) < 2:
        return {}
    intervals = unique_times.diff().dropna().dt.total_seconds().div(60).round().astype(int)
    intervals = intervals[intervals <= 240]
    return {str(minutes): int(count) for minutes, count in intervals.value_counts().sort_index().items()}


def get_trading_date_gap_count(trading_dates: pd.Series) -> int:
    unique_dates = trading_dates.dropna().drop_duplicates().sort_values()
    if len(unique_dates) < 2:
        return 0
    gaps = unique_dates.diff().dropna().dt.days
    return int((gaps > 7).sum())


def inspect_file(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    symbol = path.stem.upper()
    frame = pd.read_csv(path, low_memory=False)
    columns = set(frame.columns)
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(columns))
    missingness = [
        {
            "symbol": symbol,
            "column": column,
            "missing_rows": int(frame[column].isna().sum()),
            "missing_rate": float(frame[column].isna().mean()),
        }
        for column in frame.columns
    ]

    datetime_values = pd.to_datetime(frame.get("datetime"), errors="coerce")
    trading_dates = pd.to_datetime(frame.get("trading_date"), errors="coerce")
    duplicate_datetimes = int(datetime_values.duplicated(keep=False).sum())

    numeric = frame.reindex(columns=NUMERIC_COLUMNS).apply(pd.to_numeric, errors="coerce")
    nonpositive_ohlc = pd.Series(False, index=frame.index)
    ohlc_range_invalid = pd.Series(False, index=frame.index)
    if {"open", "high", "low", "close"}.issubset(columns):
        nonpositive_ohlc = (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
        ohlc_range_invalid = (
            (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1))
            | (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1))
        )

    nonnegative_columns = [column for column in ("volume", "total_turnover", "open_interest") if column in columns]
    negative_values = int((numeric[nonnegative_columns] < 0).sum().sum()) if nonnegative_columns else 0
    interval_counts = get_interval_counts(datetime_values)
    dominant_interval = max(interval_counts, key=interval_counts.get) if interval_counts else ""

    summary = {
        "symbol": symbol,
        "file": path.name,
        "file_size_mb": round(path.stat().st_size / 1024**2, 2),
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_required_columns": ";".join(missing_columns),
        "total_missing_cells": int(frame.isna().sum().sum()),
        "overall_missing_rate": float(frame.isna().mean().mean()),
        "first_datetime": datetime_values.min(),
        "last_datetime": datetime_values.max(),
        "first_trading_date": trading_dates.min(),
        "last_trading_date": trading_dates.max(),
        "invalid_datetime_rows": int(datetime_values.isna().sum()),
        "invalid_trading_date_rows": int(trading_dates.isna().sum()),
        "duplicate_datetime_rows": duplicate_datetimes,
        "long_trading_date_gaps": get_trading_date_gap_count(trading_dates),
        "dominant_intraday_interval_minutes": dominant_interval,
        "intraday_interval_counts": json.dumps(interval_counts, ensure_ascii=False),
        "nonpositive_adjusted_ohlc_rows": int(nonpositive_ohlc.sum()),
        "ohlc_range_invalid_rows": int(ohlc_range_invalid.sum()),
        "negative_volume_turnover_oi_values": negative_values,
        "zero_volume_rows": int((numeric["volume"] == 0).sum()) if "volume" in columns else 0,
    }
    return summary, missingness


def write_markdown(report_dir: Path, summary: pd.DataFrame, failures: list[dict[str, str]]) -> None:
    total_rows = int(summary["rows"].sum()) if not summary.empty else 0
    total_size = summary["file_size_mb"].sum() if not summary.empty else 0.0
    flagged = summary[
        (summary["missing_required_columns"] != "")
        | (summary["invalid_datetime_rows"] > 0)
        | (summary["duplicate_datetime_rows"] > 0)
        | (summary["ohlc_range_invalid_rows"] > 0)
        | (summary["negative_volume_turnover_oi_values"] > 0)
    ]
    lines = [
        "# Raw Futures Data Quality Report",
        "",
        f"- Files scanned: {len(summary)}",
        f"- Total rows: {total_rows:,}",
        f"- Total size: {total_size:.2f} MB",
        f"- Files with structural or value issues: {len(flagged)}",
        f"- Files that could not be read: {len(failures)}",
        "",
        "## Flagged Files",
        "",
    ]
    if flagged.empty:
        lines.append("No missing required fields, invalid timestamps, duplicated timestamps, OHLC range violations, or negative volume/turnover/open-interest values were found.")
    else:
        lines.extend([
            "| Symbol | Missing fields | Invalid datetimes | Duplicate datetimes | OHLC range violations | Negative values |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for _, row in flagged.iterrows():
            lines.append(
                f"| {row['symbol']} | {row['missing_required_columns'] or '-'} | "
                f"{row['invalid_datetime_rows']} | {row['duplicate_datetime_rows']} | "
                f"{row['ohlc_range_invalid_rows']} | {row['negative_volume_turnover_oi_values']} |"
            )
    adjusted_negative = summary[summary["nonpositive_adjusted_ohlc_rows"] > 0]
    if not adjusted_negative.empty:
        lines.extend([
            "",
            "## Nonpositive Adjusted Prices",
            "",
            "These are not treated as invalid OHLC values. Additive contract-roll adjustment can shift historical futures prices below zero; use unadjusted prices or return-based features when positive price levels are required.",
            "",
            "| Symbol | Nonpositive adjusted OHLC rows |",
            "|---|---:|",
        ])
        for _, row in adjusted_negative.iterrows():
            lines.append(f"| {row['symbol']} | {row['nonpositive_adjusted_ohlc_rows']} |")
    if failures:
        lines.extend(["", "## Read Failures", ""])
        lines.extend(f"- `{item['file']}`: {item['error']}" for item in failures)
    (report_dir / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    report_dir = args.report_dir or data_dir / "quality_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir.resolve()}")

    summaries: list[dict[str, object]] = []
    missingness_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            summary, missingness = inspect_file(path)
            summaries.append(summary)
            missingness_rows.extend(missingness)
            print(f"[{path.stem}] checked {summary['rows']:,} rows")
        except Exception as error:
            failures.append({"file": path.name, "error": f"{type(error).__name__}: {error}"})
            print(f"[{path.stem}] failed: {failures[-1]['error']}")

    summary_frame = pd.DataFrame(summaries).sort_values("symbol")
    missingness_frame = pd.DataFrame(missingness_rows).sort_values(["symbol", "column"])
    summary_frame.to_csv(report_dir / "quality_summary.csv", index=False)
    missingness_frame.to_csv(report_dir / "missingness_by_column.csv", index=False)
    pd.DataFrame(failures).to_csv(report_dir / "read_failures.csv", index=False)
    write_markdown(report_dir, summary_frame, failures)
    print(f"Reports written to {report_dir.resolve()}")


if __name__ == "__main__":
    main()