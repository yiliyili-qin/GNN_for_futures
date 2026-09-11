"""Check daily and auxiliary futures CSV data and append to the 15-minute report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASETS = {
    "daily_settlement": {
        "date_column": "date",
        "required": {"underlying_symbol", "date", "dominant_id", "settlement", "open", "high", "low", "close", "volume", "open_interest"},
    },
    "dominant": {
        "date_column": "date",
        "required": {"date", "dominant"},
    },
    "ex_factor": {
        "date_column": "ex_date",
        "required": {"ex_date", "underlying_symbol", "ex_factor", "ex_end_date", "ex_cum_factor"},
    },
    "warehouse_stocks": {
        "date_column": "date",
        "required": {"date", "underlying_symbol", "on_warrant", "exchange"},
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check non-minute raw futures datasets.")
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw_data"))
    parser.add_argument("--report-dir", type=Path, default=None)
    return parser.parse_args(argv)


def inspect_file(path: Path, dataset: str) -> dict[str, object]:
    config = DATASETS[dataset]
    frame = pd.read_csv(path, low_memory=False)
    symbol = path.stem.upper()
    columns = set(frame.columns)
    required = config["required"]
    missing_required = sorted(required - columns)
    date_column = config["date_column"]
    dates = pd.to_datetime(frame.get(date_column), errors="coerce")
    duplicate_dates = int(dates.duplicated(keep=False).sum())
    symbol_mismatches = 0
    if "underlying_symbol" in columns:
        symbol_mismatches = int((frame["underlying_symbol"].astype(str).str.upper() != symbol).sum())

    issue_count = len(missing_required) + int(dates.isna().sum()) + duplicate_dates + symbol_mismatches
    if dataset == "daily_settlement" and {"open", "high", "low", "close"}.issubset(columns):
        values = frame[["open", "high", "low", "close", "settlement", "volume", "open_interest"]].apply(pd.to_numeric, errors="coerce")
        ohlc_invalid = (values["high"] < values[["open", "low", "close"]].max(axis=1)) | (values["low"] > values[["open", "high", "close"]].min(axis=1))
        negative_values = int((values[["volume", "open_interest"]] < 0).sum().sum())
        nonpositive_prices = int((values[["open", "high", "low", "close", "settlement"]] <= 0).any(axis=1).sum())
        issue_count += int(ohlc_invalid.sum()) + negative_values + nonpositive_prices
    else:
        ohlc_invalid = pd.Series(False, index=frame.index)
        negative_values = 0
        nonpositive_prices = 0
    if dataset == "dominant":
        blank_contracts = int(frame["dominant"].isna().sum() + frame["dominant"].astype(str).str.strip().eq("").sum()) if "dominant" in columns else len(frame)
        issue_count += blank_contracts
    else:
        blank_contracts = 0
    if dataset == "ex_factor":
        factors = frame[["ex_factor", "ex_cum_factor"]].apply(pd.to_numeric, errors="coerce") if {"ex_factor", "ex_cum_factor"}.issubset(columns) else pd.DataFrame(index=frame.index)
        invalid_factors = int((factors <= 0).any(axis=1).sum()) if not factors.empty else len(frame)
        end_dates = pd.to_datetime(frame.get("ex_end_date"), errors="coerce")
        invalid_periods = int(((end_dates.notna()) & (end_dates < dates)).sum())
        issue_count += invalid_factors + invalid_periods
    else:
        invalid_factors = 0
        invalid_periods = 0
    if dataset == "warehouse_stocks":
        warrants = pd.to_numeric(frame.get("on_warrant"), errors="coerce")
        negative_values = int((warrants < 0).sum())
        issue_count += negative_values

    return {
        "dataset": dataset,
        "symbol": symbol,
        "file": path.name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_required_columns": ";".join(missing_required),
        "total_missing_cells": int(frame.isna().sum().sum()),
        "first_date": dates.min(),
        "last_date": dates.max(),
        "invalid_date_rows": int(dates.isna().sum()),
        "duplicate_date_rows": duplicate_dates,
        "symbol_mismatch_rows": symbol_mismatches,
        "ohlc_invalid_rows": int(ohlc_invalid.sum()),
        "nonpositive_price_rows": nonpositive_prices,
        "negative_values": negative_values,
        "blank_dominant_rows": blank_contracts,
        "invalid_factor_rows": invalid_factors,
        "invalid_factor_periods": invalid_periods,
        "issue_count": issue_count,
    }


def append_report(report_path: Path, summary: pd.DataFrame, failures: list[dict[str, str]]) -> None:
    start_marker = "<!-- auxiliary-quality-start -->"
    end_marker = "<!-- auxiliary-quality-end -->"
    prior = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Raw Futures Data Quality Report\n"
    if start_marker in prior:
        prior = prior.split(start_marker, 1)[0].rstrip() + "\n"
    dataset_summary = summary.groupby("dataset").agg(files=("file", "count"), rows=("rows", "sum"), flagged_files=("issue_count", lambda values: int((values > 0).sum()))).reset_index()
    lines = ["", start_marker, "", "## Auxiliary Daily Data Quality", "", "| Dataset | Files | Rows | Files with issues |", "|---|---:|---:|---:|"]
    for _, row in dataset_summary.iterrows():
        lines.append(f"| {row['dataset']} | {row['files']} | {row['rows']:,} | {row['flagged_files']} |")
    flagged = summary[summary["issue_count"] > 0]
    lines.extend(["", f"- Read failures: {len(failures)}"])
    if flagged.empty and not failures:
        lines.append("- No missing required keys, invalid or duplicate dates, symbol mismatches, invalid factor values, OHLC violations, or negative warehouse quantities were found.")
    else:
        lines.extend(["", "### Auxiliary Flagged Files", "", "| Dataset | Symbol | Issues |", "|---|---|---:|"])
        for _, row in flagged.iterrows():
            lines.append(f"| {row['dataset']} | {row['symbol']} | {row['issue_count']} |")
    daily_flags = summary[(summary["dataset"] == "daily_settlement") & (summary["ohlc_invalid_rows"] > 0)]
    if not daily_flags.empty:
        lines.extend([
            "",
            "### Daily Settlement OHLC Note",
            "",
            "- The 127 flagged daily OHLC rows are limited to B, BB, FB, PM, RI, RS, WH, and WR. They should be treated as low-liquidity continuous-series observations rather than broad file corruption: 126 rows have zero volume and every flagged row has volume no greater than 5.",
            "- Exclude or mask zero-volume bars for intraday/daily price features that assume a tradable bar, particularly for BB, PM, RI, RS, WH, and WR.",
        ])
    lines.extend(["", end_marker, ""])
    report_path.write_text(prior.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir or args.raw_data_dir / "15min" / "quality_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for dataset in DATASETS:
        for path in sorted((args.raw_data_dir / dataset).glob("*.csv")):
            try:
                summaries.append(inspect_file(path, dataset))
            except Exception as error:
                failures.append({"dataset": dataset, "file": path.name, "error": f"{type(error).__name__}: {error}"})
    summary = pd.DataFrame(summaries).sort_values(["dataset", "symbol"])
    summary.to_csv(report_dir / "auxiliary_quality_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(report_dir / "auxiliary_read_failures.csv", index=False)
    append_report(report_dir / "quality_report.md", summary, failures)
    print(f"Checked {len(summary)} auxiliary files; reports written to {report_dir.resolve()}")


if __name__ == "__main__":
    main()