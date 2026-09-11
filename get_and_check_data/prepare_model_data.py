"""Prepare tradability-aware futures data without changing retained raw files."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXCLUDED_SYMBOLS = ("BB", "PM", "RI", "RS", "WH", "WR", "ZC")
RAW_DATASETS = ("15min", "30min", "daily_settlement", "dominant", "ex_factor", "warehouse_stocks")
HIGH_LIQUIDITY_MAX_ZERO_VOLUME_RATE = 0.01


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move illiquid futures and build model-ready 15-minute data.")
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw_data"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/15min"))
    parser.add_argument("--excluded-dir", type=Path, default=None)
    return parser.parse_args(argv)


def move_excluded_files(raw_data_dir: Path, excluded_dir: Path) -> list[dict[str, str]]:
    moved: list[dict[str, str]] = []
    for dataset in RAW_DATASETS:
        source_dir = raw_data_dir / dataset
        if not source_dir.exists():
            continue
        destination_dir = excluded_dir / dataset
        for symbol in EXCLUDED_SYMBOLS:
            source = source_dir / f"{symbol}.csv"
            if not source.exists():
                continue
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / source.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite existing excluded file: {destination}")
            shutil.move(str(source), str(destination))
            moved.append({"dataset": dataset, "symbol": symbol, "from": str(source), "to": str(destination)})
    return moved


def prepare_symbol(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(path, low_memory=False)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    frame = frame.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
    frame["is_zero_volume"] = volume.eq(0)
    frame["is_valid_bar"] = ~frame["is_zero_volume"]
    frame["is_roll_day"] = frame["dominant_id"].ne(frame["dominant_id"].shift())
    frame.loc[0, "is_roll_day"] = False
    frame["close_for_alignment"] = pd.to_numeric(frame["close"], errors="coerce").where(frame["is_valid_bar"]).ffill()
    frame["return_15m"] = frame["close_for_alignment"].pct_change().where(frame["is_valid_bar"])
    zero_volume_rate = float(frame["is_zero_volume"].mean())
    liquidity_tier = "high" if zero_volume_rate <= HIGH_LIQUIDITY_MAX_ZERO_VOLUME_RATE else "standard"
    frame["liquidity_tier"] = liquidity_tier
    return frame, {
        "symbol": path.stem.upper(),
        "rows": len(frame),
        "zero_volume_rows": int(frame["is_zero_volume"].sum()),
        "zero_volume_rate": zero_volume_rate,
        "roll_day_rows": int(frame["is_roll_day"].sum()),
        "liquidity_tier": liquidity_tier,
    }


def main() -> None:
    args = parse_args()
    raw_data_dir = args.raw_data_dir
    excluded_dir = args.excluded_dir or raw_data_dir / "excluded_inactive_symbols"
    moved = move_excluded_files(raw_data_dir, excluded_dir)

    source_dir = raw_data_dir / "15min"
    if not source_dir.exists():
        raise FileNotFoundError(f"No retained 15-minute data directory: {source_dir.resolve()}")
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for path in sorted(source_dir.glob("*.csv")):
        frame, summary = prepare_symbol(path)
        frame.to_csv(args.processed_dir / path.name, index=False)
        summaries.append(summary)
        print(f"[{summary['symbol']}] {summary['liquidity_tier']}: {summary['rows']:,} rows")

    summary_frame = pd.DataFrame(summaries).sort_values("symbol")
    summary_frame.to_csv(args.processed_dir / "preprocessing_summary.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "excluded_symbols": list(EXCLUDED_SYMBOLS),
        "excluded_reason": "Zero-volume frequency too high for a common tradable futures universe.",
        "excluded_files_moved": moved,
        "retained_symbol_count": len(summary_frame),
        "feature_rules": {
            "is_zero_volume": "volume equals zero",
            "is_valid_bar": "not is_zero_volume",
            "is_roll_day": "dominant_id differs from preceding bar",
            "close_for_alignment": "close forward-filled only across invalid bars",
            "return_15m": "percentage change of close_for_alignment, masked on invalid bars",
            "liquidity_tier": f"high when zero-volume rate <= {HIGH_LIQUIDITY_MAX_ZERO_VOLUME_RATE:.0%}; otherwise standard",
        },
    }
    (args.processed_dir / "preprocessing_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Moved {len(moved)} raw files to {excluded_dir.resolve()}")
    print(f"Prepared {len(summary_frame)} retained symbols in {args.processed_dir.resolve()}")


if __name__ == "__main__":
    main()