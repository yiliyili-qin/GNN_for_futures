"""Download full thermal-coal data into the existing raw-data CSV layout.

Run in the authenticated PkuFinLab environment:
    python get_thermal_coal_data.py --raw-data-dir ../data/raw_data

RQData uses TC through the 2016-04 contract and ZC afterwards. Both source
series are written as ZC.csv so downstream code sees one thermal-coal symbol.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import sys
import traceback
from typing import Callable

import numpy as np
import pandas as pd


OUTPUT_SYMBOL = "ZC"
THERMAL_COAL_SEGMENTS = (
    ("TC", "2013-09-26", "2016-04-30"),
    ("ZC", "2016-05-01", "2026-08-29"),
)
ADJUST_TYPE = "pre"
ADJUST_METHOD = "prev_close_ratio"
MINUTE_FIELDS = (
    "underlying_symbol", "datetime", "trading_date", "dominant_id", "open",
    "close", "high", "low", "total_turnover", "volume", "open_interest",
)
DAILY_FIELDS = (
    "underlying_symbol", "date", "dominant_id", "settlement", "prev_settlement",
    "limit_up", "limit_down", "open", "high", "low", "close", "volume",
    "total_turnover", "open_interest",
)
DOMINANT_FIELDS = ("date", "dominant")
EX_FACTOR_FIELDS = (
    "ex_date", "underlying_symbol", "ex_factor", "ex_end_date", "ex_cum_factor",
)
WAREHOUSE_FIELDS = (
    "date", "underlying_symbol", "on_warrant", "exchange", "effective_forecast",
    "warrant_units", "contract_multiplier", "deliverable",
)
HORIZON_BARS = {"30m": 2, "1h": 4, "2h": 8, "4h": 16}


def calculate_technical_indicators(
    data: pd.DataFrame, *, bars_per_trading_day: int = 16
) -> pd.DataFrame:
    """Append the standard 15-minute raw-data technical indicator columns."""
    required_columns = {"close", "high", "low", "open", "open_interest", "total_turnover"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"15min data is missing required columns: {missing_columns}")

    features = data.copy()
    close = pd.to_numeric(features["close"], errors="coerce").where(lambda values: values > 0)
    high = pd.to_numeric(features["high"], errors="coerce")
    low = pd.to_numeric(features["low"], errors="coerce")
    open_price = pd.to_numeric(features["open"], errors="coerce").where(lambda values: values > 0)
    open_interest = pd.to_numeric(features["open_interest"], errors="coerce").where(lambda values: values > 0)
    turnover = pd.to_numeric(features["total_turnover"], errors="coerce")
    return_15m = close.pct_change(fill_method=None)
    features["return_15m"] = return_15m

    horizon_bars = {**HORIZON_BARS, "1d": bars_per_trading_day}
    for horizon, bars in horizon_bars.items():
        features[f"return_{horizon}"] = close.pct_change(periods=bars, fill_method=None)
        features[f"geometric_mean_return_{horizon}"] = (1 + return_15m).rolling(
            bars, min_periods=bars
        ).apply(np.prod, raw=True).pow(1 / bars) - 1
        features[f"rolling_volatility_{horizon}"] = return_15m.pow(2).rolling(
            bars, min_periods=bars
        ).sum().pow(0.5)

    features["high_low_range"] = (high - low) / close
    features["r_OC"] = np.log(close / open_price)
    features["delta_open_interest"] = open_interest.diff()
    features["open_interest_pct_change"] = open_interest.pct_change(fill_method=None)
    features["turnover_to_open_interest"] = turnover / open_interest
    indicator_columns = [
        "return_15m",
        *[f"return_{horizon}" for horizon in horizon_bars],
        *[f"geometric_mean_return_{horizon}" for horizon in horizon_bars],
        *[f"rolling_volatility_{horizon}" for horizon in horizon_bars],
        "high_low_range",
        "r_OC",
        "delta_open_interest",
        "open_interest_pct_change",
        "turnover_to_open_interest",
    ]
    features[indicator_columns] = features[indicator_columns].replace([np.inf, -np.inf], np.nan)
    return features


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download full TC/ZC thermal-coal data in the raw-data CSV layout."
    )
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw_data"))
    parser.add_argument("--start-date", default=THERMAL_COAL_SEGMENTS[0][1])
    parser.add_argument("--end-date", default=THERMAL_COAL_SEGMENTS[-1][2])
    parser.add_argument("--dry-run", action="store_true")
    args, _ = parser.parse_known_args(argv)
    return args


def clipped_segments(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date)
    if requested_start > requested_end:
        raise ValueError("start-date must not be later than end-date")

    segments: list[tuple[str, str, str]] = []
    for symbol, segment_start, segment_end in THERMAL_COAL_SEGMENTS:
        clipped_start = max(requested_start, pd.Timestamp(segment_start))
        clipped_end = min(requested_end, pd.Timestamp(segment_end))
        if clipped_start <= clipped_end:
            segments.append(
                (symbol, clipped_start.strftime("%Y-%m-%d"), clipped_end.strftime("%Y-%m-%d"))
            )
    if not segments:
        raise ValueError("requested dates do not overlap the supported TC/ZC history")
    return segments


def to_flat_frame(data: pd.DataFrame | pd.Series | None) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name or "dominant")
    return data.reset_index()


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], dataset: str) -> pd.DataFrame:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset}: missing RQData fields {missing}")
    return frame.loc[:, columns]


def fetch_prices_in_months(
    rq: object, source_symbol: str, start_date: str, end_date: str, frequency: str
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in pd.period_range(
        pd.Timestamp(start_date).to_period("M"),
        pd.Timestamp(end_date).to_period("M"),
        freq="M",
    ):
        month_start = max(pd.Timestamp(start_date), month.start_time).strftime("%Y-%m-%d")
        month_end = min(pd.Timestamp(end_date), month.end_time).strftime("%Y-%m-%d")
        data = rq.futures.get_dominant_price(
            source_symbol,
            start_date=month_start,
            end_date=month_end,
            frequency=frequency,
            fields=None,
            adjust_type=ADJUST_TYPE,
            adjust_method=ADJUST_METHOD,
        )
        if data is not None and not data.empty:
            frames.append(to_flat_frame(data))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_prices(rq: object, segments: list[tuple[str, str, str]], frequency: str) -> pd.DataFrame:
    frames = [
        fetch_prices_in_months(rq, symbol, start_date, end_date, frequency)
        for symbol, start_date, end_date in segments
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError(f"no {frequency} dominant-price data returned for TC/ZC")

    frame = pd.concat(frames, ignore_index=True)
    date_column = "date" if frequency == "1d" else "datetime"
    frame["underlying_symbol"] = OUTPUT_SYMBOL
    frame[date_column] = pd.to_datetime(frame[date_column])
    if "trading_date" in frame.columns:
        frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    fields = DAILY_FIELDS if frequency == "1d" else MINUTE_FIELDS
    return require_columns(frame, fields, f"{frequency} prices").sort_values(date_column).drop_duplicates(date_column, keep="last")


def fetch_dominant(rq: object, segments: list[tuple[str, str, str]]) -> pd.DataFrame:
    frames = [
        to_flat_frame(rq.futures.get_dominant(symbol, start_date=start_date, end_date=end_date))
        for symbol, start_date, end_date in segments
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("no dominant-contract data returned for TC/ZC")
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return require_columns(frame, DOMINANT_FIELDS, "dominant").sort_values("date").drop_duplicates("date", keep="last")


def fetch_ex_factor(rq: object, segments: list[tuple[str, str, str]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, start_date, end_date in segments:
        frame = to_flat_frame(rq.futures.get_ex_factor(
            symbol, start_date=start_date, end_date=end_date, adjust_method=ADJUST_METHOD
        ))
        if not frame.empty:
            frame["underlying_symbol"] = OUTPUT_SYMBOL
            frames.append(frame)
    if not frames:
        raise ValueError("no ex-factor data returned for TC/ZC")
    frame = pd.concat(frames, ignore_index=True)
    frame["ex_date"] = pd.to_datetime(frame["ex_date"])
    frame["ex_end_date"] = pd.to_datetime(frame["ex_end_date"])
    return require_columns(frame, EX_FACTOR_FIELDS, "ex_factor").sort_values("ex_date").drop_duplicates("ex_date", keep="last")


def fetch_warehouse_stocks(rq: object, segments: list[tuple[str, str, str]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, start_date, end_date in segments:
        frame = to_flat_frame(rq.futures.get_warehouse_stocks(
            symbol, start_date=start_date, end_date=end_date
        ))
        if not frame.empty:
            frame["underlying_symbol"] = OUTPUT_SYMBOL
            frames.append(frame)
    if not frames:
        raise ValueError("no warehouse-stock data returned for TC/ZC")
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    for field in WAREHOUSE_FIELDS:
        if field not in frame.columns:
            frame[field] = pd.NA
    return frame.loc[:, WAREHOUSE_FIELDS].sort_values("date").drop_duplicates("date", keep="last")


def save_dataset(
    dataset: str,
    fetcher: Callable[[], pd.DataFrame],
    raw_data_dir: Path,
    failures: dict[str, str],
) -> None:
    try:
        frame = fetcher()
        if frame.empty:
            raise ValueError("empty response")
        if dataset == "15min":
            frame = calculate_technical_indicators(frame, bars_per_trading_day=16)
        output_path = raw_data_dir / dataset / f"{OUTPUT_SYMBOL}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(f"[{OUTPUT_SYMBOL}] {dataset}: saved {len(frame):,} rows")
    except Exception as error:
        failures[dataset] = "".join(traceback.format_exception(error))
        print(f"[{OUTPUT_SYMBOL}] {dataset}: failed ({error})")


def write_metadata(
    raw_data_dir: Path,
    args: argparse.Namespace,
    segments: list[tuple[str, str, str]],
    failures: dict[str, str],
) -> None:
    metadata = {
        "source": "PkuFinLab pkufinlab.rq_init(HF=True) / Ricequant RQData",
        "output_symbol": OUTPUT_SYMBOL,
        "requested_start_date": args.start_date,
        "requested_end_date": args.end_date,
        "source_segments": [
            {"source_symbol": symbol, "start_date": start, "end_date": end}
            for symbol, start, end in segments
        ],
        "price_adjustment": {"adjust_type": ADJUST_TYPE, "adjust_method": ADJUST_METHOD},
        "datasets": ["15min", "30min", "daily_settlement", "dominant", "ex_factor", "warehouse_stocks"],
        "failures": failures,
        "created_on": date.today().isoformat(),
    }
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    (raw_data_dir / "ZC_download_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    segments = clipped_segments(args.start_date, args.end_date)
    print(f"Downloading {OUTPUT_SYMBOL} from source segments: {segments}")
    if args.dry_run:
        print("Dry run successful; no RQData calls or files written.")
        return

    try:
        from pkufinlab import rq_init
    except ImportError as error:
        raise RuntimeError(
            "pkufinlab is required for downloads. Run this script in the authenticated "
            "PkuFinLab environment."
        ) from error

    rq = rq_init(HF=True)
    failures: dict[str, str] = {}
    save_dataset("15min", lambda: fetch_prices(rq, segments, "15m"), args.raw_data_dir, failures)
    save_dataset("30min", lambda: fetch_prices(rq, segments, "30m"), args.raw_data_dir, failures)
    save_dataset("daily_settlement", lambda: fetch_prices(rq, segments, "1d"), args.raw_data_dir, failures)
    save_dataset("dominant", lambda: fetch_dominant(rq, segments), args.raw_data_dir, failures)
    save_dataset("ex_factor", lambda: fetch_ex_factor(rq, segments), args.raw_data_dir, failures)
    save_dataset("warehouse_stocks", lambda: fetch_warehouse_stocks(rq, segments), args.raw_data_dir, failures)
    write_metadata(args.raw_data_dir, args, segments, failures)
    print(f"Finished with {len(failures)} failed dataset requests.")


if __name__ == "__main__":
    main()