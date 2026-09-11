"""Download Chinese commodity futures data from PkuFinLab/RQData.

Run this script in the PkuFinLab remote environment. It overwrites the
existing 15-minute dominant continuous-contract files in ``raw_data``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import traceback

import pandas as pd
from pkufinlab import rq_init


# The 59 commodities already validated in raw_data/quality_reports.
COMMODITY_SYMBOLS = (
    "A", "AG", "AL", "AP", "AU", "B", "BB", "BC", "BU", "C", "CF",
    "CJ", "CS", "CU", "CY", "EB", "EG", "FB", "FG", "FU", "HC", "I",
    "J", "JD", "JM", "L", "LH", "LU", "M", "MA", "NI", "NR", "OI",
    "P", "PB", "PF", "PG", "PK", "PM", "PP", "RB", "RI", "RM", "RS",
    "RU", "SA", "SC", "SF", "SM", "SN", "SP", "SR", "SS", "TA", "UR",
    "V", "WH", "WR", "Y",
)

START_DATE = "2016-01-01"
END_DATE = "2026-08-25"
PRICE_FREQUENCY = "15m"
ADJUST_TYPE = "pre"
ADJUST_METHOD = "prev_close_ratio"
PRICE_FIELDS = (
    "open", "high", "low", "close", "volume", "total_turnover",
    "open_interest", "trading_date",
)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 15-minute Chinese commodity futures dominant prices."
    )
    parser.add_argument("--raw-output-dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--symbols", nargs="+", default=list(COMMODITY_SYMBOLS))
    parser.add_argument(
        "--adjust-type",
        choices=("pre", "post", "none"),
        default=ADJUST_TYPE,
        help="Dominant continuous-contract adjustment; default: pre.",
    )
    parser.add_argument(
        "--adjust-method",
        choices=(
            "prev_close_ratio",
            "open_ratio",
            "prev_close_spread",
            "open_spread",
        ),
        default=ADJUST_METHOD,
        help="Use a ratio method to retain positive price levels; default: prev_close_ratio.",
    )
    parser.add_argument(
        "--smm-quota-ids",
        nargs="+",
        default=None,
        help="After futures, download these authorized SMM physical-market indicators.",
    )
    parser.add_argument(
        "--smm-output-dir",
        type=Path,
        default=None,
        help="SMM output directory; default: <raw-output-dir>/smm.",
    )
    parser.add_argument(
        "--smm-token-cache",
        type=Path,
        default=None,
        help="SMM token cache; default: <raw-output-dir>/smm/.smm_token.json.",
    )
    parser.add_argument(
        "--smm-use-proxy",
        action="store_true",
        help="Honor HTTP(S)_PROXY for SMM; direct connections are the default.",
    )
    args, _ = parser.parse_known_args(argv)
    return args


def to_flat_frame(data: pd.DataFrame | pd.Series | None) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name or "value")
    return data.reset_index()


def fetch_dominant_15m(
    rq: object,
    symbol: str,
    start_date: str,
    end_date: str,
    adjust_type: str,
    adjust_method: str,
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
            symbol,
            start_date=month_start,
            end_date=month_end,
            frequency=PRICE_FREQUENCY,
            fields=None,
            adjust_type=adjust_type,
            adjust_method=adjust_method,
        )
        if data is not None and not data.empty:
            frames.append(to_flat_frame(data))

    if not frames:
        raise ValueError(f"{symbol}: no {PRICE_FREQUENCY} dominant price bars returned")
    frame = pd.concat(frames, ignore_index=True)
    missing = set(PRICE_FIELDS).difference(frame.columns)
    if missing:
        raise ValueError(f"{symbol}: missing 15-minute fields {sorted(missing)}")
    timestamp_column = "datetime" if "datetime" in frame.columns else "date"
    if timestamp_column not in frame.columns:
        raise ValueError(f"{symbol}: no datetime returned")
    frame = frame.rename(columns={timestamp_column: "datetime"})
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    return frame.sort_values("datetime").drop_duplicates("datetime", keep="last")


def save_dataset(
    fetcher: object,
    output_path: Path,
    failures: dict[str, str],
    dataset_name: str,
    symbol: str,
) -> None:
    try:
        frame = fetcher()
        if frame.empty:
            print(f"[{symbol}] {dataset_name}: no data")
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(f"[{symbol}] {dataset_name}: saved {len(frame):,} rows")
    except Exception as error:
        key = f"{dataset_name}/{symbol}"
        failures[key] = "".join(traceback.format_exception(error))
        print(f"[{symbol}] {dataset_name}: failed ({error})")


def write_metadata(
    raw_output_dir: Path,
    args: argparse.Namespace,
    symbols: list[str],
    failures: dict[str, str],
) -> None:
    metadata = {
        "source": "PkuFinLab pkufinlab.rq_init(HF=True) / Ricequant RQData",
        "symbols": symbols,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "dominant_price_frequency": PRICE_FREQUENCY,
        "adjust_type": args.adjust_type,
        "adjust_method": args.adjust_method,
        "raw_data_layout": {
            "15m_dominant_prices": "<symbol>.csv",
        },
        "failures": failures,
        "created_on": date.today().isoformat(),
    }
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    (raw_output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def fetch_smm_after_futures(args: argparse.Namespace) -> None:
    if not args.smm_quota_ids:
        return

    import os

    from get_smm_data import SmmClient, download_quotas, save_result

    output_dir = args.smm_output_dir or args.raw_output_dir / "smm"
    token_cache = args.smm_token_cache or output_dir / ".smm_token.json"
    client = SmmClient(
        user_name=os.environ.get("SMM_USER_NAME"),
        password=os.environ.get("SMM_PASSWORD"),
        token_cache=token_cache,
        use_proxy=args.smm_use_proxy,
    )
    result = download_quotas(
        client,
        args.smm_quota_ids,
        args.start_date,
        args.end_date,
    )
    save_result(
        result,
        output_dir,
        args.smm_quota_ids,
        args.start_date,
        args.end_date,
    )
    print(f"SMM: saved {len(result.normalized):,} rows to {output_dir}")


def main() -> None:
    args = parse_args()
    symbols = [symbol.upper() for symbol in args.symbols]
    raw_output_dir = args.raw_output_dir
    failures: dict[str, str] = {}
    rq = rq_init(HF=True)

    for symbol in symbols:
        save_dataset(
            lambda: fetch_dominant_15m(
                rq, symbol, args.start_date, args.end_date,
                args.adjust_type, args.adjust_method,
            ),
            raw_output_dir / f"{symbol}.csv", failures, "dominant_15m", symbol,
        )

    write_metadata(raw_output_dir, args, symbols, failures)
    fetch_smm_after_futures(args)
    print(f"Finished with {len(failures)} failed dataset requests.")


if __name__ == "__main__":
    main()
