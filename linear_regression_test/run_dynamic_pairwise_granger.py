"""Run a daily rolling pairwise VAR Granger network for the futures universe."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import tempfile
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import solve_triangular
from scipy.stats import f as f_distribution
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw_data" / "15min"
UNIVERSE_FILE = PROJECT_ROOT / "picture" / "futures_target_return_acf_corrected.csv"
STATIC_METADATA = Path(__file__).resolve().parent / "results" / "no_smm_pairwise_granger" / "run_metadata.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "dynamic_pairwise_granger"
RETURN_COLUMN = "return_15m"
EXPECTED_UNIVERSE_SIZE = 44
DEFAULT_WINDOW_DAYS = 20
DEFAULT_MAX_LAG = 40
DEFAULT_YEARS = 2

RESULT_COLUMNS = [
    "evaluation_date", "window_start_date", "window_end_date", "pair_id",
    "overlap_start", "overlap_end", "overlap_observations", "regression_observations",
    "bic_selected_lag", "explanatory_futures", "dependent_futures",
    "wald_f_statistic", "p_value", "significance_stars", "numerator_df",
    "denominator_df", "test_status", "fdr_bh_q_value",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", help="Comma-separated subset; default is the exact 44-node universe.")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="Latest N years to evaluate when --start-date is omitted.")
    parser.add_argument("--start-date", help="First evaluation date, inclusive; overrides --years.")
    parser.add_argument("--end-date", help="Last evaluation date, inclusive; default is the latest available date.")
    parser.add_argument("--last-days", type=int, help="Limit to the last N evaluation dates for testing.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true", help="Recompute dates that already have a daily checkpoint.")
    return parser.parse_args()


def load_universe(symbol_argument: str | None) -> list[str]:
    if UNIVERSE_FILE.exists():
        universe = pd.read_csv(UNIVERSE_FILE, usecols=["variety"])["variety"].astype(str).tolist()
    else:
        universe = json.loads(STATIC_METADATA.read_text(encoding="utf-8"))["symbols"]
    if len(universe) != EXPECTED_UNIVERSE_SIZE or len(set(universe)) != len(universe):
        raise ValueError("The authoritative futures universe must contain 44 unique symbols.")
    if symbol_argument is None:
        return universe
    requested = [item.strip() for item in symbol_argument.split(",") if item.strip()]
    unknown = sorted(set(requested).difference(universe))
    if unknown or len(requested) < 2 or len(set(requested)) != len(requested):
        raise ValueError(f"Invalid symbol subset; unknown={unknown}")
    return requested


def load_returns(symbols: list[str]) -> dict[str, pd.DataFrame]:
    returns = {}
    for symbol in symbols:
        frame = pd.read_csv(
            DATA_DIR / f"{symbol}.csv",
            usecols=["datetime", "trading_date", RETURN_COLUMN],
            parse_dates=["datetime", "trading_date"],
        )
        frame[RETURN_COLUMN] = pd.to_numeric(frame[RETURN_COLUMN], errors="coerce")
        frame = frame.dropna(subset=[RETURN_COLUMN]).drop_duplicates("datetime", keep="last").sort_values("datetime")
        returns[symbol] = frame.set_index("datetime")[["trading_date", RETURN_COLUMN]]
    return returns


def common_evaluation_dates(returns: dict[str, pd.DataFrame], args: argparse.Namespace) -> list[pd.Timestamp]:
    available = sorted(set().union(*(frame["trading_date"].dt.normalize().unique() for frame in returns.values())))
    dates = [pd.Timestamp(value) for value in available]
    latest = max(frame["trading_date"].max().normalize() for frame in returns.values())
    start = pd.Timestamp(args.start_date) if args.start_date else latest - pd.DateOffset(years=args.years) + pd.Timedelta(days=1)
    end = pd.Timestamp(args.end_date) if args.end_date else latest
    dates = [date for date in dates if start <= date <= end]
    if args.last_days:
        dates = dates[-args.last_days:]
    if not dates:
        raise ValueError("No evaluation dates remain after applying the requested range.")
    return dates


def maximum_lag_matrices(values: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    observations = len(values) - max_lag
    targets = values[max_lag:]
    design = np.ones((observations, 1 + 2 * max_lag), dtype=np.float64)
    for lag in range(1, max_lag + 1):
        design[:, 2 * lag - 1 : 2 * lag + 1] = values[max_lag - lag : -lag]
    return targets, design


def fit_ols(targets: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank != design.shape[1]:
        raise np.linalg.LinAlgError("Rank-deficient VAR design")
    residuals = targets - design @ coefficients
    return coefficients, np.linalg.inv(design.T @ design), residuals.T @ residuals


def select_bic(
    targets: np.ndarray, maximum_design: np.ndarray, max_lag: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """Select a nested VAR lag using one QR decomposition of the largest model."""
    observations = len(targets)
    orthogonal, triangular = np.linalg.qr(maximum_design, mode="reduced")
    projected_targets = orthogonal.T @ targets
    target_cross_product = targets.T @ targets
    cumulative_explained = np.cumsum(
        projected_targets[:, :, None] * projected_targets[:, None, :], axis=0
    )
    best_lag, best_bic = 0, math.inf
    for lag in range(max_lag + 1):
        parameter_count = 1 + 2 * lag
        residual_sscp = target_cross_product - cumulative_explained[parameter_count - 1]
        sign, log_determinant = np.linalg.slogdet(residual_sscp / observations)
        bic = log_determinant + math.log(observations) * (2 + 4 * lag) / observations if sign > 0 else math.inf
        if bic < best_bic:
            best_lag, best_bic = lag, bic
    return best_lag, triangular, projected_targets


def significance_stars(p_value: float) -> str:
    return "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""


def analyze_pair(first: str, second: str, window: dict[str, pd.DataFrame], max_lag: int, evaluation_date: pd.Timestamp) -> list[dict[str, object]]:
    aligned = pd.concat(
        [window[first][RETURN_COLUMN].rename(first), window[second][RETURN_COLUMN].rename(second)],
        axis=1,
        join="inner",
    ).dropna()
    window_dates = pd.concat([window[first]["trading_date"], window[second]["trading_date"]]).dropna()
    base = {
        "evaluation_date": evaluation_date.date().isoformat(),
        "window_start_date": window_dates.min().date().isoformat(),
        "window_end_date": evaluation_date.date().isoformat(),
        "pair_id": f"{first}__{second}",
        "overlap_start": aligned.index.min().isoformat() if len(aligned) else "",
        "overlap_end": aligned.index.max().isoformat() if len(aligned) else "",
        "overlap_observations": len(aligned),
    }
    minimum_observations = max(3 * max_lag + 2, 30)
    if len(aligned) <= minimum_observations:
        return [
            {**base, "regression_observations": max(0, len(aligned) - max_lag), "bic_selected_lag": np.nan,
             "explanatory_futures": cause, "dependent_futures": dependent, "wald_f_statistic": np.nan,
             "p_value": np.nan, "significance_stars": "", "numerator_df": np.nan, "denominator_df": np.nan,
             "test_status": "insufficient_observations"}
            for cause, dependent in ((second, first), (first, second))
        ]
    targets, maximum_design = maximum_lag_matrices(aligned.to_numpy(), max_lag)
    try:
        selected_lag, triangular, projected_targets = select_bic(targets, maximum_design, max_lag)
    except np.linalg.LinAlgError:
        selected_lag = None
    base.update({"regression_observations": len(targets), "bic_selected_lag": selected_lag})
    if selected_lag is None:
        return [
            {**base, "explanatory_futures": cause, "dependent_futures": dependent, "wald_f_statistic": np.nan,
             "p_value": np.nan, "significance_stars": "", "numerator_df": np.nan, "denominator_df": np.nan,
             "test_status": "singular_design"}
            for cause, dependent in ((second, first), (first, second))
        ]
    if selected_lag == 0:
        return [
            {**base, "explanatory_futures": cause, "dependent_futures": dependent, "wald_f_statistic": np.nan,
             "p_value": np.nan, "significance_stars": "", "numerator_df": 0,
             "denominator_df": 2 * (len(targets) - 1), "test_status": "bic_selected_zero_lag"}
            for cause, dependent in ((second, first), (first, second))
        ]
    parameter_count = 1 + 2 * selected_lag
    selected_triangular = triangular[:parameter_count, :parameter_count]
    coefficients = solve_triangular(
        selected_triangular, projected_targets[:parameter_count], lower=False
    )
    inverse_triangular = solve_triangular(
        selected_triangular, np.eye(parameter_count), lower=False
    )
    inverse_cross_product = inverse_triangular @ inverse_triangular.T
    residuals = targets - maximum_design[:, :parameter_count] @ coefficients
    residual_sscp = residuals.T @ residuals
    residual_df = len(targets) - parameter_count
    rows = []
    for causing_index, dependent_index in ((1, 0), (0, 1)):
        tested_columns = np.array([2 * lag - 1 + causing_index for lag in range(1, selected_lag + 1)])
        tested_coefficients = coefficients[tested_columns, dependent_index]
        restriction_covariance = inverse_cross_product[np.ix_(tested_columns, tested_columns)]
        residual_variance = residual_sscp[dependent_index, dependent_index] / residual_df
        statistic = float(tested_coefficients @ np.linalg.solve(restriction_covariance, tested_coefficients) / (selected_lag * residual_variance))
        denominator_df = 2 * residual_df
        p_value = float(f_distribution.sf(statistic, selected_lag, denominator_df))
        rows.append({
            **base, "explanatory_futures": aligned.columns[causing_index],
            "dependent_futures": aligned.columns[dependent_index], "wald_f_statistic": statistic,
            "p_value": p_value, "significance_stars": significance_stars(p_value),
            "numerator_df": selected_lag, "denominator_df": denominator_df, "test_status": "tested",
        })
    return rows


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8", newline="") as temporary:
        temporary_path = Path(temporary.name)
        frame.to_csv(temporary, index=False)
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    if args.window_days < 2 or args.max_lag < 1 or args.years < 1:
        raise ValueError("--window-days must be >= 2, --max-lag must be >= 1, and --years must be >= 1")
    symbols = load_universe(args.symbols)
    returns = load_returns(symbols)
    evaluation_dates = common_evaluation_dates(returns, args)
    all_dates = sorted(set().union(*(frame["trading_date"].dt.normalize().unique() for frame in returns.values())))
    all_dates = [pd.Timestamp(value) for value in all_dates]
    pairs = list(combinations(symbols, 2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "daily"
    checkpoint_dir.mkdir(exist_ok=True)
    for date_number, evaluation_date in enumerate(evaluation_dates, start=1):
        checkpoint_path = checkpoint_dir / f"{evaluation_date.date().isoformat()}.csv"
        if checkpoint_path.exists() and not args.no_resume:
            if date_number % args.progress_every == 0 or date_number == len(evaluation_dates):
                print(f"Available {date_number}/{len(evaluation_dates)} dates", flush=True)
            continue
        position = max(index for index, date in enumerate(all_dates) if date <= evaluation_date)
        window_dates = set(all_dates[max(0, position - args.window_days + 1) : position + 1])
        window = {symbol: frame.loc[frame["trading_date"].dt.normalize().isin(window_dates)] for symbol, frame in returns.items()}
        daily_rows = []
        for first, second in pairs:
            daily_rows.extend(analyze_pair(first, second, window, args.max_lag, evaluation_date))
        daily = pd.DataFrame(daily_rows)
        daily["fdr_bh_q_value"] = np.nan
        tested = daily["p_value"].notna()
        if tested.any():
            daily.loc[tested, "fdr_bh_q_value"] = multipletests(daily.loc[tested, "p_value"], method="fdr_bh")[1]
        daily = daily[RESULT_COLUMNS]
        expected_daily = len(symbols) * (len(symbols) - 1)
        if len(daily) != expected_daily or not daily.groupby(["explanatory_futures", "dependent_futures"]).size().eq(1).all():
            raise AssertionError(f"Expected {expected_daily} unique rows for {evaluation_date.date()}, got {len(daily)}")
        atomic_to_csv(daily, checkpoint_path)
        if date_number % args.progress_every == 0 or date_number == len(evaluation_dates):
            print(f"Completed {date_number}/{len(evaluation_dates)} dates", flush=True)
    checkpoint_paths = [checkpoint_dir / f"{date.date().isoformat()}.csv" for date in evaluation_dates]
    missing_checkpoints = [path for path in checkpoint_paths if not path.exists()]
    if missing_checkpoints:
        raise RuntimeError(f"Missing {len(missing_checkpoints)} daily checkpoints")
    results = pd.concat((pd.read_csv(path, keep_default_na=False) for path in checkpoint_paths), ignore_index=True)[RESULT_COLUMNS]
    expected = len(evaluation_dates) * len(symbols) * (len(symbols) - 1)
    if len(results) != expected or not results.groupby(["evaluation_date", "explanatory_futures", "dependent_futures"]).size().eq(1).all():
        raise AssertionError(f"Expected {expected} unique daily directional rows, got {len(results)}")
    results = results.sort_values(["explanatory_futures", "dependent_futures", "evaluation_date"], ignore_index=True)
    atomic_to_csv(results, args.output_dir / "dynamic_pairwise_granger_results.csv")
    metadata = {
        "completed_at": datetime.now().astimezone().isoformat(), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "symbols": symbols,
        "symbol_count": len(symbols), "evaluation_dates": len(evaluation_dates),
        "start_date": evaluation_dates[0].date().isoformat(), "end_date": evaluation_dates[-1].date().isoformat(),
        "window_trading_days": args.window_days, "maximum_candidate_lag": args.max_lag,
        "requested_years": args.years,
        "rows_per_date": len(symbols) * (len(symbols) - 1), "total_rows": len(results),
        "storage_order": "explanatory_futures, dependent_futures, evaluation_date",
        "fdr": "Benjamini-Hochberg independently over all finite directional p-values on each evaluation date",
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()