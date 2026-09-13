"""Measure per-symbol rolling PCA variance on 15-minute futures factors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"underlying_symbol", "datetime", "trading_date", "dominant_id", "total_turnover"}


def selected_symbols(graph_path: Path) -> list[str]:
    graph = pd.read_csv(graph_path, usecols=["code", "is_future"], dtype=str)
    symbols = graph.loc[graph["is_future"].str.lower().eq("true"), "code"].str.upper()
    if symbols.duplicated().any() or len(symbols) != 44:
        raise ValueError(f"Expected 44 distinct graph futures, found {len(symbols)}")
    return sorted(symbols.tolist())


def rolling_explained_variance(
    dates: np.ndarray, values: np.ndarray, window_days: int, max_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative explained ratios and active feature counts for each window."""
    if window_days < 2 or max_components < 1:
        raise ValueError("window_days must be >= 2 and max_components must be positive")
    if values.ndim != 2 or len(dates) != len(values):
        raise ValueError("dates and 2D values must have matching row counts")

    unique_days, day_starts, day_counts = np.unique(
        dates, return_index=True, return_counts=True
    )
    if not np.all(dates[:-1] <= dates[1:]):
        raise ValueError("dates must be sorted by trading day")
    feature_count = values.shape[1]
    # Statistics of observed values and missingness suffice for mean-imputed covariance.
    daily = np.zeros((len(unique_days), 3 + 2 * feature_count + 3 * feature_count**2))
    for day, (start, count) in enumerate(zip(day_starts, day_counts)):
        chunk = values[start : start + count]
        observed = np.isfinite(chunk)
        filled = np.where(observed, chunk, 0.0)
        missing = (~observed).astype(float)
        daily[day] = np.concatenate(
            (
                [len(chunk), 0, 0],
                filled.sum(axis=0),
                observed.sum(axis=0),
                (filled.T @ filled).ravel(),
                (missing.T @ filled).ravel(),
                (missing.T @ missing).ravel(),
            )
        )
    cumulative = np.vstack((np.zeros(daily.shape[1]), daily.cumsum(axis=0)))
    curves: list[np.ndarray] = []
    active_counts: list[int] = []
    matrix_start = 3 + 2 * feature_count
    for end in range(window_days, len(unique_days) + 1):
        stats = cumulative[end] - cumulative[end - window_days]
        row_count = int(stats[0])
        if row_count < 2:
            continue
        sums = stats[3 : 3 + feature_count]
        counts = stats[3 + feature_count : matrix_start]
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        cross, missing_cross, missing_pair = (
            stats[matrix_start + index * feature_count**2 :
                  matrix_start + (index + 1) * feature_count**2].reshape(feature_count, feature_count)
            for index in range(3)
        )
        centered = (
            cross
            + means[:, None] * missing_cross
            + means[None, :] * missing_cross.T
            + means[:, None] * means[None, :] * missing_pair
            - row_count * np.outer(means, means)
        )
        variances = np.maximum(np.diag(centered) / row_count, 0)
        active = (counts > 0) & (variances > 1e-12 * np.maximum(means**2, 1))
        active_count = int(active.sum())
        if active_count == 0:
            continue
        covariance = centered[np.ix_(active, active)] / row_count
        scales = np.sqrt(variances[active])
        correlation = covariance / np.outer(scales, scales)
        correlation = (correlation + correlation.T) / 2
        eigenvalues = np.maximum(np.linalg.eigvalsh(correlation)[::-1], 0)
        total = eigenvalues.sum()
        if total <= 0:
            continue
        ratios = np.cumsum(eigenvalues) / total
        curves.append(np.pad(ratios, (0, max(0, max_components - active_count)), constant_values=1)[:max_components])
        active_counts.append(active_count)
    return np.asarray(curves).reshape(-1, max_components), np.asarray(active_counts)


def analyze_symbol(path: Path, window_days: int, max_components: int) -> dict[str, object]:
    data = pd.read_csv(path, low_memory=False)
    if "trading_date" not in data:
        raise ValueError(f"Missing trading_date: {path}")
    dates = pd.to_datetime(data["trading_date"], errors="raise").to_numpy(dtype="datetime64[D]")
    if pd.isna(dates).any() or not np.all(dates[:-1] <= dates[1:]):
        raise ValueError(f"Unsorted or invalid trading_date: {path}")
    factor_columns = [column for column in data if column not in EXCLUDED]
    if not factor_columns:
        raise ValueError(f"No factor columns: {path}")
    factors = data[factor_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float, copy=True)
    factors[~np.isfinite(factors)] = np.nan
    available = np.isfinite(factors).any(axis=0)
    curves, active_counts = rolling_explained_variance(
        dates, factors[:, available], window_days, max_components
    )
    if len(curves) == 0:
        raise ValueError(f"No valid rolling windows: {path}")
    result: dict[str, object] = {
        "symbol": path.stem, "window_days": window_days, "step_days": 1,
        "window_count": len(curves), "input_factor_count": len(factor_columns),
        "available_factor_count": int(available.sum()),
        "min_active_factors": int(active_counts.min()),
        "mean_active_factors": float(active_counts.mean()),
        "max_active_factors": int(active_counts.max()),
    }
    for index, mean_ratio in enumerate(curves.mean(axis=0), start=1):
        result[f"pc{index}_cumulative_mean"] = float(mean_ratio)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw_data/15min")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/graph_data/economic_graph.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "pca.csv")
    parser.add_argument("--window-days", type=int, default=60)
    parser.add_argument("--max-components", type=int, default=40)
    parser.add_argument("--symbols", nargs="+", help="Subset for smoke tests")
    args = parser.parse_args()
    symbols = selected_symbols(args.graph)
    if args.symbols:
        requested = {symbol.upper() for symbol in args.symbols}
        if requested - set(symbols):
            raise ValueError(f"Not graph futures: {sorted(requested - set(symbols))}")
        symbols = [symbol for symbol in symbols if symbol in requested]
    results = []
    for symbol in symbols:
        path = args.data_dir / f"{symbol}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        result = analyze_symbol(path, args.window_days, args.max_components)
        results.append(result)
        print(f"{symbol}: {result['window_count']} windows", flush=True)
    if not args.symbols:
        global_row = {"symbol": "ALL_SYMBOLS_EQUAL_WEIGHT", "window_days": args.window_days,
                      "step_days": 1, "window_count": sum(item["window_count"] for item in results),
                      "input_factor_count": np.nan, "available_factor_count": np.nan,
                      "min_active_factors": min(item["min_active_factors"] for item in results),
                      "mean_active_factors": np.mean([item["mean_active_factors"] for item in results]),
                      "max_active_factors": max(item["max_active_factors"] for item in results)}
        for index in range(1, args.max_components + 1):
            key = f"pc{index}_cumulative_mean"
            global_row[key] = np.mean([item[key] for item in results])
        results.append(global_row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()