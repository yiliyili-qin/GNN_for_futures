"""Run pairwise bivariate VAR Granger tests for the 44-futures universe."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from scipy.stats import f as f_distribution
import statsmodels
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw_data" / "15min"
UNIVERSE_FILE = PROJECT_ROOT / "picture" / "futures_target_return_acf_corrected.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "no_smm_pairwise_granger"
RETURN_COLUMN = "return_15m"
MAX_LAG = 40
EXPECTED_UNIVERSE_SIZE = 44
COLORS = ["#FFFFFF", "#F4A3B4", "#D73027", "#6A3D9A"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        help="Comma-separated subset for a smoke test; default uses the exact 44-node universe.",
    )
    parser.add_argument("--max-lag", type=int, default=MAX_LAG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def load_universe(symbol_argument: str | None) -> list[str]:
    universe = (
        pd.read_csv(UNIVERSE_FILE, usecols=["variety"])["variety"]
        .astype(str)
        .tolist()
    )
    if len(universe) != EXPECTED_UNIVERSE_SIZE or len(set(universe)) != len(universe):
        raise ValueError("The authoritative futures universe must contain 44 unique symbols.")
    if symbol_argument is None:
        return universe
    requested = [symbol.strip() for symbol in symbol_argument.split(",") if symbol.strip()]
    unknown = sorted(set(requested).difference(universe))
    if unknown:
        raise ValueError(f"Unknown requested symbols: {unknown}")
    if len(requested) < 2 or len(set(requested)) != len(requested):
        raise ValueError("A subset must contain at least two unique symbols.")
    return requested


def load_returns(symbols: list[str]) -> dict[str, pd.Series]:
    returns: dict[str, pd.Series] = {}
    for symbol in symbols:
        frame = pd.read_csv(
            DATA_DIR / f"{symbol}.csv",
            usecols=["datetime", RETURN_COLUMN],
            parse_dates=["datetime"],
        )
        values = pd.to_numeric(frame[RETURN_COLUMN], errors="coerce")
        series = pd.Series(values.to_numpy(), index=frame["datetime"], name=symbol)
        returns[symbol] = (
            series[~series.index.duplicated(keep="last")].sort_index().dropna()
        )
    return returns


def align_pair(
    first: str, second: str, returns: dict[str, pd.Series]
) -> pd.DataFrame:
    aligned = pd.concat([returns[first], returns[second]], axis=1, join="inner").dropna()
    if len(aligned) <= 3 * MAX_LAG + 2:
        raise ValueError(f"{first}-{second} has only {len(aligned)} overlapping observations")
    return aligned


def maximum_lag_matrices(
    values: np.ndarray, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    observations = len(values) - max_lag
    targets = values[max_lag:]
    design = np.ones((observations, 1 + 2 * max_lag), dtype=np.float64)
    for lag in range(1, max_lag + 1):
        design[:, 2 * lag - 1 : 2 * lag + 1] = values[max_lag - lag : -lag]
    return targets, design


def fit_ols(
    targets: np.ndarray, design: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank != design.shape[1]:
        raise np.linalg.LinAlgError(
            f"Design rank {rank} is smaller than {design.shape[1]} columns"
        )
    residuals = targets - design @ coefficients
    inverse_cross_product = np.linalg.inv(design.T @ design)
    residual_sscp = residuals.T @ residuals
    return coefficients, residuals, inverse_cross_product, residual_sscp


def select_bic(
    targets: np.ndarray, maximum_design: np.ndarray, max_lag: int
) -> tuple[pd.DataFrame, int]:
    observations = len(targets)
    rows: list[dict[str, float | int]] = []
    for lag in range(max_lag + 1):
        parameters_per_equation = 1 + 2 * lag
        design = maximum_design[:, :parameters_per_equation]
        _, _, _, residual_sscp = fit_ols(targets, design)
        covariance_mle = residual_sscp / observations
        sign, log_determinant = np.linalg.slogdet(covariance_mle)
        if sign <= 0 or not np.isfinite(log_determinant):
            raise np.linalg.LinAlgError(f"Non-positive residual covariance at lag {lag}")
        system_parameters = 2 + 4 * lag
        bic = log_determinant + math.log(observations) * system_parameters / observations
        rows.append(
            {
                "candidate_lag": lag,
                "bic": float(bic),
                "common_effective_observations": observations,
                "parameters_per_equation": parameters_per_equation,
                "system_parameters": system_parameters,
            }
        )
    criteria = pd.DataFrame(rows)
    selected_lag = int(criteria.loc[criteria["bic"].idxmin(), "candidate_lag"])
    return criteria, selected_lag


def significance_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def wald_f_test(
    coefficients: np.ndarray,
    residual_sscp: np.ndarray,
    inverse_cross_product: np.ndarray,
    observations: int,
    parameters_per_equation: int,
    dependent_index: int,
    causing_index: int,
    selected_lag: int,
) -> tuple[float, float, int, int]:
    tested_columns = np.array(
        [2 * lag - 1 + causing_index for lag in range(1, selected_lag + 1)]
    )
    tested_coefficients = coefficients[tested_columns, dependent_index]
    restriction_covariance = inverse_cross_product[np.ix_(tested_columns, tested_columns)]
    residual_degrees_freedom = observations - parameters_per_equation
    residual_variance = residual_sscp[dependent_index, dependent_index] / residual_degrees_freedom
    statistic = float(
        tested_coefficients
        @ np.linalg.solve(restriction_covariance, tested_coefficients)
        / (selected_lag * residual_variance)
    )
    system_residual_degrees_freedom = 2 * residual_degrees_freedom
    p_value = float(
        f_distribution.sf(statistic, selected_lag, system_residual_degrees_freedom)
    )
    return statistic, p_value, selected_lag, system_residual_degrees_freedom


def analyze_pair(
    first: str,
    second: str,
    returns: dict[str, pd.Series],
    max_lag: int,
) -> tuple[list[dict[str, object]], pd.DataFrame, float]:
    aligned = align_pair(first, second, returns)
    targets, maximum_design = maximum_lag_matrices(aligned.to_numpy(), max_lag)
    criteria, selected_lag = select_bic(targets, maximum_design, max_lag)
    pair_id = f"{first}__{second}"
    criteria.insert(0, "pair_id", pair_id)
    criteria.insert(1, "first_futures", first)
    criteria.insert(2, "second_futures", second)

    base: dict[str, object] = {
        "pair_id": pair_id,
        "overlap_start": aligned.index.min().isoformat(),
        "overlap_end": aligned.index.max().isoformat(),
        "overlap_observations": len(aligned),
        "regression_observations": len(targets),
        "bic_selected_lag": selected_lag,
    }
    rows: list[dict[str, object]] = []
    if selected_lag == 0:
        for causing, dependent in ((second, first), (first, second)):
            rows.append(
                {
                    **base,
                    "explanatory_futures": causing,
                    "dependent_futures": dependent,
                    "wald_f_statistic": np.nan,
                    "p_value": np.nan,
                    "significance_stars": "",
                    "numerator_df": 0,
                    "denominator_df": 2 * (len(targets) - 1),
                    "test_status": "bic_selected_zero_lag",
                }
            )
    else:
        parameters_per_equation = 1 + 2 * selected_lag
        design = maximum_design[:, :parameters_per_equation]
        coefficients, _, inverse_cross_product, residual_sscp = fit_ols(targets, design)
        for causing_index, dependent_index in ((1, 0), (0, 1)):
            statistic, p_value, numerator_df, denominator_df = wald_f_test(
                coefficients,
                residual_sscp,
                inverse_cross_product,
                len(targets),
                parameters_per_equation,
                dependent_index,
                causing_index,
                selected_lag,
            )
            rows.append(
                {
                    **base,
                    "explanatory_futures": aligned.columns[causing_index],
                    "dependent_futures": aligned.columns[dependent_index],
                    "wald_f_statistic": statistic,
                    "p_value": p_value,
                    "significance_stars": significance_stars(p_value),
                    "numerator_df": numerator_df,
                    "denominator_df": denominator_df,
                    "test_status": "tested",
                }
            )
    correlation = float(aligned[first].corr(aligned[second]))
    return rows, criteria, correlation


def add_fdr(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["fdr_bh_q_value"] = np.nan
    tested = results["p_value"].notna()
    if tested.any():
        results.loc[tested, "fdr_bh_q_value"] = multipletests(
            results.loc[tested, "p_value"].to_numpy(), method="fdr_bh"
        )[1]
    return results


def cluster_order(
    symbols: list[str], pair_correlations: dict[tuple[str, str], float]
) -> tuple[list[str], pd.DataFrame]:
    count = len(symbols)
    correlation = np.eye(count)
    positions = {symbol: index for index, symbol in enumerate(symbols)}
    for (first, second), value in pair_correlations.items():
        first_index, second_index = positions[first], positions[second]
        correlation[first_index, second_index] = value
        correlation[second_index, first_index] = value
    distance = np.clip(1.0 - np.abs(correlation), 0.0, 1.0)
    condensed = squareform(distance, checks=True)
    tree = linkage(condensed, method="average")
    tree = optimal_leaf_ordering(tree, condensed)
    order = [symbols[index] for index in leaves_list(tree)]
    order_frame = pd.DataFrame(
        {
            "position": range(1, count + 1),
            "variety": order,
        }
    )
    return order, order_frame


def significance_level(stars: str) -> int:
    return {"": 0, "*": 1, "**": 2, "***": 3}[stars]


def plot_heatmap(results: pd.DataFrame, order: list[str], output_dir: Path) -> None:
    matrix = pd.DataFrame(np.nan, index=order, columns=order)
    for row in results.itertuples(index=False):
        matrix.loc[row.dependent_futures, row.explanatory_futures] = significance_level(
            row.significance_stars
        )
    display = matrix.fillna(0).to_numpy()
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    figure, axis = plt.subplots(figsize=(18, 17))
    axis.imshow(display, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    ticks = np.arange(len(order))
    axis.set_xticks(ticks, labels=order, rotation=90, fontsize=8)
    axis.set_yticks(ticks, labels=order, fontsize=8)
    axis.set_xlabel("Explanatory futures (candidate Granger cause)")
    axis.set_ylabel("Dependent futures")
    axis.set_title("Pairwise futures Granger significance (column -> row, no SMM)")
    axis.set_xticks(np.arange(-0.5, len(order), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    axis.grid(which="minor", color="#D0D0D0", linewidth=0.25)
    axis.tick_params(which="minor", bottom=False, left=False)
    for index in range(len(order)):
        axis.text(index, index, "-", ha="center", va="center", color="#808080", fontsize=7)
    axis.legend(
        handles=[
            Patch(facecolor=COLORS[0], edgecolor="#AAAAAA", label="Not significant"),
            Patch(facecolor=COLORS[1], label="*  p < 0.10"),
            Patch(facecolor=COLORS[2], label="**  p < 0.05"),
            Patch(facecolor=COLORS[3], label="***  p < 0.01"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "granger_significance_heatmap.png", dpi=220)
    figure.savefig(output_dir / "granger_significance_heatmap.pdf")
    plt.close(figure)


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8", newline=""
    ) as temporary:
        temporary_path = Path(temporary.name)
        frame.to_csv(temporary, index=False)
    os.replace(temporary_path, path)


def validate_results(
    results: pd.DataFrame,
    criteria: pd.DataFrame,
    symbols: list[str],
    order: list[str],
) -> None:
    expected_pairs = len(symbols) * (len(symbols) - 1) // 2
    expected_directions = len(symbols) * (len(symbols) - 1)
    if len(results) != expected_directions:
        raise AssertionError(f"Expected {expected_directions} result rows, got {len(results)}")
    if results["pair_id"].nunique() != expected_pairs:
        raise AssertionError("Unexpected number of unordered pairs")
    if not results.groupby("pair_id").size().eq(2).all():
        raise AssertionError("Each unordered pair must have exactly two directions")
    if (results["explanatory_futures"] == results["dependent_futures"]).any():
        raise AssertionError("Self-causality rows are not allowed")
    direction_count = results.groupby(
        ["explanatory_futures", "dependent_futures"]
    ).size()
    if len(direction_count) != expected_directions or not direction_count.eq(1).all():
        raise AssertionError("Each ordered direction must occur exactly once")
    tested = results["test_status"].eq("tested")
    if not np.isfinite(results.loc[tested, "wald_f_statistic"]).all():
        raise AssertionError("Tested F statistics must be finite")
    if not results.loc[tested, "p_value"].between(0, 1).all():
        raise AssertionError("Tested p-values must lie in [0, 1]")
    expected_stars = results.loc[tested, "p_value"].map(significance_stars)
    if not expected_stars.equals(results.loc[tested, "significance_stars"]):
        raise AssertionError("Significance stars do not match p-values")
    selected = criteria.loc[criteria.groupby("pair_id")["bic"].idxmin(), ["pair_id", "candidate_lag"]]
    observed = results[["pair_id", "bic_selected_lag"]].drop_duplicates()
    checked = observed.merge(selected, on="pair_id", validate="one_to_one")
    if not checked["bic_selected_lag"].eq(checked["candidate_lag"]).all():
        raise AssertionError("A selected lag is not the minimum-BIC candidate")
    if len(order) != len(symbols) or set(order) != set(symbols):
        raise AssertionError("Cluster order must contain every symbol exactly once")


def write_metadata(
    output_dir: Path,
    symbols: list[str],
    max_lag: int,
    results: pd.DataFrame,
) -> None:
    metadata = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
        "symbols": symbols,
        "symbol_count": len(symbols),
        "unordered_pair_count": len(symbols) * (len(symbols) - 1) // 2,
        "directional_test_count": len(results),
        "data_directory": str(DATA_DIR),
        "universe_file": str(UNIVERSE_FILE),
        "return_column": RETURN_COLUMN,
        "alignment": "inner join on datetime; no filling; all overlapping valid returns",
        "maximum_candidate_lag": max_lag,
        "candidate_lags": f"0..{max_lag}",
        "common_candidate_sample": f"all aligned observations after dropping the first {max_lag}",
        "bic": "log(det(residual_sscp / T)) + log(T) * (2 + 4p) / T",
        "test": "equation-level joint Wald F test of all cross-lag coefficients",
        "significance": {"*": "p < 0.10", "**": "p < 0.05", "***": "p < 0.01"},
        "fdr": "Benjamini-Hochberg over all finite directional p-values; auxiliary only",
        "cluster_distance": "1 - abs(pairwise Pearson correlation on overlapping returns)",
        "cluster_linkage": "average with optimal leaf ordering",
        "heatmap_colors": COLORS,
    }
    path = output_dir / "run_metadata.json"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=path.parent, delete=False, encoding="utf-8"
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(metadata, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    if args.max_lag < 1:
        raise ValueError("--max-lag must be at least 1")
    symbols = load_universe(args.symbols)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    returns = load_returns(symbols)
    pairs = list(combinations(symbols, 2))
    result_rows: list[dict[str, object]] = []
    criteria_frames: list[pd.DataFrame] = []
    pair_correlations: dict[tuple[str, str], float] = {}
    for pair_number, (first, second) in enumerate(pairs, start=1):
        rows, criteria, correlation = analyze_pair(
            first, second, returns, args.max_lag
        )
        result_rows.extend(rows)
        criteria_frames.append(criteria)
        pair_correlations[(first, second)] = correlation
        if pair_number % args.progress_every == 0 or pair_number == len(pairs):
            print(f"Completed {pair_number}/{len(pairs)} pairs", flush=True)

    results = add_fdr(pd.DataFrame(result_rows))
    criteria = pd.concat(criteria_frames, ignore_index=True)
    order, order_frame = cluster_order(symbols, pair_correlations)
    validate_results(results, criteria, symbols, order)
    results = results.sort_values(
        ["dependent_futures", "explanatory_futures"], ignore_index=True
    )
    criteria = criteria.sort_values(
        ["first_futures", "second_futures", "candidate_lag"], ignore_index=True
    )
    atomic_to_csv(results, args.output_dir / "pairwise_granger_results.csv")
    atomic_to_csv(criteria, args.output_dir / "pairwise_bic_candidates.csv")
    atomic_to_csv(order_frame, args.output_dir / "futures_cluster_order.csv")
    plot_heatmap(results, order, args.output_dir)
    write_metadata(args.output_dir, symbols, args.max_lag, results)
    print(
        f"Wrote {len(results)} directional results for {len(pairs)} pairs to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()