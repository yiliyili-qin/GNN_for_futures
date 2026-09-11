"""Estimate static and 20+1-day regressions for every directed economic graph edge."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import f as f_distribution
from scipy.stats import t as t_distribution


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw_data" / "15min"
GRAPH_PATH = PROJECT_ROOT / "data" / "graph_data" / "futures_graph_smm_simplified.html"
SMM_CURATED_PATH = PROJECT_ROOT / "data" / "processed" / "smm" / "curated" / "smm_indicators_long.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "graph_edge_regressions"
EDGE_BLOCK = re.compile(r"const rawEdges = `(?P<edges>.*?)`\.trim\(\);", re.DOTALL)
RETURN_COLUMN = "return_15m"

SMM_PATH_NODES = {
    "石脑油": "NAPHTHA",
    "混合二甲苯": "MX",
    "丁二烯": "BUTADIENE",
}
SMM_FEATURES = {
    "NAPHTHA": ("z00114308", "z00114313"),
    "MX": ("z00115353", "z00115373"),
    "BUTADIENE": ("z00115653", "z00115658"),
}

SUMMARY_COLUMNS = [
    "scope", "graph_edge_number", "graph_source", "graph_target", "relation_type",
    "smm_path", "directed_edge", "explanatory_futures", "dependent_futures",
    "model_variant", "forecast_date", "train_start_date", "train_end_date",
    "overlap_observations", "regression_observations", "bic_selected_lag",
    "parameter_count", "design_rank", "condition_number", "r_squared",
    "adjusted_r_squared", "overall_f_statistic", "overall_f_p_value",
    "overall_numerator_df", "overall_denominator_df", "edge_wald_f_statistic",
    "edge_wald_p_value", "edge_wald_numerator_df", "edge_wald_denominator_df",
    "residual_mean", "residual_std", "in_sample_rmse", "in_sample_mae",
    "oos_observations", "oos_r_squared", "oos_rmse", "oos_mae", "oos_error_mean",
    "oos_error_std", "forecast_direction_hits", "forecast_direction_comparisons",
    "forecast_zero_return_count", "forecast_direction_hit_rate", "smm_nodes",
    "smm_joint_f_statistic", "smm_joint_p_value", "model_status",
]

COEFFICIENT_COLUMNS = [
    "scope", "graph_edge_number", "directed_edge", "model_variant", "forecast_date",
    "term_type", "term_node", "lag", "coefficient", "standard_error",
    "t_statistic", "p_value", "ci_95_low", "ci_95_high", "significant_95",
]

FORECAST_COLUMNS = [
    "graph_edge_number", "directed_edge", "model_variant", "forecast_date", "datetime",
    "actual_y", "predicted_y", "forecast_error", "squared_error", "absolute_error",
    "forecast_direction_status", "forecast_direction_matches_actual",
    "latest_futures_information_time", "smm_latest_available_at", "smm_update_available",
    "forecast_status",
]

ABLATION_COLUMNS = [
    "scope", "graph_edge_number", "directed_edge", "forecast_date", "smm_nodes",
    "bic_selected_lag", "baseline_oos_r_squared", "smm_oos_r_squared", "delta_oos_r_squared",
    "baseline_oos_rmse", "smm_oos_rmse", "delta_oos_rmse", "baseline_oos_mae",
    "smm_oos_mae", "delta_oos_mae", "baseline_direction_hit_rate",
    "smm_direction_hit_rate", "delta_direction_hit_rate", "ablation_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "dynamic", "both"), default="both")
    parser.add_argument("--edge-numbers", help="Comma-separated graph edge numbers; default is all 57.")
    parser.add_argument("--window-days", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=40)
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--last-forecast-days", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def graph_edges(path: Path = GRAPH_PATH) -> list[dict[str, object]]:
    match = EDGE_BLOCK.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError("Could not find rawEdges in graph HTML")
    edges = []
    for number, line in enumerate(match.group("edges").strip().splitlines(), start=1):
        source, target, relation_type, smm_path = line.split("|")
        smm_nodes = []
        for name in smm_path.split(" → ") if smm_path else []:
            node = SMM_PATH_NODES.get(name)
            if node and node not in smm_nodes:
                smm_nodes.append(node)
        edges.append({
            "graph_edge_number": number,
            "graph_source": source,
            "graph_target": target,
            "relation_type": relation_type,
            "smm_path": smm_path,
            "smm_nodes": smm_nodes,
        })
    if len(edges) != 57:
        raise ValueError(f"Expected 57 graph edges, found {len(edges)}")
    return edges


def select_edges(edges: list[dict[str, object]], argument: str | None) -> list[dict[str, object]]:
    if not argument:
        return edges
    requested = {int(value.strip()) for value in argument.split(",") if value.strip()}
    selected = [edge for edge in edges if edge["graph_edge_number"] in requested]
    if len(selected) != len(requested):
        found = {int(edge["graph_edge_number"]) for edge in selected}
        raise ValueError(f"Unknown edge numbers: {sorted(requested - found)}")
    return selected


def directed_edge_registry(edges: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for edge in edges:
        for explanatory, dependent in (
            (str(edge["graph_source"]), str(edge["graph_target"])),
            (str(edge["graph_target"]), str(edge["graph_source"])),
        ):
            rows.append({
                **{key: edge[key] for key in ("graph_edge_number", "graph_source", "graph_target", "relation_type", "smm_path")},
                "directed_edge": f"{explanatory}->{dependent}",
                "explanatory_futures": explanatory,
                "dependent_futures": dependent,
                "smm_nodes": ";".join(edge["smm_nodes"]),
                "smm_ablation_applicable": bool(edge["smm_nodes"]),
            })
    return pd.DataFrame(rows)


def load_returns(symbols: set[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in sorted(symbols):
        frame = pd.read_csv(
            DATA_DIR / f"{symbol}.csv",
            usecols=["datetime", "trading_date", RETURN_COLUMN],
            parse_dates=["datetime", "trading_date"],
        )
        frame[RETURN_COLUMN] = pd.to_numeric(frame[RETURN_COLUMN], errors="coerce")
        frame = frame.dropna(subset=[RETURN_COLUMN]).drop_duplicates("datetime", keep="last")
        frames[symbol] = frame.sort_values("datetime").set_index("datetime")
    return frames


def align_pair(first: str, second: str, returns: dict[str, pd.DataFrame]) -> pd.DataFrame:
    aligned = pd.concat(
        [returns[first][RETURN_COLUMN].rename(first), returns[second][RETURN_COLUMN].rename(second)],
        axis=1,
        join="inner",
    ).dropna()
    date_lookup = returns[first]["trading_date"].combine_first(returns[second]["trading_date"])
    aligned["trading_date"] = date_lookup.reindex(aligned.index).dt.normalize()
    return aligned.dropna(subset=["trading_date"])


def maximum_lag_matrices(values: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    observations = len(values) - max_lag
    if observations <= 0:
        raise ValueError("Not enough observations for the maximum candidate lag")
    targets = values[max_lag:]
    design = np.ones((observations, 1 + 2 * max_lag), dtype=np.float64)
    for lag in range(1, max_lag + 1):
        design[:, 2 * lag - 1 : 2 * lag + 1] = values[max_lag - lag : -lag]
    return targets, design


def select_bic(values: np.ndarray, max_lag: int) -> int:
    targets, maximum_design = maximum_lag_matrices(values, max_lag)
    observations = len(targets)
    design_cross_product = maximum_design.T @ maximum_design
    design_target_product = maximum_design.T @ targets
    target_cross_product = targets.T @ targets
    best_lag, best_bic = 0, math.inf
    for lag in range(max_lag + 1):
        parameter_count = 1 + 2 * lag
        cross_product = design_cross_product[:parameter_count, :parameter_count]
        target_product = design_target_product[:parameter_count]
        coefficients = np.linalg.lstsq(cross_product, target_product, rcond=None)[0]
        residual_sscp = target_cross_product - target_product.T @ coefficients
        sign, log_determinant = np.linalg.slogdet(residual_sscp / observations)
        bic = log_determinant + math.log(observations) * (2 + 4 * lag) / observations if sign > 0 else math.inf
        if bic < best_bic:
            best_lag, best_bic = lag, bic
    return best_lag


def design_for_direction(
    values: np.ndarray,
    dependent_index: int,
    explanatory_index: int,
    lag: int,
    start: int,
    positions: np.ndarray | None = None,
    smm_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str, int]]]:
    target_positions = np.arange(start, len(values)) if positions is None else positions
    columns = [np.ones(len(target_positions))]
    terms = [("intercept", "", 0)]
    for item_lag in range(1, lag + 1):
        columns.append(values[target_positions - item_lag, dependent_index])
        terms.append(("autoregressive", "dependent", item_lag))
    for item_lag in range(1, lag + 1):
        columns.append(values[target_positions - item_lag, explanatory_index])
        terms.append(("cross_futures", "explanatory", item_lag))
    if smm_values is not None and lag > 0:
        for node_index in range(smm_values.shape[1]):
            columns.append(smm_values[target_positions - 1, node_index])
            terms.append(("smm", str(node_index), 1))
    design = np.column_stack(columns)
    targets = values[target_positions, dependent_index]
    return targets, design, terms


def fit_equation(targets: np.ndarray, design: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(targets) & np.isfinite(design).all(axis=1)
    targets = targets[finite]
    design = design[finite]
    if len(targets) <= design.shape[1]:
        raise np.linalg.LinAlgError("Insufficient finite observations")
    coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank != design.shape[1]:
        raise np.linalg.LinAlgError("Rank-deficient regression design")
    fitted = design @ coefficients
    residuals = targets - fitted
    observations, parameters = design.shape
    residual_df = observations - parameters
    residual_ss = float(residuals @ residuals)
    total_ss = float((targets - targets.mean()) @ (targets - targets.mean()))
    residual_variance = residual_ss / residual_df
    inverse_cross_product = np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.diag(inverse_cross_product) * residual_variance)
    t_statistics = coefficients / standard_errors
    p_values = 2 * t_distribution.sf(np.abs(t_statistics), residual_df)
    confidence_delta = t_distribution.ppf(0.975, residual_df) * standard_errors
    r_squared = 1 - residual_ss / total_ss if total_ss > 0 else np.nan
    adjusted = 1 - (1 - r_squared) * (observations - 1) / residual_df if np.isfinite(r_squared) else np.nan
    numerator_df = parameters - 1
    if numerator_df > 0 and residual_ss > 0:
        explained_ss = max(0.0, total_ss - residual_ss)
        overall_f = (explained_ss / numerator_df) / (residual_ss / residual_df)
        overall_p = float(f_distribution.sf(overall_f, numerator_df, residual_df))
    else:
        overall_f = overall_p = np.nan
    return {
        "targets": targets, "design": design, "finite_mask": finite, "coefficients": coefficients,
        "standard_errors": standard_errors, "t_statistics": t_statistics, "p_values": p_values,
        "ci_low": coefficients - confidence_delta, "ci_high": coefficients + confidence_delta,
        "residuals": residuals, "inverse_cross_product": inverse_cross_product,
        "residual_variance": residual_variance, "observations": observations, "parameters": parameters,
        "rank": rank, "condition_number": float(np.linalg.cond(design)), "residual_df": residual_df,
        "r_squared": r_squared, "adjusted_r_squared": adjusted, "overall_f": overall_f,
        "overall_p": overall_p, "overall_numerator_df": numerator_df,
    }


def joint_f_test(fit: dict[str, object], columns: list[int]) -> tuple[float, float, int, int]:
    if not columns:
        return np.nan, np.nan, 0, int(fit["residual_df"])
    coefficients = np.asarray(fit["coefficients"])[columns]
    inverse = np.asarray(fit["inverse_cross_product"])[np.ix_(columns, columns)]
    statistic = float(coefficients @ np.linalg.solve(inverse, coefficients) / (len(columns) * fit["residual_variance"]))
    denominator_df = int(fit["residual_df"])
    return statistic, float(f_distribution.sf(statistic, len(columns), denominator_df)), len(columns), denominator_df


def model_records(
    edge: dict[str, object],
    explanatory: str,
    dependent: str,
    variant: str,
    scope: str,
    lag: int,
    fit: dict[str, object],
    terms: list[tuple[str, str, int]],
    smm_nodes: list[str],
    dates: dict[str, str],
    oos: dict[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    edge_columns = [index for index, term in enumerate(terms) if term[0] == "cross_futures"]
    smm_columns = [index for index, term in enumerate(terms) if term[0] == "smm"]
    edge_f, edge_p, edge_num_df, edge_den_df = joint_f_test(fit, edge_columns)
    smm_f, smm_p, _, _ = joint_f_test(fit, smm_columns)
    residuals = np.asarray(fit["residuals"])
    summary = {
        "scope": scope, **{key: edge[key] for key in ("graph_edge_number", "graph_source", "graph_target", "relation_type", "smm_path")},
        "directed_edge": f"{explanatory}->{dependent}", "explanatory_futures": explanatory,
        "dependent_futures": dependent, "model_variant": variant, **dates,
        "overlap_observations": dates.pop("_overlap_observations"),
        "regression_observations": fit["observations"], "bic_selected_lag": lag,
        "parameter_count": fit["parameters"], "design_rank": fit["rank"],
        "condition_number": fit["condition_number"], "r_squared": fit["r_squared"],
        "adjusted_r_squared": fit["adjusted_r_squared"], "overall_f_statistic": fit["overall_f"],
        "overall_f_p_value": fit["overall_p"], "overall_numerator_df": fit["overall_numerator_df"],
        "overall_denominator_df": fit["residual_df"], "edge_wald_f_statistic": edge_f,
        "edge_wald_p_value": edge_p, "edge_wald_numerator_df": edge_num_df,
        "edge_wald_denominator_df": edge_den_df, "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std(ddof=1)), "in_sample_rmse": float(np.sqrt(np.mean(residuals**2))),
        "in_sample_mae": float(np.mean(np.abs(residuals))), "smm_nodes": ";".join(smm_nodes),
        "smm_joint_f_statistic": smm_f, "smm_joint_p_value": smm_p, "model_status": "fitted",
    }
    summary.update(oos or {})
    if lag == 0:
        summary["model_status"] = "zero_lag_no_smm_terms" if edge["smm_nodes"] else "bic_selected_zero_lag"
    coefficients = []
    for index, (term_type, term_node, term_lag) in enumerate(terms):
        node = smm_nodes[int(term_node)] if term_type == "smm" else term_node
        coefficients.append({
            "scope": scope, "graph_edge_number": edge["graph_edge_number"],
            "directed_edge": f"{explanatory}->{dependent}", "model_variant": variant,
            "forecast_date": dates.get("forecast_date", ""), "term_type": term_type,
            "term_node": node, "lag": term_lag, "coefficient": fit["coefficients"][index],
            "standard_error": fit["standard_errors"][index], "t_statistic": fit["t_statistics"][index],
            "p_value": fit["p_values"][index], "ci_95_low": fit["ci_low"][index],
            "ci_95_high": fit["ci_high"][index], "significant_95": bool(fit["p_values"][index] < 0.05),
        })
    return summary, coefficients


def load_smm_events() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    selected_ids = {item for values in SMM_FEATURES.values() for item in values}
    usecols = ["node_id", "feature_name", "value", "available_at", "quota_id", "quota_name", "frequency", "quality_flag"]
    curated = pd.read_csv(SMM_CURATED_PATH, usecols=usecols, parse_dates=["available_at"])
    curated = curated.loc[curated["quota_id"].isin(selected_ids) & curated["quality_flag"].eq("valid")].copy()
    registry = curated.drop_duplicates("quota_id")[usecols].drop(columns=["value", "available_at", "quality_flag"])
    registry["main_experiment"] = True
    events = {}
    for node, quota_ids in SMM_FEATURES.items():
        node_events = []
        for quota_id in quota_ids:
            frame = curated.loc[curated["quota_id"].eq(quota_id), ["available_at", "value"]].copy()
            frame = frame.drop_duplicates("available_at", keep="last").sort_values("available_at")
            frame["return"] = frame["value"].pct_change(fill_method=None)
            frame["quota_id"] = quota_id
            node_events.append(frame.dropna(subset=["return"])[["available_at", "return", "quota_id"]])
        events[node] = pd.concat(node_events, ignore_index=True).sort_values("available_at")
    return events, registry.sort_values(["node_id", "quota_id"])


def causal_smm_factors(index: pd.DatetimeIndex, events: dict[str, pd.DataFrame], nodes: list[str], bars_per_day: int) -> tuple[np.ndarray, pd.DataFrame]:
    factors = np.full((len(index), len(nodes)), np.nan)
    audit = pd.DataFrame(index=index)
    for node_index, node in enumerate(nodes):
        feature_values = []
        feature_available = []
        feature_updates = []
        for _, feature_events in events[node].groupby("quota_id"):
            values = np.full(len(index), np.nan)
            available = np.full(len(index), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
            updates = np.zeros(len(index), dtype=bool)
            active_seen = False
            for available_at, event_return, _ in feature_events.itertuples(index=False, name=None):
                position = int(index.searchsorted(available_at, side="left"))
                if position >= len(index) or event_return <= -1:
                    continue
                if not active_seen:
                    values[position:] = 0.0
                    active_seen = True
                spread = max(1, bars_per_day)
                end = min(len(index), position + spread)
                values[position:end] = math.expm1(math.log1p(float(event_return)) / spread)
                available[position:] = np.datetime64(available_at)
                updates[position] = True
            if active_seen:
                feature_values.append(values)
                feature_available.append(available)
                feature_updates.append(updates)
        if feature_values:
            stacked = np.vstack(feature_values)
            available_positions = np.isfinite(stacked).any(axis=0)
            factors[available_positions, node_index] = np.nanmedian(stacked[:, available_positions], axis=0)
            available_stack = np.vstack(feature_available)
            latest = np.max(available_stack.astype("datetime64[ns]").astype("int64"), axis=0)
            latest[latest == np.iinfo("int64").min] = np.datetime64("NaT", "ns").astype("int64")
            audit[f"{node}__latest_available_at"] = latest.astype("datetime64[ns]")
            audit[f"{node}__update_available"] = np.any(np.vstack(feature_updates), axis=0)
    return factors, audit


def typical_bars_per_day(frame: pd.DataFrame) -> int:
    counts = frame.groupby("trading_date").size()
    return max(1, int(round(float(counts.median()))))


def standardize_smm(train: np.ndarray, all_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    locations = np.nanmedian(train, axis=0)
    q75 = np.nanpercentile(train, 75, axis=0)
    q25 = np.nanpercentile(train, 25, axis=0)
    scales = q75 - q25
    valid = np.isfinite(locations) & np.isfinite(scales) & (scales > 0)
    standardized = np.full_like(all_values, np.nan, dtype=float)
    standardized[:, valid] = (all_values[:, valid] - locations[valid]) / scales[valid]
    return standardized, valid


def oos_records(
    edge: dict[str, object], explanatory: str, dependent: str, variant: str,
    test_index: pd.DatetimeIndex, actual: np.ndarray, predicted: np.ndarray,
    train_mean: float, audit: pd.DataFrame | None, forecast_date: pd.Timestamp,
    predictor_timestamps: pd.DatetimeIndex,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    finite = np.isfinite(actual) & np.isfinite(predicted)
    errors = actual[finite] - predicted[finite]
    denominator = float(np.sum((actual[finite] - train_mean) ** 2))
    direction_comparable = finite & (actual != 0) & (predicted != 0)
    direction_hits = direction_comparable & (np.sign(actual) == np.sign(predicted))
    zero_returns = finite & ((actual == 0) | (predicted == 0))
    summary = {
        "oos_observations": int(finite.sum()),
        "oos_r_squared": 1 - float(errors @ errors) / denominator if denominator > 0 else np.nan,
        "oos_rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
        "oos_mae": float(np.mean(np.abs(errors))) if len(errors) else np.nan,
        "oos_error_mean": float(np.mean(errors)) if len(errors) else np.nan,
        "oos_error_std": float(np.std(errors, ddof=1)) if len(errors) > 1 else np.nan,
        "forecast_direction_hits": int(direction_hits.sum()),
        "forecast_direction_comparisons": int(direction_comparable.sum()),
        "forecast_zero_return_count": int(zero_returns.sum()),
        "forecast_direction_hit_rate": float(direction_hits.sum() / direction_comparable.sum()) if direction_comparable.any() else np.nan,
    }
    rows = []
    for position, timestamp in enumerate(test_index):
        if not finite[position]:
            direction_status, match = "not_comparable", np.nan
        elif zero_returns[position]:
            direction_status, match = "zero_return", np.nan
        elif direction_hits[position]:
            direction_status, match = "same", True
        else:
            direction_status, match = "opposite", False
        latest_available = ""
        update_available = False
        predictor_timestamp = predictor_timestamps[position]
        if audit is not None and predictor_timestamp in audit.index:
            available_columns = [column for column in audit if column.endswith("latest_available_at")]
            update_columns = [column for column in audit if column.endswith("update_available")]
            candidates = pd.to_datetime(audit.loc[predictor_timestamp, available_columns], errors="coerce").dropna()
            latest_available = candidates.max().isoformat() if len(candidates) else ""
            update_available = bool(audit.loc[predictor_timestamp, update_columns].any()) if update_columns else False
        rows.append({
            "graph_edge_number": edge["graph_edge_number"], "directed_edge": f"{explanatory}->{dependent}",
            "model_variant": variant, "forecast_date": forecast_date.date().isoformat(), "datetime": timestamp.isoformat(),
            "actual_y": actual[position], "predicted_y": predicted[position],
            "forecast_error": actual[position] - predicted[position] if finite[position] else np.nan,
            "squared_error": (actual[position] - predicted[position]) ** 2 if finite[position] else np.nan,
            "absolute_error": abs(actual[position] - predicted[position]) if finite[position] else np.nan,
            "forecast_direction_status": direction_status, "forecast_direction_matches_actual": match,
            "latest_futures_information_time": predictor_timestamp.isoformat(),
            "smm_latest_available_at": latest_available, "smm_update_available": update_available,
            "forecast_status": "predicted" if finite[position] else "missing_predictor",
        })
    return summary, rows


def static_experiment(edges: list[dict[str, object]], returns: dict[str, pd.DataFrame], events: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, coefficients = [], []
    for edge in edges:
        aligned = align_pair(str(edge["graph_source"]), str(edge["graph_target"]), returns)
        values = aligned[[str(edge["graph_source"]), str(edge["graph_target"])]].to_numpy()
        lag = select_bic(values, arguments.max_lag)
        smm_nodes = list(edge["smm_nodes"])
        smm_values = None
        if smm_nodes:
            smm_values, _ = causal_smm_factors(aligned.index, events, smm_nodes, typical_bars_per_day(aligned))
            smm_values, valid_nodes = standardize_smm(smm_values, smm_values)
            smm_nodes = [node for node, valid in zip(smm_nodes, valid_nodes) if valid]
            smm_values = smm_values[:, valid_nodes]
        for explanatory_index, dependent_index in ((0, 1), (1, 0)):
            explanatory = str(edge["graph_source"] if explanatory_index == 0 else edge["graph_target"])
            dependent = str(edge["graph_source"] if dependent_index == 0 else edge["graph_target"])
            variants = [("baseline_full", None)]
            if smm_nodes and lag > 0:
                variants.extend((("baseline_matched", smm_values * np.nan), ("smm_enhanced", smm_values)))
            for variant, variant_smm in variants:
                actual_smm = smm_values if variant == "baseline_matched" else variant_smm
                targets, design, terms = design_for_direction(values, dependent_index, explanatory_index, lag, arguments.max_lag, smm_values=actual_smm)
                if variant == "baseline_matched":
                    smm_finite = np.isfinite(smm_values[arguments.max_lag - 1 : -1]).all(axis=1)
                    design = design[:, : 1 + 2 * lag]
                    terms = terms[: 1 + 2 * lag]
                    targets, design = targets[smm_finite], design[smm_finite]
                try:
                    fit = fit_equation(targets, design)
                except np.linalg.LinAlgError:
                    continue
                dates = {"forecast_date": "", "train_start_date": aligned["trading_date"].min().date().isoformat(),
                         "train_end_date": aligned["trading_date"].max().date().isoformat(), "_overlap_observations": len(aligned)}
                summary, rows = model_records(edge, explanatory, dependent, variant, "static", lag, fit, terms, smm_nodes, dates)
                summaries.append(summary)
                coefficients.extend(rows)
    return pd.DataFrame(summaries), pd.DataFrame(coefficients)


def dynamic_date_records(edge: dict[str, object], aligned: pd.DataFrame, forecast_date: pd.Timestamp, events: dict[str, pd.DataFrame]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    all_dates = sorted(pd.Timestamp(value) for value in aligned["trading_date"].unique())
    forecast_position = all_dates.index(forecast_date)
    train_dates = set(all_dates[forecast_position - arguments.window_days : forecast_position])
    combined = aligned.loc[aligned["trading_date"].isin(train_dates | {forecast_date})]
    train_count = int(combined["trading_date"].isin(train_dates).sum())
    values = combined[[str(edge["graph_source"]), str(edge["graph_target"])]].to_numpy()
    lag = select_bic(values[:train_count], arguments.max_lag)
    train_positions = np.arange(arguments.max_lag, train_count)
    test_positions = np.arange(train_count, len(combined))
    smm_nodes = list(edge["smm_nodes"])
    smm_values, audit = None, None
    if smm_nodes:
        raw_smm, audit = causal_smm_factors(combined.index, events, smm_nodes, typical_bars_per_day(combined.iloc[:train_count]))
        smm_values, valid_nodes = standardize_smm(raw_smm[:train_count], raw_smm)
        smm_nodes = [node for node, valid in zip(smm_nodes, valid_nodes) if valid]
        smm_values = smm_values[:, valid_nodes]
    summaries, coefficients, forecasts = [], [], []
    for explanatory_index, dependent_index in ((0, 1), (1, 0)):
        explanatory = str(edge["graph_source"] if explanatory_index == 0 else edge["graph_target"])
        dependent = str(edge["graph_source"] if dependent_index == 0 else edge["graph_target"])
        variants = [("baseline", None)]
        if smm_nodes and lag > 0:
            variants.extend((("baseline_matched", smm_values * np.nan), ("smm_enhanced", smm_values)))
        for variant, variant_smm in variants:
            actual_smm = smm_values if variant == "baseline_matched" else variant_smm
            train_y, train_x, terms = design_for_direction(values, dependent_index, explanatory_index, lag, arguments.max_lag, train_positions, actual_smm)
            test_y, test_x, _ = design_for_direction(values, dependent_index, explanatory_index, lag, arguments.max_lag, test_positions, actual_smm)
            if variant == "baseline_matched":
                train_mask = np.isfinite(smm_values[train_positions - 1]).all(axis=1)
                test_mask = np.isfinite(smm_values[test_positions - 1]).all(axis=1)
                train_x = train_x[:, : 1 + 2 * lag]
                test_x = test_x[:, : 1 + 2 * lag]
                terms = terms[: 1 + 2 * lag]
                train_y, train_x = train_y[train_mask], train_x[train_mask]
                test_x[~test_mask] = np.nan
            try:
                fit = fit_equation(train_y, train_x)
            except np.linalg.LinAlgError:
                continue
            predicted = test_x @ fit["coefficients"]
            test_index = combined.index[test_positions]
            oos, forecast_rows = oos_records(
                edge, explanatory, dependent, variant, test_index, test_y, predicted,
                float(np.mean(fit["targets"])), audit if variant == "smm_enhanced" else None,
                forecast_date, combined.index[test_positions - 1],
            )
            dates = {"forecast_date": forecast_date.date().isoformat(),
                     "train_start_date": min(train_dates).date().isoformat(), "train_end_date": max(train_dates).date().isoformat(),
                     "_overlap_observations": train_count}
            summary, coefficient_rows = model_records(edge, explanatory, dependent, variant, "dynamic", lag, fit, terms, smm_nodes, dates, oos)
            summaries.append(summary)
            coefficients.extend(coefficient_rows)
            forecasts.extend(forecast_rows)
    return summaries, coefficients, forecasts


def atomic_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8", newline="") as temporary:
        temporary_path = Path(temporary.name)
        frame.reindex(columns=columns).to_csv(temporary, index=False)
    temporary_path.replace(path)


def smm_ablation_summary(summary: pd.DataFrame) -> pd.DataFrame:
    smm_rows = summary.loc[summary["smm_nodes"].fillna("").ne("")].copy()
    keys = ["scope", "graph_edge_number", "directed_edge", "forecast_date", "smm_nodes", "bic_selected_lag"]
    matched = smm_rows.loc[smm_rows["model_variant"].eq("baseline_matched")]
    enhanced = smm_rows.loc[smm_rows["model_variant"].eq("smm_enhanced")]
    metrics = ["oos_r_squared", "oos_rmse", "oos_mae", "forecast_direction_hit_rate"]
    if not matched.empty and not enhanced.empty:
        paired = matched[keys + metrics].merge(
            enhanced[keys + metrics], on=keys, how="inner", suffixes=("_baseline", "_smm"),
            validate="one_to_one",
        )
    else:
        paired = pd.DataFrame(columns=keys)
    rows = []
    for item in paired.to_dict("records"):
        rows.append({
            **{key: item[key] for key in keys},
            "baseline_oos_r_squared": item["oos_r_squared_baseline"],
            "smm_oos_r_squared": item["oos_r_squared_smm"],
            "delta_oos_r_squared": item["oos_r_squared_smm"] - item["oos_r_squared_baseline"],
            "baseline_oos_rmse": item["oos_rmse_baseline"], "smm_oos_rmse": item["oos_rmse_smm"],
            "delta_oos_rmse": item["oos_rmse_smm"] - item["oos_rmse_baseline"],
            "baseline_oos_mae": item["oos_mae_baseline"], "smm_oos_mae": item["oos_mae_smm"],
            "delta_oos_mae": item["oos_mae_smm"] - item["oos_mae_baseline"],
            "baseline_direction_hit_rate": item["forecast_direction_hit_rate_baseline"],
            "smm_direction_hit_rate": item["forecast_direction_hit_rate_smm"],
            "delta_direction_hit_rate": item["forecast_direction_hit_rate_smm"] - item["forecast_direction_hit_rate_baseline"],
            "ablation_status": "paired",
        })
    zero_lag = smm_rows.loc[
        smm_rows["model_variant"].eq("baseline") & smm_rows["model_status"].eq("zero_lag_no_smm_terms")
    ]
    for item in zero_lag.to_dict("records"):
        rows.append({**{key: item[key] for key in keys}, "ablation_status": "zero_lag_no_smm_terms"})
    return pd.DataFrame(rows).reindex(columns=ABLATION_COLUMNS)


def dynamic_experiment(edges: list[dict[str, object]], returns: dict[str, pd.DataFrame], events: dict[str, pd.DataFrame], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aligned_edges = {int(edge["graph_edge_number"]): align_pair(str(edge["graph_source"]), str(edge["graph_target"]), returns) for edge in edges}
    available_dates = sorted(set.intersection(*(set(frame["trading_date"].unique()) for frame in aligned_edges.values())))
    dates = [pd.Timestamp(value) for value in available_dates if sum(pd.Timestamp(item) < pd.Timestamp(value) for item in available_dates) >= arguments.window_days]
    latest = max(dates)
    start = pd.Timestamp(arguments.start_date) if arguments.start_date else latest - pd.DateOffset(years=arguments.years) + pd.Timedelta(days=1)
    end = pd.Timestamp(arguments.end_date) if arguments.end_date else latest
    dates = [date for date in dates if start <= date <= end]
    if arguments.last_forecast_days:
        dates = dates[-arguments.last_forecast_days :]
    checkpoint_dir = output_dir / "daily"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for date_number, forecast_date in enumerate(dates, start=1):
        paths = {name: checkpoint_dir / f"{forecast_date.date().isoformat()}_{name}.csv" for name in ("summary", "coefficients", "forecasts")}
        if all(path.exists() for path in paths.values()) and not arguments.no_resume:
            continue
        summaries, coefficients, forecasts = [], [], []
        for edge in edges:
            edge_rows = dynamic_date_records(edge, aligned_edges[int(edge["graph_edge_number"])], forecast_date, events)
            summaries.extend(edge_rows[0]); coefficients.extend(edge_rows[1]); forecasts.extend(edge_rows[2])
        atomic_csv(pd.DataFrame(summaries), paths["summary"], SUMMARY_COLUMNS)
        atomic_csv(pd.DataFrame(coefficients), paths["coefficients"], COEFFICIENT_COLUMNS)
        atomic_csv(pd.DataFrame(forecasts), paths["forecasts"], FORECAST_COLUMNS)
        if date_number % arguments.progress_every == 0 or date_number == len(dates):
            print(f"Completed {date_number}/{len(dates)} forecast dates", flush=True)
    summaries = pd.concat((pd.read_csv(checkpoint_dir / f"{date.date().isoformat()}_summary.csv") for date in dates), ignore_index=True)
    coefficients = pd.concat((pd.read_csv(checkpoint_dir / f"{date.date().isoformat()}_coefficients.csv") for date in dates), ignore_index=True)
    forecasts = pd.concat((pd.read_csv(checkpoint_dir / f"{date.date().isoformat()}_forecasts.csv") for date in dates), ignore_index=True)
    return summaries, coefficients, forecasts


def validate_outputs(edges: list[dict[str, object]], static: pd.DataFrame | None, dynamic: pd.DataFrame | None, forecasts: pd.DataFrame | None) -> dict[str, object]:
    report = {"edge_count": len(edges), "directed_edge_count": 2 * len(edges), "checks": {}}
    if static is not None:
        full = static.loc[static["model_variant"].eq("baseline_full")]
        report["checks"]["static_baseline_complete"] = len(full) == 2 * len(edges)
    if dynamic is not None:
        report["checks"]["training_precedes_forecast"] = bool((pd.to_datetime(dynamic["train_end_date"]) < pd.to_datetime(dynamic["forecast_date"])).all())
    if forecasts is not None:
        comparable = forecasts["forecast_direction_status"].isin(["same", "opposite"])
        report["checks"]["direction_field_complete"] = bool(forecasts.loc[comparable, "forecast_direction_matches_actual"].notna().all())
        smm_rows = forecasts["smm_latest_available_at"].notna() & forecasts["smm_latest_available_at"].ne("")
        report["checks"]["smm_available_before_prediction"] = bool((pd.to_datetime(forecasts.loc[smm_rows, "smm_latest_available_at"]) <= pd.to_datetime(forecasts.loc[smm_rows, "datetime"])).all())
    report["passed"] = all(report["checks"].values())
    return report


def main() -> None:
    global arguments
    arguments = parse_args()
    if arguments.window_days < 2 or arguments.max_lag < 1:
        raise ValueError("--window-days must be >= 2 and --max-lag must be >= 1")
    edges = select_edges(graph_edges(), arguments.edge_numbers)
    symbols = {str(edge[key]) for edge in edges for key in ("graph_source", "graph_target")}
    returns = load_returns(symbols)
    events, smm_registry = load_smm_events()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    edge_registry = directed_edge_registry(edges)
    edge_registry.to_csv(arguments.output_dir / "directed_edge_registry.csv", index=False, encoding="utf-8-sig")
    smm_registry.to_csv(arguments.output_dir / "smm_feature_registry.csv", index=False, encoding="utf-8-sig")
    static_summary = dynamic_summary = forecasts = None
    if arguments.mode in ("static", "both"):
        static_summary, static_coefficients = static_experiment(edges, returns, events)
        atomic_csv(static_summary, arguments.output_dir / "static_model_summary.csv", SUMMARY_COLUMNS)
        atomic_csv(static_coefficients, arguments.output_dir / "static_coefficients.csv", COEFFICIENT_COLUMNS)
    if arguments.mode in ("dynamic", "both"):
        dynamic_summary, dynamic_coefficients, forecasts = dynamic_experiment(edges, returns, events, arguments.output_dir)
        atomic_csv(dynamic_summary, arguments.output_dir / "dynamic_model_summary.csv", SUMMARY_COLUMNS)
        atomic_csv(dynamic_coefficients, arguments.output_dir / "dynamic_coefficients.csv", COEFFICIENT_COLUMNS)
        atomic_csv(forecasts, arguments.output_dir / "dynamic_forecasts.csv", FORECAST_COLUMNS)
        atomic_csv(smm_ablation_summary(dynamic_summary), arguments.output_dir / "smm_ablation_summary.csv", ABLATION_COLUMNS)
    validation = validate_outputs(edges, static_summary, dynamic_summary, forecasts)
    (arguments.output_dir / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "completed_at": datetime.now().astimezone().isoformat(), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "mode": arguments.mode, "edge_numbers": [edge["graph_edge_number"] for edge in edges],
        "window_days": arguments.window_days, "maximum_candidate_lag": arguments.max_lag,
        "smm_lag": 1, "gnn_edge_weight_defined": False,
    }
    (arguments.output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if not validation["passed"]:
        raise AssertionError(f"Output validation failed: {validation}")
    print(f"Regression experiment completed in {arguments.output_dir}")


arguments: argparse.Namespace


if __name__ == "__main__":
    main()