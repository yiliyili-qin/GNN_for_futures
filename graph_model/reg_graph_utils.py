"""Data preparation for regression-weighted economic graph snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "linear_regression_test"
    / "results"
    / "graph_edge_regressions_full"
)

SUMMARY_KEY = [
    "scope",
    "graph_edge_number",
    "directed_edge",
    "model_variant",
    "forecast_date",
]
SUMMARY_COLUMNS = {
    *SUMMARY_KEY,
    "explanatory_futures",
    "dependent_futures",
    "train_end_date",
    "bic_selected_lag",
    "edge_wald_p_value",
    "model_status",
}
COEFFICIENT_COLUMNS = {
    *SUMMARY_KEY,
    "term_type",
    "term_node",
    "lag",
    "coefficient",
}


@dataclass(frozen=True)
class RegressionGraphSnapshot:
    """Framework-neutral inputs for one dated directed graph."""

    forecast_date: pd.Timestamp
    node_names: tuple[str, ...]
    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_frame: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_regression_outputs(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load persisted dynamic regression summaries and coefficients."""
    directory = Path(results_dir)
    summary = pd.read_csv(directory / "dynamic_model_summary.csv")
    coefficients = pd.read_csv(directory / "dynamic_coefficients.csv")
    return clean_regression_outputs(summary, coefficients)


def clean_regression_outputs(
    summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    *,
    scope: str = "dynamic",
    model_variant: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate, type, and select the causal regression rows used by the graph."""
    _require_columns(summary, SUMMARY_COLUMNS, "summary")
    _require_columns(coefficients, COEFFICIENT_COLUMNS, "coefficients")

    clean_summary = summary.loc[
        summary["scope"].eq(scope) & summary["model_variant"].eq(model_variant)
    ].copy()
    clean_coefficients = coefficients.loc[
        coefficients["scope"].eq(scope)
        & coefficients["model_variant"].eq(model_variant)
    ].copy()
    if clean_summary.empty:
        raise ValueError(f"No summary rows for scope={scope!r}, model_variant={model_variant!r}")

    for frame in (clean_summary, clean_coefficients):
        frame["forecast_date"] = pd.to_datetime(frame["forecast_date"], errors="raise").dt.normalize()
        frame["graph_edge_number"] = pd.to_numeric(
            frame["graph_edge_number"], errors="raise"
        ).astype("int64")
    clean_summary["train_end_date"] = pd.to_datetime(
        clean_summary["train_end_date"], errors="coerce"
    ).dt.normalize()
    clean_summary["bic_selected_lag"] = pd.to_numeric(
        clean_summary["bic_selected_lag"], errors="coerce"
    ).fillna(0).astype("int64")
    clean_summary["edge_wald_p_value"] = pd.to_numeric(
        clean_summary["edge_wald_p_value"], errors="coerce"
    )
    clean_coefficients["lag"] = pd.to_numeric(
        clean_coefficients["lag"], errors="coerce"
    )
    clean_coefficients["coefficient"] = pd.to_numeric(
        clean_coefficients["coefficient"], errors="coerce"
    )

    if clean_summary.duplicated(SUMMARY_KEY).any():
        raise ValueError("Summary contains duplicate directed-edge/date rows")
    noncausal = clean_summary["train_end_date"].notna() & (
        clean_summary["train_end_date"] >= clean_summary["forecast_date"]
    )
    if noncausal.any():
        raise ValueError("Found regression windows whose training end is not before forecast date")

    return clean_summary, clean_coefficients


def build_regression_edge_features(
    summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    *,
    robust_quantile: float = 0.75,
    epsilon: float = 1e-8,
    z_max: float = 2.0,
) -> pd.DataFrame:
    """Build the economic-prior and signed regression channels for every date."""
    if not 0.0 < robust_quantile <= 1.0:
        raise ValueError("robust_quantile must be in (0, 1]")
    if epsilon <= 0.0 or z_max <= 0.0:
        raise ValueError("epsilon and z_max must be positive")

    clean_summary, clean_coefficients = clean_regression_outputs(summary, coefficients)
    cross_lags = clean_coefficients.loc[
        clean_coefficients["term_type"].eq("cross_futures")
        & clean_coefficients["term_node"].eq("explanatory")
        & clean_coefficients["lag"].gt(0)
        & np.isfinite(clean_coefficients["coefficient"])
    ]
    coefficient_sums = (
        cross_lags.groupby(SUMMARY_KEY, as_index=False, dropna=False)["coefficient"]
        .sum()
        .rename(columns={"coefficient": "coefficient_sum"})
    )

    edge_features = clean_summary.merge(
        coefficient_sums,
        on=SUMMARY_KEY,
        how="left",
        validate="one_to_one",
    )
    edge_features["coefficient_sum"] = edge_features["coefficient_sum"].fillna(0.0)
    edge_features["robust_scale"] = edge_features.groupby("forecast_date")[
        "coefficient_sum"
    ].transform(lambda values: values.abs().quantile(robust_quantile))
    edge_features["scaled_coefficient"] = np.clip(
        edge_features["coefficient_sum"] / (edge_features["robust_scale"] + epsilon),
        -z_max,
        z_max,
    )

    fitted = edge_features["model_status"].eq("fitted")
    positive_lag = edge_features["bic_selected_lag"].gt(0)
    valid_p_value = edge_features["edge_wald_p_value"].between(0.0, 1.0)
    edge_features["wald_confidence"] = np.where(
        valid_p_value,
        1.0 - edge_features["edge_wald_p_value"],
        0.0,
    )
    edge_features["prior_weight"] = 1.0
    edge_features["regression_weight"] = np.where(
        fitted & positive_lag,
        edge_features["scaled_coefficient"] * edge_features["wald_confidence"],
        0.0,
    )
    return edge_features.sort_values(
        ["forecast_date", "graph_edge_number", "directed_edge"]
    ).reset_index(drop=True)


def build_graph_snapshot(
    edge_features: pd.DataFrame,
    forecast_date: str | pd.Timestamp,
    *,
    node_order: Sequence[str] | None = None,
) -> RegressionGraphSnapshot:
    """Convert one date of edge features to NumPy graph arrays."""
    date = pd.Timestamp(forecast_date).normalize()
    dated_edges = edge_features.loc[edge_features["forecast_date"].eq(date)].copy()
    if dated_edges.empty:
        raise ValueError(f"No edge features found for forecast date {date.date()}")
    if dated_edges["directed_edge"].duplicated().any():
        raise ValueError(f"Duplicate directed edges found for forecast date {date.date()}")

    observed_nodes = set(dated_edges["explanatory_futures"]).union(
        dated_edges["dependent_futures"]
    )
    names = tuple(node_order) if node_order is not None else tuple(sorted(observed_nodes))
    if len(names) != len(set(names)):
        raise ValueError("node_order contains duplicate names")
    missing_nodes = observed_nodes.difference(names)
    if missing_nodes:
        raise ValueError(f"node_order is missing graph nodes: {sorted(missing_nodes)}")

    node_index = {name: position for position, name in enumerate(names)}
    edge_index = np.asarray(
        [
            dated_edges["explanatory_futures"].map(node_index).to_numpy(dtype=np.int64),
            dated_edges["dependent_futures"].map(node_index).to_numpy(dtype=np.int64),
        ],
        dtype=np.int64,
    )
    edge_attr = dated_edges[["prior_weight", "regression_weight"]].to_numpy(
        dtype=np.float32
    )
    return RegressionGraphSnapshot(date, names, edge_index, edge_attr, dated_edges)