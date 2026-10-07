"""Build economic-prior and daily regression graph data without propagation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path(
    "linear_regression_test/results/all_node_edge_regressions"
)
DEFAULT_GRAPH_PATH = Path("data/graph_data/futures_graph_smm_simplified.html")
DEFAULT_REGRESSION_AVAILABILITY_PATH = Path(
    "data/contracts/regression_graph_availability.csv"
)
EDGE_BLOCK = re.compile(r"const rawEdges = `(?P<edges>.*?)`\.trim\(\);", re.DOTALL)

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
class GraphData:
    """One graph's aligned dense matrix and nonzero sparse representation."""

    node_names: tuple[str, ...]
    adjacency: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray


@dataclass(frozen=True)
class DailyRegressionGraphData(GraphData):
    """A dated regression graph plus its complete auditable direction table."""

    forecast_date: pd.Timestamp
    train_end_date: pd.Timestamp
    edge_frame: pd.DataFrame
    available_at: pd.Timestamp | None = None


@dataclass(frozen=True)
class DailyDualGraphSnapshot:
    """Legacy date-aligned snapshot; not safe for intraday V1 as-of use."""

    forecast_date: pd.Timestamp
    node_names: tuple[str, ...]
    economic_prior: GraphData
    regression: DailyRegressionGraphData


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_node_order(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> tuple[str, ...]:
    """Load the authoritative all-node order persisted with the experiment."""
    metadata_path = Path(results_dir) / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = tuple(str(name) for name in metadata.get("symbols", ()))
    if len(names) != 44 or len(set(names)) != len(names):
        raise ValueError("Experiment metadata must contain 44 unique symbols")
    return names


def node_order_sha256(node_order: Sequence[str]) -> str:
    """Hash an authoritative ordered node list without ambiguous separators."""
    names = _validate_node_order(node_order)
    payload = json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def load_regression_availability_manifest(
    path: str | Path = DEFAULT_REGRESSION_AVAILABILITY_PATH,
    *,
    node_order: Sequence[str],
) -> pd.DataFrame:
    """Load measured regression snapshot availability and reject inferred dates."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Regression availability manifest is required but was not found: {manifest_path}"
        )
    frame = pd.read_csv(manifest_path)
    required = {
        "forecast_date",
        "train_end_date",
        "available_at",
        "node_order_sha256",
        "measurement_source",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Regression availability manifest is missing fields: {missing}")
    if frame.empty:
        raise ValueError("Regression availability manifest must not be empty")
    frame["forecast_date"] = pd.to_datetime(
        frame["forecast_date"], errors="raise"
    ).dt.normalize()
    frame["train_end_date"] = pd.to_datetime(
        frame["train_end_date"], errors="raise"
    ).dt.normalize()
    frame["available_at"] = pd.to_datetime(frame["available_at"], errors="raise")
    if frame["available_at"].dt.tz is not None:
        raise ValueError(
            "Regression available_at must use the same timezone-naive Asia/Shanghai "
            "clock as bar datetime values"
        )
    if frame[["forecast_date", "train_end_date", "available_at"]].isna().any().any():
        raise ValueError("Regression availability dates must be non-missing")
    if frame["forecast_date"].duplicated().any():
        raise ValueError("Regression availability manifest has duplicate forecast dates")
    if (frame["train_end_date"] >= frame["forecast_date"]).any():
        raise ValueError("Regression training end must precede forecast date")
    if (
        frame["available_at"].dt.normalize() <= frame["train_end_date"]
    ).any():
        raise ValueError(
            "Regression available_at must not precede the end of its training window"
        )
    expected_hash = node_order_sha256(node_order)
    if not frame["node_order_sha256"].eq(expected_hash).all():
        raise ValueError("Regression availability manifest node hash mismatch")
    source = frame["measurement_source"].astype("string")
    if source.isna().any() or source.str.strip().eq("").any():
        raise ValueError("Regression availability measurement_source must be non-empty")
    return frame.sort_values(["available_at", "forecast_date"]).reset_index(drop=True)


def select_regression_snapshot_asof(
    availability: pd.DataFrame,
    cutoff_at: str | pd.Timestamp,
) -> pd.Series:
    """Select the latest dated snapshot truly available by an intraday cutoff."""
    cutoff = pd.Timestamp(cutoff_at)
    eligible = availability.loc[
        availability["available_at"].le(cutoff)
        & availability["forecast_date"].le(cutoff.normalize())
    ]
    if eligible.empty:
        raise LookupError(f"No regression snapshot is available by {cutoff}")
    return eligible.sort_values(["forecast_date", "available_at"]).iloc[-1]


def load_regression_outputs(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load persisted all-node dynamic regression summaries and coefficients."""
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
        frame["forecast_date"] = pd.to_datetime(
            frame["forecast_date"], errors="raise"
        ).dt.normalize()
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
    clip_max: float = 2.0,
) -> pd.DataFrame:
    """Build nonnegative daily weights from absolute cross-lag coefficient sums."""
    if not 0.0 < robust_quantile <= 1.0:
        raise ValueError("robust_quantile must be in (0, 1]")
    if epsilon <= 0.0 or clip_max <= 0.0:
        raise ValueError("epsilon and clip_max must be positive")

    clean_summary, clean_coefficients = clean_regression_outputs(summary, coefficients)
    cross_lags = clean_coefficients.loc[
        clean_coefficients["term_type"].eq("cross_futures")
        & clean_coefficients["term_node"].eq("explanatory")
        & clean_coefficients["lag"].gt(0)
        & np.isfinite(clean_coefficients["coefficient"])
    ].copy()
    cross_lags["absolute_coefficient"] = cross_lags["coefficient"].abs()
    coefficient_sums = (
        cross_lags.groupby(SUMMARY_KEY, as_index=False, dropna=False)[
            "absolute_coefficient"
        ]
        .sum()
        .rename(columns={"absolute_coefficient": "absolute_coefficient_sum"})
    )
    edge_features = clean_summary.merge(
        coefficient_sums, on=SUMMARY_KEY, how="left", validate="one_to_one"
    )
    edge_features["absolute_coefficient_sum"] = edge_features[
        "absolute_coefficient_sum"
    ].fillna(0.0)

    valid_p_value = edge_features["edge_wald_p_value"].between(0.0, 1.0)
    edge_features["eligible"] = (
        edge_features["model_status"].eq("fitted")
        & edge_features["bic_selected_lag"].gt(0)
        & valid_p_value
        & edge_features["absolute_coefficient_sum"].gt(0.0)
    )
    eligible_strength = edge_features["absolute_coefficient_sum"].where(
        edge_features["eligible"]
    )
    edge_features["robust_scale"] = eligible_strength.groupby(
        edge_features["forecast_date"]
    ).transform(lambda values: values.dropna().quantile(robust_quantile))
    edge_features["robust_scale"] = edge_features["robust_scale"].fillna(0.0)
    edge_features["normalized_strength"] = np.where(
        edge_features["eligible"],
        np.clip(
            edge_features["absolute_coefficient_sum"]
            / (edge_features["robust_scale"] + epsilon),
            0.0,
            clip_max,
        )
        / clip_max,
        0.0,
    )
    edge_features["wald_confidence"] = np.where(
        valid_p_value, 1.0 - edge_features["edge_wald_p_value"], 0.0
    )
    edge_features["regression_weight"] = np.where(
        edge_features["eligible"],
        edge_features["normalized_strength"] * edge_features["wald_confidence"],
        0.0,
    )
    return edge_features.sort_values(
        ["forecast_date", "graph_edge_number", "directed_edge"]
    ).reset_index(drop=True)


def build_economic_prior_graph_from_bytes(
    node_order: Sequence[str], graph_bytes: bytes
) -> GraphData:
    """Build the fixed graph from one immutable HTML byte snapshot."""
    names = _validate_node_order(node_order)
    try:
        graph_text = graph_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Economic graph HTML must be valid UTF-8") from error
    match = EDGE_BLOCK.search(graph_text)
    if not match:
        raise ValueError("Could not find rawEdges in economic graph HTML")
    pairs: set[tuple[str, str]] = set()
    for line in match.group("edges").strip().splitlines():
        source, target, *_ = line.split("|")
        if source == target:
            raise ValueError(f"Economic graph contains self-loop {source}")
        pairs.add(tuple(sorted((source, target))))
    if len(pairs) != 57:
        raise ValueError(f"Expected 57 unique economic pairs, found {len(pairs)}")

    node_index = {name: position for position, name in enumerate(names)}
    unknown = sorted({name for pair in pairs for name in pair}.difference(node_index))
    if unknown:
        raise ValueError(f"Economic graph contains unknown nodes: {unknown}")
    adjacency = np.zeros((len(names), len(names)), dtype=np.int8)
    for source, target in pairs:
        source_index, target_index = node_index[source], node_index[target]
        adjacency[source_index, target_index] = 1
        adjacency[target_index, source_index] = 1
    return _graph_data_from_adjacency(names, adjacency)


def load_economic_prior_graph_snapshot(
    node_order: Sequence[str], graph_path: str | Path = DEFAULT_GRAPH_PATH
) -> tuple[GraphData, str]:
    """Parse and hash one immutable economic-graph file-byte snapshot."""
    path = Path(graph_path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Economic graph must be a regular nonsymlink file: {path}")
    graph_bytes = path.read_bytes()
    graph = build_economic_prior_graph_from_bytes(node_order, graph_bytes)
    return graph, hashlib.sha256(graph_bytes).hexdigest()


def build_economic_prior_graph(
    node_order: Sequence[str], graph_path: str | Path = DEFAULT_GRAPH_PATH
) -> GraphData:
    """Build the fixed graph from one immutable file-byte snapshot."""
    graph, _ = load_economic_prior_graph_snapshot(node_order, graph_path)
    return graph


def build_daily_regression_graph(
    edge_features: pd.DataFrame,
    forecast_date: str | pd.Timestamp,
    node_order: Sequence[str],
) -> DailyRegressionGraphData:
    """Build one complete daily matrix and its strictly positive sparse edges."""
    names = _validate_node_order(node_order)
    date = pd.Timestamp(forecast_date).normalize()
    if "forecast_date" not in edge_features.columns:
        raise ValueError("edge_features is missing forecast_date")
    feature_dates = pd.to_datetime(
        edge_features["forecast_date"], errors="raise"
    ).dt.normalize()
    dated = edge_features.loc[feature_dates.eq(date)].copy()
    if dated.empty:
        raise ValueError(f"No edge features found for forecast date {date.date()}")
    required_columns = {
        "directed_edge",
        "explanatory_futures",
        "dependent_futures",
        "regression_weight",
        "train_end_date",
    }
    missing_columns = sorted(required_columns.difference(dated.columns))
    if missing_columns:
        raise ValueError(f"edge_features is missing columns: {missing_columns}")
    if dated["directed_edge"].duplicated().any():
        raise ValueError(f"Duplicate directed edges found for forecast date {date.date()}")
    expected_directions = len(names) * (len(names) - 1)
    if len(dated) != expected_directions:
        raise ValueError(
            f"Expected {expected_directions} directed rows for {date.date()}, found {len(dated)}"
        )

    node_index = {name: position for position, name in enumerate(names)}
    pairs = list(
        zip(
            dated["explanatory_futures"],
            dated["dependent_futures"],
            strict=True,
        )
    )
    observed = {name for pair in pairs for name in pair}
    unknown = sorted(observed.difference(node_index))
    if unknown:
        raise ValueError(f"Regression graph contains unknown nodes: {unknown}")
    self_loops = sorted(source for source, target in pairs if source == target)
    if self_loops:
        raise ValueError(f"Regression graph contains self-loops: {self_loops}")
    expected_pairs = {
        (source, target)
        for source in names
        for target in names
        if source != target
    }
    observed_pairs = set(pairs)
    if len(observed_pairs) != len(pairs) or observed_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs.difference(observed_pairs))
        pair_counts = Counter(pairs)
        duplicate_pairs = sorted(
            pair for pair, count in pair_counts.items() if count > 1
        )
        raise ValueError(
            "Regression directed-pair closure mismatch: "
            f"missing={missing_pairs}, duplicates={duplicate_pairs}"
        )
    expected_labels = [f"{source}->{target}" for source, target in pairs]
    observed_labels = dated["directed_edge"].astype(str).tolist()
    mismatches = [
        (observed, expected)
        for observed, expected in zip(observed_labels, expected_labels, strict=True)
        if observed != expected
    ]
    if mismatches:
        raise ValueError(
            "Regression directed_edge labels disagree with explanatory/dependent pairs: "
            f"{mismatches[:5]}"
        )
    adjacency = np.zeros((len(names), len(names)), dtype=np.float32)
    source_index = dated["explanatory_futures"].map(node_index).to_numpy(dtype=np.int64)
    target_index = dated["dependent_futures"].map(node_index).to_numpy(dtype=np.int64)
    weights = dated["regression_weight"].to_numpy(dtype=np.float32)
    if not np.isfinite(weights).all() or (weights < 0.0).any() or (weights > 1.0).any():
        raise ValueError("Regression weights must be finite and in [0, 1]")
    adjacency[source_index, target_index] = weights
    graph = _graph_data_from_adjacency(names, adjacency)

    train_dates = dated["train_end_date"].dropna().unique()
    if len(train_dates) != 1:
        raise ValueError(f"Expected one training end date for {date.date()}")
    return DailyRegressionGraphData(
        graph.node_names,
        graph.adjacency,
        graph.edge_index,
        graph.edge_weight,
        date,
        pd.Timestamp(train_dates[0]),
        dated,
    )


def build_available_regression_graph(
    edge_features: pd.DataFrame,
    cutoff_at: str | pd.Timestamp,
    node_order: Sequence[str],
    *,
    availability_path: str | Path = DEFAULT_REGRESSION_AVAILABILITY_PATH,
) -> DailyRegressionGraphData:
    """Build the latest regression graph whose measured availability is causal."""
    names = _validate_node_order(node_order)
    availability = load_regression_availability_manifest(
        availability_path, node_order=names
    )
    snapshot = select_regression_snapshot_asof(availability, cutoff_at)
    graph = build_daily_regression_graph(
        edge_features, snapshot["forecast_date"], names
    )
    if graph.train_end_date != snapshot["train_end_date"]:
        raise ValueError(
            "Regression availability train_end_date disagrees with edge outputs"
        )
    return DailyRegressionGraphData(
        graph.node_names,
        graph.adjacency,
        graph.edge_index,
        graph.edge_weight,
        graph.forecast_date,
        graph.train_end_date,
        graph.edge_frame,
        pd.Timestamp(snapshot["available_at"]),
    )


def build_daily_dual_graph_snapshot(
    edge_features: pd.DataFrame,
    forecast_date: str | pd.Timestamp,
    *,
    node_order: Sequence[str] | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
) -> DailyDualGraphSnapshot:
    """Build legacy date-aligned graphs without asserting intraday availability.

    V1 intraday consumers must use :func:`build_available_regression_graph`.
    """
    names = tuple(node_order) if node_order is not None else load_node_order(results_dir)
    prior = build_economic_prior_graph(names, graph_path)
    regression = build_daily_regression_graph(edge_features, forecast_date, names)
    return DailyDualGraphSnapshot(regression.forecast_date, tuple(names), prior, regression)


def _validate_node_order(node_order: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in node_order)
    if not names or len(names) != len(set(names)):
        raise ValueError("node_order must contain unique node names")
    return names


def _graph_data_from_adjacency(
    node_names: tuple[str, ...], adjacency: np.ndarray
) -> GraphData:
    source_index, target_index = np.nonzero(adjacency)
    edge_index = np.asarray([source_index, target_index], dtype=np.int64)
    edge_weight = adjacency[source_index, target_index].astype(np.float32, copy=False)
    return GraphData(node_names, adjacency, edge_index, edge_weight)
