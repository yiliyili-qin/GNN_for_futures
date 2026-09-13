"""Estimate static and 20+1-day regressions for every futures pair.

The statistical implementation is deliberately shared with
``run_graph_edge_regressions.py`` so the all-node and economic-prior runs use
the same alignment, BIC lag selection, OLS/Wald tests, causal rolling windows,
out-of-sample forecasts, checkpoints, and output schema.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import run_graph_edge_regressions as regression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_FILE = PROJECT_ROOT / "picture" / "futures_target_return_acf_corrected.csv"
STATIC_METADATA = (
    Path(__file__).resolve().parent
    / "results"
    / "no_smm_pairwise_granger"
    / "run_metadata.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "all_node_edge_regressions"
EXPECTED_UNIVERSE_SIZE = 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "dynamic", "both"), default="both")
    parser.add_argument(
        "--symbols",
        help="Comma-separated subset for a smoke test; default is the exact 44-node universe.",
    )
    parser.add_argument(
        "--pair-numbers",
        help="Comma-separated all-node pair numbers; default is every unordered pair.",
    )
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


def load_universe(symbol_argument: str | None) -> list[str]:
    if UNIVERSE_FILE.exists():
        universe = (
            pd.read_csv(UNIVERSE_FILE, usecols=["variety"])["variety"]
            .astype(str)
            .tolist()
        )
    else:
        universe = json.loads(STATIC_METADATA.read_text(encoding="utf-8"))["symbols"]
    if len(universe) != EXPECTED_UNIVERSE_SIZE or len(set(universe)) != len(universe):
        raise ValueError("The authoritative futures universe must contain 44 unique symbols.")
    if symbol_argument is None:
        return universe
    requested = [symbol.strip() for symbol in symbol_argument.split(",") if symbol.strip()]
    unknown = sorted(set(requested).difference(universe))
    if unknown or len(requested) < 2 or len(set(requested)) != len(requested):
        raise ValueError(f"Invalid symbol subset; unknown={unknown}")
    return requested


def all_node_pairs(symbols: list[str]) -> list[dict[str, object]]:
    return [
        {
            "graph_edge_number": pair_number,
            "graph_source": source,
            "graph_target": target,
            "relation_type": "all_node_candidate",
            "smm_path": "",
            "smm_nodes": [],
        }
        for pair_number, (source, target) in enumerate(combinations(symbols, 2), start=1)
    ]


def main() -> None:
    arguments = parse_args()
    if arguments.window_days < 2 or arguments.max_lag < 1:
        raise ValueError("--window-days must be >= 2 and --max-lag must be >= 1")

    symbols = load_universe(arguments.symbols)
    pairs = regression.select_edges(all_node_pairs(symbols), arguments.pair_numbers)
    selected_symbols = {
        str(pair[key]) for pair in pairs for key in ("graph_source", "graph_target")
    }
    returns = regression.load_returns(selected_symbols)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    regression.arguments = arguments
    registry = regression.directed_edge_registry(pairs)
    registry.to_csv(
        arguments.output_dir / "directed_edge_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )

    static_summary = dynamic_summary = forecasts = None
    empty_events: dict[str, pd.DataFrame] = {}
    if arguments.mode in ("static", "both"):
        static_summary, static_coefficients = regression.static_experiment(
            pairs, returns, empty_events
        )
        regression.atomic_csv(
            static_summary,
            arguments.output_dir / "static_model_summary.csv",
            regression.SUMMARY_COLUMNS,
        )
        regression.atomic_csv(
            static_coefficients,
            arguments.output_dir / "static_coefficients.csv",
            regression.COEFFICIENT_COLUMNS,
        )
    if arguments.mode in ("dynamic", "both"):
        dynamic_summary, dynamic_coefficients, forecasts = regression.dynamic_experiment(
            pairs, returns, empty_events, arguments.output_dir
        )
        regression.atomic_csv(
            dynamic_summary,
            arguments.output_dir / "dynamic_model_summary.csv",
            regression.SUMMARY_COLUMNS,
        )
        regression.atomic_csv(
            dynamic_coefficients,
            arguments.output_dir / "dynamic_coefficients.csv",
            regression.COEFFICIENT_COLUMNS,
        )
        regression.atomic_csv(
            forecasts,
            arguments.output_dir / "dynamic_forecasts.csv",
            regression.FORECAST_COLUMNS,
        )

    validation = regression.validate_outputs(
        pairs, static_summary, dynamic_summary, forecasts
    )
    (arguments.output_dir / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "mode": arguments.mode,
        "symbols": symbols,
        "symbol_count": len(symbols),
        "unordered_pair_count": len(pairs),
        "directed_edge_count": 2 * len(pairs),
        "pair_numbers": [pair["graph_edge_number"] for pair in pairs],
        "window_days": arguments.window_days,
        "maximum_candidate_lag": arguments.max_lag,
        "economic_prior_only": False,
        "gnn_edge_weight_defined": False,
    }
    (arguments.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not validation["passed"]:
        raise AssertionError(f"Output validation failed: {validation}")
    print(f"All-node regression experiment completed in {arguments.output_dir}")


if __name__ == "__main__":
    main()