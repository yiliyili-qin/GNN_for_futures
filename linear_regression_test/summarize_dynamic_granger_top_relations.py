"""Rank dynamic Granger directions and compare them with the economic graph."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = Path(__file__).parent / "results" / "dynamic_pairwise_granger_2y" / "dynamic_pairwise_granger_all_directions_registry.csv"
DEFAULT_GRAPH = PROJECT_ROOT / "data" / "graph_data" / "futures_graph_smm_simplified.html"
EDGE_BLOCK = re.compile(r"const rawEdges = `(?P<edges>.*?)`\.trim\(\);", re.DOTALL)
RELATION_LABELS = {
    "direct_input": "直接投入 / 加工",
    "indirect_chain": "跨级产业链",
    "peer_product": "同级商品关系",
    "grade_or_origin": "品级 / 产地",
    "market_linkage": "市场联动",
    "smm_mediated": "SMM 节点收缩边",
}
CANDIDATE_MIN_STAR_POINTS = 40
CANDIDATE_MIN_ACTIVE_MONTHS = 2
CANDIDATE_MIN_SPAN_DAYS = 180
CANDIDATE_MAX_15D_CONCENTRATION = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--statistics-output", type=Path)
    parser.add_argument("--candidates-output", type=Path)
    return parser.parse_args()


def economic_edges(graph: Path) -> dict[frozenset[str], list[str]]:
    match = EDGE_BLOCK.search(graph.read_text(encoding="utf-8"))
    if not match:
        raise ValueError("Could not find rawEdges in graph HTML")
    edges: dict[frozenset[str], list[str]] = {}
    for line in match.group("edges").strip().splitlines():
        source, target, relation_type, _ = line.split("|")
        edges.setdefault(frozenset((source, target)), []).append(RELATION_LABELS[relation_type])
    return edges


def weighted_statistics(frame: pd.DataFrame, population: str) -> dict[str, object]:
    weights = frame["total_star_points"]
    total_weight = weights.sum()
    if total_weight == 0:
        raise ValueError(f"No star points available for {population}")
    return {
        "population": population,
        "relations": len(frame),
        "total_star_points_weight": total_weight,
        "weighted_avg_10d_star_concentration": (frame["max_10d_star_concentration"] * weights).sum() / total_weight,
        "weighted_avg_15d_star_concentration": (frame["max_15d_star_concentration"] * weights).sum() / total_weight,
        "weighted_avg_active_months": (frame["active_months"] * weights).sum() / total_weight,
        "weighted_avg_significant_span_trading_days": (
            frame["significant_span_trading_days"] * weights
        ).sum() / total_weight,
    }


def main() -> None:
    args = parse_args()
    if args.top < 1:
        raise ValueError("--top must be at least 1")
    registry = pd.read_csv(args.registry, keep_default_na=False)
    required = {
        "explanatory_futures", "explanatory_futures_chinese",
        "dependent_futures", "dependent_futures_chinese", "directed_edge",
        "total_star_points", "one_star_days", "two_star_days", "three_star_days",
        "significant_days", "significant_day_ratio", "first_significant_date",
        "last_significant_date", "significant_span_trading_days",
        "longest_consecutive_significant_days", "active_months",
        "max_10d_star_concentration", "max_15d_star_concentration",
    }
    if missing := required.difference(registry.columns):
        raise ValueError(f"Missing registry columns: {sorted(missing)}")
    graph_edges = economic_edges(args.graph)
    registry["original_chinese_futures_relation"] = (
        registry["explanatory_futures_chinese"] + "（" + registry["explanatory_futures"] + "）→"
        + registry["dependent_futures_chinese"] + "（" + registry["dependent_futures"] + "）"
    )
    memberships = registry.apply(
        lambda row: graph_edges.get(frozenset((row["explanatory_futures"], row["dependent_futures"]))), axis=1
    )
    registry["in_original_economic_graph"] = memberships.map(lambda value: "是" if value else "否")
    registry["original_economic_relation_type"] = memberships.map(lambda value: "；".join(value) if value else "")
    columns = [
        "original_chinese_futures_relation", "directed_edge", "total_star_points",
        "three_star_days", "two_star_days", "one_star_days", "significant_days",
        "significant_day_ratio", "first_significant_date", "last_significant_date",
        "significant_span_trading_days", "longest_consecutive_significant_days",
        "active_months", "max_10d_star_concentration", "max_15d_star_concentration",
        "in_original_economic_graph", "original_economic_relation_type",
    ]
    ranked = registry.sort_values(
        ["total_star_points", "three_star_days", "significant_days"], ascending=False, kind="stable"
    )
    top = ranked.head(args.top)[columns].reset_index(drop=True)
    output = args.output or args.registry.with_name(f"dynamic_granger_top_{args.top}_directions_with_economic_graph.csv")
    top.to_csv(output, index=False, encoding="utf-8-sig")
    top_100 = ranked.head(100)
    statistics = pd.DataFrame([
        weighted_statistics(top_100.loc[top_100["in_original_economic_graph"] == "是"], "星数前100：原始经济图内关系"),
        weighted_statistics(top_100.loc[top_100["in_original_economic_graph"] == "否"], "星数前100：原始经济图外关系"),
    ])
    statistics_output = args.statistics_output or args.registry.with_name(
        "dynamic_granger_top_100_graph_membership_weighted_statistics.csv"
    )
    statistics.to_csv(statistics_output, index=False, encoding="utf-8-sig")
    reverse_metrics = registry.set_index("directed_edge")
    candidates = registry.loc[
        (registry["in_original_economic_graph"] == "否")
        & registry["total_star_points"].ge(CANDIDATE_MIN_STAR_POINTS)
        & registry["active_months"].ge(CANDIDATE_MIN_ACTIVE_MONTHS)
        & registry["significant_span_trading_days"].ge(CANDIDATE_MIN_SPAN_DAYS)
        & registry["max_15d_star_concentration"].le(CANDIDATE_MAX_15D_CONCENTRATION)
    ].copy()
    candidates["reverse_directed_edge"] = candidates["dependent_futures"] + "->" + candidates["explanatory_futures"]
    for column in ["total_star_points", "significant_days", "active_months", "significant_span_trading_days"]:
        candidates[f"reverse_{column}"] = candidates["reverse_directed_edge"].map(reverse_metrics[column])
    candidate_columns = [
        "original_chinese_futures_relation", "directed_edge", "total_star_points",
        "three_star_days", "two_star_days", "one_star_days", "significant_days",
        "active_months", "significant_span_trading_days", "longest_consecutive_significant_days",
        "max_10d_star_concentration", "max_15d_star_concentration", "reverse_directed_edge",
        "reverse_total_star_points", "reverse_significant_days", "reverse_active_months",
        "reverse_significant_span_trading_days",
    ]
    candidates = candidates.sort_values(
        ["total_star_points", "significant_span_trading_days"], ascending=False, kind="stable"
    )[candidate_columns]
    candidates_output = args.candidates_output or args.registry.with_name(
        "dynamic_granger_stable_graph_external_candidates.csv"
    )
    candidates.to_csv(candidates_output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(top)} rows to {output}")
    print(f"Economic-graph members in top 100: {(top_100['in_original_economic_graph'] == '是').sum()}")
    print(f"Wrote weighted statistics to {statistics_output}")
    print(f"Wrote {len(candidates)} stable graph-external candidates to {candidates_output}")


if __name__ == "__main__":
    main()