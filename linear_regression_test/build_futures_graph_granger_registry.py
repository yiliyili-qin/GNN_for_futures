"""Register dynamic Granger statistics on the simplified economic graph."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from build_granger_economic_graph import FUTURES_NODES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = Path(__file__).parent / "results" / "dynamic_pairwise_granger_2y" / "dynamic_pairwise_granger_results.csv"
DEFAULT_GRAPH = PROJECT_ROOT / "data" / "graph_data" / "futures_graph_smm_simplified.html"
STAR_LEVELS = {"": 0, "*": 1, "**": 2, "***": 3}
EDGE_BLOCK = re.compile(r"const rawEdges = `(?P<edges>.*?)`\.trim\(\);", re.DOTALL)
REGISTRY_BLOCK = re.compile(
    r"const grangerRegistry = /\* GRANGER_REGISTRY_START \*/.*?/\* GRANGER_REGISTRY_END \*/;",
    re.DOTALL,
)
EDGE_EVIDENCE = {
    frozenset(("JM", "I")): {
        "evidence_source_title": "Raw materials",
        "evidence_source_organization": "World Steel Association (worldsteel)",
        "evidence_source_url": "https://worldsteel.org/about-steel/raw-materials/",
        "economic_rationale": "焦煤与铁矿石同为高炉炼铁的关键原料，受钢铁生产需求与原料成本周期共同驱动。",
    },
    frozenset(("AG", "SC")): {
        "evidence_source_title": "Connectedness among crude oil prices, stock index and precious metal prices",
        "evidence_source_organization": "Resources Policy (Husain et al., 2019; RePEc record)",
        "evidence_source_url": "https://ideas.repec.org/a/eee/jrpoli/v62y2019icp57-65.html",
        "economic_rationale": "原油与白银可经通胀预期、地缘风险、美元及跨商品风险偏好形成市场联动，不表示直接物料投入。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-inject", action="store_true", help="Write CSV without embedding it in the graph HTML.")
    return parser.parse_args()


def longest_run(indices: list[int]) -> int:
    longest = current = 0
    previous = None
    for index in indices:
        current = current + 1 if previous is not None and index == previous + 1 else 1
        longest = max(longest, current)
        previous = index
    return longest


def maximum_window_sum(values: list[int], width: int) -> int:
    width = min(width, len(values))
    if width == 0:
        return 0
    current = maximum = sum(values[:width])
    for index in range(width, len(values)):
        current += values[index] - values[index - width]
        maximum = max(maximum, current)
    return maximum


def graph_edges(graph_text: str) -> list[dict[str, object]]:
    match = EDGE_BLOCK.search(graph_text)
    if not match:
        raise ValueError("Could not find rawEdges in graph HTML")
    edges = []
    for edge_number, line in enumerate(match.group("edges").strip().splitlines(), start=1):
        source, target, relation_type, smm_path = line.split("|")
        evidence = EDGE_EVIDENCE.get(frozenset((source, target)), {})
        edges.append({
            "graph_edge_number": edge_number,
            "graph_source": source,
            "graph_target": target,
            "relation_type": relation_type,
            "smm_path": smm_path,
            "evidence_source_title": evidence.get("evidence_source_title", ""),
            "evidence_source_organization": evidence.get("evidence_source_organization", ""),
            "evidence_source_url": evidence.get("evidence_source_url", ""),
            "economic_rationale": evidence.get("economic_rationale", ""),
        })
    return edges


def futures_names() -> dict[str, str]:
    return {line.split("|", 2)[0]: line.split("|", 2)[1] for line in FUTURES_NODES.splitlines()}


def summarize_direction(frame: pd.DataFrame, all_dates: list[str]) -> dict[str, object]:
    levels = frame.set_index("evaluation_date").reindex(all_dates)["star_level"].fillna(0).astype(int).tolist()
    significant_indices = [index for index, level in enumerate(levels) if level > 0]
    significant_dates = [all_dates[index] for index in significant_indices]
    total_points = int(sum(levels))
    max_10 = maximum_window_sum(levels, 10)
    max_15 = maximum_window_sum(levels, 15)
    return {
        "total_star_points": total_points,
        "one_star_days": levels.count(1),
        "two_star_days": levels.count(2),
        "three_star_days": levels.count(3),
        "significant_days": len(significant_indices),
        "significant_day_ratio": round(len(significant_indices) / len(all_dates), 6),
        "first_significant_date": significant_dates[0] if significant_dates else "",
        "last_significant_date": significant_dates[-1] if significant_dates else "",
        "significant_span_trading_days": significant_indices[-1] - significant_indices[0] + 1 if significant_indices else 0,
        "longest_consecutive_significant_days": longest_run(significant_indices),
        "active_months": len({date[:7] for date in significant_dates}),
        "max_10d_star_points": max_10,
        "max_15d_star_points": max_15,
        "max_10d_star_concentration": round(max_10 / total_points, 6) if total_points else 0.0,
        "max_15d_star_concentration": round(max_15 / total_points, 6) if total_points else 0.0,
    }


def direction_row(
    explanatory: str,
    dependent: str,
    directions: dict[tuple[str, str], pd.DataFrame],
    all_dates: list[str],
    names: dict[str, str],
) -> dict[str, object]:
    direction = directions[(explanatory, dependent)]
    if len(direction) != len(all_dates) or direction["evaluation_date"].nunique() != len(all_dates):
        raise ValueError(f"Incomplete direction {explanatory}->{dependent}: {len(direction)} rows")
    return {
        "explanatory_futures": explanatory,
        "explanatory_futures_chinese": names[explanatory],
        "dependent_futures": dependent,
        "dependent_futures_chinese": names[dependent],
        "directed_edge": f"{explanatory}->{dependent}",
        "evaluation_days": len(all_dates),
        **summarize_direction(direction, all_dates),
    }


def main() -> None:
    args = parse_args()
    graph_text = args.graph.read_text(encoding="utf-8")
    edges = graph_edges(graph_text)
    if len(edges) != 57:
        raise ValueError(f"Expected 57 graph edges, found {len(edges)}")
    results = pd.read_csv(args.results, keep_default_na=False)
    required = {"evaluation_date", "explanatory_futures", "dependent_futures", "significance_stars"}
    if missing := required.difference(results.columns):
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    results["evaluation_date"] = results["evaluation_date"].astype(str)
    results["star_level"] = results["significance_stars"].map(STAR_LEVELS)
    if results["star_level"].isna().any():
        raise ValueError("Unknown significance star value")
    all_dates = sorted(results["evaluation_date"].unique())
    symbols = set(results["explanatory_futures"]).union(results["dependent_futures"])
    names = futures_names()
    if unknown := symbols.difference(names):
        raise ValueError(f"Missing Chinese futures names: {sorted(unknown)}")
    directions = {
        direction: frame
        for direction, frame in results.groupby(["explanatory_futures", "dependent_futures"], sort=False)
    }

    rows = []
    for edge in edges:
        for explanatory, dependent in (
            (edge["graph_source"], edge["graph_target"]),
            (edge["graph_target"], edge["graph_source"]),
        ):
            if explanatory not in symbols or dependent not in symbols:
                raise ValueError(f"Graph direction is outside the experiment: {explanatory}->{dependent}")
            rows.append({
                **edge,
                "graph_source_chinese": names[edge["graph_source"]],
                "graph_target_chinese": names[edge["graph_target"]],
                **direction_row(explanatory, dependent, directions, all_dates, names),
            })

    registry = pd.DataFrame(rows)
    if len(registry) != 114:
        raise AssertionError(f"Expected 114 directed registry rows, got {len(registry)}")
    output = args.output or args.results.with_name("futures_graph_smm_granger_registry.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output, index=False, encoding="utf-8-sig")
    all_directions = pd.DataFrame([
        direction_row(explanatory, dependent, directions, all_dates, names)
        for explanatory in sorted(symbols)
        for dependent in sorted(symbols)
        if explanatory != dependent
    ])
    expected_directions = len(symbols) * (len(symbols) - 1)
    if len(all_directions) != expected_directions:
        raise AssertionError(f"Expected {expected_directions} directional rows, got {len(all_directions)}")
    all_output = args.results.with_name("dynamic_pairwise_granger_all_directions_registry.csv")
    all_directions.to_csv(all_output, index=False, encoding="utf-8-sig")
    if not args.no_inject:
        replacement = (
            "const grangerRegistry = /* GRANGER_REGISTRY_START */"
            + json.dumps(registry.to_dict(orient="records"), ensure_ascii=False, separators=(",", ":"))
            + "/* GRANGER_REGISTRY_END */;"
        )
        updated, count = REGISTRY_BLOCK.subn(lambda _: replacement, graph_text)
        if count != 1:
            raise ValueError("Graph HTML must contain exactly one Granger registry block")
        args.graph.write_text(updated, encoding="utf-8")
    print(f"Wrote {len(registry)} directed graph-edge rows to {output}")
    print(f"Wrote {len(all_directions)} all-pair directed rows to {all_output}")
    if not args.no_inject:
        print(f"Embedded registry in {args.graph}")


if __name__ == "__main__":
    main()