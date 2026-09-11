"""Compare active futures economic-prior edges with pairwise Granger tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIOR_EDGE_REVIEW_PATH = PROJECT_ROOT / "data" / "graph_data" / "economic_graph_edge_change_review.md"
GRANGER_RESULTS_PATH = (
    PROJECT_ROOT
    / "linear_regression_test"
    / "results"
    / "no_smm_pairwise_granger"
    / "pairwise_granger_results.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "linear_regression_test" / "results" / "no_smm_pairwise_granger"
EDGE_OUTPUT_PATH = OUTPUT_DIR / "economic_prior_granger_significance.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "economic_prior_granger_significance_summary.csv"

EDGE_SECTIONS = {
    "### 2.1 直接产业链边": ("direct_input", True),
    "### 2.2 真实但跨级的产业链边": ("indirect_chain", True),
    "### 2.3 同级商品、品级和市场联动边": (None, False),
}
STAR_RANKS = {"not_available": -1, "not_significant": 0, "*": 1, "**": 2, "***": 3}


def parse_prior_edges() -> pd.DataFrame:
    """Extract the 47 active futures-only edges from the prior-edge review table."""
    rows: list[dict[str, object]] = []
    relation_type: str | None = None
    directed = False

    for line in PRIOR_EDGE_REVIEW_PATH.read_text(encoding="utf-8").splitlines():
        if line == "## 3. 加入 SMM 实体节点后的新增边":
            break
        if line in EDGE_SECTIONS:
            relation_type, directed = EDGE_SECTIONS[line]
            continue
        if not line.startswith("| ") or "`" not in line or "->" not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells[0].isdigit() or len(cells) < 4:
            continue
        edge = cells[1].strip("`")
        is_bidirectional = "<->" in edge
        source, target = edge.replace("<->", "->").split(" -> ")
        current_relation_type = relation_type or cells[3].strip("`")
        rows.append(
            {
                "prior_edge_id": int(cells[0]),
                "source": source,
                "target": target,
                "prior_relation_type": current_relation_type,
                "prior_directionality": "directed" if directed and not is_bidirectional else "undirected",
            }
        )

    edges = pd.DataFrame(rows).sort_values("prior_edge_id").reset_index(drop=True)
    if len(edges) != 47:
        raise ValueError(f"Expected 47 active futures prior edges, found {len(edges)}.")
    return edges


def test_result_lookup(results: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    """Index one directional Granger result by explanatory and dependent contract."""
    return {
        (row.explanatory_futures, row.dependent_futures): row
        for row in results.itertuples(index=False)
    }


def result_fields(result: pd.Series | None, prefix: str) -> dict[str, object]:
    """Return report values for a single tested direction."""
    if result is None:
        return {f"{prefix}_stars": "not_available", f"{prefix}_p_value": None, f"{prefix}_lag": None, f"{prefix}_status": "not_available"}
    stars = result.significance_stars
    return {
        f"{prefix}_stars": "not_significant" if pd.isna(stars) or not stars else stars,
        f"{prefix}_p_value": result.p_value,
        f"{prefix}_lag": result.bic_selected_lag,
        f"{prefix}_status": result.test_status,
    }


def main() -> None:
    prior_edges = parse_prior_edges()
    results = pd.read_csv(GRANGER_RESULTS_PATH)
    lookup = test_result_lookup(results)
    report_rows = []

    for edge in prior_edges.itertuples(index=False):
        prior_result = lookup.get((edge.source, edge.target))
        reverse_result = lookup.get((edge.target, edge.source))
        row = edge._asdict()
        row.update(result_fields(prior_result, "prior_direction"))
        row.update(result_fields(reverse_result, "reverse_direction"))
        if edge.prior_directionality == "directed":
            row["relationship_stars"] = row["prior_direction_stars"]
        else:
            row["relationship_stars"] = max(
                (row["prior_direction_stars"], row["reverse_direction_stars"]),
                key=STAR_RANKS.__getitem__,
            )
        report_rows.append(row)

    report = pd.DataFrame(report_rows)
    report.to_csv(EDGE_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    summary = (
        report.groupby(["prior_relation_type", "prior_directionality", "relationship_stars"], dropna=False)
        .size()
        .rename("edge_count")
        .reset_index()
        .sort_values(["prior_relation_type", "prior_directionality", "relationship_stars"])
    )
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Wrote {EDGE_OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_OUTPUT_PATH}")
    print(f"Active prior edges: {len(report)}")
    print("Prior-direction significance (directed edges follow the economic-prior arrow):")
    print(report["prior_direction_stars"].value_counts().to_string())
    print("Relationship significance (undirected links use the stronger tested direction):")
    print(report["relationship_stars"].value_counts().to_string())


if __name__ == "__main__":
    main()