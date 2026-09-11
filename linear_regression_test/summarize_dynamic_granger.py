"""Summarize an annual dynamic Granger panel and rank futures pairs by stars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(__file__).parent / "results" / "dynamic_pairwise_granger" / "dynamic_pairwise_granger_results.csv"
STAR_LEVELS = {"": 0, "*": 1, "**": 2, "***": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.input, keep_default_na=False)
    results["star_level"] = results["significance_stars"].map(STAR_LEVELS)
    if results["star_level"].isna().any():
        raise ValueError("Unknown significance star value")
    results["pair"] = results.apply(
        lambda row: "__".join(sorted((row["explanatory_futures"], row["dependent_futures"]))), axis=1
    )

    date_count = results["evaluation_date"].nunique()
    pair_sizes = results.groupby("pair").size()
    if len(pair_sizes) != 946 or not pair_sizes.eq(2 * date_count).all():
        raise ValueError(f"Incomplete panel: pairs={len(pair_sizes)}, sizes={pair_sizes.value_counts().to_dict()}")

    directions = results.groupby(
        ["pair", "explanatory_futures", "dependent_futures"], as_index=False
    ).agg(
        star_points=("star_level", "sum"),
        significant_days=("star_level", lambda values: int((values > 0).sum())),
        three_star_count=("star_level", lambda values: int((values == 3).sum())),
        two_star_count=("star_level", lambda values: int((values == 2).sum())),
        one_star_count=("star_level", lambda values: int((values == 1).sum())),
    )
    pairs = results.groupby("pair", as_index=False).agg(
        star_points=("star_level", "sum"),
        significant_direction_days=("star_level", lambda values: int((values > 0).sum())),
        three_star_count=("star_level", lambda values: int((values == 3).sum())),
        two_star_count=("star_level", lambda values: int((values == 2).sum())),
        one_star_count=("star_level", lambda values: int((values == 1).sum())),
    )
    days = results.loc[results["star_level"] > 0].groupby("pair")["evaluation_date"].nunique()
    pairs["significant_calendar_days"] = pairs["pair"].map(days).fillna(0).astype(int)
    pairs["direction_breakdown"] = pairs["pair"].map(
        lambda pair: "; ".join(
            f"{row.explanatory_futures}->{row.dependent_futures}: "
            f"{row.star_points} points, {row.significant_days} days "
            f"(***={row.three_star_count}, **={row.two_star_count}, *={row.one_star_count})"
            for row in directions.loc[directions["pair"] == pair]
            .sort_values("star_points", ascending=False)
            .itertuples()
        )
    )
    top = pairs.sort_values(
        ["star_points", "significant_direction_days", "three_star_count"], ascending=False
    ).head(args.top)

    output = args.output or args.input.with_name(f"dynamic_granger_top_{args.top}_pairs.csv")
    top.to_csv(output, index=False, encoding="utf-8-sig")
    overall = {
        "rows": len(results),
        "dates": date_count,
        "unordered_pairs": len(pair_sizes),
        "directed_relations": results.groupby(["explanatory_futures", "dependent_futures"]).ngroups,
        "test_status": results["test_status"].value_counts().to_dict(),
        "selected_lag": results["bic_selected_lag"].value_counts(dropna=False).sort_index().to_dict(),
        "stars": results["significance_stars"].replace("", "none").value_counts().to_dict(),
        "fdr_q_below_0_10": int((pd.to_numeric(results["fdr_bh_q_value"], errors="coerce") < 0.10).sum()),
        "fdr_q_below_0_05": int((pd.to_numeric(results["fdr_bh_q_value"], errors="coerce") < 0.05).sum()),
    }
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(top.to_string(index=False))
    print(f"Wrote {len(top)} rows to {output}")


if __name__ == "__main__":
    main()