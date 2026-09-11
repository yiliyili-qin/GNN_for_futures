"""Match requested missing economic-graph nodes against an SMM quota catalog."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


NODE_KEYWORDS = {
    "铝土矿": ["铝土矿", "铝矾土"],
    "锰矿": ["锰矿", "氧化锰矿", "碳酸锰矿"],
    "硅石": ["硅石", "石英石"],
    "石油焦": ["石油焦"],
    "动力煤": ["动力煤", "电煤"],
    "电力": ["电价", "用电量", "发电量", "电力"],
    "铝产业供需": ["铝产量", "电解铝产量", "铝开工率", "铝库存", "铝进口", "铝出口", "铝消费"],
    "锰产业供需": ["锰产量", "锰开工率", "锰库存", "锰进口", "锰出口", "锰消费", "硅锰产量", "硅锰开工率"],
    "工业硅供需": ["工业硅产量", "工业硅开工率", "工业硅库存", "工业硅进口", "工业硅出口", "工业硅消费"],
    "兰炭": ["兰炭"],
    "天然气": ["天然气", "LNG"],
    "电石": ["电石", "碳化钙"],
    "石脑油": ["石脑油", "轻石脑油", "重石脑油"],
    "MX/混合二甲苯": ["混合二甲苯", "混二甲苯", "异构级MX", "溶剂级MX"],
    "乙烯": ["乙烯"],
    "丙烯": ["丙烯"],
    "乙苯": ["乙苯"],
    "丁二烯": ["丁二烯"],
    "EO/环氧乙烷": ["环氧乙烷", "EO"],
    "VCM": ["氯乙烯单体", "氯乙烯（VCM）", "氯乙烯(VCM)"],
}

NODE_EXCLUDES = {
    "乙烯": ["碳酸乙烯酯", "碳酸亚乙烯酯", "氟代碳酸乙烯酯", "乙烯基", "醋酸乙烯", "聚乙烯", "苯乙烯"],
    "丙烯": ["碳酸丙烯酯", "丙烯腈", "聚丙烯", "环氧丙烷", "丙烯酸"],
}

SEARCH_COLUMNS = ("quota_name", "type_all", "tags", "mark")


def match_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    text = catalog.reindex(columns=SEARCH_COLUMNS, fill_value="").fillna("")
    haystack = text.astype(str).agg(" ".join, axis=1).str.casefold()
    matches: list[pd.DataFrame] = []
    for node, keywords in NODE_KEYWORDS.items():
        mask = pd.Series(False, index=catalog.index)
        for keyword in keywords:
            if keyword.isascii():
                pattern = rf"(?<![a-z0-9]){re.escape(keyword.casefold())}(?![a-z0-9])"
                mask |= haystack.str.contains(pattern, regex=True)
            else:
                mask |= haystack.str.contains(keyword.casefold(), regex=False)
        for excluded in NODE_EXCLUDES.get(node, []):
            mask &= ~haystack.str.contains(excluded.casefold(), regex=False)
        node_matches = catalog.loc[mask].copy()
        node_matches.insert(0, "graph_node", node)
        node_matches.insert(1, "matched_keywords", ",".join(keywords))
        matches.append(node_matches)
    return pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()


def build_summary(matches: pd.DataFrame) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for node in NODE_KEYWORDS:
        node_matches = matches[matches["graph_node"] == node]
        summary.append(
            {
                "graph_node": node,
                "catalog_matches": len(node_matches),
                "quota_ids": node_matches["quota_id"].dropna().astype(str).tolist(),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog, dtype=str)
    matches = match_catalog(catalog)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output_dir / "matched_quota_catalog.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "node_match_summary.json").write_text(
        json.dumps(build_summary(matches), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Matched {len(matches):,} catalog rows across {len(NODE_KEYWORDS)} nodes")


if __name__ == "__main__":
    main()