"""Download selected SMM indicators and report coverage by economic-graph node."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from get_smm_data import MAX_QUOTA_IDS_PER_REQUEST, SmmClient, _chunks, normalize_data


def classify_series(
    payloads: list[dict[str, Any]], candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    status_rows: list[dict[str, Any]] = []
    for payload in payloads:
        for series in payload.get("data") or []:
            quota_id = str(series.get("quota_id") or "")
            unauthorized = (
                series.get("perm_state") == -1
                or series.get("state_desc") == "no authorize"
            )
            points = series.get("data") or []
            status_rows.append(
                {
                    "quota_id": quota_id,
                    "quota_name": series.get("quota_name"),
                    "perm_state": series.get("perm_state"),
                    "state_desc": series.get("state_desc"),
                    "status": "unauthorized" if unauthorized else "downloaded" if points else "empty",
                    "row_count": len(points),
                }
            )

            candidate_links = candidates[["graph_node", "quota_id"]].drop_duplicates()
            status = candidate_links.merge(pd.DataFrame(status_rows), on="quota_id", how="left")
    authorized_payloads = []
    for payload in payloads:
        permitted = [
            series
            for series in payload.get("data") or []
            if series.get("perm_state") != -1
            and series.get("state_desc") != "no authorize"
        ]
        authorized_payloads.append({**payload, "data": permitted})
    normalized = normalize_data(authorized_payloads)
    if not normalized.empty:
        normalized = candidate_links.merge(normalized, on="quota_id", how="inner")
    return normalized, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--use-proxy", action="store_true")
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates, dtype=str).drop_duplicates(
        ["graph_node", "quota_id"]
    )
    quota_ids = candidates["quota_id"].dropna().drop_duplicates().tolist()
    client = SmmClient(
        user_name=os.environ.get("SMM_USER_NAME"),
        password=os.environ.get("SMM_PASSWORD"),
        token_cache=args.token_cache,
        use_proxy=args.use_proxy,
    )
    payloads = [
        client.get_data_origin(batch, args.start_date, args.end_date)
        for batch in _chunks(quota_ids, MAX_QUOTA_IDS_PER_REQUEST)
    ]
    normalized, status = classify_series(payloads, candidates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected_raw.json").write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    normalized.to_csv(args.output_dir / "selected_normalized.csv", index=False)
    status.to_csv(args.output_dir / "download_status.csv", index=False, encoding="utf-8-sig")

    requested = candidates.groupby("graph_node").size().rename("requested")
    status_counts = status.pivot_table(
        index="graph_node", columns="status", values="quota_id", aggfunc="count", fill_value=0
    )
    coverage = requested.to_frame().join(status_counts, how="left").fillna(0).reset_index()
    coverage.to_csv(args.output_dir / "node_coverage.csv", index=False, encoding="utf-8-sig")
    print(coverage.to_string(index=False))
    print(f"Saved {len(normalized):,} observations to {args.output_dir}")


if __name__ == "__main__":
    main()