"""Assess native and model-grid quality for SMM nodes used by the graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GRAPH_NODES = ("NAPHTHA", "MX", "ETHYLENE", "EO", "PROPYLENE", "BUTADIENE")
DERIVED_TERMS = ("涨跌", "涨跌幅", "利润", "价差", "库存", "仓单", "产量", "开工率")
CORE_PRICE_TERMS = {
    "NAPHTHA": ("石脑油CFR日本", "石脑油: 直馏汽油", "石脑油: 加氢石脑油"),
    "MX": ("异构MX", "异构级二甲苯", "异构二甲苯", "溶剂级二甲苯"),
    "ETHYLENE": ("乙烯国内价格",),
    "EO": ("环氧乙烷: 国标",),
    "PROPYLENE": ("丙烯: 聚合级",),
    "BUTADIENE": ("丁二烯: 优等品",),
}

RECOMMENDATIONS = {
    "NAPHTHA": {
        "series": "石脑油CFR日本(零变动3.6%) + 山东直馏/加氢平均价(24-28%)",
        "frequency": "山东现货日更13:30(1210次发布)；CFR日本约周度批量(259次/1242日)",
        "verdict": "A·可直接用：山东现货作日度输入；CFR日本按available_at对齐后作周度外盘信号",
    },
    "MX": {
        "series": "异构级二甲苯 江苏/山东(13-27%) + 异构MXFOB韩国(5.5%)",
        "frequency": "国内二甲苯日更(1146次)；异构MXFOB韩国周度批量(259次/1242日)",
        "verdict": "A·可直接用：国内二甲苯日更；FOB韩国为周度外盘需按available_at对齐",
    },
    "BUTADIENE": {
        "series": "丁二烯优等品 华东/山东 平均价(10-14%)",
        "frequency": "全部日更现货，2021-12起，1148条",
        "verdict": "A·质量最好：日更且变动充分，直接作现货价格；仓单(SHFE)属期货衍生需另列",
    },
    "PROPYLENE": {
        "series": "丙烯聚合级 山东(23%)/东北(39%)优先；华东66%/华南80%偏粘",
        "frequency": "日更现货，2021-12起，1148条",
        "verdict": "C·可用但注意：优先山东/东北；华东/华南日度变动稀，宜降频或事件化",
    },
    "EO": {
        "series": "环氧乙烷国标 五区平均价(91-92%)",
        "frequency": "日更现货，2021-12起，1148条",
        "verdict": "D·现货几乎不日变动：适合周频或调价事件变量，不宜作日度收益；EO-1.3EG价差变动更频但属衍生",
    },
    "ETHYLENE": {
        "series": "乙烯国内价格(零变动86%，2023-01起仅904条/190次发布)",
        "frequency": "名义日度但实际约周度批量发布(190事件)，非日更",
        "verdict": "D·信息频率最低且粘性高：慎作日度收益，建议周度对齐或调价事件；EC/PC/VC/FEC为下游不能当乙烯本体",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze graph-relevant SMM data quality.")
    parser.add_argument("--smm-dir", type=Path, default=Path("data/processed/smm"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/smm/quality_reports/graph_nodes"),
    )
    return parser.parse_args()


def ratio(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else np.nan


def classify_indicator(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(column, ""))
        for column in ("quota_name", "quote_type", "quotation_definition", "unit")
    )
    for term in DERIVED_TERMS:
        if term in text:
            return "derived_or_quantity"
    return "price_or_level"


def semantic_role(row: pd.Series) -> str:
    if row["indicator_class"] == "derived_or_quantity":
        return "derived_or_quantity"
    name = str(row["quota_name"])
    if any(term in name for term in CORE_PRICE_TERMS[str(row["node_id"])]):
        return "core_product_price"
    return "related_product_or_spread"


def native_metrics(group: pd.DataFrame) -> dict[str, object]:
    group = group.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    values = pd.to_numeric(group["value"], errors="coerce")
    valid = group.loc[values.notna()].copy()
    values = pd.to_numeric(valid["value"], errors="coerce")
    dates = valid["timestamp"]
    intervals = dates.diff().dt.total_seconds().div(86400).dropna()
    changes = values.diff().dropna()
    prior = values.shift(1)
    valid_returns = prior.notna() & prior.ne(0) & values.notna()
    returns = values[valid_returns].div(prior[valid_returns]).sub(1)
    releases = pd.to_datetime(valid["available_at"], errors="coerce").dropna().drop_duplicates().sort_values()
    release_intervals = releases.diff().dt.total_seconds().div(3600).dropna()
    first = valid.iloc[0] if len(valid) else group.iloc[0]
    return {
        "node_id": first["node_id"],
        "feature_name": first["feature_name"],
        "quota_id": first["quota_id"],
        "quota_name": first["quota_name"],
        "indicator_class": classify_indicator(first),
        "unit": first["unit"],
        "declared_frequency": first["frequency"],
        "raw_frequency": first["raw_frequency"],
        "update_rule": first["update_rule"],
        "expected_update_interval_hours": first["expected_update_interval_hours"],
        "native_observations": len(valid),
        "native_start": dates.min(),
        "native_end": dates.max(),
        "native_span_days": int((dates.max() - dates.min()).days) if len(dates) else 0,
        "native_missing_value_share": ratio(values.isna()),
        "native_zero_level_share": ratio(values.eq(0)),
        "native_zero_return_share": ratio(returns.abs().le(1e-12)),
        "native_zero_change_share": ratio(changes.abs().le(1e-12)),
        "native_median_interval_days": intervals.median(),
        "native_p90_interval_days": intervals.quantile(0.9),
        "native_max_interval_days": intervals.max(),
        "release_events": len(releases),
        "release_median_interval_hours": release_intervals.median(),
        "release_p90_interval_hours": release_intervals.quantile(0.9),
        "release_max_interval_hours": release_intervals.max(),
    }


def add_tensor_metrics(row: dict[str, object], tensors_dir: Path) -> None:
    path = tensors_dir / f"{row['node_id']}.csv"
    feature = str(row["feature_name"])
    event_column = f"{feature}__update_event"
    age_column = f"{feature}__age_ratio"
    header = pd.read_csv(path, nrows=0).columns
    columns = ["timestamp", feature, event_column, age_column]
    if any(column not in header for column in columns):
        row.update({"grid_rows": np.nan, "grid_active_rows": np.nan})
        return
    frame = pd.read_csv(path, usecols=columns, parse_dates=["timestamp"])
    values = pd.to_numeric(frame[feature], errors="coerce")
    active = values.notna()
    active_values = values[active]
    grid_returns = active_values.pct_change(fill_method=None).dropna()
    events = pd.to_numeric(frame.loc[active, event_column], errors="coerce").fillna(0)
    ages = pd.to_numeric(frame.loc[active, age_column], errors="coerce")
    row.update(
        {
            "grid_rows": len(frame),
            "grid_active_rows": int(active.sum()),
            "grid_start": frame.loc[active, "timestamp"].min(),
            "grid_end": frame.loc[active, "timestamp"].max(),
            "grid_full_missing_share": ratio(~active),
            "grid_zero_return_share": ratio(grid_returns.abs().le(1e-12)),
            "grid_update_event_count": int(events.eq(1).sum()),
            "grid_update_event_share": ratio(events.eq(1)),
            "grid_age_over_expected_share": ratio(ages.gt(1)),
            "grid_age_over_2x_expected_share": ratio(ages.gt(2)),
            "grid_max_age_ratio": ages.max(),
        }
    )


def summarize_nodes(series: pd.DataFrame, data_end: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for node_id, group in series.groupby("node_id", sort=False):
        core = group[group["semantic_role"] == "core_product_price"]
        candidate = core if not core.empty else group[group["indicator_class"] == "price_or_level"]
        zero_share = candidate["native_zero_return_share"].median()
        grade = "A" if zero_share < 0.3 else "B" if zero_share < 0.5 else "C" if zero_share < 0.8 else "D"
        rows.append(
            {
                "node_id": node_id,
                "quality_grade": grade,
                "indicator_count": len(group),
                "core_product_price_count": len(core),
                "frequency_set": "; ".join(sorted(group["declared_frequency"].dropna().astype(str).unique())),
                "native_observations_min": int(group["native_observations"].min()),
                "native_observations_max": int(group["native_observations"].max()),
                "native_start_earliest": group["native_start"].min(),
                "native_start_latest": group["native_start"].max(),
                "native_end_latest": group["native_end"].max(),
                "days_behind_dataset_end": int((data_end - group["native_end"].max()).days),
                "core_median_zero_return_share": candidate["native_zero_return_share"].median(),
                "core_max_zero_return_share": candidate["native_zero_return_share"].max(),
                "median_native_interval_days": candidate["native_median_interval_days"].median(),
                "max_native_gap_days": candidate["native_max_interval_days"].max(),
                "release_events_min": int(group["release_events"].min()),
                "release_events_max": int(group["release_events"].max()),
                "grid_zero_return_share_median": candidate["grid_zero_return_share"].median(),
                "grid_age_over_expected_share_median": candidate["grid_age_over_expected_share"].median(),
            }
        )
    return pd.DataFrame(rows)


def fmt(value: object, percent: bool = False) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.1%}" if percent else f"{float(value):.1f}"


def write_report(path: Path, nodes: pd.DataFrame, series: pd.DataFrame, data_end: pd.Timestamp) -> None:
    lines = [
        "# 图谱相关 SMM 节点数据质量报告",
        "",
        f"- 分析节点：{', '.join(GRAPH_NODES)}",
        f"- curated 数据截至：{data_end.date()}",
        "- 口径：源观测、实际发布事件、15 分钟模型网格分别统计；模型网格行数不代表原生观测数。",
        "",
        "## 节点汇总",
        "",
        "| 节点 | 等级 | 指标数 | 核心本体价格 | 原生观测数范围 | 最早起点 | 最新终点 | 核心原生零收益率中位数 | 最大原生间隔(天) | 15分钟零收益率中位数 |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for _, row in nodes.iterrows():
        lines.append(
            f"| {row['node_id']} | {row['quality_grade']} | {row['indicator_count']} | {row['core_product_price_count']} | "
            f"{row['native_observations_min']}-{row['native_observations_max']} | "
            f"{row['native_start_earliest'].date()} | {row['native_end_latest'].date()} | "
            f"{fmt(row['core_median_zero_return_share'], True)} | {fmt(row['max_native_gap_days'])} | "
            f"{fmt(row['grid_zero_return_share_median'], True)} |"
        )
    lines.extend(
        [
            "",
            "## 解读规则",
            "",
            "- 质量等级仅衡量核心本体价格的有效变化密度：原生零收益率低于 30% 为 A，30%-50% 为 B，50%-80% 为 C，80% 及以上为 D。",
            "- `原生零收益率`只比较相邻源观测，直接反映序列是否长期不变；这是判断有效变化密度的主要指标。",
            "- `15分钟零收益率`基于前向填充后的交易网格。日度、周度或批量发布数据接近 100% 属于频率转换的机械结果，不应单独判为坏数据。",
            "- `release_events`按 `available_at` 去重。同一发布日期一次补发多天历史值时，多条源观测只构成一次当时可用的信息事件。",
            "- `age_over_expected`表示距最近发布超过声明更新周期，节假日和批量发布规则也会推高该值，建模时应保留更新事件和 age 特征。",
            "",
            "## 使用建议",
            "",
            "- 节点汇总只使用 `core_product_price` 计算变化质量；下游产品、加工费和价差标为 `related_product_or_spread`，不能冒充节点本体价格。",
            "- 利润、涨跌、库存、仓单、产量等 `derived_or_quantity` 指标应独立建模，不能与现货价格无差别聚合。",
            "- 15 分钟模型中不要对前向填充值重复计算并当作高频价格发现。应使用发布时点对齐的 `update_event`、`last_delta` 或 `raw_return_daily`。",
            "- 回测切分应以 `available_at` 为准，尤其是每周集中补发多条日度数据的指标，避免使用尚未发布的观测日期造成前视偏差。",
            "- 指标级异常和具体更新规则见 `graph_smm_series_quality.csv`，节点汇总见 `graph_smm_node_quality.csv`。",
            "",
            "## 高零变化指标",
            "",
            "以下列出原生零收益率不低于 50% 的指标；若为空，说明没有大面积原生零收益问题。",
            "",
            "| 节点 | 指标 | 语义角色 | 观测数 | 原生零收益率 |",
            "|---|---|---|---:|---:|",
        ]
    )
    flagged = series[series["native_zero_return_share"].ge(0.5)].sort_values(
        "native_zero_return_share", ascending=False
    )
    if flagged.empty:
        lines.append("| - | 无 | - | - | - |")
    else:
        for _, row in flagged.iterrows():
            lines.append(
                f"| {row['node_id']} | {row['quota_name']} | {row['semantic_role']} | "
                f"{row['native_observations']} | {fmt(row['native_zero_return_share'], True)} |"
            )
    lines.extend(["", "## 结论与接入建议", ""])
    grade_map = nodes.set_index("node_id")["quality_grade"].to_dict()
    lines.append("| 节点 | 等级 | 推荐核心序列(原生零变动参考) | 信息发布频率 | 判定与用法 |")
    lines.append("|---|---|---|---|---|")
    for node_id in GRAPH_NODES:
        rec = RECOMMENDATIONS[node_id]
        lines.append(
            f"| {node_id} | {grade_map[node_id]} | {rec['series']} | {rec['frequency']} | {rec['verdict']} |"
        )
    lines.extend(
        [
            "",
            "补充说明：",
            "",
            "- 全库 SMM 数据截至 2026-08-31；乙烯国内价格、石脑油CFR日本、异构MXFOB韩国等周度批量序列最近一次实际发布在 2026-08-27，实时接入前需重新下载。",
            "- 15 分钟网格 96%-100% 的零收益率来自日/周频数据前向填充，是频率转换结果，不代表源数据损坏。",
            "- `related_product_or_spread` 与 `derived_or_quantity` 指标已排除在等级判定之外，只能作参考或独立特征。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    long_path = args.smm_dir / "curated" / "smm_indicators_long.csv"
    usecols = [
        "timestamp", "node_id", "feature_name", "value", "unit", "frequency", "available_at",
        "quota_id", "quota_name", "quote_type", "raw_frequency", "update_rule",
        "expected_update_interval_hours", "quotation_definition",
    ]
    long = pd.read_csv(long_path, usecols=usecols, low_memory=False)
    long = long[long["node_id"].isin(GRAPH_NODES)].copy()
    long["timestamp"] = pd.to_datetime(long["timestamp"], errors="coerce")
    rows = [native_metrics(group) for _, group in long.groupby(["node_id", "feature_name"], sort=False)]
    tensors_dir = args.smm_dir / "model" / "node_feature_tensors"
    for row in rows:
        add_tensor_metrics(row, tensors_dir)
    series = pd.DataFrame(rows)
    series["semantic_role"] = series.apply(semantic_role, axis=1)
    data_end = long["timestamp"].max()
    nodes = summarize_nodes(series, data_end)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(args.output_dir / "graph_smm_series_quality.csv", index=False)
    nodes.to_csv(args.output_dir / "graph_smm_node_quality.csv", index=False)
    write_report(args.output_dir / "graph_smm_quality_report.md", nodes, series, data_end)
    print(f"Analyzed {len(series)} indicators across {len(nodes)} nodes; output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()