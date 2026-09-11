"""Create provenance-preserving derived SMM data without changing raw files.

The script treats four cases separately: copied files with the same quota ID,
identical time series with different IDs, quote-family variants, and regional
observations.  Only the first two are collapsed; quote-family and regional
features are retained as additional derived observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw_data" / "smm" / "raw_smm_data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "smm"
DEFAULT_SPECIFICATION = DEFAULT_INPUT / "price_specification.csv"
DEFAULT_CALENDAR = PROJECT_ROOT / "data" / "processed" / "15min" / "PP.csv"
FILENAME_PATTERN = re.compile(r"^(?P<quota_id>[^_]+)_(?P<label>.+)\.csv$")
REGIONS = (
    "华东港口", "华东", "华南", "华北", "华中", "西北", "西南", "东北", "山东",
    "江苏", "浙江", "广东", "福建", "河北", "河南", "天津", "上海", "全国",
)
QUOTE_TYPES = ("最高价", "最低价", "平均价", "涨跌", "涨跌幅")
PRICE_BASIS_COLUMNS = ("currency", "quotation_definition", "tax_description", "vat_rate", "grade_specification", "trade_term")
PRICE_BASIS_METADATA = ("currency", "quotation_definition", "delivery_location", "tax_description", "vat_rate", "grade_specification", "trade_term")
UNDISCLOSED_VALUES = {"", "未披露", "nan", "None"}
PRODUCT_NODE_IDS = {
    "铝土矿": "BAUXITE", "锰矿": "MANGANESE_ORE", "硅石": "SILICA_STONE",
    "石油焦": "PETCOKE", "动力煤": "THERMAL_COAL", "电力": "POWER",
    "铝产业供需": "AL_SUPPLY_DEMAND", "锰产业供需": "MN_SUPPLY_DEMAND",
    "兰炭": "CHARCOAL", "天然气": "NATURAL_GAS", "电石": "CALCIUM_CARBIDE",
    "石脑油": "NAPHTHA", "混合二甲苯": "MX", "MX": "MX", "乙烯": "ETHYLENE",
    "丙烯": "PROPYLENE", "丁二烯": "BUTADIENE", "环氧乙烷": "EO", "VCM": "VCM",
}
UNIT_MAP = {"元/吨": "CNY/t", "美元/吨": "USD/t", "元/吨度": "CNY/t_degree", "美元/桶": "USD/bbl", "吨": "t", "%": "%", "-": "ratio"}
FREQUENCY_MAP = {"日": "daily", "周": "weekly", "月": "monthly", "季": "quarterly"}
FREQUENCY_INTERVAL_HOURS = {"日": 24.0, "周": 168.0, "月": 730.5, "季": 2191.5}
WEEKDAY_NAMES = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6}


@dataclass(frozen=True)
class RawSeries:
    path: Path
    product: str
    quota_id: str
    label: str
    content_hash: str
    observations: int
    date_start: str | None
    date_end: str | None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--specification", type=Path, default=DEFAULT_SPECIFICATION)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--atol", type=float, default=1e-9)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_observations(path: Path) -> pd.DataFrame | None:
    frame = pd.read_csv(path)
    if not {"date", "value"}.issubset(frame.columns):
        return None
    frame = frame.loc[:, ["date", "value"]]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).drop_duplicates("date", keep="last")
    return frame.sort_values("date").reset_index(drop=True)


def inspect_raw_series(input_dir: Path) -> tuple[list[RawSeries], dict[Path, pd.DataFrame], list[dict[str, str]]]:
    records: list[RawSeries] = []
    observations: dict[Path, pd.DataFrame] = {}
    unsupported: list[dict[str, str]] = []
    for path in sorted(input_dir.rglob("*.csv")):
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            unsupported.append({"raw_path": str(path.relative_to(input_dir)), "reason": "unrecognized_filename"})
            continue
        frame = read_observations(path)
        if frame is None:
            unsupported.append({"raw_path": str(path.relative_to(input_dir)), "reason": "missing_date_value_columns"})
            continue
        observations[path] = frame
        records.append(
            RawSeries(
                path=path,
                product=path.parent.name,
                quota_id=match.group("quota_id"),
                label=match.group("label").removesuffix(".csv"),
                content_hash=file_hash(path),
                observations=len(frame),
                date_start=frame["date"].min().strftime("%Y-%m-%d") if not frame.empty else None,
                date_end=frame["date"].max().strftime("%Y-%m-%d") if not frame.empty else None,
            )
        )
    return records, observations, unsupported


def load_specification(path: Path) -> pd.DataFrame:
    specification = pd.read_csv(path, dtype={"quota_id": str})
    specification = specification.drop_duplicates("quota_id", keep="first")
    return specification.set_index("quota_id")


def metadata_for(series: RawSeries, specification: pd.DataFrame) -> dict[str, str]:
    row = specification.loc[series.quota_id] if series.quota_id in specification.index else None
    quota_name = str(row["quota_name"]) if row is not None else series.label.replace("_", ": ")
    unit = str(row["unit"]) if row is not None else ""
    frequency = str(row["frequency"]) if row is not None else ""
    metadata = {"quota_name": quota_name, "unit": unit, "frequency": frequency}
    for column in PRICE_BASIS_METADATA:
        metadata[column] = str(row[column]) if row is not None and pd.notna(row[column]) else "未披露"
    metadata["update_rule"] = str(row["update_rule"]) if row is not None and pd.notna(row.get("update_rule")) else "未披露"
    return metadata


def quote_metadata(quota_name: str) -> tuple[str, str, str, str]:
    parts = [part.strip() for part in quota_name.replace("SMM:", "").split(":") if part.strip()]
    quote_type = next((item for item in QUOTE_TYPES if item in parts), "")
    frequency = next((part for part in parts if part in {"日度", "周度", "月度", "日", "周", "月"}), "")
    region = next((part for part in parts if part in REGIONS), "全国")
    key_parts = [part for part in parts if part not in {quote_type, frequency, region}]
    base_name = ":".join(key_parts)
    return base_name, region, quote_type, frequency


def choose_canonical(group: list[RawSeries]) -> RawSeries:
    return min(group, key=lambda item: ("已知指标" in item.path.name, len(str(item.path)), str(item.path)))


def compare_series(left: pd.DataFrame, right: pd.DataFrame, atol: float) -> dict[str, float | int | bool]:
    overlap = left.merge(right, on="date", suffixes=("_left", "_right"))
    if overlap.empty:
        return {"overlap_observations": 0, "equal_observations": 0, "equal_share": 0.0, "max_abs_difference": None, "identical": False}
    difference = (overlap["value_left"] - overlap["value_right"]).abs()
    equal = difference.le(atol)
    return {
        "overlap_observations": len(overlap),
        "equal_observations": int(equal.sum()),
        "equal_share": float(equal.mean()),
        "max_abs_difference": float(difference.max()),
        "identical": bool(equal.all()),
    }


def replace_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def feature_name(quota_id: str, quote_type: str, region: str) -> str:
    parts = ["smm", quota_id.lower()]
    if quote_type:
        parts.append(quote_type)
    if region and region != "全国":
        parts.append(region)
    return "_".join(re.sub(r"[^a-z0-9]+", "_", part.casefold()) if part.isascii() else part for part in parts)


def next_weekday(date: pd.Timestamp, weekday: int, include_today: bool) -> pd.Timestamp:
    days = (weekday - date.weekday()) % 7
    if days == 0 and not include_today:
        days = 7
    return date + pd.Timedelta(days=days)


def timestamp_at(date: pd.Timestamp, hour: int, minute: int) -> pd.Timestamp:
    return date.normalize() + pd.Timedelta(hours=hour, minutes=minute)


def parse_time(rule: str) -> tuple[int, int] | None:
    match = re.search(r"(\d{1,2}):(\d{2})", rule)
    return (int(match.group(1)), int(match.group(2))) if match else None


def availability_from_rule(timestamp: pd.Timestamp, frequency: str, rule: str) -> tuple[pd.Timestamp, str, float]:
    date = timestamp.normalize()
    rule = str(rule or "未披露")
    time = parse_time(rule)
    expected_hours = FREQUENCY_INTERVAL_HOURS.get(frequency, 24.0)
    if rule.startswith("每日") and time:
        return timestamp_at(date, *time), "daily_fixed_time", expected_hours
    if rule.startswith("次日") and time:
        return timestamp_at(date + pd.Timedelta(days=1), *time), "next_day_fixed_time", expected_hours
    weekday_match = re.search(r"周([一二三四五六日])", rule)
    if "每两周" in rule and weekday_match and time:
        return timestamp_at(next_weekday(date, WEEKDAY_NAMES[weekday_match.group(1)], True), *time), "biweekly_weekday_time", 336.0
    if (rule.startswith("每周") or rule.startswith("次周")) and weekday_match:
        update_date = next_weekday(date, WEEKDAY_NAMES[weekday_match.group(1)], not rule.startswith("次周"))
        if rule.startswith("每周一") or rule.startswith("每周四更新"):
            update_date = next_weekday(date, WEEKDAY_NAMES[weekday_match.group(1)], False)
        is_batch_backfill = "上周" in rule or "本周" in rule
        if time:
            return timestamp_at(update_date, *time), "weekly_batch_time" if is_batch_backfill else "weekly_fixed_time", 168.0
        return timestamp_at(update_date, 23, 59), "weekly_batch_end_of_day_assumed" if is_batch_backfill else "weekly_end_of_day_assumed", 168.0
    if frequency == "月":
        if rule.startswith("次月"):
            day_match = re.search(r"(\d{1,2})日", rule)
            if day_match and time:
                next_month = date + pd.offsets.MonthBegin(1)
                return timestamp_at(next_month.replace(day=int(day_match.group(1))), *time), "next_month_fixed_day", expected_hours
        if rule.startswith("每月末") and time:
            return timestamp_at(date + pd.offsets.MonthEnd(0), *time), "month_end_time", expected_hours
        if rule.startswith("每月"):
            day_match = re.search(r"每月(\d{1,2})日", rule)
            if day_match and time:
                candidate = date.replace(day=min(int(day_match.group(1)), date.days_in_month))
                if candidate < date:
                    candidate = candidate + pd.offsets.MonthBegin(1)
                return timestamp_at(candidate, *time), "monthly_fixed_day", expected_hours
    if frequency == "季" and rule.startswith("次季"):
        day_match = re.search(r"(\d{1,2})日", rule)
        if day_match and time:
            next_quarter = date + pd.offsets.QuarterBegin(startingMonth=1)
            return timestamp_at(next_quarter.replace(day=int(day_match.group(1))), *time), "next_quarter_fixed_day", expected_hours
    fallback_offsets = {"日": pd.DateOffset(days=1), "周": pd.DateOffset(days=7), "月": pd.DateOffset(months=1), "季": pd.DateOffset(months=3)}
    return date + fallback_offsets.get(frequency, pd.DateOffset(days=1)), "frequency_fallback", expected_hours


def availability_for_frame(frame: pd.DataFrame) -> pd.DataFrame:
    records = [availability_from_rule(timestamp, frequency, rule) for timestamp, frequency, rule in frame[["timestamp", "raw_frequency", "update_rule"]].itertuples(index=False, name=None)]
    result = pd.DataFrame(records, columns=["available_at", "availability_rule", "expected_update_interval_hours"], index=frame.index)
    return result


def has_complete_price_basis(frame: pd.DataFrame) -> pd.Series:
    return ~frame.loc[:, PRICE_BASIS_COLUMNS].astype(str).isin(UNDISCLOSED_VALUES).any(axis=1)


def build_verified_regional_averages(retained: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = retained.loc[retained["quote_type"].eq("平均价")].copy()
    prices["price_basis_complete"] = has_complete_price_basis(prices)
    group_columns = ["base_name", "frequency", "unit", *PRICE_BASIS_COLUMNS]
    candidates: list[dict[str, object]] = []
    aggregates: list[pd.DataFrame] = []
    for keys, group in prices.groupby(group_columns, dropna=False):
        series_count = group["quota_id"].nunique()
        regions = sorted(group["region"].dropna().unique())
        if series_count < 2:
            continue
        complete = bool(group["price_basis_complete"].all())
        decision = "aggregated_verified_same_basis" if complete else "not_aggregated_incomplete_price_basis"
        candidates.append(dict(zip(group_columns, keys)) | {"series_count": series_count, "regions": ";".join(regions), "decision": decision})
        if not complete:
            continue
        aggregate = group.groupby(["date", *group_columns], as_index=False).agg(
            regional_average=("value", "mean"), regional_minimum=("value", "min"), regional_maximum=("value", "max"), regional_standard_deviation=("value", "std"), regional_count=("value", "count"),
        )
        aggregate["regional_spread"] = aggregate["regional_maximum"] - aggregate["regional_minimum"]
        aggregates.append(aggregate)
    columns = ["date", *group_columns, "regional_average", "regional_minimum", "regional_maximum", "regional_standard_deviation", "regional_count", "regional_spread"]
    return (pd.concat(aggregates, ignore_index=True) if aggregates else pd.DataFrame(columns=columns), pd.DataFrame(candidates))


def build_curated(retained: pd.DataFrame) -> pd.DataFrame:
    curated = retained.copy()
    forecast = curated["quota_name"].str.contains("预测", na=False)
    curated["timestamp"] = pd.to_datetime(curated.pop("date")).dt.normalize()
    curated["node_id"] = curated["product"].map(PRODUCT_NODE_IDS).fillna("UNMAPPED_" + curated["product"].astype(str))
    curated["feature_name"] = [feature_name(*values) for values in curated[["quota_id", "quote_type", "region"]].itertuples(index=False, name=None)]
    curated["unit"] = curated["unit"].map(UNIT_MAP).fillna(curated["unit"].fillna(""))
    curated["raw_frequency"] = curated["frequency"]
    curated["frequency"] = curated["raw_frequency"].map(FREQUENCY_MAP).fillna("unknown")
    curated = pd.concat([curated, availability_for_frame(curated)], axis=1)
    curated["source"] = "SMM"
    curated["quality_flag"] = "valid"
    curated.loc[~retained["unit"].isin(UNIT_MAP), "quality_flag"] = "unit_unstandardized"
    curated.loc[forecast, "quality_flag"] = "forecast_excluded"
    columns = ["timestamp", "node_id", "feature_name", "value", "unit", "frequency", "available_at", "source", "quality_flag", "quota_id", "quota_name", "product", "region", "quote_type", "raw_frequency", "update_rule", "availability_rule", "expected_update_interval_hours", *PRICE_BASIS_METADATA]
    return curated.loc[:, columns].sort_values(["available_at", "node_id", "feature_name"]).reset_index(drop=True)


def build_model(curated: pd.DataFrame, calendar_path: Path, output_dir: Path) -> pd.DataFrame:
    if not calendar_path.is_file():
        raise FileNotFoundError(f"15-minute calendar file not found: {calendar_path}")
    eligible = curated.loc[curated["quality_flag"].eq("valid") & ~curated["node_id"].str.startswith("UNMAPPED_")].copy()
    calendar = pd.read_csv(calendar_path, usecols=["datetime"])
    calendar["timestamp"] = pd.to_datetime(calendar.pop("datetime"))
    calendar = calendar.loc[:, ["timestamp"]].drop_duplicates().sort_values("timestamp")
    tensor_dir = output_dir / "model" / "node_feature_tensors"
    tensor_dir.mkdir(parents=True)
    manifest: list[dict[str, object]] = []
    state_manifest = pd.DataFrame([
        {"state_feature_suffix": "update_event", "description": "1 only on the first available 15-minute trading timestamp after an SMM update; otherwise 0", "unit": "dummy"},
        {"state_feature_suffix": "age_log1p_hours", "description": "log(1 + elapsed hours since the last available update), compressing long monthly or quarterly gaps", "unit": "log_hours"},
        {"state_feature_suffix": "age_ratio", "description": "elapsed hours since update divided by the series expected update interval", "unit": "ratio"},
        {"state_feature_suffix": "last_delta", "description": "value change at the most recent SMM update, forward-filled until the next update", "unit": "same_as_value"},
        {"state_feature_suffix": "raw_return_daily", "description": "simple net return between consecutive source observations, released with the newer observation and forward-filled until the next update", "unit": "ratio"},
    ])
    state_manifest.to_csv(output_dir / "model" / "smm_update_feature_schema.csv", index=False, encoding="utf-8-sig")
    for node_id in sorted(eligible["node_id"].unique()):
        node_data = eligible.loc[eligible["node_id"].eq(node_id)].sort_values("available_at")
        tensor = calendar.copy()
        for feature, observations in node_data.groupby("feature_name"):
            updates = observations[["available_at", "value", "expected_update_interval_hours"]].drop_duplicates("available_at", keep="last").sort_values("available_at").copy()
            updates["last_delta"] = updates["value"].diff()
            updates["raw_return_daily"] = updates["value"].pct_change(fill_method=None)
            filled = pd.merge_asof(calendar, updates, left_on="timestamp", right_on="available_at", direction="backward")
            tensor[feature] = filled["value"]
            update_event = pd.Series(0, index=calendar.index, dtype="int8")
            calendar_positions = calendar["timestamp"].searchsorted(updates["available_at"], side="left")
            valid_positions = calendar_positions[calendar_positions < len(calendar)]
            update_event.iloc[valid_positions] = 1
            age_hours = (calendar["timestamp"] - filled["available_at"]).dt.total_seconds() / 3600
            tensor[f"{feature}__update_event"] = update_event
            tensor[f"{feature}__age_log1p_hours"] = age_hours.clip(lower=0).map(lambda value: math.log1p(value) if pd.notna(value) else float("nan"))
            tensor[f"{feature}__age_ratio"] = age_hours.clip(lower=0) / filled["expected_update_interval_hours"]
            tensor[f"{feature}__last_delta"] = filled["last_delta"]
            tensor[f"{feature}__raw_return_daily"] = filled["raw_return_daily"]
        tensor_path = tensor_dir / f"{node_id}.csv"
        tensor.to_csv(tensor_path, index=False, encoding="utf-8-sig")
        manifest.append({"node_id": node_id, "tensor_path": str(tensor_path.relative_to(output_dir)), "value_features": len(node_data["feature_name"].unique()), "state_features_per_value": 5, "total_features": len(tensor.columns) - 1, "timestamps": len(tensor), "calendar_source": str(calendar_path.relative_to(PROJECT_ROOT))})
    return pd.DataFrame(manifest)


def main() -> None:
    arguments = parse_arguments()
    if not arguments.input_dir.is_dir():
        raise FileNotFoundError(f"Raw SMM directory not found: {arguments.input_dir}")
    raw_series, observations, unsupported = inspect_raw_series(arguments.input_dir)
    if not raw_series:
        raise RuntimeError("No recognizable SMM CSV files were found.")
    specification = load_specification(arguments.specification)
    replace_output_directory(arguments.output_dir)

    by_quota_id: dict[str, list[RawSeries]] = defaultdict(list)
    for item in raw_series:
        by_quota_id[item.quota_id].append(item)

    canonical_by_id = {quota_id: choose_canonical(group) for quota_id, group in by_quota_id.items()}
    aliases: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    canonical_rows: list[pd.DataFrame] = []
    series_info: list[dict[str, str]] = []
    for quota_id, group in sorted(by_quota_id.items()):
        canonical = canonical_by_id[quota_id]
        metadata = metadata_for(canonical, specification)
        base_name, region, quote_type, name_frequency = quote_metadata(metadata["quota_name"])
        for item in group:
            is_canonical = item.path == canonical.path
            manifest.append({
                "raw_path": str(item.path.relative_to(arguments.input_dir)), "product": item.product,
                "quota_id": item.quota_id, "label": item.label, "content_sha256": item.content_hash,
                "observations": item.observations, "date_start": item.date_start, "date_end": item.date_end,
                "canonical_raw_path": str(canonical.path.relative_to(arguments.input_dir)),
                "is_canonical": is_canonical, "duplicate_type": "unique" if len(group) == 1 else "same_quota_id_copy",
            })
            if not is_canonical:
                aliases.append({"alias_quota_id": quota_id, "alias_raw_path": str(item.path.relative_to(arguments.input_dir)), "canonical_quota_id": quota_id, "canonical_raw_path": str(canonical.path.relative_to(arguments.input_dir)), "reason": "same_quota_id_copy"})
        frame = observations[canonical.path].copy()
        frame["quota_id"] = quota_id
        frame["quota_name"] = metadata["quota_name"]
        frame["product"] = canonical.product
        frame["unit"] = metadata["unit"]
        frame["frequency"] = metadata["frequency"]
        frame["update_rule"] = metadata["update_rule"]
        frame["base_name"] = base_name
        frame["region"] = region
        frame["quote_type"] = quote_type
        for column in PRICE_BASIS_METADATA:
            frame[column] = metadata[column]
        canonical_rows.append(frame)
        series_info.append({"quota_id": quota_id, "quota_name": metadata["quota_name"], "base_name": base_name, "region": region, "quote_type": quote_type, "frequency": name_frequency or metadata["frequency"], "unit": metadata["unit"], **{column: metadata[column] for column in PRICE_BASIS_COLUMNS}})

    canonical_data = pd.concat(canonical_rows, ignore_index=True)
    series_frame = pd.DataFrame(series_info)
    semantic_report: list[dict[str, object]] = []
    semantic_alias_ids: set[str] = set()
    grouping_columns = ["base_name", "region", "quote_type", "frequency", "unit", *PRICE_BASIS_COLUMNS]
    for _, group in series_frame.groupby(grouping_columns, dropna=False):
        identifiers = sorted(group["quota_id"].tolist())
        if len(identifiers) < 2:
            continue
        canonical_id = identifiers[0]
        for candidate_id in identifiers[1:]:
            comparison = compare_series(
                canonical_data.loc[canonical_data["quota_id"].eq(canonical_id), ["date", "value"]],
                canonical_data.loc[canonical_data["quota_id"].eq(candidate_id), ["date", "value"]],
                arguments.atol,
            )
            identical = comparison["identical"] and comparison["overlap_observations"] > 0
            semantic_report.append({"canonical_quota_id": canonical_id, "candidate_quota_id": candidate_id, "decision": "semantic_duplicate" if identical else "retain_distinct", **comparison})
            if identical:
                semantic_alias_ids.add(candidate_id)
                aliases.append({"alias_quota_id": candidate_id, "alias_raw_path": str(canonical_by_id[candidate_id].path.relative_to(arguments.input_dir)), "canonical_quota_id": canonical_id, "canonical_raw_path": str(canonical_by_id[canonical_id].path.relative_to(arguments.input_dir)), "reason": "identical_semantic_series"})

    retained = canonical_data.loc[~canonical_data["quota_id"].isin(semantic_alias_ids)].copy()
    base_prices = retained.loc[retained["quote_type"].eq("平均价")].copy()
    regional, regional_audit = build_verified_regional_averages(retained)

    raw_dir = arguments.output_dir / "raw"
    curated_dir = arguments.output_dir / "curated"
    raw_dir.mkdir()
    curated_dir.mkdir()
    raw_reference = pd.DataFrame(manifest)
    raw_reference["raw_root"] = str(arguments.input_dir)
    raw_reference.to_csv(raw_dir / "raw_file_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(unsupported).to_csv(raw_dir / "unsupported_raw_files.csv", index=False, encoding="utf-8-sig")
    curated = build_curated(retained)
    curated.to_csv(curated_dir / "smm_indicators_long.csv", index=False, encoding="utf-8-sig")
    retained.to_csv(curated_dir / "canonical_long_legacy.csv", index=False, encoding="utf-8-sig")
    base_prices.to_csv(curated_dir / "base_average_prices_legacy.csv", index=False, encoding="utf-8-sig")
    regional.to_csv(curated_dir / "regional_features_verified.csv", index=False, encoding="utf-8-sig")
    regional_audit.to_csv(curated_dir / "regional_average_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(aliases).to_csv(curated_dir / "alias_mapping.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(semantic_report).to_csv(curated_dir / "semantic_duplicate_report.csv", index=False, encoding="utf-8-sig")
    model_manifest = build_model(curated, arguments.calendar, arguments.output_dir)
    model_manifest.to_csv(arguments.output_dir / "model" / "node_feature_manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "raw_timeseries_files": len(raw_series), "unsupported_raw_files": len(unsupported), "unique_quota_ids": len(canonical_by_id),
        "same_quota_id_aliases": len(raw_series) - len(canonical_by_id),
        "semantic_duplicate_aliases": len(semantic_alias_ids), "retained_quota_ids": int(retained["quota_id"].nunique()),
        "raw_observations": int(sum(item.observations for item in raw_series)), "retained_observations": int(len(retained)),
        "curated_observations": int(len(curated)), "model_nodes": int(len(model_manifest)),
        "raw_layer": str(arguments.input_dir), "calendar_source": str(arguments.calendar),
        "verified_regional_average_rows": int(len(regional)), "regional_average_candidates": int(len(regional_audit)),
        "output_files": ["raw/raw_file_manifest.csv", "raw/unsupported_raw_files.csv", "curated/smm_indicators_long.csv", "curated/regional_features_verified.csv", "curated/regional_average_audit.csv", "curated/alias_mapping.csv", "curated/semantic_duplicate_report.csv", "model/node_feature_manifest.csv", "model/node_feature_tensors/*.csv"],
    }
    (arguments.output_dir / "processing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()