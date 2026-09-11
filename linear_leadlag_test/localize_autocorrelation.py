"""Localize futures return autocorrelation by clock time and session boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw_data" / "15min"
ACF_FILE = PROJECT_ROOT / "picture" / "futures_target_return_acf_corrected.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "autocorrelation_timing_results"
RETURN_COLUMN = "return_15m"
MAX_OBSERVATIONS = 10_000
MIN_GROUP_OBSERVATIONS = 80


def parse_lags(value: object) -> list[int]:
    """Parse semicolon-delimited lag lists from the existing ACF report."""

    if pd.isna(value) or str(value).strip() == "":
        return []
    return [int(lag) for lag in str(value).split(";")]


def load_returns(symbol: str) -> pd.DataFrame:
    """Load the same final 10,000 valid returns used by the ACF analysis."""

    frame = pd.read_csv(
        DATA_DIR / f"{symbol}.csv",
        usecols=["datetime", "trading_date", RETURN_COLUMN],
        parse_dates=["datetime", "trading_date"],
    )
    frame[RETURN_COLUMN] = pd.to_numeric(frame[RETURN_COLUMN], errors="coerce")
    frame = frame.dropna(subset=["datetime", "trading_date", RETURN_COLUMN])
    frame = frame.drop_duplicates("datetime", keep="last").sort_values("datetime")
    return frame.tail(MAX_OBSERVATIONS).reset_index(drop=True)


def correlation(values: pd.DataFrame) -> tuple[float, float]:
    """Return Pearson correlation and its two-sided p-value."""

    if len(values) < MIN_GROUP_OBSERVATIONS:
        return np.nan, np.nan
    result = pearsonr(values["current_return"], values["lagged_return"])
    return float(result.statistic), float(result.pvalue)


def build_pairs(frame: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Construct lagged pairs and boundary labels for one observation lag."""

    previous_gap = frame["datetime"].diff().dt.total_seconds().div(60)
    transition = pd.Series("continuous_15m", index=frame.index)
    transition.loc[previous_gap.eq(30)] = "after_morning_break"
    transition.loc[previous_gap.eq(135)] = "after_lunch_break"
    transition.loc[previous_gap.gt(135) & frame["datetime"].dt.hour.ge(18)] = "night_open"
    transition.loc[previous_gap.gt(135) & frame["datetime"].dt.hour.lt(18)] = "day_open"
    transition.loc[previous_gap.isna()] = "sample_start"
    pairs = pd.DataFrame(
        {
            "current_datetime": frame["datetime"],
            "current_trading_date": frame["trading_date"],
            "current_return": frame[RETURN_COLUMN],
            "lagged_datetime": frame["datetime"].shift(lag),
            "lagged_trading_date": frame["trading_date"].shift(lag),
            "lagged_return": frame[RETURN_COLUMN].shift(lag),
            "current_transition": transition,
            "lagged_transition": transition.shift(lag),
        }
    ).dropna()
    pairs["current_time"] = pairs["current_datetime"].dt.strftime("%H:%M")
    pairs["lagged_time"] = pairs["lagged_datetime"].dt.strftime("%H:%M")
    pairs["elapsed_minutes"] = (
        pairs["current_datetime"] - pairs["lagged_datetime"]
    ).dt.total_seconds().div(60).astype(int)
    pairs["same_clock"] = pairs["current_time"] == pairs["lagged_time"]
    pairs["cross_trading_date"] = (
        pairs["current_trading_date"] != pairs["lagged_trading_date"]
    )
    pairs["cross_natural_date"] = (
        pairs["current_datetime"].dt.date != pairs["lagged_datetime"].dt.date
    )
    pairs["centered_product"] = (
        pairs["current_return"] - pairs["current_return"].mean()
    ) * (pairs["lagged_return"] - pairs["lagged_return"].mean())
    return pairs


def timing_rows(symbol: str, lag: int, pairs: pd.DataFrame) -> list[dict[str, object]]:
    """Calculate conditional correlation and covariance contribution by clock pair."""

    total_product = pairs["centered_product"].sum()
    total_absolute_product = pairs["centered_product"].abs().sum()
    rows: list[dict[str, object]] = []
    for (current_time, lagged_time), group in pairs.groupby(
        ["current_time", "lagged_time"], sort=False
    ):
        conditional_acf, pvalue = correlation(group)
        rows.append(
            {
                "variety": symbol,
                "lag": lag,
                "current_time": current_time,
                "lagged_time": lagged_time,
                "n_pairs": len(group),
                "conditional_acf": conditional_acf,
                "pvalue": pvalue,
                "significant_95pct": bool(pvalue < 0.05) if np.isfinite(pvalue) else False,
                "mean_centered_product": group["centered_product"].mean(),
                "sum_centered_product": group["centered_product"].sum(),
                "signed_covariance_share": (
                    group["centered_product"].sum() / total_product
                    if total_product
                    else np.nan
                ),
                "absolute_product_share": (
                    group["centered_product"].abs().sum() / total_absolute_product
                    if total_absolute_product
                    else np.nan
                ),
                "same_clock_share": group["same_clock"].mean(),
                "cross_trading_date_share": group["cross_trading_date"].mean(),
                "cross_natural_date_share": group["cross_natural_date"].mean(),
                "median_elapsed_minutes": group["elapsed_minutes"].median(),
                "current_transition": group["current_transition"].mode().iat[0],
                "lagged_transition": group["lagged_transition"].mode().iat[0],
            }
        )
    return rows


def summarize_lag(
    symbol: str,
    lag: int,
    pairs: pd.DataFrame,
    timing: pd.DataFrame,
    peak_lag: int,
    bonferroni_lags: set[int],
) -> dict[str, object]:
    """Summarize whether one lag is daily alignment or time-localized."""

    overall_acf, overall_pvalue = correlation(pairs)
    eligible = timing.dropna(subset=["conditional_acf"])
    strongest = eligible.loc[eligible["conditional_acf"].abs().idxmax()] if not eligible.empty else None
    positive_covariance = timing.loc[timing["sum_centered_product"] > 0]
    negative_covariance = timing.loc[timing["sum_centered_product"] < 0]
    covariance_source = (
        positive_covariance.loc[positive_covariance["sum_centered_product"].idxmax()]
        if overall_acf >= 0 and not positive_covariance.empty
        else negative_covariance.loc[negative_covariance["sum_centered_product"].idxmin()]
        if overall_acf < 0 and not negative_covariance.empty
        else None
    )
    return {
        "variety": symbol,
        "lag": lag,
        "is_peak_lag": lag == peak_lag,
        "is_bonferroni_lag": lag in bonferroni_lags,
        "n_pairs": len(pairs),
        "overall_acf": overall_acf,
        "overall_pvalue": overall_pvalue,
        "same_clock_share": pairs["same_clock"].mean(),
        "cross_trading_date_share": pairs["cross_trading_date"].mean(),
        "cross_natural_date_share": pairs["cross_natural_date"].mean(),
        "median_elapsed_minutes": pairs["elapsed_minutes"].median(),
        "strongest_clock_pair": (
            f"{strongest.current_time}<-{strongest.lagged_time}" if strongest is not None else ""
        ),
        "strongest_conditional_acf": strongest.conditional_acf if strongest is not None else np.nan,
        "strongest_clock_pvalue": strongest.pvalue if strongest is not None else np.nan,
        "covariance_source_clock_pair": (
            f"{covariance_source.current_time}<-{covariance_source.lagged_time}"
            if covariance_source is not None
            else ""
        ),
        "covariance_source_mean_product": (
            covariance_source.mean_centered_product if covariance_source is not None else np.nan
        ),
        "covariance_source_signed_share": (
            covariance_source.signed_covariance_share if covariance_source is not None else np.nan
        ),
        "covariance_source_transition": (
            covariance_source.current_transition if covariance_source is not None else ""
        ),
    }


def write_report(summary: pd.DataFrame) -> None:
    """Write a compact report for peak and family-wise significant lags."""

    focus = summary.loc[summary["is_peak_lag"] | summary["is_bonferroni_lag"]].copy()
    daily = focus.loc[focus["same_clock_share"] >= 0.8]
    cross_trading = focus.loc[focus["cross_trading_date_share"] >= 0.8]
    lines = [
        "# 期货自相关滞后的时段定位",
        "",
        "## 方法",
        "",
        "- 样本与原 ACF 分析一致：每个品种最后 10,000 个有效 `return_15m`。",
        "- 滞后按有效观测行计数，不把午休、夜盘休市或周末补成15分钟网格。",
        "- 对每个显著滞后按 `当前钟点 <- 滞后端钟点` 分组计算条件相关。",
        "- `trading_date` 用于识别交易日切换；自然日期变化单独报告。",
        "",
        "## 汇总",
        "",
        f"- 峰值或 Bonferroni 显著滞后共 {len(focus)} 个。",
        f"- 至少80%的配对连接同一钟点：{len(daily)}/{len(focus)}。",
        f"- 至少80%的配对跨 `trading_date`：{len(cross_trading)}/{len(focus)}。",
        "",
        "## 逐品种重点滞后",
        "",
    ]
    for row in focus.sort_values(["variety", "lag"]).itertuples(index=False):
        lines.append(
            f"- {row.variety} lag {row.lag}: ACF={row.overall_acf:.4f}; "
            f"同钟点={row.same_clock_share:.1%}, 跨交易日={row.cross_trading_date_share:.1%}, "
            f"跨自然日={row.cross_natural_date_share:.1%}; "
            f"最强条件时段={row.strongest_clock_pair} "
            f"(r={row.strongest_conditional_acf:.4f}, p={row.strongest_clock_pvalue:.3g}); "
            f"总体协方差主来源={row.covariance_source_clock_pair} "
            f"({row.covariance_source_transition}, 方向内贡献={row.covariance_source_signed_share:.1%})。"
        )
    (OUTPUT_DIR / "autocorrelation_timing_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Localize all pointwise-significant lags in the existing ACF report."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    acf_summary = pd.read_csv(ACF_FILE)
    all_timing: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    for acf_row in acf_summary.itertuples(index=False):
        frame = load_returns(acf_row.variety)
        pointwise_lags = parse_lags(acf_row.pointwise_lags)
        bonferroni_lags = set(parse_lags(acf_row.bonferroni_lags))
        for lag in pointwise_lags:
            pairs = build_pairs(frame, lag)
            rows = timing_rows(acf_row.variety, lag, pairs)
            timing = pd.DataFrame(rows)
            all_timing.extend(rows)
            all_summary.append(
                summarize_lag(
                    acf_row.variety,
                    lag,
                    pairs,
                    timing,
                    int(acf_row.peak_lag),
                    bonferroni_lags,
                )
            )
        print(f"{acf_row.variety}: localized {len(pointwise_lags)} significant lags")

    timing_frame = pd.DataFrame(all_timing)
    summary_frame = pd.DataFrame(all_summary)
    timing_frame.to_csv(OUTPUT_DIR / "lag_clock_pair_details.csv", index=False)
    summary_frame.to_csv(OUTPUT_DIR / "lag_timing_summary.csv", index=False)
    write_report(summary_frame)
    print(f"Wrote timing results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()