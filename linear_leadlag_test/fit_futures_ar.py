"""Fit plain AR models to futures returns and test residual autocorrelation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw_data" / "15min"
UNIVERSE_FILE = PROJECT_ROOT / "picture" / "futures_target_return_acf_corrected.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "ar_results"
RETURN_COLUMN = "return_15m"
MAX_OBSERVATIONS = 10_000
MAX_AR_LAG = 40
LB_LAGS = (50, 60, 80)


def load_universe() -> list[str]:
    """Use the exact futures universe represented in the corrected plots."""

    universe = pd.read_csv(UNIVERSE_FILE, usecols=["variety"])["variety"]
    return universe.astype(str).tolist()


def load_returns(symbol: str) -> pd.Series:
    """Load the final common-length sample of observed 15-minute returns."""

    path = DATA_DIR / f"{symbol}.csv"
    frame = pd.read_csv(path, usecols=["datetime", RETURN_COLUMN], parse_dates=["datetime"])
    values = pd.to_numeric(frame[RETURN_COLUMN], errors="coerce")
    series = pd.Series(values.to_numpy(), index=frame["datetime"], name=symbol).dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index().tail(MAX_OBSERVATIONS)
    if len(series) <= MAX_AR_LAG + max(LB_LAGS):
        raise ValueError(f"{symbol} has only {len(series)} valid returns")
    return series


def fit_candidates(values: pd.Series) -> tuple[pd.DataFrame, int]:
    """Fit AR(0)..AR(40) on a common effective sample and select by AIC."""

    observations = values.to_numpy()
    rows: list[dict[str, float | int]] = []
    for lag in range(MAX_AR_LAG + 1):
        result = AutoReg(observations, lags=lag, trend="c", hold_back=MAX_AR_LAG).fit()
        rows.append({"lag": lag, "aic": result.aic})
    criteria = pd.DataFrame(rows)
    aic_lag = int(criteria.loc[criteria["aic"].idxmin(), "lag"])
    return criteria, aic_lag


def analyze_symbol(
    symbol: str,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Select the AIC order, fit it, and run residual Ljung-Box tests."""

    values = load_returns(symbol)
    criteria, aic_lag = fit_candidates(values)
    selected_lag = aic_lag
    result = AutoReg(
        values.to_numpy(), lags=selected_lag, trend="c", hold_back=MAX_AR_LAG
    ).fit()
    coefficient_names = ["const", *(f"lag_{lag}" for lag in range(1, selected_lag + 1))]
    coefficients = pd.DataFrame(
        {
            "variety": symbol,
            "term": coefficient_names,
            "coefficient": result.params,
            "standard_error": result.bse,
            "pvalue": result.pvalues,
        }
    )

    tests = acorr_ljungbox(
        result.resid,
        lags=list(LB_LAGS),
        model_df=selected_lag,
        return_df=True,
    ).reset_index(names="test_lag")
    tests.insert(0, "variety", symbol)
    tests["pass_95pct"] = tests["lb_pvalue"] >= 0.05
    tests["pass_99pct"] = tests["lb_pvalue"] >= 0.01

    criteria.insert(0, "variety", symbol)
    summary: dict[str, object] = {
        "variety": symbol,
        "n": len(values),
        "aic_lag": aic_lag,
        "selected_lag": selected_lag,
        "residual_std": float(np.std(result.resid, ddof=1)),
        "all_lb_pass_95pct": bool(tests["pass_95pct"].all()),
        "all_lb_pass_99pct": bool(tests["pass_99pct"].all()),
        "minimum_lb_pvalue": float(tests["lb_pvalue"].min()),
    }
    return criteria, summary, tests, coefficients


def write_report(summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    """Write a compact Chinese report emphasizing residual test failures."""

    failed_95 = summary.loc[~summary["all_lb_pass_95pct"], "variety"].tolist()
    failed_99 = summary.loc[~summary["all_lb_pass_99pct"], "variety"].tolist()
    lines = [
        "# 期货收益率 AR 与残差 Ljung-Box 检验",
        "",
        "## 方法",
        "",
        f"- 样本：图中 {len(summary)} 个期货品种，各取最后 {MAX_OBSERVATIONS:,} 个有效 `return_15m`。",
        f"- 候选模型：含截距的 AR(0)-AR({MAX_AR_LAG})，不加入季节性或其他回归变量。",
        "- 所有品种统一以 AIC 最小值选择最终 AR 阶数。",
        f"- 对最终模型残差在 {', '.join(map(str, LB_LAGS))} 阶做 Ljung-Box 累计检验；检验自由度扣除所选 AR 阶数。",
        "- 原假设为截至该阶残差不存在自相关；p >= 0.05 表示通过95%置信水平，p >= 0.01 表示通过99%置信水平。",
        "",
        "## 汇总",
        "",
        f"- 所有 Ljung-Box 窗口均通过95%检验：{summary['all_lb_pass_95pct'].sum()}/{len(summary)}。",
        f"- 所有 Ljung-Box 窗口均通过99%检验：{summary['all_lb_pass_99pct'].sum()}/{len(summary)}。",
        f"- 连95%都未全部通过：{', '.join(failed_95) if failed_95 else '无'}。",
        f"- 未全部通过99%：{', '.join(failed_99) if failed_99 else '无'}。",
        "",
        "## 连95%都未通过的具体窗口",
        "",
    ]
    failures = tests.loc[~tests["pass_95pct"]]
    if failures.empty:
        lines.append("无。")
    else:
        lines.extend(
            f"- {row.variety}: LB({row.test_lag})={row.lb_stat:.4f}, p={row.lb_pvalue:.6g}"
            for row in failures.itertuples(index=False)
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            "- `ar_order_summary.csv`：AIC 阶数、最终阶数和总体通过情况。",
            "- `ar_information_criteria.csv`：每个品种 AR(0)-AR(40) 的完整 AIC。",
            "- `ar_coefficients.csv`：最终所选 AR 模型的截距和滞后系数。",
            "- `ljung_box_results.csv`：每个检验窗口的统计量、p值和95%/99%结论。",
        ]
    )
    (OUTPUT_DIR / "ar_ljung_box_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the complete futures AR analysis."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_criteria: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    all_tests: list[pd.DataFrame] = []
    all_coefficients: list[pd.DataFrame] = []
    for symbol in load_universe():
        criteria, summary, tests, coefficients = analyze_symbol(symbol)
        all_criteria.append(criteria)
        summaries.append(summary)
        all_tests.append(tests)
        all_coefficients.append(coefficients)
        print(f"{symbol}: AIC={summary['aic_lag']}, selected={summary['selected_lag']}")

    criteria_frame = pd.concat(all_criteria, ignore_index=True)
    summary_frame = pd.DataFrame(summaries).sort_values("variety")
    tests_frame = pd.concat(all_tests, ignore_index=True).sort_values(["variety", "test_lag"])
    criteria_frame.to_csv(OUTPUT_DIR / "ar_information_criteria.csv", index=False)
    summary_frame.to_csv(OUTPUT_DIR / "ar_order_summary.csv", index=False)
    pd.concat(all_coefficients, ignore_index=True).to_csv(
        OUTPUT_DIR / "ar_coefficients.csv", index=False
    )
    tests_frame.to_csv(OUTPUT_DIR / "ljung_box_results.csv", index=False)
    write_report(summary_frame, tests_frame)
    print(f"Wrote results for {len(summary_frame)} futures varieties to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()