"""Create return-series diagnostics for every SMM node tensor."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import acf, adfuller, pacf


PROJECT_ROOT = Path(__file__).resolve().parent
SMM_DIR = PROJECT_ROOT / "data" / "processed" / "smm" / "model" / "node_feature_tensors"
OUTPUT_DIR = PROJECT_ROOT / "picture"
MAX_ADF_OBSERVATIONS = 20_000
MAX_CORRELATION_OBSERVATIONS = 10_000
MAX_LAGS = 40


def raw_price_column(columns: pd.Index) -> str:
    """Select the first un-suffixed raw price column from a node tensor."""

    candidates = [
        column
        for column in columns
        if column != "timestamp" and "__" not in column
    ]
    if not candidates:
        raise ValueError("No raw price column found.")
    return candidates[0]


def load_returns(path: Path, source: str) -> tuple[pd.Series, pd.Series, str]:
    """Load raw prices and derive 15-minute and daily log-return series."""

    header = pd.read_csv(path, nrows=0).columns
    price_column = raw_price_column(header)
    frame = pd.read_csv(path, usecols=["timestamp", price_column], parse_dates=["timestamp"])
    prices = pd.to_numeric(frame[price_column], errors="coerce").where(lambda values: values > 0)
    indexed_prices = pd.Series(prices.to_numpy(), index=frame["timestamp"], name=path.stem).sort_index()
    indexed_prices = indexed_prices[~indexed_prices.index.duplicated(keep="last")]
    return_15m = np.log(indexed_prices).diff().replace([np.inf, -np.inf], np.nan).dropna()
    daily_prices = indexed_prices.groupby(indexed_prices.index.normalize()).last()
    daily_return = np.log(daily_prices).diff().replace([np.inf, -np.inf], np.nan).dropna()
    return return_15m, daily_return, price_column


def sample_series(values: pd.Series, maximum: int) -> pd.Series:
    """Evenly subsample long series without changing the full-series summary statistics."""

    if len(values) <= maximum:
        return values
    positions = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return values.iloc[positions]


def adf_result(values: pd.Series) -> dict[str, float | int | str | bool]:
    """Run an ADF test; reject the unit-root null at the 5% significance level."""

    sampled = sample_series(values, MAX_ADF_OBSERVATIONS)
    statistic, p_value, used_lags, observations, critical_values, _ = adfuller(
        sampled.to_numpy(), autolag="AIC"
    )
    stationary = p_value < 0.05
    return {
        "adf_statistic": statistic,
        "adf_p_value": p_value,
        "adf_used_lags": used_lags,
        "adf_observations": observations,
        "adf_critical_value_5pct": critical_values["5%"],
        "stationary_at_5pct": stationary,
        "stationarity_label": "stationary" if stationary else "non_stationary",
    }


def autocorrelation_result(values: pd.Series) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray]:
    """Calculate ACF/PACF and label the dominant short-lag dependence."""

    sampled = sample_series(values, MAX_CORRELATION_OBSERVATIONS)
    allowed_lags = min(MAX_LAGS, max(1, len(sampled) // 10 - 1))
    acf_values = acf(sampled.to_numpy(), nlags=allowed_lags, fft=True)
    pacf_values = pacf(sampled.to_numpy(), nlags=allowed_lags, method="ywm")
    confidence = 1.96 / math.sqrt(len(sampled))
    significant_lags = np.flatnonzero(np.abs(acf_values[1:]) > confidence) + 1
    if not len(significant_lags):
        label = "no_significant_short_lag_autocorrelation"
    elif abs(acf_values[1]) >= confidence:
        label = "positive_lag1_autocorrelation" if acf_values[1] > 0 else "negative_lag1_autocorrelation"
    else:
        label = "higher_lag_autocorrelation"
    return (
        {
            "acf_lag_1": acf_values[1] if len(acf_values) > 1 else np.nan,
            "pacf_lag_1": pacf_values[1] if len(pacf_values) > 1 else np.nan,
            "acf_confidence_bound_95pct": confidence,
            "significant_acf_lags": ";".join(map(str, significant_lags.tolist())),
            "autocorrelation_label": label,
        },
        acf_values,
        pacf_values,
    )


def plot_diagnostics(name: str, source: str, returns: pd.Series, acf_values: np.ndarray, pacf_values: np.ndarray) -> None:
    """Write one raw-return plot plus its ACF/PACF diagnostic panel per variety."""

    plot_name = f"{source.lower()}_{name}"
    figure, axis = plt.subplots(figsize=(14, 4.5))
    axis.plot(returns.index, returns.to_numpy(), linewidth=0.45, color="#176b87")
    axis.axhline(0, color="#333333", linewidth=0.6)
    axis.set(title=f"{source} {name}: raw 15-minute log returns", xlabel="Timestamp", ylabel="Log return")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "return_series" / f"{plot_name}_returns.png", dpi=160)
    plt.close(figure)

    lags = np.arange(len(acf_values))
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for axis, values, title in zip(axes, (acf_values, pacf_values), ("ACF", "PACF"), strict=True):
        axis.vlines(lags, 0, values, color="#c65d07", linewidth=1.2)
        axis.scatter(lags, values, color="#c65d07", s=10)
        axis.axhline(0, color="#333333", linewidth=0.6)
        axis.set(title=f"{source} {name}: {title}", xlabel="Lag", ylabel="Correlation")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "acf_pacf" / f"{plot_name}_acf_pacf.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Analyze all node tensors and persist reports, charts, and reproducible metadata."""

    (OUTPUT_DIR / "return_series").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "acf_pacf").mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    sources = {"SMM": SMM_DIR}
    for source, directory in sources.items():
        for path in sorted(directory.glob("*.csv")):
            try:
                returns_15m, returns_daily, price_column = load_returns(path, source)
            except ValueError as error:
                excluded.append({"source": source, "variety": path.stem, "reason": str(error)})
                print(f"Skipped {source} {path.stem}: {error}")
                continue
            if len(returns_15m) < MAX_LAGS + 10:
                raise ValueError(f"Too few valid 15-minute returns for {path}.")
            adf_metrics = adf_result(returns_15m)
            autocorrelation_metrics, acf_values, pacf_values = autocorrelation_result(returns_15m)
            record: dict[str, object] = {
                "source": source,
                "variety": path.stem,
                "raw_price_column": price_column,
                "observations_15m": len(returns_15m),
                "mean_return_15m": returns_15m.mean(),
                "std_return_15m": returns_15m.std(ddof=1),
                "skewness_15m": skew(returns_15m, bias=False),
                "excess_kurtosis_15m": kurtosis(returns_15m, fisher=True, bias=False),
                "observations_daily": len(returns_daily),
                "mean_return_daily": returns_daily.mean(),
                "std_return_daily": returns_daily.std(ddof=1),
                **adf_metrics,
                **autocorrelation_metrics,
            }
            records.append(record)
            plot_diagnostics(path.stem, source, returns_15m, acf_values, pacf_values)
            print(f"Completed {source} {path.stem}: {len(returns_15m)} 15-minute returns")

    summary = pd.DataFrame(records).sort_values(["source", "variety"])
    summary.to_csv(OUTPUT_DIR / "return_series_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(excluded, columns=["source", "variety", "reason"]).to_csv(
        OUTPUT_DIR / "excluded_series.csv", index=False, encoding="utf-8-sig"
    )
    non_stationary = summary.loc[~summary["stationary_at_5pct"], ["source", "variety", "adf_p_value"]]
    report_lines = [
        "# Return Series Analysis",
        "",
        "## Method",
        "",
        "Each variety uses the first raw, unsuffixed price column in its node tensor. 15-minute and daily log returns are recomputed from that raw price series. ADF uses the unit-root null hypothesis and flags p-values at or above 0.05 as non-stationary. ACF/PACF diagnostics use up to 40 lags; long series are evenly sampled only for ADF/ACF/PACF estimation.",
        "",
        "## Coverage",
        "",
        f"- Varieties analyzed: {len(summary)}",
        f"- Excluded varieties with no raw price series: {len(excluded)}",
        f"- SMM varieties: {(summary['source'] == 'SMM').sum()}",
        f"- Futures varieties: {(summary['source'] == 'FUTURES').sum()}",
        f"- Non-stationary 15-minute return series at 5%: {len(non_stationary)}",
        "",
        "## Non-stationary Series",
        "",
    ]
    if non_stationary.empty:
        report_lines.append("None. All ADF tests reject the unit-root null at the 5% level.")
    else:
        report_lines.extend(
            f"- {row.source} {row.variety}: ADF p-value={row.adf_p_value:.6g}"
            for row in non_stationary.itertuples(index=False)
        )
    if excluded:
        report_lines.extend(
            [
                "",
                "## Excluded Series",
                "",
                *(
                    f"- {entry['source']} {entry['variety']}: {entry['reason']}"
                    for entry in excluded
                ),
            ]
        )
    report_lines.extend(
        [
            "",
            "## Files",
            "",
            "- `return_series_summary.csv`: complete descriptive statistics, ADF results, and autocorrelation labels.",
            "- `excluded_series.csv`: node tensors omitted because they do not contain an original raw price series.",
            "- `return_series/`: one raw 15-minute return time-series chart for each variety.",
            "- `acf_pacf/`: one ACF/PACF chart for each variety.",
        ]
    )
    (OUTPUT_DIR / "return_series_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary)} varieties to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()