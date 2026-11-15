"""Retrospective daily regressions for the approved 20261010 protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def validate_daily_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Daily data require a DatetimeIndex")
    if frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Trading dates must be unique, sorted, and nonmissing")
    if not frame.index.equals(frame.index.normalize()):
        raise ValueError("Trading dates must be normalized")
    if not frame.columns.is_unique or frame.empty:
        raise ValueError("Daily data require unique nodes and nonempty rows")


def trading_day_closes(bars: pd.DataFrame) -> pd.Series:
    required = {"datetime", "trading_date", "close"}
    if not required.issubset(bars.columns):
        raise ValueError("Bars require datetime, trading_date, close")
    frame = bars.loc[:, ["datetime", "trading_date", "close"]].copy()
    for column in ("datetime", "trading_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    if frame[["datetime", "trading_date"]].isna().any().any() or frame.datetime.duplicated().any():
        raise ValueError("Bar timestamps must be unique and dates nonmissing")
    frame["trading_date"] = frame.trading_date.dt.normalize()
    last = frame.sort_values("datetime").groupby("trading_date", sort=True).tail(1)
    return pd.to_numeric(last.set_index("trading_date")["close"], errors="raise").sort_index()


def daily_close_returns(closes: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Use an explicit trading calendar; never bridge or fill missing closes."""
    validate_daily_frame(closes)
    aligned = closes.reindex(calendar)
    validate_daily_frame(aligned)
    valid = aligned.where(np.isfinite(aligned) & aligned.gt(0.0))
    return valid.div(valid.shift(1)).sub(1.0)


def fit_daily_graph(returns: pd.DataFrame, forecast_date: str | pd.Timestamp) -> pd.DataFrame:
    """Fit both directions using exactly 20 prior calendar rows and 17 targets."""
    validate_daily_frame(returns)
    forecast = pd.Timestamp(forecast_date)
    if pd.isna(forecast) or forecast != forecast.normalize():
        raise ValueError("forecast_date must be a normalized trading date")
    window = returns.loc[returns.index < forecast].tail(20)
    records = []
    for source in returns.columns:
        for target in returns.columns:
            if source == target:
                continue
            record = dict(
                forecast_date=forecast, source=source, target=target,
                window_start=window.index.min(), train_end_date=window.index.max(),
                available=False, reason="insufficient_history", selected_lag=0,
                nobs=0, df_resid=0, bic=np.nan, condition_number=np.nan,
                beta_1=np.nan, beta_2=np.nan, abs_beta_sum=np.nan,
                beta_sum=np.nan, wald_p_value=np.nan,
            )
            records.append(record)
            if len(window) != 20:
                continue
            pair = window.loc[:, [source, target]].to_numpy(dtype=float)
            if not np.isfinite(pair).all():
                record["reason"] = "missing_returns"
                continue
            if np.any(np.ptp(pair, axis=0) == 0.0):
                record["reason"] = "zero_variance"
                continue
            candidates = []
            target_values = pair[3:, 1]
            for lag_order in (1, 2):
                design = np.column_stack(
                    [np.ones(17)]
                    + [pair[3 - lag:20 - lag, 1] for lag in range(1, lag_order + 1)]
                    + [pair[3 - lag:20 - lag, 0] for lag in range(1, lag_order + 1)]
                )
                if np.linalg.matrix_rank(design) != design.shape[1]:
                    continue
                try:
                    fitted = sm.OLS(target_values, design, missing="raise").fit()
                    restriction = np.eye(design.shape[1])[1 + lag_order:]
                    wald = fitted.wald_test(restriction, use_f=False, scalar=True)
                    cross = fitted.params[1 + lag_order:]
                    diagnostics = [fitted.bic, fitted.condition_number, float(wald.pvalue)]
                    if not np.isfinite(diagnostics).all() or not np.isfinite(cross).all():
                        continue
                    candidates.append((float(fitted.bic), lag_order, fitted, cross, float(wald.pvalue)))
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    continue
            if not candidates:
                record["reason"] = "invalid_fit"
                continue
            bic, lag_order, fitted, cross, p_value = min(candidates, key=lambda item: item[:2])
            record.update(
                available=True, reason="fitted", selected_lag=lag_order,
                nobs=int(fitted.nobs), df_resid=int(fitted.df_resid), bic=bic,
                condition_number=float(fitted.condition_number), beta_1=float(cross[0]),
                beta_2=float(cross[1]) if lag_order == 2 else 0.0,
                abs_beta_sum=float(np.abs(cross).sum()), beta_sum=float(cross.sum()),
                wald_p_value=p_value,
            )
    return pd.DataFrame.from_records(records)