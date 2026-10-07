"""Utility functions for constructing technical features from 15-minute futures bars."""

from __future__ import annotations

import numpy as np
import pandas as pd


HORIZON_BARS = {
	"30m": 2,
	"1h": 4,
	"2h": 8,
	"4h": 16,
}

FUNDAMENTAL_FACTOR_COLUMNS = (
	"ma_5",
	"ma_10",
	"ma_20",
	"momentum_10",
	"rsi_14",
	"macd",
	"macd_signal",
	"macd_hist",
	"atr_14",
	"rolling_std_20",
	"rolling_skew_20",
)

INVENTORY_FACTOR_COLUMNS = (
	"on_warrant",
	"effective_forecast",
	"deliverable",
)


def calculate_wilder_average(values: pd.Series, period: int) -> pd.Series:
	"""Return Wilder's moving average with an initial simple average."""
	if not isinstance(values, pd.Series):
		raise TypeError("values must be a pandas Series.")
	if not isinstance(period, int) or isinstance(period, bool):
		raise TypeError("period must be an integer.")
	if period < 1:
		raise ValueError("period must be positive.")

	numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
	result = np.full(len(numeric_values), np.nan, dtype=float)
	previous = np.nan
	consecutive_values: list[float] = []
	for position, value in enumerate(numeric_values):
		if not np.isfinite(value):
			previous = np.nan
			consecutive_values.clear()
			continue
		if np.isnan(previous):
			consecutive_values.append(value)
			if len(consecutive_values) < period:
				continue
			previous = float(np.mean(consecutive_values[-period:]))
		else:
			previous = (previous * (period - 1) + value) / period
		result[position] = previous
	return pd.Series(result, index=values.index, name=values.name)


def calculate_fundamental_factors(
	data: pd.DataFrame,
	*,
	close_column: str = "close",
	high_column: str = "high",
	low_column: str = "low",
) -> pd.DataFrame:
	"""Append the approved bar-frequency price factors to 15-minute bars."""
	if not isinstance(data, pd.DataFrame):
		raise TypeError("data must be a pandas DataFrame.")
	required_columns = {close_column, high_column, low_column}
	missing_columns = sorted(required_columns - set(data.columns))
	if missing_columns:
		raise ValueError(f"data is missing required columns: {missing_columns}")

	features = data.copy()
	close = pd.to_numeric(features[close_column], errors="coerce").where(lambda values: values > 0)
	high = pd.to_numeric(features[high_column], errors="coerce")
	low = pd.to_numeric(features[low_column], errors="coerce")
	bar_return = close.pct_change(fill_method=None)

	for period in (5, 10, 20):
		features[f"ma_{period}"] = close.rolling(period, min_periods=period).mean()
	features["momentum_10"] = close.pct_change(periods=10, fill_method=None)

	close_change = close.diff()
	average_gain = calculate_wilder_average(close_change.clip(lower=0), 14)
	average_loss = calculate_wilder_average(-close_change.clip(upper=0), 14)
	rsi = 100 - 100 / (1 + average_gain / average_loss)
	rsi = rsi.mask(average_loss.eq(0) & average_gain.gt(0), 100.0)
	rsi = rsi.mask(average_gain.eq(0) & average_loss.gt(0), 0.0)
	features["rsi_14"] = rsi.mask(average_gain.eq(0) & average_loss.eq(0), 50.0)

	ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
	ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
	features["macd"] = ema_12 - ema_26
	features["macd_signal"] = features["macd"].ewm(
		span=9, adjust=False, min_periods=9
	).mean()
	features["macd_hist"] = features["macd"] - features["macd_signal"]

	previous_close = close.shift(1)
	true_range = pd.concat(
		[
			high - low,
			(high - previous_close).abs(),
			(low - previous_close).abs(),
		],
		axis=1,
	).max(axis=1, skipna=True)
	features["atr_14"] = calculate_wilder_average(true_range, 14)

	rolling = bar_return.rolling(20, min_periods=20)
	rolling_mean = rolling.mean()
	second_raw_moment = bar_return.pow(2).rolling(20, min_periods=20).mean()
	third_raw_moment = bar_return.pow(3).rolling(20, min_periods=20).mean()
	variance = (second_raw_moment - rolling_mean.pow(2)).clip(lower=0)
	standard_deviation = variance.pow(0.5)
	third_central_moment = (
		third_raw_moment
		- 3 * rolling_mean * second_raw_moment
		+ 2 * rolling_mean.pow(3)
	)
	features["rolling_std_20"] = rolling.std(ddof=0)
	rolling_skew = third_central_moment / standard_deviation.pow(3)
	features["rolling_skew_20"] = rolling_skew.mask(standard_deviation.eq(0), 0.0)

	features[list(FUNDAMENTAL_FACTOR_COLUMNS)] = features[
		list(FUNDAMENTAL_FACTOR_COLUMNS)
	].replace([np.inf, -np.inf], np.nan)
	return features


def merge_daily_inventory_factors(
	bars: pd.DataFrame,
	inventory: pd.DataFrame,
	*,
	trading_date_column: str = "trading_date",
	inventory_date_column: str = "date",
	lag_trading_days: int = 1,
) -> pd.DataFrame:
	"""Merge inventory values after lagging on the bars' trading-day calendar."""
	if not isinstance(bars, pd.DataFrame) or not isinstance(inventory, pd.DataFrame):
		raise TypeError("bars and inventory must be pandas DataFrames.")
	if not isinstance(lag_trading_days, int) or isinstance(lag_trading_days, bool):
		raise TypeError("lag_trading_days must be an integer.")
	if lag_trading_days < 0:
		raise ValueError("lag_trading_days must be nonnegative.")
	if trading_date_column not in bars.columns:
		raise ValueError(f"bars is missing required column: {trading_date_column}")
	required_inventory_columns = {
		inventory_date_column,
		"on_warrant",
		"effective_forecast",
	}
	missing_columns = sorted(required_inventory_columns - set(inventory.columns))
	if missing_columns:
		raise ValueError(f"inventory is missing required columns: {missing_columns}")

	features = bars.copy()
	bar_dates = pd.to_datetime(features[trading_date_column], errors="coerce").dt.normalize()
	if bar_dates.isna().any():
		raise ValueError(f"bars contains invalid {trading_date_column} values.")

	daily_inventory = inventory.loc[
		:, [inventory_date_column, "on_warrant", "effective_forecast"]
	].copy()
	daily_inventory[inventory_date_column] = pd.to_datetime(
		daily_inventory[inventory_date_column].astype("string").str.strip(),
		errors="coerce",
	).dt.normalize()
	if daily_inventory[inventory_date_column].isna().any():
		raise ValueError(f"inventory contains invalid {inventory_date_column} values.")
	if daily_inventory[inventory_date_column].duplicated().any():
		raise ValueError("inventory contains duplicate dates.")
	for column in ("on_warrant", "effective_forecast"):
		daily_inventory[column] = pd.to_numeric(daily_inventory[column], errors="coerce")

	trading_calendar = pd.DataFrame(
		{trading_date_column: pd.Index(bar_dates.unique()).sort_values()}
	)
	daily_inventory = daily_inventory.rename(columns={inventory_date_column: trading_date_column})
	lagged_inventory = trading_calendar.merge(
		daily_inventory, on=trading_date_column, how="left", validate="one_to_one"
	)
	value_columns = ["on_warrant", "effective_forecast"]
	lagged_inventory[value_columns] = lagged_inventory[value_columns].shift(lag_trading_days)
	lagged_inventory["deliverable"] = lagged_inventory["on_warrant"] + lagged_inventory[
		"effective_forecast"
	]

	features[trading_date_column] = bar_dates
	features = features.drop(columns=list(INVENTORY_FACTOR_COLUMNS), errors="ignore").merge(
		lagged_inventory,
		on=trading_date_column,
		how="left",
		validate="many_to_one",
		sort=False,
	)
	return features


def calculate_technical_indicators(
	data: pd.DataFrame,
	*,
	bars_per_trading_day: int = 16,
	close_column: str = "close",
	high_column: str = "high",
	low_column: str = "low",
	open_column: str = "open",
	open_interest_column: str = "open_interest",
	turnover_column: str = "total_turnover",
) -> pd.DataFrame:
	"""Append technical indicators to sorted 15-minute futures bars.

	Returns use the existing 15-minute convention: ``close_t / close_{t-k} - 1``.
	The one-day horizon is a configurable number of 15-minute trading bars because
	Chinese futures trading sessions differ by product and include session breaks.
	Rolling geometric return is ``prod(1 + r_15m) ** (1 / n) - 1``; rolling
	volatility is realized volatility, ``sqrt(sum(r_15m ** 2))`` over the
	same 15-minute return window.
	Invalid divisions and non-positive prices/open interest become ``NaN``.
	"""
	if not isinstance(data, pd.DataFrame):
		raise TypeError("data must be a pandas DataFrame.")
	if not isinstance(bars_per_trading_day, int) or isinstance(bars_per_trading_day, bool):
		raise TypeError("bars_per_trading_day must be an integer.")
	if bars_per_trading_day < 1:
		raise ValueError("bars_per_trading_day must be positive.")

	required_columns = {
		close_column,
		high_column,
		low_column,
		open_column,
		open_interest_column,
		turnover_column,
	}
	missing_columns = sorted(required_columns - set(data.columns))
	if missing_columns:
		raise ValueError(f"data is missing required columns: {missing_columns}")

	features = data.copy()
	close = pd.to_numeric(features[close_column], errors="coerce").where(lambda values: values > 0)
	high = pd.to_numeric(features[high_column], errors="coerce")
	low = pd.to_numeric(features[low_column], errors="coerce")
	open_price = pd.to_numeric(features[open_column], errors="coerce").where(lambda values: values > 0)
	open_interest = pd.to_numeric(features[open_interest_column], errors="coerce").where(lambda values: values > 0)
	turnover = pd.to_numeric(features[turnover_column], errors="coerce")

	return_15m = close.pct_change(fill_method=None)
	features["return_15m"] = return_15m

	horizon_bars = {**HORIZON_BARS, "1d": bars_per_trading_day}
	for horizon, bars in horizon_bars.items():
		features[f"return_{horizon}"] = close.pct_change(periods=bars, fill_method=None)

	for horizon, bars in horizon_bars.items():
		features[f"geometric_mean_return_{horizon}"] = (1 + return_15m).rolling(
			bars, min_periods=bars
		).apply(np.prod, raw=True).pow(1 / bars) - 1
		features[f"rolling_volatility_{horizon}"] = return_15m.pow(2).rolling(
			bars, min_periods=bars
		).sum().pow(0.5)

	features["high_low_range"] = (high - low) / close
	features["r_OC"] = np.log(close / open_price)
	features["delta_open_interest"] = open_interest.diff()
	features["open_interest_pct_change"] = open_interest.pct_change(fill_method=None)
	features["turnover_to_open_interest"] = turnover / open_interest

	feature_columns = [
		"return_15m",
		*[f"return_{horizon}" for horizon in horizon_bars],
		*[f"geometric_mean_return_{horizon}" for horizon in horizon_bars],
		*[f"rolling_volatility_{horizon}" for horizon in horizon_bars],
		"high_low_range",
		"r_OC",
		"delta_open_interest",
		"open_interest_pct_change",
		"turnover_to_open_interest",
	]
	features[feature_columns] = features[feature_columns].replace([np.inf, -np.inf], np.nan)
	return features
