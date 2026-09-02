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
