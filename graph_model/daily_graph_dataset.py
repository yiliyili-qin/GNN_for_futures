"""Trading-date lookup for retrospective daily regression graphs."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DailyGraphDataset(Dataset):
    def __init__(self, base, graph_dir: Path, node_order, fixed_scale: float | None):
        if fixed_scale is not None and (not np.isfinite(fixed_scale) or fixed_scale <= 0):
            raise ValueError("fixed_scale must be finite and positive")
        self.base = base
        self.graph_dir = Path(graph_dir)
        self.node_order = tuple(node_order)
        if len(set(self.node_order)) != len(self.node_order):
            raise ValueError("Duplicate nodes")
        self.fixed_scale = None if fixed_scale is None else float(fixed_scale)
        self.cache = {}

    def __len__(self):
        return len(self.base)

    def graph_for_date(self, trading_date):
        date = pd.Timestamp(trading_date).normalize()
        if date in self.cache:
            return self.cache[date]
        frame = pd.read_csv(self.graph_dir / f"{date:%Y-%m-%d}.csv")
        expected = {(source, target) for source in self.node_order
                    for target in self.node_order if source != target}
        actual = list(zip(frame.source, frame.target))
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Daily graph has duplicate or mismatched directed pairs")
        if not pd.to_datetime(frame.forecast_date).eq(date).all():
            raise ValueError("Daily graph forecast date mismatch")
        if not frame.available.isin([True, False]).all():
            raise ValueError("Daily graph availability must be boolean")
        fitted = frame.loc[frame.available.astype(bool)]
        train_ends = pd.to_datetime(fitted.train_end_date)
        if train_ends.isna().any() or not train_ends.lt(date).all():
            raise ValueError("Daily graph uses nonhistorical inputs")
        raw = fitted.abs_beta_sum.to_numpy(dtype=float)
        probability = fitted.wald_p_value.to_numpy(dtype=float)
        if (not np.isfinite(raw).all() or (raw < 0).any()
                or not np.isfinite(probability).all()
                or ((probability < 0) | (probability > 1)).any()):
            raise ValueError("Invalid fitted regression strengths or p values")
        positions = {symbol: index for index, symbol in enumerate(self.node_order)}
        weights = torch.zeros((len(positions), len(positions)), dtype=torch.float32)
        available = torch.zeros_like(weights, dtype=torch.bool)
        if self.fixed_scale is None:
            positive = raw[raw > 0]
            scale = float(np.quantile(positive, .75)) if positive.size else 0.
            denominator = 2 * (scale + 1e-8)
        else:
            denominator = 2 * self.fixed_scale
        strength = np.clip(raw / denominator, 0, 1) * (1 - probability)
        for row, value in zip(fitted.itertuples(), strength):
            source, target = positions[row.source], positions[row.target]
            weights[source, target] = float(value)
            available[source, target] = True
        self.cache[date] = (weights, available)
        return weights, available

    def __getitem__(self, index):
        sample = self.base[index]
        weights, available = self.graph_for_date(sample.trading_date)
        return replace(sample, regression_weight=weights.clone(),
                       regression_available=available.clone())