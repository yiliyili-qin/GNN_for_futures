"""Causal History graph dataset used by the frozen baseline and ablations."""
from dataclasses import replace
import numpy as np
import torch

class HistoricalGraphDataset:
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def graph_for_cutoff(self, cutoff):
        panel = self.base.panel
        size = len(panel.node_order)
        weights = np.zeros((size, size), dtype=np.float32)
        available = np.zeros((size, size), dtype=bool)
        previous = cutoff - 1
        if previous >= 0 and panel.timestamps[cutoff] - panel.timestamps[previous] == np.timedelta64(15, "m") and panel.trading_dates[previous] == panel.trading_dates[cutoff]:
            valid = self.base.input_valid
            available = valid[previous, :, None] & valid[cutoff, None, :] & ~np.eye(size, dtype=bool)
            threshold = self.base.threshold_matrix[previous]
            source = np.abs(panel.signed_returns[previous]) >= threshold
            target = np.abs(panel.signed_returns[cutoff]) >= threshold
            weights = (source[:, None] & target[None, :] & available).astype(np.float32)
        return torch.from_numpy(weights), torch.from_numpy(available)

    def __getitem__(self, index):
        sample = self.base[index]
        weights, available = self.graph_for_cutoff(sample.panel_index)
        return replace(sample, regression_weight=weights, regression_available=available)
