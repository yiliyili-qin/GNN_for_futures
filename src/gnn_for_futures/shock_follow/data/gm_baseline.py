"""Causal realized-edge histories for the GraphMixer classification baseline."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from gnn_for_futures.shock_follow.data.labels import build_shock_follow_labels_for_cutoff
from gnn_for_futures.shock_follow.data.panel import ShockFollowBarPanel
from gnn_for_futures.shock_follow.data.prepared import ShockFollowPreparedData


class CausalEventHistory:
    """Index all realized O/S edges, including events without a training sample.

    A source at t and follower at t+15m form an event available at t+15m,
    never at t. Attributes are the two endpoints' nine features at source t.
    Sampling is undirected, but attributes retain original source/target order.
    """

    def __init__(self, panel: ShockFollowBarPanel, prepared: ShockFollowPreparedData) -> None:
        if prepared.panel_materialization_sha256 != panel.materialization_sha256:
            raise ValueError("event history and prepared panel identities differ")
        self.panel = panel
        self.prepared = prepared
        source_rows, sources, targets = [], [], []
        at, dates = panel.timestamps, panel.trading_dates
        adjacent = np.flatnonzero(
            (np.diff(at) == np.timedelta64(15, "m")) & (dates[:-1] == dates[1:])
        )
        off_diagonal = ~np.eye(len(panel.node_order), dtype=bool)
        for row in adjacent:
            threshold = prepared.threshold_matrix[row]
            valid = prepared.input_valid[row]
            leaders = valid & (np.abs(panel.signed_returns[row]) >= threshold)
            followers = valid & np.isfinite(panel.signed_returns[row + 1]) & (
                np.abs(panel.signed_returns[row + 1]) >= threshold
            )
            source, target = np.nonzero(leaders[:, None] & followers[None, :] & off_diagonal)
            if len(source):
                source_rows.append(np.full(len(source), row, dtype=np.int64))
                sources.append(source)
                targets.append(target)
        self.source_rows = np.concatenate(source_rows) if source_rows else np.empty(0, np.int64)
        self.realized_rows = self.source_rows + 1
        self.sources = np.concatenate(sources) if sources else np.empty(0, np.int64)
        self.targets = np.concatenate(targets) if targets else np.empty(0, np.int64)
        self.incident = [np.flatnonzero((self.sources == node) | (self.targets == node))
                         for node in range(len(panel.node_order))]

    def snapshot(self, cutoff: int, history_start: int, max_events: int, structure_events: int):
        """Return recent per-node events and the last S global-event adjacency.

        No event may start before the query's existing 40-bar history boundary.
        Events realized exactly at the query close are observable and included.
        Simultaneous events are deterministically ordered by source then target.
        """
        if not 0 <= history_start <= cutoff < len(self.panel.timestamps):
            raise ValueError("invalid query/history indices")
        if max_events < 1 or structure_events < 1:
            raise ValueError("event capacities must be positive")
        nodes, features = len(self.panel.node_order), self.prepared.node_features.shape[-1]
        edge_features = np.zeros((nodes, max_events, 2 * features), dtype=np.float32)
        ages = np.zeros((nodes, max_events), dtype=np.float32)
        mask = np.zeros((nodes, max_events), dtype=bool)
        adjacency = np.zeros((nodes, nodes), dtype=bool)
        lower = int(np.searchsorted(self.source_rows, history_start, side="left"))
        upper = int(np.searchsorted(self.realized_rows, cutoff, side="right"))
        node_features = self.prepared.node_features
        for node, incident in enumerate(self.incident):
            left, right = np.searchsorted(incident, [lower, upper])
            events = incident[max(left, right - max_events):right]
            count = len(events)
            rows = self.source_rows[events]
            edge_features[node, :count] = np.concatenate((
                node_features[rows, self.sources[events]],
                node_features[rows, self.targets[events]],
            ), axis=-1)
            elapsed = self.panel.timestamps[cutoff] - self.panel.timestamps[self.realized_rows[events]]
            ages[node, :count] = elapsed / np.timedelta64(15, "m")
            mask[node, :count] = True
        recent = slice(max(lower, upper - structure_events), upper)
        source, target = self.sources[recent], self.targets[recent]
        adjacency[source, target] = True
        adjacency[target, source] = True
        return dict(edge_features=edge_features, edge_ages=ages, edge_mask=mask, adjacency=adjacency)


class GMBaselineDataset(Dataset):
    """Reuse exact prepared cutoff/label/mask memberships without graph dependencies."""

    def __init__(
        self, events: CausalEventHistory, cutoffs: np.ndarray, histories: np.ndarray,
        target_class: np.ndarray, loss_mask: np.ndarray, *, nominal_start: str,
        node_input: str = "identity", max_events: int = 50, structure_events: int = 55,
    ) -> None:
        if node_input not in ("identity", "current"):
            raise ValueError("node_input must be identity (GM) or current (GM-TNF-style)")
        self.events = events
        self.cutoffs = np.asarray(cutoffs)
        self.histories = np.asarray(histories)
        self.targets = np.asarray(target_class)
        self.masks = np.asarray(loss_mask)
        self.node_input, self.max_events, self.structure_events = node_input, max_events, structure_events
        panel, prepared = events.panel, events.prepared
        nodes, samples = len(panel.node_order), len(cutoffs)
        if self.cutoffs.dtype.kind not in "iu" or self.histories.dtype.kind not in "iu":
            raise ValueError("cutoffs and histories must be integer panel indices")
        if not samples or self.cutoffs.shape != (samples,) or np.any(np.diff(self.cutoffs) <= 0):
            raise ValueError("cutoffs must be nonempty, increasing and unique")
        if self.histories.shape != (samples, prepared.sequence_length):
            raise ValueError("history shape must match the existing sequence length")
        if (self.histories < 0).any() or (self.histories >= len(panel.timestamps)).any():
            raise ValueError("history indices outside panel")
        if np.any(np.diff(self.histories, axis=1) <= 0) or not np.array_equal(self.histories[:, -1], self.cutoffs):
            raise ValueError("histories must increase and end at their cutoff")
        start = np.datetime64(nominal_start)
        if np.any(panel.timestamps[self.histories[:, 0]] <= start) or np.any(
            panel.trading_dates[self.histories[:, 0]] < start.astype("datetime64[D]")
        ):
            raise ValueError("history crosses nominal split boundary")
        if self.targets.shape != (samples, nodes, nodes) or self.masks.shape != self.targets.shape:
            raise ValueError("labels/masks must match cutoff source-target cells")
        if self.masks.dtype != np.bool_ or np.diagonal(self.masks, axis1=1, axis2=2).any():
            raise ValueError("supervision must be boolean and exclude self pairs")
        if not self.masks.any():
            raise ValueError("split must contain supervised pairs")
        # Replay the existing label function, rather than inventing GM-specific negatives.
        for item, cutoff in enumerate(self.cutoffs):
            labels = build_shock_follow_labels_for_cutoff(
                panel.timestamps, panel.trading_dates, panel.signed_returns, panel.node_order,
                prepared.threshold_matrix[cutoff], panel.timestamps[cutoff],
                source_input_valid=prepared.input_valid, target_input_valid=prepared.input_valid,
                validate_timestamps=False,
            )
            if not np.array_equal(labels.target_class, self.targets[item]) or not np.array_equal(
                labels.loss_mask, self.masks[item]
            ):
                raise ValueError(f"prepared label/mask mismatch at cutoff {cutoff}")
        self.identity = np.eye(nodes, dtype=np.float32)

    def __len__(self):
        return len(self.cutoffs)

    def __getitem__(self, item):
        cutoff = int(self.cutoffs[item])
        result = self.events.snapshot(cutoff, int(self.histories[item, 0]), self.max_events, self.structure_events)
        result["node_features"] = (
            self.identity.copy() if self.node_input == "identity"
            else self.events.prepared.node_features[cutoff].copy()
        )
        result["target_class"] = self.targets[item].astype(np.int64, copy=True)
        result["loss_mask"] = self.masks[item].copy()
        result["sample_index"] = item
        return result
