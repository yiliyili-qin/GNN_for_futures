"""Audited validity-aware 15-minute bar panel for shock-follow V1."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from gnn_for_futures.shock_follow.contract import (
    canonical_json_sha256,
    node_order_sha256,
    validate_node_order,
)


BAR_PANEL_CONTRACT_VERSION = "validity_aware_exact_15m_returns_v1"
BAR_COLUMNS = (
    "underlying_symbol",
    "datetime",
    "trading_date",
    "dominant_id",
    "close",
    "volume",
    "open_interest",
)


def bar_panel_contract_payload() -> dict[str, object]:
    """Return the frozen rules used to reconstruct non-z-scored returns."""
    return {
        "bar_panel_contract_version": BAR_PANEL_CONTRACT_VERSION,
        "timestamp_order": "strictly ascending and unique within each symbol",
        "return": "current_close / preceding_close - 1",
        "preceding_endpoint": "observed row exactly 15 minutes before current row",
        "trading_date": "current and preceding rows have identical exchange trading_date",
        "valid_endpoint": "finite close, finite strictly positive volume, non-missing dominant_id",
        "roll_transition": "current dominant_id must equal preceding dominant_id",
        "invalid_encoding": "NaN signed_return and open_interest_change",
        "row_observed": "true only when the symbol source file contains that exact timestamp",
        "alignment": "union of authoritative node timestamps; no inferred or forward-filled bars",
        "data_scope": "retrospective mutable history research only",
    }


def bar_panel_contract_sha256() -> str:
    """Hash the exact return reconstruction contract."""
    return canonical_json_sha256(bar_panel_contract_payload())


def _array_digest_update(digest, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())


@dataclass(frozen=True)
class SymbolBarSource:
    """Immutable source-file facts retained with a materialized panel."""

    symbol: str
    path: str
    sha256: str
    size_bytes: int
    row_count: int
    first_timestamp: str
    last_timestamp: str

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "row_count": self.row_count,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
        }


@dataclass(frozen=True)
class ShockFollowBarPanel:
    """Union-aligned raw fields and exact validity-aware returns."""

    node_order: tuple[str, ...]
    timestamps: np.ndarray
    trading_dates: np.ndarray
    signed_returns: np.ndarray
    volumes: np.ndarray
    open_interest_changes: np.ndarray
    row_observed: np.ndarray
    bar_available: np.ndarray
    return_valid: np.ndarray
    source_files: tuple[SymbolBarSource, ...]
    bar_contract_sha256: str
    materialization_sha256: str

    def validate(self) -> None:
        node_count = len(self.node_order)
        row_count = len(self.timestamps)
        if self.timestamps.shape != (row_count,) or self.trading_dates.shape != (row_count,):
            raise ValueError("timestamps and trading_dates must be one-dimensional")
        if np.isnat(self.timestamps).any() or np.isnat(self.trading_dates).any():
            raise ValueError("panel timestamps and trading_dates must be valid")
        if row_count and np.any(self.timestamps[1:] <= self.timestamps[:-1]):
            raise ValueError("panel timestamps must be strictly ascending and unique")
        expected = (row_count, node_count)
        for name, values in (
            ("signed_returns", self.signed_returns),
            ("volumes", self.volumes),
            ("open_interest_changes", self.open_interest_changes),
            ("row_observed", self.row_observed),
            ("bar_available", self.bar_available),
            ("return_valid", self.return_valid),
        ):
            if values.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if (
            self.row_observed.dtype != np.bool_
            or self.bar_available.dtype != np.bool_
            or self.return_valid.dtype != np.bool_
        ):
            raise TypeError("row_observed, bar_available, and return_valid must be boolean")
        if (self.bar_available & ~self.row_observed).any():
            raise ValueError("available bars must be observed rows")
        if not np.array_equal(self.return_valid, np.isfinite(self.signed_returns)):
            raise ValueError("return_valid must exactly match finite signed_returns")
        if np.isfinite(self.volumes[~self.bar_available]).any():
            raise ValueError("unavailable bars must not carry finite volume")
        if len(self.source_files) != node_count:
            raise ValueError("source_files must contain one entry per node")
        if tuple(source.symbol for source in self.source_files) != self.node_order:
            raise ValueError("source_files must follow node_order")
        if self.bar_contract_sha256 != bar_panel_contract_sha256():
            raise ValueError("bar panel contract hash mismatch")
        observed_hash = _panel_materialization_sha256(
            self.node_order,
            self.timestamps,
            self.trading_dates,
            self.signed_returns,
            self.volumes,
            self.open_interest_changes,
            self.row_observed,
            self.bar_available,
            self.return_valid,
            self.source_files,
            self.bar_contract_sha256,
        )
        if observed_hash != self.materialization_sha256:
            raise ValueError("bar panel materialization hash mismatch")


def _panel_materialization_sha256(
    node_order: tuple[str, ...],
    timestamps: np.ndarray,
    trading_dates: np.ndarray,
    signed_returns: np.ndarray,
    volumes: np.ndarray,
    open_interest_changes: np.ndarray,
    row_observed: np.ndarray,
    bar_available: np.ndarray,
    return_valid: np.ndarray,
    source_files: tuple[SymbolBarSource, ...],
    contract_hash: str,
) -> str:
    digest = sha256()
    metadata = {
        "bar_contract_sha256": contract_hash,
        "node_order_sha256": node_order_sha256(node_order),
        "source_files": [source.as_dict() for source in source_files],
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for values in (
        timestamps,
        trading_dates,
        signed_returns,
        volumes,
        open_interest_changes,
        row_observed,
        bar_available,
        return_valid,
    ):
        _array_digest_update(digest, values)
    return digest.hexdigest()


def _relative_source_path(path: Path, project_root: Path | None) -> str:
    if project_root is not None:
        try:
            return path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def _asia_shanghai_naive_timestamps(values: pd.Series) -> np.ndarray:
    """Preserve the analytical Asia/Shanghai clock for offset-aware inputs."""
    timestamps = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if timestamps.tz is not None:
        timestamps = timestamps.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamps.to_numpy(dtype="datetime64[ns]")


def _load_symbol_bars(
    path: Path,
    symbol: str,
    *,
    project_root: Path | None,
) -> tuple[dict[str, np.ndarray], SymbolBarSource]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"Missing regular nonsymlink authoritative 15-minute file: {path}"
        )
    source_bytes = path.read_bytes()
    frame = pd.read_csv(
        BytesIO(source_bytes),
        usecols=list(BAR_COLUMNS),
        low_memory=False,
    )
    if frame.empty:
        raise ValueError(f"{path} contains no rows")
    observed_symbols = frame["underlying_symbol"].astype("string").dropna().unique()
    if len(observed_symbols) != 1 or str(observed_symbols[0]).upper() != symbol:
        raise ValueError(f"{path} does not contain exactly underlying_symbol={symbol}")

    timestamps = _asia_shanghai_naive_timestamps(frame["datetime"])
    trading_dates = pd.to_datetime(frame["trading_date"], errors="raise").to_numpy(dtype="datetime64[D]")
    if np.isnat(timestamps).any() or np.isnat(trading_dates).any():
        raise ValueError(f"{path} contains missing timestamps or trading dates")
    if np.any(timestamps[1:] <= timestamps[:-1]):
        raise ValueError(f"{path} timestamps must be strictly ascending and unique")

    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(dtype=np.float64)
    open_interest = pd.to_numeric(frame["open_interest"], errors="coerce").to_numpy(dtype=np.float64)
    dominant = frame["dominant_id"].astype("string")
    endpoint_valid = (
        np.isfinite(close)
        & np.isfinite(volume)
        & (volume > 0.0)
        & dominant.notna().to_numpy()
        & dominant.str.strip().ne("").fillna(False).to_numpy()
    )

    previous_exact = np.zeros(len(frame), dtype=bool)
    previous_exact[1:] = timestamps[1:] - timestamps[:-1] == np.timedelta64(15, "m")
    same_trading_date = np.zeros(len(frame), dtype=bool)
    same_trading_date[1:] = trading_dates[1:] == trading_dates[:-1]
    same_contract = np.zeros(len(frame), dtype=bool)
    same_contract[1:] = (
        dominant.iloc[1:].reset_index(drop=True)
        == dominant.iloc[:-1].reset_index(drop=True)
    ).fillna(False).to_numpy()
    predecessor_valid = np.zeros(len(frame), dtype=bool)
    predecessor_valid[1:] = endpoint_valid[:-1]
    return_valid = (
        endpoint_valid
        & predecessor_valid
        & previous_exact
        & same_trading_date
        & same_contract
    )
    valid_denominator = np.zeros(len(frame), dtype=bool)
    valid_denominator[1:] = np.isfinite(close[:-1]) & (close[:-1] != 0.0)
    return_valid &= valid_denominator

    signed_returns = np.full(len(frame), np.nan, dtype=np.float64)
    valid_rows = np.flatnonzero(return_valid)
    signed_returns[valid_rows] = close[valid_rows] / close[valid_rows - 1] - 1.0
    finite_returns = np.isfinite(signed_returns)
    return_valid &= finite_returns
    signed_returns[~return_valid] = np.nan

    open_interest_changes = np.full(len(frame), np.nan, dtype=np.float64)
    oi_valid = return_valid.copy()
    oi_valid[1:] &= np.isfinite(open_interest[1:]) & np.isfinite(open_interest[:-1])
    oi_denominator = np.zeros(len(frame), dtype=bool)
    oi_denominator[1:] = open_interest[:-1] != 0.0
    oi_valid &= oi_denominator
    oi_rows = np.flatnonzero(oi_valid)
    open_interest_changes[oi_rows] = (
        open_interest[oi_rows] / open_interest[oi_rows - 1] - 1.0
    )
    open_interest_changes[~np.isfinite(open_interest_changes)] = np.nan

    valid_volume = np.where(endpoint_valid, volume, np.nan)
    source = SymbolBarSource(
        symbol=symbol,
        path=_relative_source_path(path, project_root),
        sha256=sha256(source_bytes).hexdigest(),
        size_bytes=len(source_bytes),
        row_count=len(frame),
        first_timestamp=np.datetime_as_string(timestamps[0], unit="s"),
        last_timestamp=np.datetime_as_string(timestamps[-1], unit="s"),
    )
    return {
        "timestamps": timestamps,
        "trading_dates": trading_dates,
        "signed_returns": signed_returns,
        "volumes": valid_volume,
        "open_interest_changes": open_interest_changes,
        "bar_available": endpoint_valid,
        "return_valid": return_valid,
    }, source


def materialize_shock_follow_bar_panel(
    raw_15min_dir: str | Path,
    node_order: Sequence[str],
    *,
    project_root: str | Path | None = None,
) -> ShockFollowBarPanel:
    """Load authoritative nodes and align exact returns on their timestamp union."""
    names = validate_node_order(node_order)
    source_dir = Path(raw_15min_dir)
    root = Path(project_root) if project_root is not None else None
    loaded: list[dict[str, np.ndarray]] = []
    sources: list[SymbolBarSource] = []
    for symbol in names:
        values, source = _load_symbol_bars(
            source_dir / f"{symbol}.csv", symbol, project_root=root
        )
        loaded.append(values)
        sources.append(source)

    timestamps = np.unique(np.concatenate([values["timestamps"] for values in loaded]))
    trading_dates = np.full(len(timestamps), np.datetime64("NaT", "D"))
    shape = (len(timestamps), len(names))
    signed_returns = np.full(shape, np.nan, dtype=np.float64)
    volumes = np.full(shape, np.nan, dtype=np.float64)
    open_interest_changes = np.full(shape, np.nan, dtype=np.float64)
    row_observed = np.zeros(shape, dtype=bool)
    bar_available = np.zeros(shape, dtype=bool)
    return_valid = np.zeros(shape, dtype=bool)

    for node, values in enumerate(loaded):
        positions = np.searchsorted(timestamps, values["timestamps"])
        prior_dates = trading_dates[positions]
        conflict = ~np.isnat(prior_dates) & (prior_dates != values["trading_dates"])
        if conflict.any():
            conflict_at = values["timestamps"][np.flatnonzero(conflict)[0]]
            raise ValueError(f"Conflicting trading_date at {conflict_at}")
        trading_dates[positions] = values["trading_dates"]
        signed_returns[positions, node] = values["signed_returns"]
        volumes[positions, node] = values["volumes"]
        open_interest_changes[positions, node] = values["open_interest_changes"]
        row_observed[positions, node] = True
        bar_available[positions, node] = values["bar_available"]
        return_valid[positions, node] = values["return_valid"]
    if np.isnat(trading_dates).any():
        raise AssertionError("every union timestamp must carry one trading_date")

    contract_hash = bar_panel_contract_sha256()
    source_tuple = tuple(sources)
    materialization_hash = _panel_materialization_sha256(
        names,
        timestamps,
        trading_dates,
        signed_returns,
        volumes,
        open_interest_changes,
        row_observed,
        bar_available,
        return_valid,
        source_tuple,
        contract_hash,
    )
    panel = ShockFollowBarPanel(
        names,
        timestamps,
        trading_dates,
        signed_returns,
        volumes,
        open_interest_changes,
        row_observed,
        bar_available,
        return_valid,
        source_tuple,
        contract_hash,
        materialization_hash,
    )
    panel.validate()
    return panel


def save_shock_follow_bar_panel(
    panel: ShockFollowBarPanel,
    npz_path: str | Path,
    manifest_path: str | Path,
) -> None:
    """Atomically persist a panel and its human-readable provenance manifest."""
    panel.validate()
    data_path = Path(npz_path)
    metadata_path = Path(manifest_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = data_path.with_name(data_path.name + ".tmp")
    temporary_manifest = metadata_path.with_name(metadata_path.name + ".tmp")
    with temporary_data.open("wb") as stream:
        np.savez_compressed(
            stream,
            node_order=np.asarray(panel.node_order, dtype="U16"),
            timestamps=panel.timestamps,
            trading_dates=panel.trading_dates,
            signed_returns=panel.signed_returns,
            volumes=panel.volumes,
            open_interest_changes=panel.open_interest_changes,
            row_observed=panel.row_observed,
            bar_available=panel.bar_available,
            return_valid=panel.return_valid,
        )
    manifest = {
        "bar_panel_contract_version": BAR_PANEL_CONTRACT_VERSION,
        "bar_panel_contract": bar_panel_contract_payload(),
        "bar_contract_sha256": panel.bar_contract_sha256,
        "materialization_sha256": panel.materialization_sha256,
        "node_order": list(panel.node_order),
        "node_order_sha256": node_order_sha256(panel.node_order),
        "row_count": len(panel.timestamps),
        "first_timestamp": np.datetime_as_string(panel.timestamps[0], unit="s"),
        "last_timestamp": np.datetime_as_string(panel.timestamps[-1], unit="s"),
        "source_files": [source.as_dict() for source in panel.source_files],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "bar_data_mode": "retrospective-finalized",
    }
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_data.replace(data_path)
    temporary_manifest.replace(metadata_path)


def load_shock_follow_bar_panel(
    npz_path: str | Path,
    manifest_path: str | Path,
) -> ShockFollowBarPanel:
    """Reload and fully hash-validate a persisted panel."""
    metadata = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    required = {
        "bar_contract_sha256",
        "materialization_sha256",
        "node_order",
        "source_files",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"bar panel manifest is missing fields: {missing}")
    with np.load(npz_path, allow_pickle=False) as archive:
        panel = ShockFollowBarPanel(
            tuple(str(value) for value in archive["node_order"].tolist()),
            archive["timestamps"],
            archive["trading_dates"],
            archive["signed_returns"],
            archive["volumes"],
            archive["open_interest_changes"],
            archive["row_observed"],
            archive["bar_available"],
            archive["return_valid"],
            tuple(SymbolBarSource(**value) for value in metadata["source_files"]),
            str(metadata["bar_contract_sha256"]),
            str(metadata["materialization_sha256"]),
        )
    if list(panel.node_order) != metadata["node_order"]:
        raise ValueError("bar panel NPZ and manifest node order disagree")
    panel.validate()
    return panel
