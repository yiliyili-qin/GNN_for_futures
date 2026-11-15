"""Causal monthly expanding raw-return shock threshold materialization."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from gnn_for_futures.shock_follow.contract import (
    MIN_ELIGIBLE_FINITE_RETURNS,
    MIN_ELIGIBLE_TRADING_DAYS,
    THRESHOLD_CONTRACT_VERSION,
    THRESHOLD_QUANTILE,
    canonical_json_sha256,
    node_order_sha256,
    threshold_contract_sha256,
    validate_node_order,
)

SOURCE_OWN = "own"
SOURCE_MARKET_FALLBACK = "market_fallback"
SOURCE_UNAVAILABLE = "unavailable"


def _timestamps(values: Sequence[object] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values).astype("datetime64[ns]")
    if result.ndim != 1 or np.isnat(result).any():
        raise ValueError(f"{name} must be a one-dimensional sequence of valid timestamps")
    return result


def _months(values: np.ndarray) -> np.ndarray:
    return values.astype("datetime64[M]")


def threshold_input_sha256(
    timestamps: np.ndarray, trading_dates: np.ndarray, returns: np.ndarray
) -> str:
    digest = sha256()
    for value in (timestamps, trading_dates, returns):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MonthlyThresholdTable:
    """Thresholds and symbol-month causal provenance by exchange-date month."""

    node_order: tuple[str, ...]
    months: np.ndarray
    month_starts: np.ndarray
    thresholds: np.ndarray
    sources: np.ndarray
    own_thresholds: np.ndarray
    finite_return_counts: np.ndarray
    distinct_trading_day_counts: np.ndarray
    fit_starts: np.ndarray
    fit_ends: np.ndarray
    market_fallbacks: np.ndarray
    input_sha256: str
    threshold_contract_sha256: str
    materialization_sha256: str

    def thresholds_for(
        self, cutoff_at: object, *, effective_trading_date: object | None = None
    ) -> np.ndarray:
        """Return the threshold vector for the normalized exchange-date month."""
        cutoff = np.asarray(cutoff_at).astype("datetime64[ns]")
        if cutoff.ndim != 0 or np.isnat(cutoff):
            raise ValueError("cutoff_at must be one valid timestamp")
        effective_date = cutoff if effective_trading_date is None else np.asarray(
            effective_trading_date
        ).astype("datetime64[D]")
        if effective_date.ndim != 0 or np.isnat(effective_date):
            raise ValueError("effective_trading_date must be one valid date")
        month = effective_date.astype("datetime64[M]")
        matches = np.flatnonzero(self.months == month)
        if len(matches) != 1:
            raise KeyError(f"No materialized threshold month for {month}")
        index = int(matches[0])
        if cutoff < self.month_starts[index]:
            raise ValueError(
                "cutoff_at precedes the selected effective threshold month"
            )
        available = np.isfinite(self.thresholds[index])
        fit_end = self.fit_ends[index].astype("datetime64[ns]")
        if available.any() and (
            np.isnat(fit_end[available]).any()
            or not (fit_end[available] < self.month_starts[index]).all()
        ):
            raise AssertionError("threshold provenance violates fit_end < month_start")
        return self.thresholds[index].copy()

    def to_frame(self) -> pd.DataFrame:
        """Return one auditable row per effective month and symbol."""
        month_count, node_count = self.thresholds.shape
        month_rows = np.repeat(np.arange(month_count), node_count)
        symbol_rows = np.tile(np.arange(node_count), month_count)
        return pd.DataFrame(
            {
                "symbol": np.asarray(self.node_order, dtype="U16")[symbol_rows],
                "effective_month": self.months[month_rows].astype(str),
                "month_start": self.month_starts[month_rows].astype("datetime64[ns]"),
                "threshold": self.thresholds[month_rows, symbol_rows],
                "source": self.sources[month_rows, symbol_rows],
                "own_threshold": self.own_thresholds[month_rows, symbol_rows],
                "finite_return_count": self.finite_return_counts[month_rows, symbol_rows],
                "distinct_trading_day_count": self.distinct_trading_day_counts[
                    month_rows, symbol_rows
                ],
                "fit_start": self.fit_starts[month_rows, symbol_rows],
                "fit_end": self.fit_ends[month_rows, symbol_rows],
                "market_fallback": self.market_fallbacks[month_rows],
            }
        )

    def validate(self) -> None:
        """Fail closed on malformed arrays or a changed materialization hash."""
        names = validate_node_order(self.node_order)
        month_count, node_count = len(self.months), len(names)
        if self.months.shape != (month_count,) or self.month_starts.shape != (month_count,):
            raise ValueError("months and month_starts must be one-dimensional")
        if np.isnat(self.months).any() or np.isnat(self.month_starts).any():
            raise ValueError("threshold months and month starts must be valid")
        if month_count and np.any(self.months[1:] <= self.months[:-1]):
            raise ValueError("threshold months must be strictly ascending and unique")
        matrix_shape = (month_count, node_count)
        for name, values in (
            ("thresholds", self.thresholds),
            ("sources", self.sources),
            ("own_thresholds", self.own_thresholds),
            ("finite_return_counts", self.finite_return_counts),
            ("distinct_trading_day_counts", self.distinct_trading_day_counts),
            ("fit_starts", self.fit_starts),
            ("fit_ends", self.fit_ends),
        ):
            if values.shape != matrix_shape:
                raise ValueError(f"{name} must have shape {matrix_shape}")
        if self.market_fallbacks.shape != (month_count,):
            raise ValueError("market_fallbacks must have one value per month")
        allowed_sources = {SOURCE_OWN, SOURCE_MARKET_FALLBACK, SOURCE_UNAVAILABLE}
        if not set(np.unique(self.sources)).issubset(allowed_sources):
            raise ValueError("threshold table contains an unknown source")
        available = np.isfinite(self.thresholds)
        if (self.thresholds[available] <= 0.0).any():
            raise ValueError("available thresholds must be strictly positive")
        if not np.array_equal(available, self.sources != SOURCE_UNAVAILABLE):
            raise ValueError("threshold availability and source disagree")
        if available.any():
            repeated_starts = np.broadcast_to(self.month_starts[:, None], matrix_shape)
            if np.isnat(self.fit_ends[available]).any() or not (
                self.fit_ends[available] < repeated_starts[available]
            ).all():
                raise ValueError("available thresholds must satisfy fit_end < month_start")
        if self.threshold_contract_sha256 != threshold_contract_sha256():
            raise ValueError("threshold contract hash mismatch")
        expected_hash = _materialization_sha256(
            names,
            self.months,
            self.month_starts,
            self.thresholds,
            self.sources,
            self.own_thresholds,
            self.finite_return_counts,
            self.distinct_trading_day_counts,
            self.fit_starts,
            self.fit_ends,
            self.market_fallbacks,
            self.input_sha256,
            self.threshold_contract_sha256,
        )
        if expected_hash != self.materialization_sha256:
            raise ValueError("threshold materialization hash mismatch")


def _materialization_payload(
    names: tuple[str, ...],
    months: np.ndarray,
    month_starts: np.ndarray,
    thresholds: np.ndarray,
    sources: np.ndarray,
    own_thresholds: np.ndarray,
    finite_counts: np.ndarray,
    day_counts: np.ndarray,
    fit_starts: np.ndarray,
    fit_ends: np.ndarray,
    fallbacks: np.ndarray,
    input_hash: str,
    contract_hash: str,
) -> dict[str, object]:
    return {
        "threshold_contract_sha256": contract_hash,
        "node_order_sha256": node_order_sha256(names),
        "months": months.astype(str).tolist(),
        "month_starts": month_starts.astype(str).tolist(),
        "thresholds": thresholds.tolist(),
        "sources": sources.tolist(),
        "own_thresholds": own_thresholds.tolist(),
        "finite_return_counts": finite_counts.tolist(),
        "distinct_trading_day_counts": day_counts.tolist(),
        "fit_starts": fit_starts.astype(str).tolist(),
        "fit_ends": fit_ends.astype(str).tolist(),
        "market_fallbacks": fallbacks.tolist(),
        "input_sha256": input_hash,
        "threshold_contract_version": THRESHOLD_CONTRACT_VERSION,
    }


def _materialization_sha256(
    names: tuple[str, ...],
    months: np.ndarray,
    month_starts: np.ndarray,
    thresholds: np.ndarray,
    sources: np.ndarray,
    own_thresholds: np.ndarray,
    finite_counts: np.ndarray,
    day_counts: np.ndarray,
    fit_starts: np.ndarray,
    fit_ends: np.ndarray,
    fallbacks: np.ndarray,
    input_hash: str,
    contract_hash: str,
) -> str:
    return canonical_json_sha256(
        _materialization_payload(
            names,
            months,
            month_starts,
            thresholds,
            sources,
            own_thresholds,
            finite_counts,
            day_counts,
            fit_starts,
            fit_ends,
            fallbacks,
            input_hash,
            contract_hash,
        )
    )


def materialize_causal_monthly_thresholds(
    timestamps: Sequence[object] | np.ndarray,
    trading_dates: Sequence[object] | np.ndarray,
    signed_returns: np.ndarray,
    node_order: Sequence[str],
    *,
    months: Sequence[object] | np.ndarray | None = None,
) -> MonthlyThresholdTable:
    """Materialize q95 thresholds using only records strictly before each month.

    ``signed_returns`` is aligned ``[timestamp, symbol]``. A symbol is eligible
    only with at least 20 distinct trading dates and 500 finite returns. An
    ineligible symbol receives that month's arithmetic mean eligible own
    threshold, or an unavailable value if no symbol is eligible.
    """
    names = validate_node_order(node_order)
    at = _timestamps(timestamps, "timestamps")
    dates = _timestamps(trading_dates, "trading_dates").astype("datetime64[D]")
    values = np.asarray(signed_returns, dtype=np.float64)
    if values.shape != (len(at), len(names)):
        raise ValueError("signed_returns must have shape [timestamps, node_order]")
    if len(dates) != len(at):
        raise ValueError("trading_dates must align with timestamps")
    if len(at) and np.any(at[1:] <= at[:-1]):
        raise ValueError("timestamps must be strictly ascending and unique")

    # The exchange-normalized trading date, rather than wall-clock timestamp,
    # selects the effective threshold month for overnight-session observations.
    if months is None:
        requested_months = np.unique(_months(dates))
    else:
        requested_months = np.unique(_months(_timestamps(months, "months")))
    date_months = _months(dates)
    starts = requested_months.astype("datetime64[ns]")
    for row, month in enumerate(requested_months):
        observed = date_months == month
        if observed.any():
            starts[row] = at[observed].min()
        elif months is not None:
            raise ValueError(f"No observed bar defines month_start for {month}")
    count_months, node_count = len(requested_months), len(names)
    threshold_values = np.full((count_months, node_count), np.nan, dtype=np.float64)
    own_values = np.full_like(threshold_values, np.nan)
    sources = np.full((count_months, node_count), SOURCE_UNAVAILABLE, dtype="U16")
    finite_counts = np.zeros((count_months, node_count), dtype=np.int64)
    day_counts = np.zeros((count_months, node_count), dtype=np.int64)
    fit_starts = np.full((count_months, node_count), np.datetime64("NaT", "ns"))
    fit_ends = np.full((count_months, node_count), np.datetime64("NaT", "ns"))
    fallbacks = np.full((count_months,), np.nan, dtype=np.float64)

    for row, month_start in enumerate(starts):
        history = at < month_start
        eligible = np.zeros(node_count, dtype=bool)
        for symbol in range(node_count):
            usable = history & np.isfinite(values[:, symbol])
            finite_counts[row, symbol] = int(usable.sum())
            day_counts[row, symbol] = len(np.unique(dates[usable]))
            if usable.any():
                fit_starts[row, symbol] = at[usable][0]
                fit_ends[row, symbol] = at[usable][-1]
            if (
                finite_counts[row, symbol] >= MIN_ELIGIBLE_FINITE_RETURNS
                and day_counts[row, symbol] >= MIN_ELIGIBLE_TRADING_DAYS
            ):
                threshold = float(
                    np.quantile(
                        np.abs(values[usable, symbol]),
                        THRESHOLD_QUANTILE,
                        method="linear",
                    )
                )
                if np.isfinite(threshold) and threshold > 0.0:
                    own_values[row, symbol] = threshold
                    threshold_values[row, symbol] = threshold
                    sources[row, symbol] = SOURCE_OWN
                    eligible[symbol] = True
        if eligible.any():
            fallback = float(np.mean(own_values[row, eligible]))
            if not np.isfinite(fallback) or fallback <= 0.0:
                raise AssertionError("eligible thresholds must yield a positive finite mean")
            fallbacks[row] = fallback
            use_fallback = ~eligible
            threshold_values[row, use_fallback] = fallback
            sources[row, use_fallback] = SOURCE_MARKET_FALLBACK
            fallback_start = fit_starts[row, eligible].min()
            fallback_end = fit_ends[row, eligible].max()
            fit_starts[row, use_fallback] = fallback_start
            fit_ends[row, use_fallback] = fallback_end

    input_hash = threshold_input_sha256(at, dates, values)
    contract_hash = threshold_contract_sha256()
    materialization_hash = _materialization_sha256(
        names,
        requested_months,
        starts,
        threshold_values,
        sources,
        own_values,
        finite_counts,
        day_counts,
        fit_starts,
        fit_ends,
        fallbacks,
        input_hash,
        contract_hash,
    )
    table = MonthlyThresholdTable(
        names, requested_months, starts, threshold_values, sources, own_values,
        finite_counts, day_counts, fit_starts, fit_ends, fallbacks, input_hash,
        contract_hash, materialization_hash,
    )
    table.validate()
    return table


def save_monthly_threshold_table(
    table: MonthlyThresholdTable,
    npz_path: str | Path,
    csv_path: str | Path,
    manifest_path: str | Path,
) -> None:
    """Atomically write numeric arrays, long-form audit rows, and provenance."""
    table.validate()
    data_path, rows_path, metadata_path = Path(npz_path), Path(csv_path), Path(manifest_path)
    for path in (data_path, rows_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = data_path.with_name(data_path.name + ".tmp")
    temporary_rows = rows_path.with_name(rows_path.name + ".tmp")
    temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
    with temporary_data.open("wb") as stream:
        np.savez_compressed(
            stream,
            node_order=np.asarray(table.node_order, dtype="U16"),
            months=table.months,
            month_starts=table.month_starts,
            thresholds=table.thresholds,
            sources=table.sources,
            own_thresholds=table.own_thresholds,
            finite_return_counts=table.finite_return_counts,
            distinct_trading_day_counts=table.distinct_trading_day_counts,
            fit_starts=table.fit_starts,
            fit_ends=table.fit_ends,
            market_fallbacks=table.market_fallbacks,
        )
    table.to_frame().to_csv(temporary_rows, index=False)
    source_counts = {
        source: int((table.sources == source).sum())
        for source in (SOURCE_OWN, SOURCE_MARKET_FALLBACK, SOURCE_UNAVAILABLE)
    }
    metadata = {
        "threshold_contract_version": THRESHOLD_CONTRACT_VERSION,
        "threshold_contract_sha256": table.threshold_contract_sha256,
        "materialization_sha256": table.materialization_sha256,
        "input_sha256": table.input_sha256,
        "node_order": list(table.node_order),
        "node_order_sha256": node_order_sha256(table.node_order),
        "month_count": len(table.months),
        "row_count": int(table.thresholds.size),
        "source_counts": source_counts,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_data.replace(data_path)
    temporary_rows.replace(rows_path)
    temporary_metadata.replace(metadata_path)


def load_monthly_threshold_table(
    npz_path: str | Path,
    manifest_path: str | Path,
) -> MonthlyThresholdTable:
    """Reload a table and verify all contract and materialization hashes."""
    metadata = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    required = {
        "threshold_contract_sha256",
        "materialization_sha256",
        "input_sha256",
        "node_order",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"threshold manifest is missing fields: {missing}")
    with np.load(npz_path, allow_pickle=False) as archive:
        table = MonthlyThresholdTable(
            tuple(str(value) for value in archive["node_order"].tolist()),
            archive["months"],
            archive["month_starts"],
            archive["thresholds"],
            archive["sources"],
            archive["own_thresholds"],
            archive["finite_return_counts"],
            archive["distinct_trading_day_counts"],
            archive["fit_starts"],
            archive["fit_ends"],
            archive["market_fallbacks"],
            str(metadata["input_sha256"]),
            str(metadata["threshold_contract_sha256"]),
            str(metadata["materialization_sha256"]),
        )
    if list(table.node_order) != metadata["node_order"]:
        raise ValueError("threshold NPZ and manifest node order disagree")
    table.validate()
    return table
