"""Machine-checkable availability contracts for time-causal graph inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

import pandas as pd


SUPPORTED_SCHEMA_VERSION = 1
RETROSPECTIVE_SCHEMA_VERSION = 3
SUPPORTED_TIMESTAMP_SEMANTICS = "bar_close"
SUPPORTED_AVAILABLE_AT_RULE = "datetime"
CONFIRMED_BAR_DATA_MODE = "confirmed-point-in-time"
RETROSPECTIVE_BAR_DATA_MODE = "retrospective-finalized"
BAR_DATA_MODES = (CONFIRMED_BAR_DATA_MODE, RETROSPECTIVE_BAR_DATA_MODE)
CONFIRMED_CONTRACT_FILENAME = "bar_availability.json"
RETROSPECTIVE_CONTRACT_FILENAME = "bar_availability.retrospective.json"
CONFIRMED_METADATA_MODE = "confirmed_point_in_time"
RETROSPECTIVE_METADATA_MODE = "retrospective_finalized"


@dataclass(frozen=True)
class BarAvailabilityContract:
    """Confirmed point-in-time interpretation of the 15-minute bar files."""

    schema_version: int
    timestamp_column: str
    timestamp_semantics: str
    available_at_rule: str
    available_at_delay_seconds: int
    bar_duration_minutes: int
    timezone: str
    historical_revision_policy: str
    historical_values_point_in_time_safe: bool
    point_in_time_replay_available: bool
    confirmation_status: str
    source: str
    confirmed_by: str
    confirmed_at: pd.Timestamp
    sha256: str

    def available_at(self, timestamps: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Map bar timestamps to their confirmed first usable information times."""
        values = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
        if values.tz is not None:
            values = values.tz_convert(self.timezone).tz_localize(None)
        return values


@dataclass(frozen=True)
class RetrospectiveBarDataContract:
    """Research-only interpretation of mutable historical bars."""

    schema_version: int
    usage_scope: str
    timestamp_column: str
    timestamp_semantics_status: str
    aggregation_interval: str
    analytical_timestamp_rule: str
    bar_duration_minutes: int
    bar_construction: str
    timezone: str
    timezone_status: str
    trading_date_rule: str
    trading_hours_no_trade_rule: str
    nontrading_hours_rule: str
    auction_rule: str
    typical_regular_delay_seconds: float
    typical_day_close_delay_seconds: float
    delay_status: str
    point_in_time_availability: str
    price_volume_point_in_time: bool
    point_in_time_replay_available: bool
    historical_revision: str
    dominant_series_revision: str
    adjustment_revision: str
    evidence_status: str
    source_urls: tuple[str, ...]
    support_response_document: str
    support_response_document_sha256: str
    evidence_document: str
    evidence_document_sha256: str
    limitations: tuple[str, ...]
    evidence_review_date: str
    sha256: str
    identity_sha256: str

    def analytical_timestamps(
        self, timestamps: pd.Series | pd.DatetimeIndex
    ) -> pd.DatetimeIndex:
        """Normalize timestamps for offline ordering, not availability claims."""
        values = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
        if values.tz is not None:
            values = values.tz_convert("Asia/Shanghai").tz_localize(None)
        return values


def _read_json_contract(path: str | Path) -> tuple[Path, bytes, dict[str, object]]:
    contract_path = Path(path)
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Bar availability contract is required but was not found: {contract_path}"
        )
    raw_content = contract_path.read_bytes()
    payload = json.loads(raw_content.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Bar availability contract must contain a JSON object")
    return contract_path, raw_content, payload


def _require_fields(payload: dict[str, object], required: set[str]) -> None:
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Bar availability contract is missing fields: {missing}")


def _nonempty_string(payload: dict[str, object], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Bar availability contract field {field!r} must be non-empty")
    return value.strip()


def _string_sequence(payload: dict[str, object], field: str) -> tuple[str, ...]:
    values = payload[field]
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise ValueError(
            f"Bar availability contract field {field!r} must contain non-empty strings"
        )
    return tuple(value.strip() for value in values)


def load_bar_availability_contract(path: str | Path) -> BarAvailabilityContract:
    """Load a confirmed point-in-time contract, rejecting research assumptions."""
    contract_path, raw_content, payload = _read_json_contract(path)
    if contract_path.name == RETROSPECTIVE_CONTRACT_FILENAME:
        raise ValueError("A research-only contract cannot be used as a confirmed contract")
    required = {
        "schema_version",
        "timestamp_column",
        "timestamp_semantics",
        "available_at_rule",
        "bar_duration_minutes",
        "timezone",
        "historical_revision_policy",
        "historical_values_point_in_time_safe",
        "point_in_time_replay_available",
        "confirmation_status",
        "source",
        "confirmed_by",
        "confirmed_at",
    }
    _require_fields(payload, required)
    schema_version = payload["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SUPPORTED_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported bar availability schema_version={schema_version!r}"
        )
    contract_type = payload.get("contract_type", "confirmed_point_in_time")
    if contract_type != "confirmed_point_in_time":
        raise ValueError("A research-only contract cannot be used as a confirmed contract")
    usage_scope = payload.get("usage_scope", "admission_diagnostics")
    if usage_scope != "admission_diagnostics":
        raise ValueError("Confirmed contract usage_scope must be 'admission_diagnostics'")
    if payload["timestamp_column"] != "datetime":
        raise ValueError("Bar availability timestamp_column must be 'datetime'")
    if payload["timestamp_semantics"] != SUPPORTED_TIMESTAMP_SEMANTICS:
        raise ValueError("Bar datetime must be explicitly confirmed as a bar-close time")
    if payload["available_at_rule"] != SUPPORTED_AVAILABLE_AT_RULE:
        raise ValueError("Only the confirmed available_at_rule='datetime' is supported")
    delay = payload.get("available_at_delay_seconds", 0)
    if not isinstance(delay, int) or isinstance(delay, bool) or delay != 0:
        raise ValueError("available_at_rule='datetime' requires zero delay seconds")
    duration = payload["bar_duration_minutes"]
    if not isinstance(duration, int) or isinstance(duration, bool) or duration != 15:
        raise ValueError("bar_duration_minutes must be exactly 15")
    if payload["confirmation_status"] != "confirmed":
        raise ValueError("Bar availability contract has not been confirmed")
    if payload["timezone"] != "Asia/Shanghai":
        raise ValueError("Bar availability timezone must be 'Asia/Shanghai'")
    historical_safe = payload["historical_values_point_in_time_safe"]
    if historical_safe is not True:
        raise ValueError(
            "Historical bar values must be confirmed point-in-time safe for admission"
        )
    replay_available = payload["point_in_time_replay_available"]
    if replay_available is not True:
        raise ValueError(
            "Confirmed admission requires point_in_time_replay_available=true"
        )
    if payload.get("price_volume_point_in_time") is False:
        raise ValueError("Confirmed admission cannot use non-PIT price/volume history")
    if payload.get("delay_status") == "typical_examples_not_guaranteed_no_fixed_delay":
        raise ValueError("Typical publication delays cannot authorize confirmed admission")
    source = _nonempty_string(payload, "source")
    confirmed_by = _nonempty_string(payload, "confirmed_by")
    revision_policy = str(payload["historical_revision_policy"]).strip()
    if not revision_policy:
        raise ValueError("historical_revision_policy must be non-empty when supplied")
    try:
        confirmed_at = pd.Timestamp(payload["confirmed_at"])
    except (TypeError, ValueError) as error:
        raise ValueError("Bar availability contract confirmed_at is invalid") from error
    if pd.isna(confirmed_at):
        raise ValueError("Bar availability contract confirmed_at is invalid")
    if confirmed_at.tzinfo is None:
        raise ValueError("Bar availability contract confirmed_at must include a timezone")
    return BarAvailabilityContract(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        timestamp_column="datetime",
        timestamp_semantics=SUPPORTED_TIMESTAMP_SEMANTICS,
        available_at_rule=SUPPORTED_AVAILABLE_AT_RULE,
        available_at_delay_seconds=delay,
        bar_duration_minutes=duration,
        timezone="Asia/Shanghai",
        historical_revision_policy=revision_policy,
        historical_values_point_in_time_safe=True,
        point_in_time_replay_available=replay_available,
        confirmation_status="confirmed",
        source=source,
        confirmed_by=confirmed_by,
        confirmed_at=confirmed_at,
        sha256=hashlib.sha256(raw_content).hexdigest(),
    )


def load_retrospective_bar_data_contract(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> RetrospectiveBarDataContract:
    """Load the mutable-history contract and its repository evidence.

    Repository-relative evidence paths resolve from ``project_root`` when supplied,
    otherwise from the process working directory.  No checkout location is inferred
    from this installed module's path.
    """
    contract_path, raw_content, payload = _read_json_contract(path)
    supplied_root = Path(project_root) if project_root is not None else Path.cwd()
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise ValueError("project_root must be a regular nonsymlink directory")
    evidence_root = supplied_root.resolve()
    if contract_path.name != RETROSPECTIVE_CONTRACT_FILENAME:
        raise ValueError(
            f"Retrospective contract filename must be {RETROSPECTIVE_CONTRACT_FILENAME!r}"
        )
    required = {
        "schema_version",
        "contract_type",
        "usage_scope",
        "timestamp_column",
        "timestamp_semantics_status",
        "aggregation_interval",
        "analytical_timestamp_rule",
        "bar_duration_minutes",
        "bar_construction",
        "timezone",
        "timezone_status",
        "trading_date_rule",
        "trading_hours_no_trade_rule",
        "nontrading_hours_rule",
        "auction_rule",
        "typical_regular_delay_seconds",
        "typical_day_close_delay_seconds",
        "delay_status",
        "point_in_time_availability",
        "price_volume_point_in_time",
        "point_in_time_replay_available",
        "historical_revision",
        "dominant_series_revision",
        "adjustment_revision",
        "evidence_status",
        "source_urls",
        "support_response_document",
        "evidence_document",
        "limitations",
        "evidence_review_date",
    }
    _require_fields(payload, required)
    expected = {
        "schema_version": RETROSPECTIVE_SCHEMA_VERSION,
        "contract_type": "retrospective_mutable_history",
        "usage_scope": "retrospective_diagnostics_only",
        "timestamp_column": "datetime",
        "timestamp_semantics_status": "support_confirmed_interval_right_edge_not_availability",
        "aggregation_interval": "(t,t+15m]",
        "analytical_timestamp_rule": "stored_datetime_as_confirmed_right_edge_for_offline_ordering_only",
        "bar_duration_minutes": 15,
        "bar_construction": "composed_from_one_minute_bars",
        "timezone": "Asia/Shanghai",
        "timezone_status": "support_confirmed_default",
        "trading_date_rule": "exchange_trading_date_unchanged_by_display_timezone_night_session_next_day",
        "trading_hours_no_trade_rule": "emit_zero_volume_bar_with_previous_price",
        "nontrading_hours_rule": "no_bar",
        "auction_rule": "auction_in_first_minute_bar_no_separate_auction_bar",
        "typical_regular_delay_seconds": 0.5,
        "typical_day_close_delay_seconds": 5.5,
        "delay_status": "typical_examples_not_guaranteed_no_fixed_delay",
        "point_in_time_availability": "unknown_variable_no_per_bar_available_at",
        "price_volume_point_in_time": False,
        "point_in_time_replay_available": False,
        "historical_revision": "all_fields_revisable_without_per_row_lineage_small_changes_may_be_silent",
        "dominant_series_revision": "mapping_version_unavailable",
        "adjustment_revision": "pre_adjusted_dominant_history_version_unavailable",
        "evidence_status": "user_supplied_followup_support_response_partial",
    }
    for field, value in expected.items():
        supplied = payload[field]
        if isinstance(value, bool):
            matches = isinstance(supplied, bool) and supplied is value
        elif isinstance(value, int):
            matches = (
                isinstance(supplied, int)
                and not isinstance(supplied, bool)
                and supplied == value
            )
        elif isinstance(value, float):
            matches = (
                isinstance(supplied, (int, float))
                and not isinstance(supplied, bool)
                and float(supplied) == value
            )
        else:
            matches = isinstance(supplied, type(value)) and supplied == value
        if not matches:
            raise ValueError(
                f"Retrospective bar contract requires {field}={value!r}"
            )
    source_urls = _string_sequence(payload, "source_urls")
    if any(
        not url.startswith("https://www.ricequant.com/doc/") for url in source_urls
    ):
        raise ValueError("Retrospective source_urls must be Ricequant documentation URLs")

    def repository_evidence_path(field: str) -> tuple[str, str]:
        document = _nonempty_string(payload, field)
        relative_path = PurePosixPath(document)
        windows_path = PureWindowsPath(document)
        if (
            "\\" in document
            or any(ord(character) < 32 for character in document)
            or relative_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or document != relative_path.as_posix()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise ValueError(f"{field} must be a canonical repository-relative path")
        evidence_path = evidence_root
        for part in relative_path.parts:
            evidence_path = evidence_path / part
            if evidence_path.is_symlink():
                raise ValueError(
                    f"Retrospective evidence path traverses a symlink: {relative_path}"
                )
        if not evidence_path.is_file():
            raise FileNotFoundError(
                "Retrospective evidence document was not found as a regular "
                f"nonsymlink file: {evidence_path}"
            )
        return document, hashlib.sha256(evidence_path.read_bytes()).hexdigest()

    support_response_document, support_response_document_sha256 = (
        repository_evidence_path("support_response_document")
    )
    evidence_document, evidence_document_sha256 = repository_evidence_path(
        "evidence_document"
    )
    contract_sha256 = hashlib.sha256(raw_content).hexdigest()
    identity_sha256 = hashlib.sha256(
        json.dumps(
            {
                "contract_sha256": contract_sha256,
                "support_response_document": support_response_document,
                "support_response_document_sha256": support_response_document_sha256,
                "evidence_document": evidence_document,
                "evidence_document_sha256": evidence_document_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    limitations = _string_sequence(payload, "limitations")
    if len(limitations) < 4:
        raise ValueError("Retrospective contract must document at least four limitations")
    review_date = _nonempty_string(payload, "evidence_review_date")
    try:
        parsed_review_date = pd.Timestamp(review_date)
    except (TypeError, ValueError) as error:
        raise ValueError("evidence_review_date is invalid") from error
    if (
        pd.isna(parsed_review_date)
        or parsed_review_date.tzinfo is not None
        or parsed_review_date.strftime("%Y-%m-%d") != review_date
    ):
        raise ValueError("evidence_review_date must be an ISO calendar date")
    return RetrospectiveBarDataContract(
        schema_version=RETROSPECTIVE_SCHEMA_VERSION,
        usage_scope="retrospective_diagnostics_only",
        timestamp_column="datetime",
        timestamp_semantics_status=expected["timestamp_semantics_status"],
        aggregation_interval=expected["aggregation_interval"],
        analytical_timestamp_rule=expected["analytical_timestamp_rule"],
        bar_duration_minutes=15,
        bar_construction=expected["bar_construction"],
        timezone="Asia/Shanghai",
        timezone_status=expected["timezone_status"],
        trading_date_rule=expected["trading_date_rule"],
        trading_hours_no_trade_rule=expected["trading_hours_no_trade_rule"],
        nontrading_hours_rule=expected["nontrading_hours_rule"],
        auction_rule=expected["auction_rule"],
        typical_regular_delay_seconds=0.5,
        typical_day_close_delay_seconds=5.5,
        delay_status=expected["delay_status"],
        point_in_time_availability=expected["point_in_time_availability"],
        price_volume_point_in_time=False,
        point_in_time_replay_available=False,
        historical_revision=expected["historical_revision"],
        dominant_series_revision=expected["dominant_series_revision"],
        adjustment_revision=expected["adjustment_revision"],
        evidence_status=expected["evidence_status"],
        source_urls=source_urls,
        support_response_document=support_response_document,
        support_response_document_sha256=support_response_document_sha256,
        evidence_document=evidence_document,
        evidence_document_sha256=evidence_document_sha256,
        limitations=limitations,
        evidence_review_date=review_date,
        sha256=contract_sha256,
        identity_sha256=identity_sha256,
    )
