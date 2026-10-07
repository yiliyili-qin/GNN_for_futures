"""Canonical, versioned contract for the shock-follow direction experiment."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Sequence

LABEL_CONTRACT_VERSION = "shock_follow_direction_v1"
MATERIALIZATION_SCHEMA_VERSION = 3
LEGACY_MATERIALIZATION_SCHEMA_VERSION = 2
THRESHOLD_CONTRACT_VERSION = (
    "causal_monthly_expanding_per_symbol_q95_with_market_mean_fallback_v1"
)
CLASS_NO_FOLLOW = 0
CLASS_OPPOSITE_DIRECTION = 1
CLASS_SAME_DIRECTION = 2
CLASS_ORDER = ("no_follow", "opposite_direction", "same_direction")
LABEL_UNAVAILABLE = -1
HORIZON_MINUTES = 15
THRESHOLD_QUANTILE = 0.95
MIN_ELIGIBLE_TRADING_DAYS = 20
MIN_ELIGIBLE_FINITE_RETURNS = 500
MATERIALIZATION_SOURCE_PATHS = (
    "src/gnn_for_futures/__init__.py",
    "src/gnn_for_futures/cli/__init__.py",
    "src/gnn_for_futures/cli/materialize_shock_follow.py",
    "src/gnn_for_futures/graph/__init__.py",
    "src/gnn_for_futures/graph/regression.py",
    "src/gnn_for_futures/shock_follow/__init__.py",
    "src/gnn_for_futures/shock_follow/contract.py",
    "src/gnn_for_futures/shock_follow/data/__init__.py",
    "src/gnn_for_futures/shock_follow/data/availability.py",
    "src/gnn_for_futures/shock_follow/data/panel.py",
    "src/gnn_for_futures/shock_follow/data/thresholds.py",
    "src/gnn_for_futures/shock_follow/provenance/__init__.py",
    "src/gnn_for_futures/shock_follow/provenance/source.py",
)
LEGACY_MATERIALIZATION_SOURCE_PATHS = (
    "graph_model/availability.py",
    "graph_model/materialize_shock_follow_data.py",
    "graph_model/reg_graph_utils.py",
    "graph_model/shock_follow_contract.py",
    "graph_model/shock_follow_data.py",
    "graph_model/shock_follow_thresholds.py",
)
CANONICAL_NODE_ORDER = (
    "A",
    "AG",
    "AL",
    "AO",
    "AU",
    "B",
    "BR",
    "BU",
    "C",
    "CF",
    "CS",
    "CY",
    "EB",
    "EG",
    "FU",
    "HC",
    "I",
    "J",
    "JM",
    "L",
    "LH",
    "LU",
    "M",
    "MA",
    "NR",
    "OI",
    "P",
    "PF",
    "PG",
    "PK",
    "PP",
    "PX",
    "RB",
    "RM",
    "RU",
    "SC",
    "SF",
    "SH",
    "SI",
    "SM",
    "TA",
    "UR",
    "V",
    "Y",
)


def canonical_json_sha256(value: object) -> str:
    """Return a stable SHA-256 for JSON-compatible contract/provenance data."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def validate_node_order(node_order: Sequence[str]) -> tuple[str, ...]:
    """Validate and canonicalize the source-by-target node order."""
    names = tuple(str(name) for name in node_order)
    if not names:
        raise ValueError("node_order must not be empty")
    if len(names) != len(set(names)):
        raise ValueError("node_order must contain unique symbols")
    return names


def node_order_sha256(node_order: Sequence[str]) -> str:
    """Hash the ordered symbol list, not an unordered set of symbols."""
    return canonical_json_sha256(list(validate_node_order(node_order)))


def shock_follow_contract_payload() -> dict[str, object]:
    """Return every frozen semantic rule that defines a V1 label."""
    return {
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "class_order": list(CLASS_ORDER),
        "class_ids": {
            "no_follow": CLASS_NO_FOLLOW,
            "opposite_direction": CLASS_OPPOSITE_DIRECTION,
            "same_direction": CLASS_SAME_DIRECTION,
        },
        "label_unavailable": LABEL_UNAVAILABLE,
        "matrix_orientation": "row=source,column=target",
        "source_trigger": "source_input_valid and finite positive source_threshold and abs(source_return) >= source_threshold",
        "target_class_rule": {
            "no_follow": "valid target return with abs(target_return) < target_threshold",
            "opposite_direction": "valid target shock and target_sign != source_sign",
            "same_direction": "valid target shock and target_sign == source_sign",
        },
        "loss_mask_rule": (
            "source is a valid shock; target input is valid at cutoff; target return is finite; "
            "source and target thresholds are finite and positive; source != target; exact target exists; "
            "cutoff and target share trading_date"
        ),
        "return_input_policy": (
            "non_z_scored validity-aware returns; current and preceding close "
            "endpoints exactly 15 minutes apart with same trading_date; "
            "zero-volume and roll-transition rows excluded"
        ),
        "threshold_comparison": ">=",
        "horizon_minutes": HORIZON_MINUTES,
        "timestamp_policy": "target timestamp must equal cutoff + 15 minutes",
        "trading_date_policy": "cutoff and target must have the same trading_date",
        "missing_target_policy": "unavailable; never no_follow",
        "diagonal_policy": "unavailable and excluded from loss",
    }


def shock_follow_contract_sha256() -> str:
    """Hash the canonical V1 label semantics."""
    return canonical_json_sha256(shock_follow_contract_payload())


def threshold_contract_payload() -> dict[str, object]:
    """Return the frozen causal monthly threshold semantics."""
    return {
        "threshold_contract_version": THRESHOLD_CONTRACT_VERSION,
        "quantile": THRESHOLD_QUANTILE,
        "quantile_method": "linear",
        "effective_month_policy": "normalized exchange trading_date month",
        "month_start_policy": "first observed bar timestamp carrying the effective trading_date month",
        "history": "expanding; all records with timestamp strictly before month_start",
        "fit_end_policy": "per symbol, last valid return timestamp < month_start",
        "eligibility": {
            "min_distinct_trading_days": MIN_ELIGIBLE_TRADING_DAYS,
            "min_finite_returns": MIN_ELIGIBLE_FINITE_RETURNS,
        },
        "fallback": "arithmetic mean of eligible own thresholds in same month",
        "no_eligible_symbols": "all thresholds unavailable",
        "threshold_validity": "finite and strictly positive only",
    }


def threshold_contract_sha256() -> str:
    """Hash the canonical V1 monthly threshold semantics."""
    return canonical_json_sha256(threshold_contract_payload())
