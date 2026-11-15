"""Pre-registered confirmation gate for shock-follow GraphMixer V2."""

from __future__ import annotations

import math
from typing import Mapping


GATE_SCHEMA_VERSION = 1
FOLLOW_AP_MINIMUM = 0.3066258009
FOLLOW_RECALL_MINIMUM = 0.20
FOLLOW_PRECISION_MINIMUM = 0.10
FOLLOW_F1_MINIMUM = 0.15
PREDICTED_FOLLOW_RATE_MINIMUM = 0.01
PREDICTED_FOLLOW_RATE_MAXIMUM = 0.30
DIRECTION_RECALL_MINIMUM = 0.10
CONDITIONAL_DIRECTION_MACRO_F1_MINIMUM = 0.52
CONDITIONAL_DIRECTION_ACCURACY_MINIMUM = 0.52
NO_FOLLOW_RECALL_MINIMUM = 0.80
THREE_CLASS_MACRO_F1_MINIMUM = 0.39
SCORE_IDENTITY_MAXIMUM = 1e-6


def _finite_number(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"confirmation metric {key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"confirmation metric {key} must be finite")
    return result


def _nonnegative_integer(metrics: Mapping[str, object], key: str) -> int:
    value = metrics.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"confirmation metric {key} must be a nonnegative integer")
    return value


def evaluate_confirmation_gate(metrics: Mapping[str, object]) -> dict[str, object]:
    """Evaluate the frozen V2 thresholds without searching or changing them."""
    values = {
        key: _finite_number(metrics, key)
        for key in (
            "follow_average_precision",
            "follow_precision",
            "follow_recall",
            "follow_f1",
            "predicted_follow_rate",
            "opposite_recall",
            "same_recall",
            "conditional_direction_macro_f1",
            "conditional_direction_accuracy",
            "no_follow_recall",
            "three_class_macro_f1",
            "joint_score_sum_max_abs_error",
            "joint_follow_identity_max_abs_error",
        )
    }
    opposite_count = _nonnegative_integer(metrics, "predicted_opposite_count")
    same_count = _nonnegative_integer(metrics, "predicted_same_count")

    checks: list[dict[str, object]] = []

    def minimum(name: str, observed: float, threshold: float) -> None:
        checks.append(
            {
                "name": name,
                "observed": observed,
                "operator": ">=",
                "threshold": threshold,
                "passed": observed >= threshold,
            }
        )

    def maximum(name: str, observed: float, threshold: float) -> None:
        checks.append(
            {
                "name": name,
                "observed": observed,
                "operator": "<=",
                "threshold": threshold,
                "passed": observed <= threshold,
            }
        )

    minimum("follow_average_precision", values["follow_average_precision"], FOLLOW_AP_MINIMUM)
    minimum("follow_recall", values["follow_recall"], FOLLOW_RECALL_MINIMUM)
    minimum("follow_precision", values["follow_precision"], FOLLOW_PRECISION_MINIMUM)
    minimum("follow_f1", values["follow_f1"], FOLLOW_F1_MINIMUM)
    minimum(
        "predicted_follow_rate_lower",
        values["predicted_follow_rate"],
        PREDICTED_FOLLOW_RATE_MINIMUM,
    )
    maximum(
        "predicted_follow_rate_upper",
        values["predicted_follow_rate"],
        PREDICTED_FOLLOW_RATE_MAXIMUM,
    )
    minimum("predicted_opposite_count", float(opposite_count), 1.0)
    minimum("predicted_same_count", float(same_count), 1.0)
    minimum("opposite_recall", values["opposite_recall"], DIRECTION_RECALL_MINIMUM)
    minimum("same_recall", values["same_recall"], DIRECTION_RECALL_MINIMUM)
    minimum(
        "conditional_direction_macro_f1",
        values["conditional_direction_macro_f1"],
        CONDITIONAL_DIRECTION_MACRO_F1_MINIMUM,
    )
    minimum(
        "conditional_direction_accuracy",
        values["conditional_direction_accuracy"],
        CONDITIONAL_DIRECTION_ACCURACY_MINIMUM,
    )
    minimum("no_follow_recall", values["no_follow_recall"], NO_FOLLOW_RECALL_MINIMUM)
    minimum(
        "three_class_macro_f1",
        values["three_class_macro_f1"],
        THREE_CLASS_MACRO_F1_MINIMUM,
    )
    maximum(
        "joint_score_sum_max_abs_error",
        values["joint_score_sum_max_abs_error"],
        SCORE_IDENTITY_MAXIMUM,
    )
    maximum(
        "joint_follow_identity_max_abs_error",
        values["joint_follow_identity_max_abs_error"],
        SCORE_IDENTITY_MAXIMUM,
    )
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": "shock_follow_graphmixer_v2_confirmation_v1",
        "threshold_search_performed": False,
        "checks": checks,
        "all_passed": passed,
        "conclusion": (
            "V2 hierarchical repair provisionally confirmed"
            if passed
            else "V2 hierarchical repair not confirmed"
        ),
    }
