"""Focused tests for graph edge regression mechanics."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from run_graph_edge_regressions import (
    causal_smm_factors,
    directed_edge_registry,
    fit_equation,
    graph_edges,
    maximum_lag_matrices,
    oos_records,
    select_bic,
)


class GraphEdgeRegressionTests(unittest.TestCase):
    def test_graph_expands_to_expected_directed_registry(self) -> None:
        registry = directed_edge_registry(graph_edges())
        self.assertEqual(len(registry), 114)
        self.assertEqual(registry["directed_edge"].nunique(), 114)
        self.assertEqual(int(registry["smm_ablation_applicable"].sum()), 10)

    def test_bic_selects_nonzero_lag_for_var_one(self) -> None:
        generator = np.random.default_rng(7)
        values = np.zeros((800, 2))
        noise = generator.normal(scale=0.1, size=(800, 2))
        for position in range(1, len(values)):
            values[position, 0] = 0.75 * values[position - 1, 0] + noise[position, 0]
            values[position, 1] = 0.55 * values[position - 1, 0] + 0.3 * values[position - 1, 1] + noise[position, 1]
        self.assertEqual(select_bic(values, 5), 1)

    def test_bic_matches_direct_least_squares_reference(self) -> None:
        generator = np.random.default_rng(19)
        values = generator.normal(size=(240, 2))
        targets, maximum_design = maximum_lag_matrices(values, 6)
        reference_scores = []
        for lag in range(7):
            parameter_count = 1 + 2 * lag
            design = maximum_design[:, :parameter_count]
            coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
            residuals = targets - design @ coefficients
            covariance = residuals.T @ residuals / len(targets)
            reference_scores.append(np.linalg.slogdet(covariance)[1] + np.log(len(targets)) * (2 + 4 * lag) / len(targets))
        self.assertEqual(select_bic(values, 6), int(np.argmin(reference_scores)))

    def test_ols_reports_expected_coefficients(self) -> None:
        explanatory = np.arange(1.0, 21.0)
        design = np.column_stack([np.ones(len(explanatory)), explanatory])
        targets = 2.0 + 3.0 * explanatory
        targets[::2] += 0.01
        fit = fit_equation(targets, design)
        np.testing.assert_allclose(fit["coefficients"], [2.0, 3.0], atol=0.01)
        self.assertGreater(fit["r_squared"], 0.999)

    def test_smm_return_starts_at_first_available_grid(self) -> None:
        index = pd.date_range("2026-01-05 09:00", periods=8, freq="15min")
        events = {
            "NAPHTHA": pd.DataFrame({
                "available_at": [pd.Timestamp("2026-01-05 09:22")],
                "return": [0.04],
                "quota_id": ["test"],
            })
        }
        factors, audit = causal_smm_factors(index, events, ["NAPHTHA"], bars_per_day=4)
        self.assertTrue(np.isnan(factors[:2, 0]).all())
        expected = np.expm1(np.log1p(0.04) / 4)
        np.testing.assert_allclose(factors[2:6, 0], expected)
        self.assertEqual(factors[6, 0], 0.0)
        self.assertTrue(audit.loc[index[2], "NAPHTHA__update_available"])
        self.assertLessEqual(audit.loc[index[2], "NAPHTHA__latest_available_at"], index[2])

    def test_forecast_direction_treats_zero_separately(self) -> None:
        edge = {"graph_edge_number": 1}
        index = pd.date_range("2026-01-05 09:00", periods=4, freq="15min")
        summary, rows = oos_records(
            edge, "X", "Y", "baseline", index,
            np.array([1.0, -1.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.3, -0.2]),
            0.0, None, pd.Timestamp("2026-01-05"),
            pd.DatetimeIndex([pd.Timestamp("2026-01-02 15:00"), *index[:3]]),
        )
        self.assertEqual(summary["forecast_direction_hits"], 1)
        self.assertEqual(summary["forecast_direction_comparisons"], 3)
        self.assertEqual(summary["forecast_zero_return_count"], 1)
        self.assertAlmostEqual(summary["forecast_direction_hit_rate"], 1 / 3)
        self.assertEqual([row["forecast_direction_status"] for row in rows], ["same", "opposite", "zero_return", "opposite"])
        self.assertEqual(rows[0]["latest_futures_information_time"], "2026-01-02T15:00:00")


if __name__ == "__main__":
    unittest.main()