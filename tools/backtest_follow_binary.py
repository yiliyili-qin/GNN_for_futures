"""Binary-primary replay; fixed source direction by default, optional trained dual head."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np

from graph_model.intraday_ledger import execute_ledger
from tools.backtest_gm_baseline import (
    BASELINES, QUOTE_FIELDS, fixed_path_stress, load_quotes, quote_boundaries,
    shared_schedule, write_csv,
)
from tools.model_performance_registry import marks_and_risk
from tools.model_performance_registry_audit import macro_f1, summarize
from tools.research_alpha39_shifted_mapping import shifted_weights
from tools.train_gm_baseline import sha, write_json

CLASS_LABELS = ["no_follow", "follow"]


def probabilities(q):
    q = np.asarray(q)
    if q.dtype.kind not in "biuf" or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("Invalid q probabilities; every value must be finite and in [0, 1]")
    return q


def native_utilities(panel, prepared, schedule, q, *, direction=None):
    """Sum source shocks; optional native conditional direction preserves dual-head replay."""
    q = probabilities(q)
    rows, nodes = len(schedule["cutoffs"]), len(panel.node_order)
    if q.shape != (rows, nodes, nodes):
        raise ValueError("Scheduled native q shape mismatch")
    returns = np.asarray(panel.signed_returns[schedule["cutoffs"]], dtype=np.float64)
    thresholds = np.asarray(prepared.threshold_matrix[schedule["cutoffs"]], dtype=np.float64)
    sigma = np.asarray(schedule["sigma"], dtype=np.float64)
    if (returns.shape != (rows, nodes) or thresholds.shape != returns.shape
        or not np.isfinite(returns).all() or not np.isfinite(thresholds).all() or not (thresholds > 0).all()):
        raise ValueError("Scheduled source inputs must align and be finite with positive thresholds")
    if sigma.shape != returns.shape or not np.isfinite(sigma).all() or not (sigma > 0).all():
        raise ValueError("Scheduled sigma must align and be finite and positive")
    shock = np.abs(returns) >= thresholds
    eligible = shock[:, :, None] & ~np.eye(nodes, dtype=bool)[None] & (q > .5)
    contributions = q.astype(np.float64) * returns[:, :, None]
    if direction is not None:
        direction = probabilities(direction)
        if direction.shape != q.shape:
            raise ValueError("Scheduled direction must have the same shape as q")
        contributions = contributions * (2 * direction.astype(np.float64) - 1)
    alpha = np.where(eligible, contributions, 0.).sum(axis=1)
    utility = 10 * alpha / sigma
    if not np.isfinite(utility).all():
        raise ValueError("Nonfinite scheduled utility")
    return utility


def evaluate_case(directory, name, quotes, schedule, dataset, keys, row_q, scheduled_q,
                  *, row_d=None, scheduled_d=None):
    # The runner imports these replay helpers, so metrics must be imported only at call time.
    from tools.run_follow_binary_experiment import binary_classification

    directory = Path(directory)
    panel, prepared = dataset.events.panel, dataset.events.prepared
    coordinates = np.nonzero(dataset.masks)
    original = np.asarray(dataset.targets[dataset.masks])
    labels = (original != 0).astype(np.int64)
    keys = np.asarray(keys)
    if original.ndim != 1 or not np.isin(original, [0, 1, 2]).all() or keys.shape != original.shape:
        raise ValueError("Original classification labels/row keys must align with the complete mask")
    if (row_q is None) != (scheduled_q is None) or (row_q is None) != (name in ("cash", "EW")):
        raise ValueError("Only cash/EW omit classification and scheduled q together")
    if (row_d is None) != (scheduled_d is None) or (row_d is not None and row_q is None):
        raise ValueError("Dual-head replay requires both row and scheduled direction with q")
    if row_d is not None:
        row_d = probabilities(row_d)
        if row_d.shape != labels.shape:
            raise ValueError("Row direction must cover every classification mask row")
    if row_q is not None:
        row_q = probabilities(row_q)
        if row_q.shape != labels.shape:
            raise ValueError("Row q must cover every original classification mask row")
        utilities = native_utilities(panel, prepared, schedule, scheduled_q, direction=scheduled_d)
        scheduled_q = probabilities(scheduled_q)
    execution = np.asarray(schedule["execution_rows"])
    if (execution.shape != (len(schedule["cutoffs"]),) or execution.dtype.kind not in "iu"
        or (execution < 0).any() or (execution >= len(quotes["timestamps"])).any()
        or len(np.unique(execution)) != len(execution)):
        raise ValueError("Scheduled execution rows must be unique aligned quote indices")
    inputs = dict(quotes, utilities=np.zeros_like(quotes["opens"], dtype=np.float64),
                  signals=np.zeros(len(quotes["timestamps"]), bool))
    inputs["signals"][execution] = True
    if scheduled_q is not None:
        inputs["utilities"][execution] = utilities
    allocator = (lambda values: np.full_like(values, 1 / len(values))) if name == "EW" else shifted_weights
    result = execute_ledger(**inputs, allocation=allocator, constrained=False)
    opened, closed, entry, exits, risk = marks_and_risk(inputs, result)
    if result["fallback"].any():
        raise ValueError("Normalization fallback is forbidden")
    day_end = np.r_[quotes["trading_dates"][1:] != quotes["trading_dates"][:-1], True]
    if np.any((result["units"] + result["close_trade_units"])[day_end]):
        raise ValueError("Position carried over panel trading-date boundary")
    gross_target = np.abs(result["weights"][inputs["signals"]]).sum(axis=1)
    if not np.all(np.isclose(gross_target, 1, rtol=0, atol=1e-12) | (gross_target == 0)):
        raise ValueError("Nonzero scheduled targets must have gross one")
    observed = np.concatenate([result[key] for key in ("open_pre_trade_weights", "weights", "close_pre_exit_weights")])
    gross = np.abs(observed).sum(axis=1)
    shares = np.divide(np.abs(observed), gross[:, None], out=np.zeros_like(observed), where=gross[:, None] > 0)
    risk.update(net_exposure_mean=float(observed.sum(1).mean()), net_exposure_min=float(observed.sum(1).min()),
                net_exposure_max=float(observed.sum(1).max()), gross_exposure_mean=float(gross.mean()), gross_exposure_max=float(gross.max()),
                cash_fraction=float((result["units"] == 0).all(1).mean()), max_abs_weight=float(np.abs(observed).max()),
                max_abs_weight_over_gross=float(shares.max()), HHI_mean=float(np.square(shares).sum(1).mean()),
                exposure_sampling="open_pretrade_open_posttrade_close_preexit_equal_weighted")
    days, daily = result["trading_dates"], result["daily_returns"]
    row_cutoffs = dataset.cutoffs[coordinates[0]]
    row_dates = panel.trading_dates[row_cutoffs]
    if not np.isin(row_dates, days).all():
        raise ValueError("Classification rows fall outside the full-calendar quotes")
    classification, counts = None, None
    directory.mkdir(parents=True)
    if row_q is not None:
        classification = binary_classification(original, row_q)
        predicted = (row_q > .5).astype(np.int64)
        counts = np.stack([np.bincount(2 * labels[row_dates == day] + predicted[row_dates == day], minlength=4).reshape(2, 2) for day in days])
        write_csv(directory / "confusion.csv", dict(true_class=CLASS_LABELS,
                  **dict(zip(CLASS_LABELS, np.asarray(classification["confusion"]).T))))
        np.savez_compressed(directory / "predictions_with_row_keys.npz", row_keys=keys, target_binary=labels,
                            target_original=original, q=row_q, predicted_binary=predicted,
                            cutoff_timestamp=panel.timestamps[row_cutoffs], sample_index=coordinates[0],
                            source=coordinates[1], target_node=coordinates[2], trading_date=row_dates,
                            daily_confusion=counts, scheduled_q=scheduled_q, scheduled_cutoffs=schedule["cutoffs"],
                            class_labels=np.asarray(CLASS_LABELS),
                            **(dict(d=row_d, scheduled_d=scheduled_d) if row_d is not None else {}))
        if row_d is not None:
            from tools.ten_year_train import classification as three_classification
            write_json(directory / "three_class_classification.json", three_classification(original, row_q, row_d))
    else:
        write_csv(directory / "confusion.csv", dict(true_class=[], no_follow=[], follow=[]))
    write_json(directory / "classification.json", classification if classification is not None
               else dict(status="not_applicable", reason="cash_or_EW_has_no_classification_head"))
    monthly = []
    for month in np.unique(days.astype("datetime64[M]")):
        chosen_days = days.astype("datetime64[M]") == month
        chosen_rows = row_dates.astype("datetime64[M]") == month
        classified = binary_classification(original[chosen_rows], row_q[chosen_rows]) if row_q is not None and chosen_rows.any() else None
        monthly.append(dict(month=str(month), monthly_return=float(np.prod(1 + daily[chosen_days]) - 1),
                            trading_days=int(chosen_days.sum()), rows=int(chosen_rows.sum()),
                            support=np.bincount(labels[chosen_rows], minlength=2).tolist(), classification=classified,
                            classification_null_reason=None if classified is not None else "empty_month" if row_q is not None else "not_applicable"))
    write_json(directory / "monthly_classification.json", monthly)
    write_csv(directory / "monthly.csv", {key: [item[key] for item in monthly] for key in ("month", "monthly_return", "trading_days", "rows")})
    write_json(directory / "risk.json", risk)
    np.savez_compressed(directory / "ledger_quotes_units_trades_marks.npz", **inputs,
                        **{key: value for key, value in result.items() if isinstance(value, np.ndarray) and key not in inputs},
                        open_equity=opened, close_equity=closed, entry_notional=entry, exit_notional=exits)
    write_csv(directory / "equity.csv", dict(timestamp=quotes["timestamps"], trading_date=quotes["trading_dates"],
                                            open_equity=opened, close_equity=closed, turnover=result["turnover"]))
    write_csv(directory / "daily.csv", dict(trading_date=days, daily_return=daily, equity=closed[day_end],
                                           turnover=[result["turnover"][quotes["trading_dates"] == day].sum() for day in days]))
    stress, status = fixed_path_stress(opened, closed, entry, exits)
    write_csv(directory / "fixed_path_cost_stress.csv", dict(timestamp=quotes["timestamps"], **stress))
    write_json(directory / "cost_stress.json", status)
    return dict(classification=classification, risk=risk, daily=daily, counts=counts)


def paired_bootstrap(cases, days, seed, directory):
    """Shared five-day blocks; HH/HF are one model, and seeds are never pooled."""
    days = np.asarray(days)
    repetitions = 10000
    if days.ndim != 1:
        raise ValueError("Bootstrap requires a one-dimensional full trading-day calendar")
    if len(days) < 5:
        return dict(status="unavailable", null_reason="fewer_than_5_days", comparisons={})
    if np.any(days[1:] <= days[:-1]) or (days.dtype.kind == "M" and np.isnat(days).any()):
        raise ValueError("Bootstrap trading-day calendar must be ordered, unique and nonmissing")
    learned, aliases = {}, {}
    for name in cases:
        if name in BASELINES:
            continue
        family = name.removesuffix(f"_{seed}").lower()
        if family not in ("full", "hh", "gm", "hf") or family in learned:
            raise ValueError("Bootstrap requires one case per binary model from the same seed")
        learned[family] = name
    if any(name not in cases for name in BASELINES) or any(family not in learned for family in ("full", "hh", "gm")):
        raise ValueError("Bootstrap requires full, HH, gm and all four deterministic baselines")
    if "hf" in learned:
        alias, source = learned.pop("hf"), learned["hh"]
        left, right = cases[alias], cases[source]
        if (not np.array_equal(left["daily"], right["daily"]) or not np.array_equal(left["counts"], right["counts"])
            or left["classification"] != right["classification"] or left["risk"] != right["risk"]):
            raise ValueError("HF must be identical to HH, not an independent binary model")
        aliases[alias] = source
    unique = [learned[family] for family in ("full", "hh", "gm")]
    for name in (*unique, *BASELINES):
        case = cases[name]
        returns, counts = np.asarray(case["daily"]), case["counts"]
        if returns.shape != days.shape or not np.isfinite(returns).all():
            raise ValueError("Bootstrap returns must cover every full-calendar day without dropping")
        if counts is None:
            if name not in ("cash", "EW") or case["classification"] is not None:
                raise ValueError("Only cash/EW have no binary classification counts")
        else:
            counts = np.asarray(counts)
            if (name in ("cash", "EW") or counts.shape != (len(days), 2, 2)
                or counts.dtype.kind not in "iu" or (counts < 0).any() or counts.sum() == 0):
                raise ValueError("Binary confusion counts must cover all days with fixed 2-class axes")
            if not np.isclose(macro_f1(counts.sum(0)), case["classification"]["macro_f1"], rtol=0, atol=1e-15):
                raise ValueError("Binary macro-F1 differs from full-calendar pooled confusion")
    rng = np.random.default_rng(20261023)
    starts = rng.integers(0, len(days) - 4, size=(repetitions, (len(days) + 4) // 5))
    indices = (starts[..., None] + np.arange(5)).reshape(repetitions, -1)[:, :len(days)]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "paired_day_indices.npz"
    if not index_path.exists():
        np.savez_compressed(index_path, indices=indices, trading_dates=days)
    else:
        with np.load(index_path, allow_pickle=False) as archive:
            if not np.array_equal(archive["indices"], indices) or not np.array_equal(archive["trading_dates"], days):
                raise ValueError("Bootstrap shared day sample changed between seeds")
    statistics = {}
    for name in (*unique, *BASELINES):
        case = cases[name]
        returns = np.asarray(case["daily"], dtype=np.float64)
        sampled = returns[indices]
        deviation = sampled.std(axis=1, ddof=1)
        variable = (deviation > 0) & (np.ptp(sampled, axis=1) > 0)
        sharpes = np.divide(np.sqrt(252) * sampled.mean(1), deviation, out=np.full(repetitions, np.nan), where=variable)
        observed_deviation = returns.std(ddof=1)
        observed_sharpe = float(np.sqrt(252) * returns.mean() / observed_deviation) if observed_deviation > 0 and np.ptp(returns) > 0 else np.nan
        stats = dict(mean_daily_return_difference=(sampled.mean(1), float(returns.mean())),
                     sharpe_difference=(sharpes, observed_sharpe))
        if case["counts"] is not None:
            counts = np.asarray(case["counts"])
            draws = np.empty(repetitions)
            for offset in range(0, repetitions, 200):
                totals = counts[indices[offset:offset + 200]].sum(1)
                values = macro_f1(totals)  # Both classes always contribute, including absent classes as zero.
                values[totals.sum(axis=(1, 2)) == 0] = np.nan
                draws[offset:offset + 200] = values
            stats["macro_f1_difference"] = draws, float(macro_f1(counts.sum(0)))
        statistics[name] = stats
    comparisons = {}
    for name in unique:
        for other in (*unique, *BASELINES):
            if name == other:
                continue
            comparisons[f"{name}-minus-{other}"] = {
                metric: summarize(values[0] - statistics[other][metric][0], values[1] - statistics[other][metric][1])
                if metric in statistics[other] else dict(estimate=None, percentile_95=None, null_reason="classification_not_applicable")
                for metric, values in statistics[name].items()
            }
    return dict(method="paired_non_circular_moving_trading_day_blocks", block_days=5, repetitions=repetitions,
                seed=20261023, model_seed=seed, days=len(days), confidence=.95, indices_sha256=sha(index_path),
                annual_days=252, standard_deviation_ddof=1, macro_f1_denominator=2,
                unique_learned_models=unique, aliases=aliases,
                alias_policy="HF is identical to HH after removing direction; no independent training or bootstrap evidence",
                multiplicity="unadjusted_descriptive_not_confirmatory", comparisons=comparisons)
