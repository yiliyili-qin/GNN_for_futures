"""Frozen alpha39 GM replay and deterministic baselines; never trains or reads legacy scores."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch

from graph_model.intraday_ledger import execute_ledger
from gnn_for_futures.shock_follow.data.thresholds import load_monthly_threshold_table
from gnn_for_futures.shock_follow.models.gm_baseline import GraphMixerBaseline
from tools.model_performance_registry import array_hash, marks_and_risk
from tools.model_performance_registry_audit import macro_f1, summarize
from tools.research_alpha39_shifted_mapping import shifted_weights
from tools.train_gm_baseline import INPUT_FIELDS, FROZEN, classification_report, load_inputs, sha, write_json

QUOTE_FIELDS = ("timestamps", "trading_dates", "opens", "closes", "contracts", "quote_valid")
BASELINES = ("cash", "no_follow", "train_prior", "EW")
SCOPE = "GM three-seed macro-F1-selected evaluation plus deterministic baselines; not the full/history/FH suite"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def validate_protocol(protocol):
    if (protocol["schema"], protocol["version"]) != ("alpha39_unified_evaluation", "1.0.0"):
        raise ValueError("Only frozen alpha39 v1.0.0 is supported")
    backtest = protocol["backtest"]
    expected = dict(primary_allocator="shifted_c0_L1_uncapped", multiplier=10.,
                    initial_equity=1., base_fee_bps=0, fee_stress_bps=[1, 2, 5],
                    single_name_cap=None, duplicate_execution_target="fail_until_shared_policy_versioned",
                    all_zero="explicit_liquidation_to_cash", denominator_fallback="none")
    if any(backtest[key] != value for key, value in expected.items()):
        raise ValueError("Unsupported frozen backtest settings")
    if backtest["sigma"] != dict(prior_trading_days=10, input="signed_bar_returns", ddof=1,
                                all_nodes_finite_positive_required=True):
        raise ValueError("Unsupported sigma definition")
    frozen = read_json(ROOT / "frozen-base.json")
    if any(protocol[key] != frozen[key] for key in ("data", "interface", "classification", "selection", "backtest", "risk", "baselines")):
        raise ValueError("Evaluation definitions differ from the frozen protocol")
    bootstrap = protocol["risk"]["bootstrap"]
    if (protocol["risk"]["annual_days"], bootstrap["block_days"], bootstrap["repetitions"],
        bootstrap["seed"], bootstrap["confidence"]) != (252, 5, 10000, 20261023, .95):
        raise ValueError("Unsupported risk/bootstrap settings")


def load_quotes(directory, split, panel, protocol, sources):
    manifest_path = directory / "quote_manifest.json"
    manifest = read_json(manifest_path)
    sources[str(manifest_path.resolve())] = sha(manifest_path)
    if manifest["panel_identity"] != panel.materialization_sha256 or manifest["node_order"] != list(panel.node_order):
        raise ValueError("Quote provenance panel identity/node order mismatch")
    expected_sources = {source.symbol: source.sha256 for source in panel.source_files}
    if manifest["source_sha256"] != expected_sources:
        raise ValueError("Quote raw-source hashes differ from panel provenance")
    path = directory / f"{split}_ledger_inputs.npz"
    check_hash(path, manifest["files"][path.name])
    sources[str(path.resolve())] = sha(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            quotes = {key: archive[key] for key in QUOTE_FIELDS}
            if not np.array_equal(archive["node_order"], panel.node_order):
                raise ValueError("Quote node order mismatch")
    except ValueError as error:
        raise ValueError(f"Unsafe/invalid quotes {path}; re-export contracts as Unicode, never pickle: {error}") from error
    times, dates = quotes["timestamps"], quotes["trading_dates"]
    lower, upper = map(np.datetime64, protocol["data"]["splits"][split])
    selected = (panel.trading_dates >= lower) & (panel.trading_dates < upper)
    if (times.dtype.kind != "M" or dates.dtype.kind != "M" or times.ndim != 1 or not len(times)
        or not np.array_equal(times, panel.timestamps[selected])
        or not np.array_equal(dates, panel.trading_dates[selected])):
        raise ValueError(f"Quotes must exactly cover the full {split} panel trading-date calendar")
    shape = (len(times), len(panel.node_order))
    if any(quotes[key].shape != shape for key in ("opens", "closes", "contracts", "quote_valid")):
        raise ValueError("Quote matrices must align with calendar and nodes")
    if quotes["quote_valid"].dtype != bool or quotes["contracts"].dtype.kind not in "US":
        raise ValueError("Expected boolean quote_valid and native string contracts")
    if np.isnat(times).any() or np.isnat(dates).any() or np.any(np.diff(times) <= np.timedelta64(0, "ns")) or np.any(dates[1:] < dates[:-1]):
        raise ValueError("Quote timestamps/dates must be ordered, unique and nonmissing")
    prices_valid = np.isfinite(quotes["opens"]) & np.isfinite(quotes["closes"]) & (quotes["opens"] > 0) & (quotes["closes"] > 0)
    if np.any(quotes["quote_valid"] & ~prices_valid):
        raise ValueError("Declared-valid quotes must have finite positive opens/closes")
    quotes["contracts"] = quotes["contracts"].astype(str)
    return quotes


def quote_boundaries(quotes):
    contracts, dates = quotes["contracts"], quotes["trading_dates"]
    complete = (quotes["quote_valid"] & np.isfinite(quotes["opens"]) & np.isfinite(quotes["closes"])
                & (quotes["opens"] > 0) & (quotes["closes"] > 0)
                & ~np.isin(contracts, ["", "nan", "None"])).all(axis=1)
    carry = np.zeros(len(dates), bool)
    carry[:-1] = (complete[:-1] & complete[1:] & (dates[:-1] == dates[1:])
                  & (contracts[:-1] == contracts[1:]).all(axis=1))
    reasons = np.full(len(dates), "", dtype="U32")
    reasons[complete & ~carry] = "quote_or_roll_boundary"
    day_end = np.r_[dates[1:] != dates[:-1], True]
    reasons[complete & day_end] = "trading_day_close"
    return complete, carry, reasons


def shared_schedule(panel, prepared, quotes, protocol, split):
    """The trade clock deliberately never consults labels, source shocks or q/d."""
    cutoffs, histories = prepared.common_cutoffs, prepared.common_histories
    lower, upper = map(np.datetime64, protocol["data"]["splits"][split])
    times, dates = panel.timestamps[cutoffs], panel.trading_dates[cutoffs]
    selected = ((times >= lower) & (times < upper) & (dates >= lower) & (dates < upper)
                & prepared.input_valid[cutoffs].all(axis=1))
    previous = "train" if split == "validation" else "validation"
    end = np.datetime64(protocol["data"]["splits"][previous][1])
    selected &= ((panel.timestamps[histories[:, 0]] > end) & (panel.trading_dates[histories[:, 0]] > end))
    cutoffs, histories = cutoffs[selected], histories[selected]
    complete, carry, reasons = quote_boundaries(quotes)
    all_days = np.unique(panel.trading_dates)
    sigma_by_day, executable_by_day = {}, {}
    for day in np.unique(panel.trading_dates[cutoffs]):
        position = np.searchsorted(all_days, day)
        sigma = np.full(len(panel.node_order), np.nan)
        if position >= 10:
            returns = panel.signed_returns[np.isin(panel.trading_dates, all_days[position - 10:position])]
            if (np.isfinite(returns).sum(axis=0) >= 2).all():
                sigma = np.nanstd(returns, axis=0, ddof=1)
        sigma_by_day[day] = sigma
        executable_by_day[day] = np.flatnonzero(complete & (quotes["trading_dates"] == day))
    chosen, execution, sigmas = [], [], []
    for item, cutoff in enumerate(cutoffs):
        day = panel.trading_dates[cutoff]
        sigma = sigma_by_day[day]
        if not np.isfinite(sigma).all() or not (sigma > 0).all():
            continue
        candidates = executable_by_day[day]
        position = np.searchsorted(quotes["timestamps"][candidates], panel.timestamps[cutoff], side="right")
        if position == len(candidates):
            continue
        chosen.append(item)
        execution.append(candidates[position])
        sigmas.append(sigma)
    if len(set(execution)) != len(execution):
        raise ValueError(f"Duplicate execution target in shared {split} schedule; no implicit collision policy")
    if not chosen:
        raise ValueError(f"Empty shared {split} signal schedule")
    return dict(cutoffs=cutoffs[chosen], histories=histories[chosen], execution_rows=np.asarray(execution, dtype=np.int64),
                sigma=np.asarray(sigmas), quote_complete=complete, carry=carry, boundary_reason=reasons,
                signal_timestamp=panel.timestamps[cutoffs[chosen]], trading_date=panel.trading_dates[cutoffs[chosen]])


def validate_training(directory, protocol_path, datasets, keys, report, sources):
    required = ("protocol_snapshot.json", "protocol_sha256", "source_hashes.json", "data_manifest.json",
                "split_and_mask_hashes.json", "thresholds_with_fit_provenance.npz", "thresholds_manifest.json",
                "candidate_registry.json", "gm_contract.json", "environment.json", "training_completed.json")
    for name in required:
        path = directory / name
        sources[str(path.resolve())] = sha(path)
    check_hash(directory / "protocol_snapshot.json", sha(protocol_path))
    if (directory / "protocol_sha256").read_text().strip() != sha(protocol_path):
        raise ValueError("Training protocol hash differs from requested frozen protocol")
    source_hashes = read_json(directory / "source_hashes.json")
    if "tools/train_gm_baseline.py" not in source_hashes or "src/gnn_for_futures/shock_follow/models/gm_baseline.py" not in source_hashes:
        raise ValueError("Training source manifest is incomplete")
    for relative, expected in source_hashes.items():
        path = ROOT / relative
        check_hash(path, expected)
        sources[str(path.resolve())] = expected
    manifest = read_json(directory / "data_manifest.json")
    for key in ("panel_sha256", "panel_manifest_sha256", "panel_materialization_sha256", "nodes", "boundaries", "threshold_materialization_sha256"):
        if manifest[key] != json.loads(json.dumps(report[key])):
            raise ValueError(f"Training data identity mismatch: {key}")
    masks = read_json(directory / "split_and_mask_hashes.json")
    for split, dataset in datasets.items():
        for field, values in dict(cutoffs=dataset.cutoffs, history=dataset.histories, target=dataset.targets,
                                  mask=dataset.masks, row_keys=keys[split]).items():
            actual = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
            if actual != masks[split][field]:
                raise ValueError(f"Training split/mask hash mismatch: {split}/{field}")
    thresholds = load_monthly_threshold_table(directory / "thresholds_with_fit_provenance.npz", directory / "thresholds_manifest.json")
    if thresholds.materialization_sha256 != report["threshold_materialization_sha256"]:
        raise ValueError("Training threshold reconstruction mismatch")
    protocol, contract = read_json(protocol_path), read_json(directory / "gm_contract.json")
    registry, completed = read_json(directory / "candidate_registry.json"), read_json(directory / "training_completed.json")
    if registry["model_kwargs"] != contract["model_kwargs"] or registry["seeds"] != protocol["selection"]["seeds"]:
        raise ValueError("Training candidate registry mismatch")
    if contract["selection"] != FROZEN or contract["preflight"]["protocol_sha256"] != sha(protocol_path):
        raise ValueError("GM contract selection/protocol mismatch")
    if [trial["seed"] for trial in completed["trials"]] != protocol["selection"]["seeds"]:
        raise ValueError("Exactly all three frozen seeds are required, without seed selection")
    for trial in completed["trials"]:
        folder = directory / f"gm_{trial['seed']}"
        for name in ("best.pt", "checkpoint_hashes.json", "training_completed.json", "history.json", "all_epoch_validation_metrics.csv"):
            path = folder / name
            sources[str(path.resolve())] = sha(path)
        checkpoint = read_json(folder / "checkpoint_hashes.json")
        if checkpoint["sha256"] != trial["checkpoint_sha256"] or checkpoint["best_epoch"] != trial["best_epoch"]:
            raise ValueError("Checkpoint selection manifests disagree")
        check_hash(folder / "best.pt", trial["checkpoint_sha256"])
        history = read_json(folder / "history.json")
        if (not history or len(history) > protocol["selection"]["max_epochs"]
            or [item["epoch"] for item in history] != list(range(1, len(history) + 1))
            or trial["epochs"] != len(history)
            or not np.isfinite([item["validation"]["macro_f1"] for item in history]).all()):
            raise ValueError("Invalid training epoch budget/validation history")
        best = max(history, key=lambda item: item["validation"]["macro_f1"])
        if best["epoch"] != trial["best_epoch"] or best["validation"]["macro_f1"] != trial["validation_macro_f1"]:
            raise ValueError("Checkpoint was not earliest validation macro-F1 maximum")
        if read_json(folder / "training_completed.json") != trial:
            raise ValueError("Per-seed training completion disagrees")
        for split in ("validation", "test"):
            path = folder / f"{split}_predictions.npz"
            check_hash(path, trial["prediction_sha256"][split])
            sources[str(path.resolve())] = sha(path)
    return contract, completed


def probabilities(q, d):
    q, d = np.asarray(q), np.asarray(d)
    if q.shape != d.shape or any(not np.isfinite(value).all() or ((value < 0) | (value > 1)).any() for value in (q, d)):
        raise ValueError("Invalid probabilities; no model-specific dropping permitted")
    return q, d


def load_prediction(path, dataset, keys, seed, sources):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    sources[str(path.resolve())] = sha(path)
    coordinates = np.argwhere(dataset.masks)
    expected = dict(row_keys=keys, target=dataset.targets[dataset.masks], dense_cutoffs=dataset.cutoffs,
                    dense_history_indices=dataset.histories, node_order=dataset.events.panel.node_order,
                    dense_sample_index=np.arange(len(dataset)), sample_index=coordinates[:, 0],
                    source=coordinates[:, 1], target_node=coordinates[:, 2], seed=seed,
                    cutoff_timestamp=dataset.events.panel.timestamps[dataset.cutoffs[coordinates[:, 0]]])
    if any(not np.array_equal(values[key], value) for key, value in expected.items()):
        raise ValueError(f"Classification export row/mask/seed alignment mismatch: {path}")
    q, d = probabilities(values["dense_q"], values["dense_d"])
    if q.shape != dataset.masks.shape:
        raise ValueError("Dense probabilities must cover every original classification candidate")
    if not np.array_equal(values["q"], q[dataset.masks]) or not np.array_equal(values["d"], d[dataset.masks]):
        raise ValueError("Dense and masked native probabilities disagree")
    return values


def infer_schedule(model, events, schedule, contract, device):
    match = re.fullmatch(r"last (\d+) global realized edges; undirected unique neighbors", contract["structure_rule"])
    if not match or contract["node_input"] not in ("identity", "current"):
        raise ValueError("Unsupported GM event/node input contract")
    outputs = [[], []]
    model.eval()
    identity = np.eye(len(events.panel.node_order), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(schedule["cutoffs"]), 16):
            batch = []
            for cutoff, history in zip(schedule["cutoffs"][start:start + 16], schedule["histories"][start:start + 16]):
                sample = events.snapshot(int(cutoff), int(history[0]), contract["model_kwargs"]["max_events"], int(match[1]))
                sample["node_features"] = identity if contract["node_input"] == "identity" else events.prepared.node_features[cutoff]
                batch.append(sample)
            inputs = {key: torch.as_tensor(np.stack([sample[key] for sample in batch]), device=device) for key in INPUT_FIELDS}
            result = model(**inputs)
            for index, logits in enumerate((result.follow_logits, result.direction_logits)):
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite scheduled GM logits")
                outputs[index].append(logits.softmax(-1)[..., 1].cpu().numpy())
    return probabilities(*(np.concatenate(values) for values in outputs))


def native_utilities(panel, prepared, schedule, q, d):
    q, d = probabilities(q, d)
    rows, nodes = len(schedule["cutoffs"]), len(panel.node_order)
    if q.shape != (rows, nodes, nodes):
        raise ValueError("Scheduled native q/d shape mismatch")
    returns = panel.signed_returns[schedule["cutoffs"]]
    thresholds = prepared.threshold_matrix[schedule["cutoffs"]]
    if not np.isfinite(returns).all() or not np.isfinite(thresholds).all() or not (thresholds > 0).all():
        raise ValueError("Scheduled source inputs must be finite with positive thresholds")
    shock = np.abs(returns) >= thresholds
    eligible = shock[:, :, None] & ~np.eye(nodes, dtype=bool)[None] & (q > .5)
    # Cast the native heads before arithmetic: float32 joint subtraction changes zero/sign decisions.
    alpha = np.where(eligible, q.astype(np.float64) * (2 * d.astype(np.float64) - 1) * returns[:, :, None], 0).sum(axis=1)
    utility = 10 * alpha / schedule["sigma"]
    if not np.isfinite(utility).all():
        raise ValueError("Nonfinite scheduled utility")
    return utility


def fixed_path_stress(opened, closed, entry, exits):
    cumulative_open = np.cumsum(entry + exits) - exits
    cumulative_close = np.cumsum(entry + exits)
    arrays, statuses = {}, {}
    for bps in (1, 2, 5):
        stressed_open, stressed_close = opened - bps / 10000 * cumulative_open, closed - bps / 10000 * cumulative_close
        failed = not np.isfinite(stressed_open).all() or not np.isfinite(stressed_close).all() or min(stressed_open.min(), stressed_close.min()) <= 0
        arrays[f"open_equity_{bps}bp"], arrays[f"close_equity_{bps}bp"] = stressed_open, stressed_close
        statuses[str(bps)] = dict(status="nonpositive_equity_failed" if failed else "positive_fixed_path_stress",
                                 final_equity=float(stressed_close[-1]), total_absolute_cost=float(bps / 10000 * cumulative_close[-1]))
    return arrays, dict(mode="fixed_zero_cost_path_not_self_financing_net_ledger", scenarios=statuses)


def write_csv(path, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def evaluate_case(directory, name, quotes, schedule, dataset, keys, row_qd, scheduled_qd):
    directory.mkdir(parents=True)
    panel, prepared = dataset.events.panel, dataset.events.prepared
    inputs = dict(quotes, utilities=np.zeros_like(quotes["opens"], dtype=float), signals=np.zeros(len(quotes["timestamps"]), bool))
    inputs["signals"][schedule["execution_rows"]] = True
    if scheduled_qd is not None:
        inputs["utilities"][schedule["execution_rows"]] = native_utilities(panel, prepared, schedule, *scheduled_qd)
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
    labels = dataset.targets[dataset.masks]
    row_dates = panel.trading_dates[dataset.cutoffs[np.nonzero(dataset.masks)[0]]]
    if not np.isin(row_dates, days).all():
        raise ValueError("Classification rows fall outside the full-calendar quotes")
    classification, counts = None, None
    if row_qd is not None:
        q, d = probabilities(*row_qd)
        classification = classification_report(labels, q, d)
        predicted = np.where(q > .5, np.where(d > .5, 2, 1), 0)
        counts = np.stack([np.bincount(3 * labels[row_dates == day] + predicted[row_dates == day], minlength=9).reshape(3, 3) for day in days])
        write_csv(directory / "confusion.csv", dict(true_class=["N", "O", "S"], **dict(zip(("N", "O", "S"), np.asarray(classification["confusion"]).T))))
        np.savez_compressed(directory / "predictions_with_row_keys.npz", row_keys=keys, target=labels, q=q, d=d,
                            cutoff_timestamp=panel.timestamps[dataset.cutoffs[np.nonzero(dataset.masks)[0]]],
                            source=np.nonzero(dataset.masks)[1], target_node=np.nonzero(dataset.masks)[2], trading_date=row_dates,
                            daily_confusion=counts, scheduled_q=scheduled_qd[0], scheduled_d=scheduled_qd[1], scheduled_cutoffs=schedule["cutoffs"])
    else:
        write_csv(directory / "confusion.csv", dict(true_class=[], N=[], O=[], S=[]))
    write_json(directory / "classification.json", classification if classification is not None else dict(status="not_applicable", reason="cash_or_EW_has_no_classification_head"))
    monthly = []
    for month in np.unique(days.astype("datetime64[M]")):
        chosen_days, chosen_rows = days.astype("datetime64[M]") == month, row_dates.astype("datetime64[M]") == month
        classified = classification_report(labels[chosen_rows], q[chosen_rows], d[chosen_rows]) if row_qd is not None and chosen_rows.any() else None
        monthly.append(dict(month=str(month), monthly_return=float(np.prod(1 + daily[chosen_days]) - 1), trading_days=int(chosen_days.sum()),
                            rows=int(chosen_rows.sum()), support=np.bincount(labels[chosen_rows], minlength=3).tolist(), classification=classified,
                            classification_null_reason=None if classified is not None else "empty_month" if row_qd is not None else "not_applicable"))
    write_json(directory / "monthly_classification.json", monthly)
    write_csv(directory / "monthly.csv", {key: [item[key] for item in monthly] for key in ("month", "monthly_return", "trading_days", "rows")})
    write_json(directory / "risk.json", risk)
    np.savez_compressed(directory / "ledger_quotes_units_trades_marks.npz", **inputs,
                        **{key: value for key, value in result.items() if isinstance(value, np.ndarray) and key not in inputs},
                        open_equity=opened, close_equity=closed, entry_notional=entry, exit_notional=exits)
    write_csv(directory / "equity.csv", dict(timestamp=quotes["timestamps"], trading_date=quotes["trading_dates"], open_equity=opened, close_equity=closed, turnover=result["turnover"]))
    write_csv(directory / "daily.csv", dict(trading_date=days, daily_return=daily, equity=closed[day_end],
                                         turnover=[result["turnover"][quotes["trading_dates"] == day].sum() for day in days]))
    stress, status = fixed_path_stress(opened, closed, entry, exits)
    write_csv(directory / "fixed_path_cost_stress.csv", dict(timestamp=quotes["timestamps"], **stress))
    write_json(directory / "cost_stress.json", status)
    return dict(classification=classification, risk=risk, daily=daily, counts=counts)


def paired_bootstrap(cases, days, seed, directory):
    """One non-circular day-index sample shared by all models, never pooling seeds."""
    repetitions = 10000
    if len(days) < 5:
        return dict(status="unavailable", null_reason="fewer_than_5_days", comparisons={})
    rng = np.random.default_rng(20261023)
    starts = rng.integers(0, len(days) - 4, size=(repetitions, (len(days) + 4) // 5))
    indices = (starts[..., None] + np.arange(5)).reshape(repetitions, -1)[:, :len(days)]
    index_path = directory / "paired_day_indices.npz"
    if not index_path.exists():
        np.savez_compressed(index_path, indices=indices, trading_dates=days)
    else:
        with np.load(index_path, allow_pickle=False) as archive:
            if not np.array_equal(archive["indices"], indices) or not np.array_equal(archive["trading_dates"], days):
                raise ValueError("Bootstrap shared day sample changed between seeds")
    statistics = {}
    for name, case in cases.items():
        returns, counts = case["daily"], case["counts"]
        sampled = returns[indices]
        deviation = sampled.std(axis=1, ddof=1)
        sharpes = np.divide(np.sqrt(252) * sampled.mean(1), deviation, out=np.full(repetitions, np.nan), where=deviation > 0)
        stats = dict(mean_daily_return_difference=(sampled.mean(1), float(returns.mean())),
                     sharpe_difference=(sharpes, case["risk"]["sharpe"] if case["risk"]["sharpe"] is not None else np.nan))
        if counts is not None:
            draws = np.empty(repetitions)
            for offset in range(0, repetitions, 200):
                totals = counts[indices[offset:offset + 200]].sum(1)
                values = macro_f1(totals)
                values[totals.sum(axis=(1, 2)) == 0] = np.nan
                draws[offset:offset + 200] = values
            stats["macro_f1_difference"] = draws, case["classification"]["macro_f1"]
        statistics[name] = stats
    comparisons = {}
    gm = f"gm_{seed}"
    for baseline in BASELINES:
        comparisons[f"{gm}-minus-{baseline}"] = {
            metric: summarize(statistics[gm][metric][0] - statistics[baseline][metric][0],
                              statistics[gm][metric][1] - statistics[baseline][metric][1])
            if metric in statistics[baseline] else dict(estimate=None, percentile_95=None, null_reason="classification_not_applicable")
            for metric in statistics[gm]
        }
    return dict(method="paired_non_circular_moving_trading_day_blocks", block_days=5, repetitions=repetitions,
                seed=20261023, model_seed=seed, days=len(days), confidence=.95, indices_sha256=sha(index_path),
                multiplicity="unadjusted_descriptive_not_confirmatory", comparisons=comparisons)


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to reuse output: {args.output}")
    protocol = read_json(args.protocol)
    validate_protocol(protocol)
    sources = {str(args.protocol.resolve()): sha(args.protocol)}
    for relative, expected in protocol["source_sha256"].items():
        path = ROOT / relative
        check_hash(path, expected)
        sources[str(path.resolve())] = expected
    for path in (Path(__file__), ROOT / "graph_model/intraday_ledger.py", ROOT / "tools/research_alpha39_shifted_mapping.py",
                 ROOT / "tools/model_performance_registry.py", ROOT / "tools/model_performance_registry_audit.py"):
        sources[str(path.resolve())] = sha(path)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).suffix == ".py" and Path(filename).is_file() and Path(filename).resolve().is_relative_to(ROOT):
            path = Path(filename).resolve()
            sources[str(path)] = sha(path)
    contract = read_json(args.training_output / "gm_contract.json") if args.training_output is not None and not args.preflight else None
    node_input = contract["node_input"] if contract else "identity"
    max_events = contract["model_kwargs"]["max_events"] if contract else 50
    structure_match = re.fullmatch(r"last (\d+) global realized edges; undirected unique neighbors", contract["structure_rule"]) if contract else None
    if contract and not structure_match:
        raise ValueError("Unsupported GM structure rule")
    structure = int(structure_match[1]) if contract else 55
    datasets, keys, report = load_inputs(args.protocol, None, node_input, max_events, structure, args.device, data_dir=args.data_dir)
    panel, prepared = datasets["train"].events.panel, datasets["train"].events.prepared
    for name in ("bar_panel.npz", "bar_panel_manifest.json"):
        path = args.data_dir / name
        sources[str(path.resolve())] = sha(path)
    quotes, schedules = {}, {}
    for split in ("validation", "test"):
        quotes[split] = load_quotes(args.quotes_dir, split, panel, protocol, sources)
        schedules[split] = shared_schedule(panel, prepared, quotes[split], protocol, split)
    completed = None
    if not args.preflight:
        contract, completed = validate_training(args.training_output, args.protocol, datasets, keys, report, sources)
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.joinpath("protocol_snapshot.json").write_bytes(args.protocol.read_bytes())
    args.output.joinpath("protocol_sha256").write_text(sha(args.protocol) + "\n", encoding="utf-8")
    schedule_hashes = {}
    for split, schedule in schedules.items():
        directory = args.output / split
        directory.mkdir()
        np.savez_compressed(directory / "signal_schedule_and_boundaries.npz", **schedule, node_order=np.asarray(panel.node_order))
        schedule_hashes[split] = dict(schedule_sha256=array_hash(*(schedule[key] for key in sorted(schedule))),
                                     quotes_sha256=array_hash(*(quotes[split][key] for key in QUOTE_FIELDS)), signals=len(schedule["cutoffs"]),
                                     classification_sha256=array_hash(datasets[split].cutoffs, datasets[split].targets, datasets[split].masks, keys[split]))
    write_json(args.output / "split_and_mask_hashes.json", schedule_hashes)
    write_json(args.output / "data_manifest.json", dict(report, quote_manifest=read_json(args.quotes_dir / "quote_manifest.json")))
    write_json(args.output / "environment.json", dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, torch=torch.__version__, device=args.device, threads=args.threads, no_training=True))
    if args.preflight:
        write_json(args.output / "preflight.json", dict(status="ready", scope=SCOPE, schedules=schedule_hashes, training_output_required=False))
    else:
        for name in ("thresholds_with_fit_provenance.npz", "thresholds_manifest.json", "candidate_registry.json", "gm_contract.json", "training_completed.json"):
            shutil.copyfile(args.training_output / name, args.output / name)
        checkpoint_hashes = {str(item["seed"]): item["checkpoint_sha256"] for item in completed["trials"]}
        write_json(args.output / "checkpoint_hashes.json", checkpoint_hashes)
        with (args.output / "all_epoch_validation_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("seed", "epoch", "macro_f1", "rows"))
            writer.writeheader()
            for trial in completed["trials"]:
                with (args.training_output / f"gm_{trial['seed']}" / "all_epoch_validation_metrics.csv").open(newline="", encoding="utf-8") as incoming:
                    writer.writerows(dict(row, seed=trial["seed"]) for row in csv.DictReader(incoming))
        train_labels = datasets["train"].targets[datasets["train"].masks]
        frequencies = np.bincount(train_labels, minlength=3) / len(train_labels)
        prior_q = float(frequencies[1:].sum())
        prior_d = float(frequencies[2] / prior_q) if prior_q else .5
        comparison, bootstrap, summaries = [], {}, {}
        for split, schedule in schedules.items():
            directory, dataset = args.output / split, datasets[split]
            cases = {}
            shape = (len(schedule["cutoffs"]), len(panel.node_order), len(panel.node_order))
            for name in BASELINES:
                row_qd = None if name in ("cash", "EW") else (np.full(dataset.masks.sum(), prior_q if name == "train_prior" else 0.), np.full(dataset.masks.sum(), prior_d if name == "train_prior" else .5))
                scheduled_qd = None if row_qd is None else (np.full(shape, prior_q if name == "train_prior" else 0.), np.full(shape, prior_d if name == "train_prior" else .5))
                cases[name] = evaluate_case(directory / name, name, quotes[split], schedule, dataset, keys[split], row_qd, scheduled_qd)
            for seed in protocol["selection"]["seeds"]:
                name, folder = f"gm_{seed}", args.training_output / f"gm_{seed}"
                trial = next(item for item in completed["trials"] if item["seed"] == seed)
                check_hash(folder / f"{split}_predictions.npz", trial["prediction_sha256"][split])
                exported = load_prediction(folder / f"{split}_predictions.npz", dataset, keys[split], seed, sources)
                check_hash(folder / "best.pt", checkpoint_hashes[str(seed)])
                model = GraphMixerBaseline(**contract["model_kwargs"]).to(args.device)
                model.load_state_dict(torch.load(folder / "best.pt", weights_only=True, map_location=args.device))
                model.requires_grad_(False)
                q, d = infer_schedule(model, dataset.events, schedule, contract, args.device)
                overlap = np.isin(schedule["cutoffs"], dataset.cutoffs)
                positions = np.searchsorted(dataset.cutoffs, schedule["cutoffs"][overlap])
                if not overlap.any():
                    raise ValueError("No shared scheduled/classification overlap to verify GM inference")
                for actual, field in ((q, "dense_q"), (d, "dense_d")):
                    reference = exported[field][positions]
                    np.testing.assert_allclose(actual[overlap], reference, atol=1e-6, rtol=0, err_msg="Scheduled re-inference differs from dense classification export")
                    if not np.array_equal(actual[overlap] > .5, reference > .5):
                        raise ValueError("Scheduled/classification inference crosses a frozen probability threshold")
                cases[name] = evaluate_case(directory / name, name, quotes[split], schedule, dataset, keys[split], (exported["q"], exported["d"]), (q, d))
                bootstrap[f"{split}/{name}"] = paired_bootstrap({key: cases[key] for key in (*BASELINES, name)}, np.unique(quotes[split]["trading_dates"]), seed, directory)
            for name, case in cases.items():
                comparison.append(dict(split=split, model=name, macro_f1=case["classification"]["macro_f1"] if case["classification"] else None, **case["risk"]))
            summaries[split] = {metric: dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)), seeds=protocol["selection"]["seeds"])
                                for metric, values in (("macro_f1", [cases[f"gm_{seed}"]["classification"]["macro_f1"] for seed in protocol["selection"]["seeds"]]),
                                                       ("total_return", [cases[f"gm_{seed}"]["risk"]["total_return"] for seed in protocol["selection"]["seeds"]]))}
        write_csv(args.output / "baseline_comparison.csv", {key: [item[key] for item in comparison] for key in comparison[0]})
        write_json(args.output / "paired_daily_bootstrap.json", bootstrap)
        write_json(args.output / "seed_summary.json", dict(by_split=summaries, std_definition="sample ddof=1; seeds never pooled or selected"))
        write_json(args.output / "unavailable_baselines.json", {name: "Fresh same-seed macro-F1-selected components not supplied; legacy outputs never admitted" for name in ("history", "full", "frozenFH")})
    write_json(args.output / "admission.json", dict(status="preflight_only" if args.preflight else "gm_and_deterministic_baselines_only",
               full_protocol_suite_admitted=False, available_models=[] if args.preflight else ["gm", *BASELINES],
               unavailable_models=["history", "full", "frozenFH"], scope=SCOPE))
    for path, expected in sources.items():
        check_hash(Path(path), expected)
    write_json(args.output / "source_hashes.json", sources)
    artifacts = {path.relative_to(args.output).as_posix(): sha(path) for path in sorted(args.output.rglob("*")) if path.is_file()}
    write_json(args.output / "artifact_hashes.json", artifacts)
    write_json(args.output / ("preflight_completed.json" if args.preflight else "completed.json"), dict(
        status="preflight_ready" if args.preflight else "completed", scope=SCOPE, full_protocol_suite_admitted=False,
        admission_status="preflight_only" if args.preflight else "gm_and_deterministic_baselines_only", strict_pit=protocol["strict_pit"], fresh_oos=protocol["fresh_oos"],
        previously_inspected_test=protocol["previously_inspected_test"], protocol_sha256=sha(args.protocol), artifact_hashes_sha256=sha(args.output / "artifact_hashes.json")))
    return dict(status="preflight_ready" if args.preflight else "completed", output=str(args.output.resolve()), scope=SCOPE,
                full_protocol_suite_admitted=False, admission_status="preflight_only" if args.preflight else "gm_and_deterministic_baselines_only", schedules=schedule_hashes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "frozen-base.json")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--training-output", type=Path)
    parser.add_argument("--quotes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or (not args.preflight and args.training_output is None):
        parser.error("Positive --threads and, except for --preflight, --training-output are required")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    try:
        print(json.dumps(run(args), indent=2), flush=True)
    except (OSError, ValueError, KeyError, TypeError, AssertionError, RuntimeError) as error:
        print(json.dumps(dict(status="blocked", reason=str(error)), indent=2), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
