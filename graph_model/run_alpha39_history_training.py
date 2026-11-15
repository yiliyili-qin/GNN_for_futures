"""Frozen prediction/ledger export helpers; historical fusion sweeps are excluded."""
import numpy as np

def alpha_from_predictions(joint_scores, leader_returns, eligible, predicted):
    scores = np.asarray(joint_scores)
    returns = np.asarray(leader_returns)
    if scores.shape != (len(returns), len(returns), 3) or not np.isfinite(scores).all():
        raise ValueError("Invalid joint scores")
    mask = np.asarray(eligible, dtype=bool) & np.asarray(predicted, dtype=bool)
    mask = mask.copy()
    np.fill_diagonal(mask, False)
    return np.sum(np.where(mask, (scores[..., 2] - scores[..., 1]) * returns[:, None], 0.), axis=0)


def export_split(output_dir, split, context, prediction_fn, raw_dir):
    import pandas as pd

    panel, prepared = context["panel"], context["prepared"]
    frames = []
    for symbol in panel.node_order:
        frame = pd.read_csv(raw_dir / f"{symbol}.csv", usecols=["datetime", "open", "close", "dominant_id", "volume"])
        frame["datetime"] = pd.to_datetime(frame.datetime)
        frames.append(frame.set_index("datetime").reindex(pd.DatetimeIndex(panel.timestamps)))
    opens = np.column_stack([frame.open.to_numpy() for frame in frames])
    closes = np.column_stack([frame.close.to_numpy() for frame in frames])
    contracts = np.column_stack([frame.dominant_id.fillna("").astype(str).to_numpy() for frame in frames])
    quote_valid = panel.bar_available & np.isfinite(opens) & np.isfinite(closes) & (opens > 0) & (closes > 0) & (contracts != "")
    utilities = np.zeros_like(opens)
    signals = np.zeros(len(opens), dtype=bool)
    days = np.unique(panel.trading_dates)
    sigma_cache, records = {}, []
    for cutoff in context["cutoffs"][context["masks"][split]]:
        day = panel.trading_dates[cutoff]
        prediction = prediction_fn(int(cutoff))
        joint = np.asarray(prediction["joint_scores"])
        follow = np.asarray(prediction["follow_probability"])
        predicted = np.asarray(prediction["predicted_follow"], dtype=bool)
        shock = np.abs(panel.signed_returns[cutoff]) >= prepared.threshold_matrix[cutoff]
        eligible = np.broadcast_to(shock[:, None], joint.shape[:2]).copy()
        alpha = alpha_from_predictions(joint, panel.signed_returns[cutoff], eligible, predicted)
        if day not in sigma_cache:
            prior_days = days[days < day][-10:]
            sigma_cache[day] = (np.nanstd(panel.signed_returns[np.isin(panel.trading_dates, prior_days)], axis=0, ddof=1)
                                if len(prior_days) == 10 else np.full(len(panel.node_order), np.nan))
        sigma = sigma_cache[day]
        valid_sigma = np.isfinite(sigma).all() and (sigma > 0).all()
        utility = 10. * alpha / sigma if valid_sigma else np.full_like(alpha, np.nan)
        targets = np.flatnonzero((np.arange(len(opens)) > cutoff) & (panel.trading_dates == day) & quote_valid.all(axis=1))
        target = int(targets[0]) if len(targets) else -1
        scheduled = target >= 0 and valid_sigma and prepared.input_valid[cutoff].all()
        if not np.isfinite(follow).all() or not np.isfinite(alpha).all():
            raise ValueError("Nonfinite exported prediction")
        if scheduled:
            if signals[target]:
                raise ValueError("Multiple signals map to one execution row; explicit policy required")
            utilities[target], signals[target] = utility, True
        records.append(dict(cutoff=int(cutoff), target=target, signal_time=panel.timestamps[cutoff],
                            trading_date=day, joint_scores=joint, follow_probability=follow,
                            predicted_follow=predicted, eligible_pairs=eligible,
                            alpha=alpha, sigma=sigma, utilities=utility, scheduled=scheduled))
    if not records:
        raise ValueError("No prediction records")
    np.savez_compressed(output_dir / f"{split}_predictions.npz", node_order=panel.node_order,
                        **{key: np.asarray([record[key] for record in records]) for key in records[0]})
    bounds = context["bounds"][split]
    rows = (panel.trading_dates >= np.datetime64(bounds[0])) & (panel.trading_dates < np.datetime64(bounds[1]))
    np.savez_compressed(output_dir / f"{split}_ledger_inputs.npz", node_order=panel.node_order,
                        opens=opens[rows], closes=closes[rows], contracts=contracts[rows],
                        timestamps=panel.timestamps[rows], trading_dates=panel.trading_dates[rows],
                        quote_valid=quote_valid[rows], utilities=utilities[rows], signals=signals[rows])
    return dict(predictions=len(records), scheduled_signals=int(signals[rows].sum()))
