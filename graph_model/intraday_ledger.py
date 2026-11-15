"""Continuous-price units, not integer futures contracts; zero-cost execution."""

import numpy as np

from graph_model.alpha_backtest import InvalidNormalization, utility_weights


SCHEMA = "alpha39_intraday_units_v2"


def l1_weights(utilities):
    utilities = np.asarray(utilities, dtype=float)
    if utilities.ndim != 1 or not utilities.size or not np.isfinite(utilities).all():
        raise ValueError("Utilities must be a nonempty finite vector")
    raw = np.ones_like(utilities)
    negative = utilities < 0
    positive = utilities > 0
    raw[negative] = np.expm1(utilities[negative])
    raw[positive] = 3. - np.exp(-utilities[positive])
    denominator = np.abs(raw).sum()
    if not np.isfinite(denominator):
        raise ValueError("Nonfinite L1 denominator")
    if denominator == 0.:
        raise InvalidNormalization("Zero L1 normalization denominator")
    return raw / denominator


def constrain_weights(target, instrument_cap=.1):
    if not np.isfinite(instrument_cap) or instrument_cap <= 0:
        raise ValueError("Positive finite instrument cap required")
    target = np.asarray(target, dtype=float)
    if not np.isfinite(target).all() or not np.isclose(target.sum(), 1.):
        raise ValueError("Finite net-one target required")
    long_capacity = instrument_cap * np.count_nonzero(target > 0)
    short_capacity = instrument_cap * np.count_nonzero(target < 0)
    if long_capacity < 1. - 1e-10:
        raise ValueError(f"Infeasible long capacity for net 100% under {instrument_cap:.0%} instrument cap")
    short_budget = max(0., min(float(-target[target < 0].sum()), .5,
                              short_capacity, long_capacity - 1.))
    result = np.zeros_like(target)
    for direction, budget in ((1., 1. + short_budget), (-1., short_budget)):
        active = target * direction > 0
        strength = np.abs(target)
        if active.sum() * instrument_cap < budget - 1e-10:
            raise ValueError(f"Infeasible side capacity under {instrument_cap:.0%} instrument cap")
        remaining = budget
        while remaining > 1e-12 and active.any():
            indices = np.flatnonzero(active)
            proposed = remaining * strength[indices] / strength[indices].sum()
            capped = proposed > instrument_cap
            if not capped.any():
                result[indices] = direction * proposed
                break
            fixed = indices[capped]
            result[fixed] = direction * instrument_cap
            remaining -= instrument_cap * len(fixed)
            active[fixed] = False
    if (abs(result.sum() - 1.) > 1e-9 or np.abs(result).sum() > 2. + 1e-9
            or np.abs(result).max() > instrument_cap + 1e-9):
        raise ValueError("Constrained allocation invariant failed")
    return result


def execute_ledger(opens, closes, timestamps, trading_dates, contracts, quote_valid,
                   utilities, signals, *, equal_weight=False, constrained=False, instrument_cap=.1,
                   allocation=None):
    if allocation is not None and (not callable(allocation) or constrained):
        raise ValueError("Allocation must be callable and cannot combine with constrained=True")
    opens, closes, utilities = [np.asarray(value, dtype=float) for value in (opens, closes, utilities)]
    times = np.asarray(timestamps, dtype="datetime64[ns]")
    dates = np.asarray(trading_dates, dtype="datetime64[D]")
    contracts = np.asarray(contracts).astype(str)
    valid, signals = np.asarray(quote_valid), np.asarray(signals)
    if opens.ndim != 2 or not opens.size or any(value.shape != opens.shape for value in (closes, utilities, contracts, valid)):
        raise ValueError("Aligned nonempty quote/utility matrices required")
    rows, nodes = opens.shape
    if (times.shape != (rows,) or dates.shape != (rows,) or signals.shape != (rows,)
            or signals.dtype != bool or valid.dtype != bool or np.isnat(times).any()
            or np.isnat(dates).any() or np.any(times[1:] <= times[:-1])
            or np.any(dates[1:] < dates[:-1])):
        raise ValueError("Ordered timestamps and panel trading dates required")
    if not np.isfinite(utilities).all():
        raise ValueError("Nonfinite utility input")
    if np.any(valid & (~np.isfinite(opens) | ~np.isfinite(closes) | (opens <= 0) | (closes <= 0))):
        raise ValueError("Observed quotes must be finite and positive")
    complete = valid.all(axis=1) & np.all((contracts != "") & (contracts != "nan") & (contracts != "None"), axis=1)
    if np.any(signals & ~complete):
        raise ValueError("Signal scheduled at an unexecutable quote")
    carry = np.zeros(rows, dtype=bool)
    carry[:-1] = (complete[:-1] & complete[1:] & (dates[:-1] == dates[1:])
                  & np.all(contracts[:-1] == contracts[1:], axis=1))
    units = np.zeros(nodes)
    equity = np.ones(rows + 1)
    unit_history = np.zeros_like(opens)
    weights = np.zeros_like(opens)
    traded = np.zeros_like(opens)
    close_traded = np.zeros_like(opens)
    pre_trade_weights = np.zeros_like(opens)
    pre_exit_weights = np.zeros_like(opens)
    turnover = np.zeros(rows)
    returns = np.zeros(rows)
    fallback = np.zeros(rows, dtype=bool)
    exits = np.full(rows, "", dtype="U32")
    longest = streak = 0
    for index in range(rows):
        capital = equity[index]
        if index and np.any(units):
            capital += float(units @ (opens[index] - closes[index - 1]))
        if not np.isfinite(capital) or capital <= 0:
            raise ValueError(f"Equity exhausted at open {times[index]}")
        if complete[index]:
            pre_trade_weights[index] = units * opens[index] / capital
        if signals[index]:
            try:
                allocator = utility_weights if allocation is None else allocation
                target = np.asarray(allocator(np.zeros(nodes) if equal_weight else utilities[index].copy()), dtype=float)
                if target.shape != (nodes,) or not np.isfinite(target).all():
                    raise ValueError("Allocation must return a finite node-aligned vector")
            except InvalidNormalization:
                fallback[index] = True
                streak += 1
                longest = max(longest, streak)
            else:
                streak = 0
                if constrained:
                    try:
                        target = constrain_weights(target, instrument_cap=instrument_cap)
                    except ValueError as error:
                        raise ValueError(f"{error} at {times[index]}") from error
                desired = capital * target / opens[index]
                traded[index] = desired - units
                turnover[index] += float(np.abs(traded[index]) @ opens[index]) / capital
                units = desired
        unit_history[index] = units
        if complete[index]:
            weights[index] = units * opens[index] / capital
            capital += float(units @ (closes[index] - opens[index]))
        elif np.any(units):
            raise ValueError("Cannot mark an existing position with missing quotes")
        if not np.isfinite(capital) or capital <= 0:
            raise ValueError(f"Equity exhausted at close {times[index]}")
        if complete[index]:
            pre_exit_weights[index] = units * closes[index] / capital
        if not carry[index] and np.any(units):
            close_traded[index] = -units
            turnover[index] += float(np.abs(units) @ closes[index]) / capital
            exits[index] = ("trading_day_close" if index == rows - 1 or dates[index + 1] != dates[index]
                            else "quote_or_roll_boundary")
            units = np.zeros(nodes)
        equity[index + 1] = capital
        returns[index] = capital / equity[index] - 1.
    days = np.unique(dates)
    daily = np.array([np.prod(1. + returns[dates == day]) - 1. for day in days])
    deviation = daily.std(ddof=1) if len(days) > 1 else 0.
    maximum_drawdown = float(np.max(1. - equity / np.maximum.accumulate(equity)))
    annual = float(equity[-1] ** (252. / len(days)) - 1.)
    gross = np.abs(weights).sum(axis=1)
    metrics = dict(total_return=float(equity[-1] - 1.), annual_return=annual,
                   sharpe=float(np.sqrt(252.) * daily.mean() / deviation) if deviation > 0 else None,
                   max_drawdown=maximum_drawdown,
                   calmar=annual / maximum_drawdown if maximum_drawdown > 0 else None,
                   traded_notional_turnover=float(turnover.sum()),
                   maximum_gross_exposure=float(gross.max()),
                   maximum_open_post_trade_gross_exposure=float(gross.max()),
                   maximum_open_pre_trade_gross_exposure=float(np.abs(pre_trade_weights).sum(axis=1).max()),
                   maximum_close_pre_exit_gross_exposure=float(np.abs(pre_exit_weights).sum(axis=1).max()),
                   maximum_observed_gross_exposure=float(max(gross.max(), np.abs(pre_trade_weights).sum(axis=1).max(),
                                                              np.abs(pre_exit_weights).sum(axis=1).max())),
                   minimum_net_exposure=float(weights.sum(axis=1).min()),
                   maximum_net_exposure=float(weights.sum(axis=1).max()),
                   trading_days=len(days), intervals=rows,
                   cash_days=int(sum(not np.any(unit_history[dates == day]) for day in days)))
    anomalies = dict(count=int(fallback.sum()), signal_count=int(signals.sum()),
                     fraction=float(fallback.sum() / signals.sum()) if signals.any() else 0.,
                     times=[str(value) for value in times[fallback]], longest_consecutive_signals=longest)
    return dict(metrics=metrics, equity=equity, interval_returns=returns, daily_returns=daily,
                trading_dates=days, units=unit_history, weights=weights, trade_units=traded,
                close_trade_units=close_traded, open_pre_trade_weights=pre_trade_weights,
                close_pre_exit_weights=pre_exit_weights,
                turnover=turnover, fallback=fallback, exits=exits, anomalies=anomalies)