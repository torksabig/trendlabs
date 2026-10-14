"""Phase D/E — economic paper portfolio from OOS predictions + cost stress."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.prediction import HORIZON_DAYS
from trendlabs.prediction.metrics_ic import assign_quintiles

TRADING_DAYS = 252


def one_way_turnover(prev_weights: dict[str, float], next_weights: dict[str, float]) -> float:
    """One-way turnover = 0.5 * Σ|Δw| over the union of names."""
    names = set(prev_weights) | set(next_weights)
    if not names:
        return 0.0
    return 0.5 * float(sum(abs(next_weights.get(n, 0.0) - prev_weights.get(n, 0.0)) for n in names))


def cost_from_turnover(turnover: float, round_trip_bps: float) -> float:
    """Cost as fraction of NAV: one-way turnover × round-trip bps / 1e4.

    Round-trip bps means the full buy+sell cost of turning over 100% of NAV once.
    """
    return float(turnover) * float(round_trip_bps) / 10_000.0


def _equal_weights(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    w = 1.0 / len(symbols)
    return {s: w for s in symbols}


def _period_metrics(
    period_log_returns: pd.Series,
    *,
    turnovers: pd.Series,
    horizon_days: int = HORIZON_DAYS,
) -> dict[str, Any]:
    """Metrics for a series of non-overlapping horizon log returns."""
    r = period_log_returns.dropna()
    if r.empty:
        return {
            "n_periods": 0,
            "total_return": None,
            "CAGR": None,
            "annualized_return": None,
            "Sharpe": None,
            "max_drawdown": None,
            "mean_turnover": None,
            "mean_period_return": None,
        }
    # Compound in simple space from log returns
    equity = np.exp(r.cumsum())
    total_return = float(equity.iloc[-1] - 1.0)
    n = len(r)
    years = (n * horizon_days) / TRADING_DAYS
    if years > 0 and equity.iloc[-1] > 0:
        cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    else:
        cagr = None
    # Annualize Sharpe on period returns (simple), periods_per_year = 252/horizon
    simple = np.expm1(r)
    mu = float(simple.mean())
    sigma = float(simple.std(ddof=1)) if len(simple) > 1 else 0.0
    ppy = TRADING_DAYS / float(horizon_days)
    sharpe = (mu / sigma) * math.sqrt(ppy) if sigma > 0 else None
    peak = np.maximum.accumulate(equity.to_numpy())
    dd = equity.to_numpy() / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else None
    tov = turnovers.reindex(r.index).dropna()
    return {
        "n_periods": int(n),
        "total_return": total_return,
        "CAGR": cagr,
        "annualized_return": cagr,
        "Sharpe": sharpe,
        "max_drawdown": max_dd,
        "mean_turnover": float(tov.mean()) if len(tov) else None,
        "mean_period_return": float(simple.mean()),
    }


def build_quintile_holdings(
    oos: pd.DataFrame,
    *,
    score_col: str = "z_hat",
    date_col: str = "ts",
    min_names: int = 10,
    long_quintile: int = 5,
    short_quintile: int | None = None,
) -> pd.DataFrame:
    """Per-date equal-weight long (and optional short) membership + gross basket R."""
    rows: list[dict[str, Any]] = []
    work = oos.copy()
    work[date_col] = pd.to_datetime(work[date_col], utc=True)

    for ts, g in work.groupby(date_col, sort=True):
        sub = g.dropna(subset=[score_col, "R", "symbol"])
        if len(sub) < min_names:
            continue
        q = assign_quintiles(sub[score_col])
        if q.nunique(dropna=True) < 5:
            continue
        long_syms = sub.loc[q == float(long_quintile), "symbol"].astype(str).tolist()
        if not long_syms:
            continue
        long_r = float(sub.loc[q == float(long_quintile), "R"].mean())
        short_syms: list[str] = []
        short_r = 0.0
        if short_quintile is not None:
            short_syms = sub.loc[q == float(short_quintile), "symbol"].astype(str).tolist()
            if short_syms:
                short_r = float(sub.loc[q == float(short_quintile), "R"].mean())
        # Long-only gross = mean R of Q5; LS gross = Q5 mean R − Q1 mean R
        if short_quintile is None:
            gross = long_r
        else:
            if not short_syms:
                continue
            gross = long_r - short_r
        rows.append(
            {
                "ts": ts,
                "long_symbols": tuple(sorted(long_syms)),
                "short_symbols": tuple(sorted(short_syms)),
                "gross_R": gross,
                "n_long": len(long_syms),
                "n_short": len(short_syms),
            }
        )
    return pd.DataFrame(rows)


def simulate_paper_portfolio(
    holdings: pd.DataFrame,
    *,
    round_trip_bps: float,
    horizon_days: int = HORIZON_DAYS,
    rebalance_every: int | None = None,
) -> dict[str, Any]:
    """Equal-weight Q5 paper book; entry open[t+1] / exit close[t+5] via panel R.

    Rebalances on the prediction schedule spaced by ``rebalance_every`` trading
    prediction rows (default = horizon_days) to avoid overlapping compounding.
    Turnover measured between consecutive rebalance books; cost = turnover × RT bps.
    """
    step = rebalance_every if rebalance_every is not None else horizon_days
    if holdings.empty:
        empty = _period_metrics(pd.Series(dtype=float), turnovers=pd.Series(dtype=float))
        empty.update({"round_trip_bps": round_trip_bps, "rebalance_every": step, "positive_expectancy": False})
        return empty

    h = holdings.sort_values("ts").reset_index(drop=True)
    # Space rebalances by `step` rows in the chronological prediction schedule
    sched = h.iloc[::step].copy().reset_index(drop=True)

    prev_w: dict[str, float] = {}
    log_rets: list[float] = []
    turnovers: list[float] = []
    dates: list[Any] = []

    for _, row in sched.iterrows():
        long_w = _equal_weights(list(row["long_symbols"]))
        short_syms = list(row["short_symbols"]) if row["short_symbols"] else []
        if short_syms:
            # Dollar-neutral: +0.5 long book / −0.5 short book (equal within side)
            lw = _equal_weights(list(row["long_symbols"]))
            sw = _equal_weights(short_syms)
            next_w = {s: 0.5 * w for s, w in lw.items()}
            for s, w in sw.items():
                next_w[s] = next_w.get(s, 0.0) - 0.5 * w
        else:
            next_w = long_w

        tov = one_way_turnover(prev_w, next_w) if prev_w else 1.0  # initial full deploy
        cost = cost_from_turnover(tov, round_trip_bps)
        # Gross R is already log return of the basket over the hold; subtract cost in simple space
        gross_simple = math.expm1(float(row["gross_R"]))
        net_simple = gross_simple - cost
        net_log = math.log1p(net_simple) if net_simple > -0.999999 else float("-inf")
        if not math.isfinite(net_log):
            continue
        log_rets.append(net_log)
        turnovers.append(tov)
        dates.append(row["ts"])
        prev_w = next_w

    series = pd.Series(log_rets, index=pd.to_datetime(dates, utc=True), dtype=float)
    tov_s = pd.Series(turnovers, index=series.index, dtype=float)
    metrics = _period_metrics(series, turnovers=tov_s, horizon_days=horizon_days)
    mean_net = metrics.get("mean_period_return")
    metrics.update(
        {
            "round_trip_bps": round_trip_bps,
            "rebalance_every": step,
            "positive_expectancy": bool(mean_net is not None and mean_net > 0),
        }
    )
    return metrics


def run_economic_simulation(
    oos: pd.DataFrame,
    *,
    score_col: str = "z_hat",
    cost_bps_list: tuple[float, ...] = (10.0, 20.0),
    min_names: int = 10,
    horizon_days: int = HORIZON_DAYS,
) -> dict[str, Any]:
    """Long Q5 and optional Q5−Q1 spread portfolios net of cost models."""
    long_h = build_quintile_holdings(
        oos, score_col=score_col, min_names=min_names, long_quintile=5, short_quintile=None
    )
    ls_h = build_quintile_holdings(
        oos, score_col=score_col, min_names=min_names, long_quintile=5, short_quintile=1
    )

    long_by_cost = {
        f"{int(bps)}bps": simulate_paper_portfolio(
            long_h, round_trip_bps=bps, horizon_days=horizon_days
        )
        for bps in cost_bps_list
    }
    ls_by_cost = {
        f"{int(bps)}bps": simulate_paper_portfolio(
            ls_h, round_trip_bps=bps, horizon_days=horizon_days
        )
        for bps in cost_bps_list
    }

    return {
        "execution": {
            "entry": "open[t+1]",
            "exit": "close[t+5]",
            "same_bar_entry": False,
            "weighting": "equal-weight within quintile",
            "rebalance": f"every {horizon_days} prediction dates (non-overlapping hold)",
            "rank_exclude_spy": True,
        },
        "long_Q5": long_by_cost,
        "long_Q5_short_Q1": ls_by_cost,
        "n_signal_dates_long": int(len(long_h)),
        "n_signal_dates_ls": int(len(ls_h)),
    }
