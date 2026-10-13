"""Risk-adjusted performance metrics for backtests (no live trading)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _equity_frame(equity_curve: list[dict[str, Any]]) -> pd.DataFrame:
    if not equity_curve:
        return pd.DataFrame(columns=["ts", "equity"])
    df = pd.DataFrame(equity_curve).copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["equity"] = df["equity"].astype(float)
    return df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _period_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float | None:
    if len(returns) < 2:
        return None
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    if sigma <= 0 or not math.isfinite(sigma):
        return None
    return (mu / sigma) * math.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: float) -> float | None:
    if len(returns) < 2:
        return None
    downside = returns[returns < 0]
    if downside.empty:
        return None
    downside_std = float(downside.std(ddof=1))
    if downside_std <= 0 or not math.isfinite(downside_std):
        return None
    mu = float(returns.mean())
    return (mu / downside_std) * math.sqrt(periods_per_year)


def infer_periods_per_year(timestamps: pd.Series, timeframe: str) -> float:
    mapping = {"15m": 252 * 26, "1h": 252 * 6.5, "4h": 252 * 1.625, "1d": 252.0}
    if timeframe in mapping:
        return float(mapping[timeframe])
    if len(timestamps) < 2:
        return float(TRADING_DAYS)
    deltas = timestamps.diff().dt.total_seconds().dropna()
    med = float(deltas.median()) if len(deltas) else 3600.0
    if med <= 0:
        return float(TRADING_DAYS)
    seconds_year = 365.25 * 24 * 3600
    return seconds_year / med


def trade_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair BUY/SELL chronologically per symbol for round-trip stats."""
    open_buys: dict[str, list[dict[str, Any]]] = {}
    rounds: list[dict[str, Any]] = []
    for trade in sorted(trades, key=lambda t: str(t["ts"])):
        sym = trade["symbol"]
        side = trade["side"]
        if side == "BUY":
            open_buys.setdefault(sym, []).append(trade)
            continue
        if side != "SELL" or not open_buys.get(sym):
            continue
        buy = open_buys[sym].pop(0)
        qty = float(trade["qty"])
        entry = float(buy["price"])
        exit_px = float(trade["price"])
        fees = float(buy.get("fee", 0)) + float(trade.get("fee", 0))
        pnl = (exit_px - entry) * qty - fees
        hold_bars = None
        try:
            t0 = pd.Timestamp(buy["ts"])
            t1 = pd.Timestamp(trade["ts"])
            hold_bars = max((t1 - t0).total_seconds() / 3600.0, 0.0)
        except Exception:  # noqa: BLE001
            hold_bars = None
        rounds.append({"pnl": pnl, "hold_hours": hold_bars, "symbol": sym})

    if not rounds:
        return {
            "round_trips": 0,
            "win_rate": None,
            "profit_factor": None,
            "avg_winner": None,
            "avg_loser": None,
            "avg_holding_hours": None,
            "worst_losing_streak": 0,
        }

    pnls = [r["pnl"] for r in rounds]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    streak = best = 0
    for p in pnls:
        if p <= 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    holds = [r["hold_hours"] for r in rounds if r["hold_hours"] is not None]
    return {
        "round_trips": len(rounds),
        "win_rate": len(winners) / len(rounds) if rounds else None,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "avg_winner": (sum(winners) / len(winners)) if winners else None,
        "avg_loser": (sum(losers) / len(losers)) if losers else None,
        "avg_holding_hours": (sum(holds) / len(holds)) if holds else None,
        "worst_losing_streak": best,
    }


def exposure_pct(equity_curve: list[dict[str, Any]]) -> float | None:
    if not equity_curve:
        return None
    invested = 0
    for row in equity_curve:
        eq = float(row["equity"])
        cash = float(row.get("cash", eq))
        if eq > 0 and (eq - cash) > 1e-9:
            invested += 1
    return invested / len(equity_curve)


def compute_metrics(
    *,
    equity_curve: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    starting_cash: float,
    timeframe: str,
    benchmark_curve: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    df = _equity_frame(equity_curve)
    if df.empty:
        return {
            "starting_cash": starting_cash,
            "ending_equity": starting_cash,
            "total_return_pct": 0.0,
            "cagr_pct": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown_pct": 0.0,
            "volatility_pct": None,
            "trade_count": 0,
            "turnover": 0.0,
            "exposure_pct": None,
            "benchmark_return_pct": None,
            "alpha_pct": None,
            **trade_stats([]),
        }

    start_eq = float(df["equity"].iloc[0])
    end_eq = float(df["equity"].iloc[-1])
    total_return = end_eq / starting_cash - 1.0
    periods_per_year = infer_periods_per_year(df["ts"], timeframe)
    returns = _period_returns(df["equity"])
    vol = float(returns.std(ddof=1) * math.sqrt(periods_per_year)) if len(returns) > 1 else None

    elapsed_years = max((df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    cagr = (end_eq / starting_cash) ** (1 / elapsed_years) - 1 if starting_cash > 0 and end_eq > 0 else None

    buy_notional = sum(float(t["qty"]) * float(t["price"]) for t in trades if t["side"] == "BUY")
    turnover = buy_notional / starting_cash if starting_cash else 0.0

    bench_ret = None
    alpha = None
    if benchmark_curve:
        bdf = _equity_frame(benchmark_curve)
        if len(bdf) >= 2 and float(bdf["equity"].iloc[0]) > 0:
            bench_ret = float(bdf["equity"].iloc[-1] / bdf["equity"].iloc[0] - 1.0)
            alpha = total_return - bench_ret

    stats = trade_stats(trades)
    return {
        "starting_cash": starting_cash,
        "ending_equity": end_eq,
        "start_equity": start_eq,
        "total_return_pct": total_return * 100,
        "cagr_pct": (cagr * 100) if cagr is not None else None,
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown_pct": max_drawdown(df["equity"]) * 100,
        "volatility_pct": (vol * 100) if vol is not None else None,
        "trade_count": len(trades),
        "turnover": turnover,
        "exposure_pct": exposure_pct(equity_curve),
        "benchmark_return_pct": (bench_ret * 100) if bench_ret is not None else None,
        "alpha_pct": (alpha * 100) if alpha is not None else None,
        "periods_per_year": periods_per_year,
        **stats,
    }
