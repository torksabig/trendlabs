"""Unit tests for risk-adjusted metrics."""

from __future__ import annotations

import unittest

from trendlabs.metrics import compute_metrics, max_drawdown, trade_stats
import pandas as pd


class MetricsTests(unittest.TestCase):
    def test_max_drawdown(self) -> None:
        eq = pd.Series([100.0, 110.0, 90.0, 95.0])
        self.assertAlmostEqual(max_drawdown(eq), (90 / 110) - 1.0, places=6)

    def test_trade_stats_round_trips(self) -> None:
        trades = [
            {"symbol": "AAPL", "side": "BUY", "qty": 1, "price": 100, "fee": 0, "ts": "2024-01-01"},
            {"symbol": "AAPL", "side": "SELL", "qty": 1, "price": 110, "fee": 0, "ts": "2024-01-02"},
            {"symbol": "AAPL", "side": "BUY", "qty": 1, "price": 110, "fee": 0, "ts": "2024-01-03"},
            {"symbol": "AAPL", "side": "SELL", "qty": 1, "price": 100, "fee": 0, "ts": "2024-01-04"},
        ]
        stats = trade_stats(trades)
        self.assertEqual(stats["round_trips"], 2)
        self.assertAlmostEqual(stats["win_rate"] or 0, 0.5)
        self.assertEqual(stats["worst_losing_streak"], 1)

    def test_compute_metrics_reproducible_shape(self) -> None:
        curve = [
            {"ts": "2024-01-01T00:00:00+00:00", "equity": 10000, "cash": 10000},
            {"ts": "2024-01-02T00:00:00+00:00", "equity": 10100, "cash": 5000},
            {"ts": "2024-01-03T00:00:00+00:00", "equity": 9900, "cash": 5000},
            {"ts": "2024-01-04T00:00:00+00:00", "equity": 10200, "cash": 10200},
        ]
        m = compute_metrics(
            equity_curve=curve,
            trades=[],
            starting_cash=10000,
            timeframe="1d",
            benchmark_curve=[
                {"ts": "2024-01-01T00:00:00+00:00", "equity": 10000, "cash": 0},
                {"ts": "2024-01-04T00:00:00+00:00", "equity": 10100, "cash": 0},
            ],
        )
        self.assertAlmostEqual(m["total_return_pct"], 2.0, places=6)
        self.assertIsNotNone(m["sharpe"])
        self.assertLess(m["max_drawdown_pct"], 0)
        self.assertIsNotNone(m["benchmark_return_pct"])
        self.assertIsNotNone(m["alpha_pct"])


if __name__ == "__main__":
    unittest.main()
