"""Unit tests for market-regime classification."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trendlabs.indicators import enrich
from trendlabs.regimes import (
    attribute_trades_by_regime,
    classify_row,
    classify_series,
    score_regime_fit,
)


class RegimeTests(unittest.TestCase):
    def test_classify_row_priorities(self) -> None:
        self.assertEqual(
            classify_row(
                close=90,
                ema_fast=95,
                ema_slow=100,
                ema_long=110,
                adx=30,
                vol=0.4,
                ret_20=-0.08,
                vol_hi=0.25,
                vol_lo=0.1,
            ),
            "risk_off",
        )
        self.assertEqual(
            classify_row(
                close=110,
                ema_fast=108,
                ema_slow=100,
                ema_long=90,
                adx=28,
                vol=0.15,
                ret_20=0.03,
                vol_hi=0.25,
                vol_lo=0.1,
            ),
            "trending_bullish",
        )
        self.assertEqual(
            classify_row(
                close=90,
                ema_fast=88,
                ema_slow=100,
                ema_long=110,
                adx=28,
                vol=0.15,
                ret_20=-0.01,
                vol_hi=0.25,
                vol_lo=0.1,
            ),
            "trending_bearish",
        )
        self.assertEqual(
            classify_row(
                close=100,
                ema_fast=101,
                ema_slow=100,
                ema_long=99,
                adx=12,
                vol=0.35,
                ret_20=0.0,
                vol_hi=0.25,
                vol_lo=0.1,
            ),
            "high_volatility",
        )

    def test_enrich_adds_market_regime(self) -> None:
        rng = np.random.default_rng(0)
        n = 260
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, size=n))
        df = pd.DataFrame(
            {
                "ts": ts,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": rng.uniform(1e6, 2e6, size=n),
            }
        )
        out = enrich(df)
        self.assertIn("market_regime", out.columns)
        self.assertEqual(out["trend_label"].tolist(), out["market_regime"].tolist())
        labels = set(classify_series(out).unique())
        self.assertTrue(labels)

    def test_attribute_and_score(self) -> None:
        trades = [
            {
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 1,
                "price": 100,
                "fee": 0,
                "ts": "2024-01-01T10:00:00+00:00",
            },
            {
                "symbol": "AAPL",
                "side": "SELL",
                "qty": 1,
                "price": 110,
                "fee": 0,
                "ts": "2024-01-01T12:00:00+00:00",
            },
        ]
        regime_map = {"AAPL": {"2024-01-01T10:00:00+00:00": "trending_bullish"}}
        stats = attribute_trades_by_regime(trades, regime_map)
        self.assertIn("trending_bullish", stats)
        self.assertEqual(stats["trending_bullish"]["trade_count"], 1)
        fit = score_regime_fit(stats)
        self.assertEqual(fit["trending_bullish"], "insufficient")


if __name__ == "__main__":
    unittest.main()
