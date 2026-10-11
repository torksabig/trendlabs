from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("TRENDLAB_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.getenv("TRENDLAB_DB", DATA_DIR / "trendlabs.db"))
SNAPSHOT_PATH = Path(os.getenv("TRENDLAB_SNAPSHOT", DATA_DIR / "snapshots" / "latest.json"))

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
# IEX is free-tier friendly; SIP may require paid plan — record source metadata either way.
ALPACA_FEED = os.getenv("ALPACA_FEED", "iex")

STARTING_CASH = float(os.getenv("STARTING_CASH", "10000"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
FEE_BPS = float(os.getenv("FEE_BPS", "5"))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "5"))

DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1h")
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "400"))

INSTRUMENTS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
    {"symbol": "NVDA", "name": "NVIDIA Corp.", "sector": "Semiconductors"},
    {"symbol": "AMD", "name": "Advanced Micro Devices", "sector": "Semiconductors"},
    {"symbol": "JPM", "name": "JPMorgan Chase", "sector": "Financial services"},
    {"symbol": "GS", "name": "Goldman Sachs", "sector": "Financial services"},
    {"symbol": "XOM", "name": "Exxon Mobil", "sector": "Energy"},
    {"symbol": "CVX", "name": "Chevron", "sector": "Energy"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
    {"symbol": "UNH", "name": "UnitedHealth Group", "sector": "Healthcare"},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "sector": "Market benchmarks"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "sector": "Market benchmarks"},
]

SYMBOLS = [i["symbol"] for i in INSTRUMENTS]

TIMEFRAMES = {
    "15m": {"yf": "15m", "alpaca": "15Min", "period": "60d"},
    "1h": {"yf": "60m", "alpaca": "1Hour", "period": "730d"},
    "4h": {"yf": "60m", "alpaca": "1Hour", "period": "730d"},  # resampled from 1h
    "1d": {"yf": "1d", "alpaca": "1Day", "period": "5y"},
}

AGENTS = [
    {
        "id": "ema_trend",
        "name": "Agent A — EMA Trend",
        "strategy": "ema_trend",
        "description": "EMA 20/50/200 trend confirmation with protective stops.",
    },
    {
        "id": "breakout",
        "name": "Agent B — Breakout",
        "strategy": "breakout",
        "description": "Enter above 20-bar highs; exit below 10-bar lows.",
    },
    {
        "id": "momentum",
        "name": "Agent C — Momentum",
        "strategy": "momentum",
        "description": "Momentum + ADX + trend filter alignment.",
    },
    {
        "id": "supertrend",
        "name": "Agent D — Supertrend",
        "strategy": "supertrend",
        "description": "Volatility-adjusted Supertrend direction follower.",
    },
]
