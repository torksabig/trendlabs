from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DB_PATH = Path(os.getenv("TRADING_SIM_DB", DATA_DIR / "paper_trading.db"))

# Binance public market data (no auth required).
# data-api.binance.vision is preferred in restricted regions where api.binance.com returns 451.
BINANCE_REST = os.getenv("BINANCE_REST", "https://data-api.binance.vision")
BINANCE_REST_FALLBACKS = tuple(
    u.strip()
    for u in os.getenv(
        "BINANCE_REST_FALLBACKS",
        "https://data-api.binance.vision,https://api.binance.us,https://api.binance.com",
    ).split(",")
    if u.strip()
)
DEFAULT_SYMBOL = os.getenv("TRADING_SYMBOL", "BTCUSDT")
DEFAULT_INTERVAL = os.getenv("TRADING_INTERVAL", "1h")
DEFAULT_LOOKBACK = int(os.getenv("TRADING_LOOKBACK", "500"))

# Paper trading
STARTING_CASH = float(os.getenv("STARTING_CASH", "10000"))
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "0.95"))  # fraction of cash per entry
FEE_BPS = float(os.getenv("FEE_BPS", "10"))  # 10 bps = 0.10%

# Optional LLM research agent
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

AGENTS = (
    "ema_crossover",
    "adx_trend",
    "momentum",
    "supertrend",
)

AGENT_LABELS = {
    "ema_crossover": "EMA Crossover",
    "adx_trend": "ADX Trend",
    "momentum": "Momentum",
    "supertrend": "Supertrend",
}
