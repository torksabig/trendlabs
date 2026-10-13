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

def _env(name: str, default: str = "") -> str:
    """Read env var; treat missing/blank as default (Actions empty secrets → blank)."""
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


# Research LLM providers (never used for trade execution).
# TRENDLAB_LLM_PROVIDER: auto | google | groq | openrouter | cloudflare | cerebras | mistral | ollama | fallback
TRENDLAB_LLM_PROVIDER = _env("TRENDLAB_LLM_PROVIDER", "auto").lower() or "auto"

# Local Ollama (Mac)
OLLAMA_HOST = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen3:14b-q4_K_M")

# Google AI Studio (Gemini) — free tier
GOOGLE_AI_API_KEY = _env("GOOGLE_AI_API_KEY") or _env("GEMINI_API_KEY")
GOOGLE_AI_MODEL = _env("GOOGLE_AI_MODEL", "gemini-2.0-flash")

# Groq — free tier, fast inference
GROQ_API_KEY = _env("GROQ_API_KEY")
GROQ_BASE_URL = _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = _env("GROQ_MODEL", "llama-3.3-70b-versatile")

# OpenRouter — free model variants (:free / openrouter/free)
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = _env("OPENROUTER_MODEL", "openrouter/free")

# Cloudflare Workers AI — free daily neuron allocation
CLOUDFLARE_ACCOUNT_ID = _env("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = _env("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_AI_MODEL = _env("CLOUDFLARE_AI_MODEL", "@cf/meta/llama-3.1-8b-instruct")

# Cerebras — free / experimentation tier
CEREBRAS_API_KEY = _env("CEREBRAS_API_KEY")
CEREBRAS_BASE_URL = _env("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_MODEL = _env("CEREBRAS_MODEL", "llama3.1-8b")

# Mistral — experimentation tier
MISTRAL_API_KEY = _env("MISTRAL_API_KEY")
MISTRAL_BASE_URL = _env("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
MISTRAL_MODEL = _env("MISTRAL_MODEL", "mistral-small-latest")

SUPABASE_URL = os.getenv("SUPABASE_URL", os.getenv("NEXT_PUBLIC_SUPABASE_URL", ""))
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""))

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

# Yahoo period limits (free): 15m≈60d, 60m≈730d, 1d≈max/10y.
# Alpaca Basic uses history_start when full_history extract is requested.
TIMEFRAMES = {
    "15m": {
        "yf": "15m",
        "alpaca": "15Min",
        "period": "60d",
        "max_period": "60d",
        "history_start": "2024-01-01T00:00:00Z",
        "seconds": 15 * 60,
        "ml_priority": 2,
    },
    "1h": {
        "yf": "60m",
        "alpaca": "1Hour",
        "period": "730d",
        "max_period": "730d",
        "history_start": "2022-01-01T00:00:00Z",
        "seconds": 60 * 60,
        "ml_priority": 3,
    },
    "4h": {
        "yf": "60m",
        "alpaca": "1Hour",
        "period": "730d",
        "max_period": "730d",
        "history_start": "2022-01-01T00:00:00Z",
        "seconds": 4 * 60 * 60,
        "ml_priority": 4,
    },  # resampled from 1h
    "1d": {
        "yf": "1d",
        "alpaca": "1Day",
        "period": "10y",
        "max_period": "max",
        "history_start": "2015-01-01T00:00:00Z",
        "seconds": 24 * 60 * 60,
        "ml_priority": 1,
    },
}

# Default extract order: daily first (deepest history), then intraday.
EXTRACT_TIMEFRAMES = ["1d", "1h", "15m", "4h"]

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
