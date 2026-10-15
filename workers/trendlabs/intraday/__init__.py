"""Intraday Prediction Engine track — 15m / 1h / 4h horizons.

Rebuild of the `cursor/pe-intraday-v1-0eae` track inside this repo, mirroring the
daily track conventions in `trendlabs.prediction`:

  - Execution rule: signal at close of bar t, enter at open of bar t+1,
    exit at close of bar t+h (h = horizon in panel bars).
  - Target z = R / (sigma_N * sqrt(h)); sigma_N = std of past N bar log returns at t only.
  - Purged walk-forward in BARS; purge gap >= horizon so train labels cannot overlap test.
  - Data: base 5m bars cached under data/intraday/bars/. Alpaca v2 when keyed
    (historical SIP since 2016, 200 calls/min on the free Basic plan), Yahoo fallback
    (keyless: 1h ~730d, 5m/15m ~60d).

The daily PE track (FAIL result) and HAR-vol stay method benchmarks; they are never
part of this intraday product metric.
"""

from __future__ import annotations

from trendlabs.config import DATA_DIR, SYMBOLS

INTRA_DIR = DATA_DIR / "intraday"
BARS_DIR = INTRA_DIR / "bars"

# Liquid universe — same set as the daily PE track for comparability.
DEFAULT_UNIVERSE = list(SYMBOLS)
BENCHMARK_SYMBOL = "SPY"

PANEL_TFS = ("15m", "1h", "4h")
TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240}
# Regular session 9:30–16:00 ET = 390 minutes.
BARS_PER_DAY = {"15m": 26, "1h": 7, "4h": 2}
SESSION_MINUTES = 390

VOL_WINDOW = 20  # bars, trailing sigma window (same spirit as daily sigma_20)
DEFAULT_HORIZON_BARS = 1  # per-panel: 15m panel → 15m move, 1h panel → 1h move, 4h panel → 4h move

DATASET_VERSION = "pe_intra_v1"
DISCLAIMER = "EXPLORATORY — SURVIVORSHIP BIAS NOT FULLY CONTROLLED"

PURGE_NOTE = (
    "Purge gap >= horizon_bars between train_end and test_start so labels from train "
    "cannot overlap the test window. Bar-count purge, like the daily track's trading-day purge."
)

EXECUTION_RULE = (
    "Signal at close of bar t (no future bars); enter at open of bar t+1; exit at close of "
    "bar t+h. R = ln(close_{t+h} / open_{t+1}); z = R / (sigma_vol_window * sqrt(h)) with "
    "sigma from past vol_window bar log returns at t only. Panel bars are RTH-only, "
    "session-anchored at 9:30 ET (last 4h bar of a session is 2.5h: 13:30-16:00)."
)
