#!/bin/bash
# TrendLab daily ML loop — run after market close (16:10 ET via launchd).
# Fresh daily candles → feature snapshots → predictions (paper trades for approved
# symbols) → mature paper trades. Logs to data/ml/logs/.
set -u
cd "$(dirname "$0")/../.."

export DYLD_FALLBACK_LIBRARY_PATH="$HOME/libomp/lib"
export PYTHONPATH=workers
PY=./.venv/bin/python
LOG_DIR=data/ml/logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y-%m-%d)
LOG="$LOG_DIR/daily_$STAMP.log"

{
  echo "=== daily_ml $STAMP $(date -u +%FT%TZ) ==="
  echo "--- collect"
  "$PY" -m trendlabs.collect --timeframe 1d --full-history --no-snapshot 2>&1 | grep -viE "notopenssl|warnings.warn"
  echo "--- ml-features (snapshot new bars)"
  "$PY" -m trendlabs.ml_features --timeframe 1d 2>&1 | grep -viE "notopenssl|warnings.warn" | tail -2
  echo "--- ml-predict"
  "$PY" -m trendlabs.ml_predict --timeframe 1d --horizon 1 --persist 2>&1 | grep -viE "notopenssl|warnings.warn|coef_|@ "
  echo "--- ml-paper"
  "$PY" -m trendlabs.ml_paper 2>&1 | grep -viE "notopenssl|warnings.warn"
  echo "=== done $(date -u +%FT%TZ) ==="
} >> "$LOG" 2>&1

# Trim logs older than 60 days
find "$LOG_DIR" -name "daily_*.log" -mtime +60 -delete
