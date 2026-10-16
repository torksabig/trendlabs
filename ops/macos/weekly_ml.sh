#!/bin/bash
# TrendLab weekly candidate retrain — Fri after close (17:00 ET via launchd).
# Fresh candles → retrain all symbols (embargoed split) → scorecard writes the
# approval registry → predictions persist against the NEW approved set → mature book.
set -u
cd "$(dirname "$0")/../.."

export DYLD_FALLBACK_LIBRARY_PATH="$HOME/libomp/lib"
export PYTHONPATH=workers
PY=./.venv/bin/python
LOG_DIR=data/ml/logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y-%m-%d)
LOG="$LOG_DIR/weekly_$STAMP.log"

{
  echo "=== weekly_ml $STAMP $(date -u +%FT%TZ) ==="
  echo "--- collect"
  "$PY" -m trendlabs.collect --timeframe 1d --full-history --no-snapshot 2>&1 | grep -viE "notopenssl|warnings.warn" | tail -3
  echo "--- ml-train"
  "$PY" -m trendlabs.ml_train --timeframe 1d 2>&1 | grep -viE "notopenssl|warnings.warn|coef_|@ " | grep -E "accuracy=|ERROR" || true
  echo "--- ml-features"
  "$PY" -m trendlabs.ml_features --timeframe 1d 2>&1 | grep -viE "notopenssl|warnings.warn" | tail -1
  echo "--- ml-scorecard (writes approved_models.json)"
  "$PY" -m trendlabs.ml_scorecard --timeframe 1d 2>&1 | grep -viE "notopenssl|warnings.warn|coef_|@ " | tail -3
  echo "--- ml-predict against new approvals"
  "$PY" -m trendlabs.ml_predict --timeframe 1d --horizon 1 --persist 2>&1 | grep -viE "notopenssl|warnings.warn|coef_|@ " | tail -13
  echo "--- ml-paper"
  "$PY" -m trendlabs.ml_paper 2>&1 | grep -viE "notopenssl|warnings.warn" | tail -6
  echo "=== done $(date -u +%FT%TZ) ==="
} >> "$LOG" 2>&1

find "$LOG_DIR" -name "weekly_*.log" -mtime +120 -delete
