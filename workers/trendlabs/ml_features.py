"""Backfill frozen feature snapshots into market_features (Phase B persistence).

Writes the PIT feature rows for every bar with complete features (warm-up head and
incomplete rows skipped) so predictions/train data can be audited against exactly
what the model saw. Deterministic and versioned via FEATURE_VERSION.
"""

from __future__ import annotations

import argparse

from trendlabs.config import SYMBOLS
from trendlabs.db import Store
from trendlabs.features import FEATURE_COLUMNS, FEATURE_VERSION, build_feature_frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill market_features snapshots")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument(
        "--min-history",
        type=int,
        default=60,
        help="Skip the first N bars (indicator warm-up, mirrors dataset_from_candles)",
    )
    args = parser.parse_args()

    store = Store()
    symbols = args.symbols or SYMBOLS
    total = 0
    for symbol in symbols:
        candles = store.load_candles(symbol, args.timeframe)
        if candles.empty:
            print(f"{symbol} {args.timeframe}: no candles, skipped")
            continue
        feat = build_feature_frame(candles)
        feat = feat.iloc[args.min_history :].dropna(subset=FEATURE_COLUMNS)
        n = store.upsert_market_features(
            feat,
            symbol=symbol,
            timeframe=args.timeframe,
            feature_version=FEATURE_VERSION,
            columns=FEATURE_COLUMNS,
        )
        total += n
        print(f"{symbol} {args.timeframe}: {n} snapshots [{feat['ts'].iloc[0]} → {feat['ts'].iloc[-1]}] ({FEATURE_VERSION})")
    print(f"Done. total={total} feature_version={FEATURE_VERSION}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
