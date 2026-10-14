# Repeating Trend Detection

## Method

Normalized close-path motif matching on extracted OHLCV:

1. Take completed close windows of length `W` (default 20 bars).
2. Z-score each window (shape only).
3. Match a recent window to earlier non-overlapping windows with Pearson correlation ≥ `min_corr` (default 0.90).
4. Score historical matches by the forward return over the next `H` bars (default 10) after each match ended.
5. Rank by occurrence count × continuation edge × mean shape correlation.

No missing prices are invented. Forward returns use only bars after each historical window.

## Command

```bash
npm run repeating-trends
# or
python -m trendlabs.repeating_trends --timeframe 1d --window 20 --horizon 10

npm run test:repeating-trends
```

Output: `data/repeating_trends/latest.json` (mirrored to `public/data/repeating_trends/latest.json`).

## Reading results

| Field | Meaning |
| --- | --- |
| `direction` | Net move of matched windows (`up` / `down` / `mixed`) |
| `occurrences` | Current shape + historical matches |
| `continuation_rate` | Share of historical matches where the next `H` bars continued the window direction |
| `avg_forward_return_pct` | Mean return over the next `H` bars after matches |
| `mean_shape_corr` | Average shape correlation of matches |
| `latest_match_ts` | Start of the most recent window used as an anchor |

Continuation near 50% means the repeating shape is common but not a reliable directional edge.
