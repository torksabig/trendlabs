import type { Candle } from "@/lib/types";
import type { Timeframe } from "@/lib/instruments";

type YahooMeta = {
  interval: string;
  range: string;
};

const YAHOO_META: Record<Timeframe, YahooMeta> = {
  "15m": { interval: "15m", range: "60d" },
  "1h": { interval: "60m", range: "6mo" },
  "4h": { interval: "60m", range: "6mo" },
  "1d": { interval: "1d", range: "2y" },
};

const LOOKBACK: Record<Timeframe, number> = {
  "15m": 180,
  "1h": 180,
  "4h": 180,
  "1d": 260,
};

type YahooChartResponse = {
  chart?: {
    result?: Array<{
      timestamp?: number[];
      indicators?: {
        quote?: Array<{
          open?: Array<number | null>;
          high?: Array<number | null>;
          low?: Array<number | null>;
          close?: Array<number | null>;
          volume?: Array<number | null>;
        }>;
      };
    }>;
    error?: { description?: string } | null;
  };
};

function resample4h(candles: Candle[]): Candle[] {
  if (candles.length === 0) return [];
  const buckets = new Map<number, Candle[]>();
  for (const c of candles) {
    const date = new Date(c.time * 1000);
    const hour = Math.floor(date.getUTCHours() / 4) * 4;
    date.setUTCHours(hour, 0, 0, 0);
    const key = Math.floor(date.getTime() / 1000);
    const list = buckets.get(key) ?? [];
    list.push(c);
    buckets.set(key, list);
  }
  return [...buckets.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([time, bars]) => ({
      time,
      open: bars[0].open,
      high: Math.max(...bars.map((b) => b.high)),
      low: Math.min(...bars.map((b) => b.low)),
      close: bars[bars.length - 1].close,
      volume: bars.reduce((sum, b) => sum + b.volume, 0),
    }));
}

/** Fetch OHLCV candles from Yahoo Finance chart API (no API key). */
export async function fetchYahooCandles(
  symbol: string,
  timeframe: Timeframe,
  opts: { bustCache?: boolean } = {},
): Promise<{ candles: Candle[]; source: string }> {
  const meta = YAHOO_META[timeframe];
  const url = new URL(
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol.toUpperCase())}`,
  );
  url.searchParams.set("interval", meta.interval);
  url.searchParams.set("range", meta.range);
  url.searchParams.set("includePrePost", "false");
  url.searchParams.set("events", "div|split");

  const res = await fetch(url.toString(), {
    headers: {
      "User-Agent": "TrendLab/1.0 (+https://github.com/torksabig/trendlabs)",
      Accept: "application/json",
    },
    ...(opts.bustCache
      ? { cache: "no-store" as const }
      : { next: { revalidate: 120 } }),
  });

  if (!res.ok) {
    throw new Error(`Yahoo chart HTTP ${res.status} for ${symbol}`);
  }

  const payload = (await res.json()) as YahooChartResponse;
  if (payload.chart?.error) {
    throw new Error(payload.chart.error.description || "Yahoo chart error");
  }

  const result = payload.chart?.result?.[0];
  const timestamps = result?.timestamp ?? [];
  const quote = result?.indicators?.quote?.[0];
  if (!timestamps.length || !quote) {
    throw new Error(`No Yahoo candles for ${symbol} ${timeframe}`);
  }

  let candles: Candle[] = [];
  for (let i = 0; i < timestamps.length; i += 1) {
    const open = quote.open?.[i];
    const high = quote.high?.[i];
    const low = quote.low?.[i];
    const close = quote.close?.[i];
    const volume = quote.volume?.[i] ?? 0;
    if (
      open == null ||
      high == null ||
      low == null ||
      close == null ||
      !Number.isFinite(open) ||
      !Number.isFinite(high) ||
      !Number.isFinite(low) ||
      !Number.isFinite(close)
    ) {
      continue;
    }
    candles.push({
      time: timestamps[i],
      open,
      high,
      low,
      close,
      volume: Number(volume) || 0,
    });
  }

  if (timeframe === "4h") {
    candles = resample4h(candles);
  }

  candles = candles.slice(-LOOKBACK[timeframe]);
  if (candles.length < 5) {
    throw new Error(`Insufficient Yahoo candles for ${symbol} ${timeframe}`);
  }

  return { candles, source: "yfinance" };
}

export function classifyTrend(candles: Candle[]): string {
  if (candles.length < 20) return "neutral";
  const closes = candles.map((c) => c.close);
  const ema = (period: number) => {
    const k = 2 / (period + 1);
    let value = closes[0];
    for (let i = 1; i < closes.length; i += 1) {
      value = closes[i] * k + value * (1 - k);
    }
    return value;
  };
  const ema20 = ema(20);
  const ema50 = closes.length >= 50 ? ema(50) : ema20;
  const last = closes[closes.length - 1];
  if (last > ema20 && ema20 >= ema50) return "bullish";
  if (last < ema20 && ema20 <= ema50) return "bearish";
  return "neutral";
}
