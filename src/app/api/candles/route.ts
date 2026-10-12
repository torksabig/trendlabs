import { NextRequest, NextResponse } from "next/server";

import { instrumentFor, isTimeframe, SYMBOLS, type Timeframe } from "@/lib/instruments";
import { loadSnapshot } from "@/lib/snapshot";
import { classifyTrend, fetchYahooCandles } from "@/lib/yahoo-candles";
import type { Market } from "@/lib/types";

export const dynamic = "force-dynamic";
export const maxDuration = 30;

function ema(values: number[], period: number): number | null {
  if (values.length < period) return null;
  const k = 2 / (period + 1);
  let value = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < values.length; i += 1) {
    value = values[i] * k + value * (1 - k);
  }
  return value;
}

function enrichMarket(
  symbol: string,
  timeframe: Timeframe,
  candles: Market["candles"],
  source: string,
): Market {
  const instrument = instrumentFor(symbol);
  const closes = candles.map((c) => c.close);
  const last = candles[candles.length - 1];
  return {
    symbol: instrument.symbol,
    name: instrument.name,
    sector: instrument.sector,
    timeframe,
    last_close: last.close,
    trend_label: classifyTrend(candles),
    ema_20: ema(closes, 20),
    ema_50: ema(closes, 50),
    ema_200: ema(closes, 200),
    rsi_14: null,
    adx_14: null,
    atr_14: null,
    source,
    as_of: new Date(last.time * 1000).toISOString(),
    candles,
  };
}

export async function GET(request: NextRequest) {
  const symbol = (request.nextUrl.searchParams.get("symbol") || "AAPL").toUpperCase();
  const timeframeRaw = request.nextUrl.searchParams.get("timeframe") || "1h";
  const live = request.nextUrl.searchParams.get("live") === "1";

  if (!SYMBOLS.includes(symbol)) {
    return NextResponse.json(
      { error: `Unsupported symbol ${symbol}. Followed: ${SYMBOLS.join(", ")}` },
      { status: 400 },
    );
  }
  if (!isTimeframe(timeframeRaw)) {
    return NextResponse.json(
      { error: `Unsupported timeframe ${timeframeRaw}` },
      { status: 400 },
    );
  }
  const timeframe = timeframeRaw;

  try {
    if (!live) {
      const snapshot = await loadSnapshot();
      const market = snapshot?.markets.find((m) => m.symbol === symbol);
      if (
        market &&
        market.timeframe === timeframe &&
        market.candles.length >= 20
      ) {
        return NextResponse.json({
          ...market,
          live: false,
        });
      }
    }

    const { candles, source } = await fetchYahooCandles(symbol, timeframe, {
      bustCache: live,
    });
    const market = enrichMarket(symbol, timeframe, candles, source);
    return NextResponse.json({
      ...market,
      live: true,
    });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : String(err) },
      { status: 502 },
    );
  }
}
