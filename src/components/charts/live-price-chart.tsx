"use client";

import { useEffect, useState } from "react";

import { CandleChart } from "@/components/charts/candle-chart";
import type { Market } from "@/lib/types";
import { formatMoney } from "@/lib/utils";

type Props = {
  symbol: string;
  timeframe: string;
  fallback?: Market | null;
  height?: number;
  /** Increment to force a Yahoo live refresh. */
  refreshToken?: number;
};

type CandleResponse = Market & { live?: boolean; error?: string };

export function LivePriceChart({
  symbol,
  timeframe,
  fallback = null,
  height = 300,
  refreshToken = 0,
}: Props) {
  const [market, setMarket] = useState<Market | null>(fallback);
  const [live, setLive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [asOf, setAsOf] = useState<string | null>(fallback?.as_of ?? null);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const forceLive = refreshToken > 0;

    async function load() {
      setLoading(true);
      setError(null);
      // Optimistic paint from snapshot while network catches up
      if (
        !forceLive &&
        fallback &&
        fallback.symbol === symbol &&
        fallback.timeframe === timeframe
      ) {
        setMarket(fallback);
      }

      try {
        const params = new URLSearchParams({
          symbol,
          timeframe,
        });
        if (forceLive) params.set("live", "1");
        const res = await fetch(`/api/candles?${params.toString()}`, {
          cache: "no-store",
          signal: controller.signal,
        });
        const data = (await res.json()) as CandleResponse;
        if (cancelled) return;
        if (!res.ok) {
          setError(data.error || `Failed to load ${symbol}`);
          setLoading(false);
          return;
        }
        setMarket(data);
        setLive(Boolean(data.live));
        setAsOf(data.as_of);
        setLoading(false);
      } catch (err) {
        if (cancelled || controller.signal.aborted) return;
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      }
    }

    void load();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [symbol, timeframe, refreshToken, fallback]);

  const trendClass =
    market?.trend_label === "bullish"
      ? "text-mint"
      : market?.trend_label === "bearish"
        ? "text-crimson"
        : "text-slateui";

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="font-display text-2xl text-ink">
            Price · {symbol}
            {market ? (
              <span className="ml-2 text-base font-sans text-slateui-dim">{market.name}</span>
            ) : null}
          </h2>
          <p className="text-sm text-slateui">
            {market ? (
              <>
                <span className={`capitalize ${trendClass}`}>{market.trend_label}</span>
                {" · "}
                {formatMoney(market.last_close)}
                {" · "}
                {timeframe}
                {" · "}
                source {market.source}
                {live ? " (live)" : " (snapshot)"}
              </>
            ) : loading ? (
              <>Loading {symbol} · {timeframe}…</>
            ) : (
              <>No candles for {symbol}</>
            )}
          </p>
          {asOf ? (
            <p className="mt-1 text-xs text-slateui-dim">
              As of {new Date(asOf).toLocaleString()}
            </p>
          ) : null}
        </div>
        {loading ? (
          <span className="rounded-md border border-white/10 px-2 py-1 text-xs text-slateui">
            Updating…
          </span>
        ) : null}
      </div>

      {error ? (
        <p className="mb-3 rounded-md border border-crimson/40 bg-crimson/10 px-3 py-2 text-sm text-crimson">
          {error}
          {fallback?.candles?.length ? " Showing last snapshot bars." : ""}
        </p>
      ) : null}

      {market?.candles?.length ? (
        <CandleChart
          key={`${symbol}-${timeframe}-${market.candles.length}-${market.as_of}`}
          candles={market.candles}
          height={height}
        />
      ) : (
        <div
          className="flex items-center justify-center rounded-md border border-dashed border-white/10 text-sm text-slateui"
          style={{ height }}
        >
          {loading ? "Fetching candles…" : "No candle data"}
        </div>
      )}
    </div>
  );
}
