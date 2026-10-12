"use client";

import Link from "next/link";
import { useState } from "react";

import { LivePriceChart } from "@/components/charts/live-price-chart";
import { INSTRUMENTS, TIMEFRAMES } from "@/lib/instruments";
import type { Market, Snapshot } from "@/lib/types";

type Props = {
  snapshot: Snapshot | null;
  initialSymbol: string;
};

export function MarketsView({ snapshot, initialSymbol }: Props) {
  const [symbol, setSymbol] = useState(initialSymbol);
  const [timeframe, setTimeframe] = useState(snapshot?.timeframe || "1h");
  const [refreshToken, setRefreshToken] = useState(0);

  const fallback: Market | null =
    snapshot?.markets.find((m) => m.symbol === symbol && m.timeframe === timeframe) ??
    null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl text-ink">Market analysis</h1>
        <p className="mt-2 text-slateui">
          Live candlesticks for every followed symbol. Switch ticker or interval to refresh the
          chart.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <label className="text-xs uppercase tracking-wide text-slateui-dim">Interval</label>
        <select
          value={timeframe}
          onChange={(e) => {
            setTimeframe(e.target.value);
            setRefreshToken((n) => n + 1);
          }}
          className="rounded-md border border-white/10 bg-charcoal px-3 py-2 text-sm text-ink outline-none focus:border-mint"
        >
          {TIMEFRAMES.map((tf) => (
            <option key={tf} value={tf}>
              {tf}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => setRefreshToken((n) => n + 1)}
          className="rounded-md border border-mint/40 bg-mint/10 px-3 py-2 text-sm text-mint hover:bg-mint/15"
        >
          Refresh live
        </button>
      </div>

      <div className="flex flex-wrap gap-2">
        {INSTRUMENTS.map((m) => (
          <Link
            key={m.symbol}
            href={`/markets?symbol=${m.symbol}`}
            onClick={() => setSymbol(m.symbol)}
            className={`rounded-full border px-3 py-1 text-sm transition ${
              m.symbol === symbol
                ? "border-mint bg-mint text-black"
                : "border-white/10 bg-charcoal text-slateui hover:border-white/20 hover:text-ink"
            }`}
          >
            {m.symbol}
          </Link>
        ))}
      </div>

      <section className="panel p-5">
        <LivePriceChart
          symbol={symbol}
          timeframe={timeframe}
          fallback={fallback}
          height={420}
          refreshToken={refreshToken}
        />
      </section>
    </div>
  );
}
