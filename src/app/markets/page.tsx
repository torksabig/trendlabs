import Link from "next/link";

import { CandleChart } from "@/components/charts/candle-chart";
import { loadSnapshot } from "@/lib/snapshot";
import { formatMoney } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function MarketsPage({
  searchParams,
}: {
  searchParams: Promise<{ symbol?: string }>;
}) {
  const snapshot = await loadSnapshot();
  const params = await searchParams;

  if (!snapshot || snapshot.markets.length === 0) {
    return <p className="panel p-6">No market data. Run `npm run seed`.</p>;
  }

  const symbol = (params.symbol || "AAPL").toUpperCase();
  const market = snapshot.markets.find((m) => m.symbol === symbol) ?? snapshot.markets[0];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl">Market analysis</h1>
        <p className="mt-2 text-ink/65">
          Interactive candlesticks with trend classification and indicator snapshots.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {snapshot.markets.map((m) => (
          <Link
            key={m.symbol}
            href={`/markets?symbol=${m.symbol}`}
            className={`rounded-full border px-3 py-1 text-sm ${
              m.symbol === market.symbol
                ? "border-teal-800 bg-teal-800 text-paper"
                : "border-ink/10 bg-white/60 text-ink/70 hover:bg-white"
            }`}
          >
            {m.symbol}
          </Link>
        ))}
      </div>

      <section className="panel p-5">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="font-display text-3xl">
              {market.symbol}{" "}
              <span className="text-ink/45">{market.name}</span>
            </h2>
            <p className="text-sm text-ink/55">
              {market.sector} · {market.timeframe} · {market.source}
            </p>
          </div>
          <div className="text-right">
            <p className="font-mono text-2xl">{formatMoney(market.last_close)}</p>
            <p className="text-sm capitalize text-ink/60">Trend: {market.trend_label}</p>
          </div>
        </div>
        <CandleChart candles={market.candles} height={420} />
        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-6">
          {[
            ["EMA 20", market.ema_20],
            ["EMA 50", market.ema_50],
            ["EMA 200", market.ema_200],
            ["RSI 14", market.rsi_14],
            ["ADX 14", market.adx_14],
            ["ATR 14", market.atr_14],
          ].map(([label, value]) => (
            <div key={label as string} className="rounded-xl bg-white/60 px-3 py-2">
              <p className="text-xs uppercase tracking-wide text-ink/45">{label}</p>
              <p className="font-mono text-sm">
                {typeof value === "number" ? value.toFixed(2) : "—"}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
