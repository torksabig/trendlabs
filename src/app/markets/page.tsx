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
    return <p className="panel p-6 text-slateui">No market data. Run `npm run seed`.</p>;
  }

  const symbol = (params.symbol || "AAPL").toUpperCase();
  const market = snapshot.markets.find((m) => m.symbol === symbol) ?? snapshot.markets[0];
  const trendClass =
    market.trend_label === "bullish"
      ? "text-mint"
      : market.trend_label === "bearish"
        ? "text-crimson"
        : "text-slateui";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl text-ink">Market analysis</h1>
        <p className="mt-2 text-slateui">
          Interactive candlesticks with trend classification and indicator snapshots.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {snapshot.markets.map((m) => (
          <Link
            key={m.symbol}
            href={`/markets?symbol=${m.symbol}`}
            className={`rounded-full border px-3 py-1 text-sm transition ${
              m.symbol === market.symbol
                ? "border-mint bg-mint text-black"
                : "border-white/10 bg-charcoal text-slateui hover:border-white/20 hover:text-ink"
            }`}
          >
            {m.symbol}
          </Link>
        ))}
      </div>

      <section className="panel p-5">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="font-display text-3xl text-ink">
              {market.symbol} <span className="text-slateui-dim">{market.name}</span>
            </h2>
            <p className="text-sm text-slateui">
              {market.sector} · {market.timeframe} · {market.source}
            </p>
          </div>
          <div className="text-right">
            <p className="font-mono text-2xl text-ink">{formatMoney(market.last_close)}</p>
            <p className={`text-sm capitalize ${trendClass}`}>Trend: {market.trend_label}</p>
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
            <div key={label as string} className="rounded-xl border border-white/10 bg-charcoal px-3 py-2">
              <p className="text-xs uppercase tracking-wide text-slateui-dim">{label}</p>
              <p className="font-mono text-sm text-ink">
                {typeof value === "number" ? value.toFixed(2) : "—"}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
