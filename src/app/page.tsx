import Link from "next/link";

import { CandleChart } from "@/components/charts/candle-chart";
import { EquityChart } from "@/components/charts/equity-chart";
import { loadSnapshot } from "@/lib/snapshot";
import { formatMoney, formatPct, pctTone } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const snapshot = await loadSnapshot();

  if (!snapshot) {
    return (
      <section className="panel fade-up p-8">
        <h1 className="font-display text-5xl text-ink">TrendLab</h1>
        <p className="mt-4 max-w-2xl text-slateui">
          No market snapshot yet. Collect candles and run the four paper agents, then refresh.
        </p>
        <pre className="mt-6 overflow-x-auto rounded-xl bg-charcoal-2 px-4 py-3 font-mono text-sm text-mint">
          npm run seed
        </pre>
      </section>
    );
  }

  const featured = snapshot.markets.find((m) => m.symbol === "AAPL") ?? snapshot.markets[0];
  const labels = Object.fromEntries(snapshot.leaderboard.map((a) => [a.id, a.name]));

  return (
    <div className="space-y-8">
      <section className="fade-up grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-mint">Multi-agent research</p>
          <h1 className="mt-2 font-display text-5xl leading-none text-ink sm:text-6xl">
            TrendLab
          </h1>
          <p className="mt-4 max-w-xl text-lg text-slateui">
            Twelve instruments. Four competing trend agents. Simulated capital only — built for a
            $0/month research loop on Vercel, GitHub Actions, and free market data.
          </p>
          <div className="mt-6 flex flex-wrap gap-3 text-sm text-slateui">
            <span className="rounded-full border border-white/10 bg-charcoal px-3 py-1">
              Source: {snapshot.data_source}
            </span>
            <span className="rounded-full border border-white/10 bg-charcoal px-3 py-1">
              TF: {snapshot.timeframe}
            </span>
            <span className="rounded-full border border-white/10 bg-charcoal px-3 py-1">
              Updated: {new Date(snapshot.generated_at).toLocaleString()}
            </span>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="metric">
            <p className="text-xs uppercase tracking-wider text-slateui-dim">Virtual capital</p>
            <p className="mt-2 font-mono text-2xl text-ink">
              {formatMoney(snapshot.totals.total_equity)}
            </p>
          </div>
          <div className="metric">
            <p className="text-xs uppercase tracking-wider text-slateui-dim">Avg return</p>
            <p className={`mt-2 font-mono text-2xl ${pctTone(snapshot.totals.avg_return_pct)}`}>
              {formatPct(snapshot.totals.avg_return_pct)}
            </p>
          </div>
          <div className="metric">
            <p className="text-xs uppercase tracking-wider text-slateui-dim">Agents</p>
            <p className="mt-2 font-mono text-2xl text-ink">{snapshot.totals.agents}</p>
          </div>
          <div className="metric">
            <p className="text-xs uppercase tracking-wider text-slateui-dim">Instruments</p>
            <p className="mt-2 font-mono text-2xl text-ink">{snapshot.totals.instruments}</p>
          </div>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
        <div className="panel fade-up p-5">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-display text-2xl text-ink">Agent equity</h2>
            <Link href="/agents" className="text-sm text-mint hover:underline">
              Leaderboard
            </Link>
          </div>
          <EquityChart curves={snapshot.equity_curves} labels={labels} />
        </div>
        <div className="panel fade-up p-5">
          <h2 className="font-display text-2xl text-ink">Leaderboard</h2>
          <div className="mt-4 space-y-3">
            {snapshot.leaderboard.map((agent, idx) => (
              <div
                key={agent.id}
                className="flex items-center justify-between rounded-xl border border-white/10 bg-charcoal px-3 py-3"
              >
                <div>
                  <p className="text-sm font-medium text-ink">
                    #{idx + 1} {agent.name}
                  </p>
                  <p className="text-xs text-slateui-dim">
                    {agent.open_positions} open · {agent.trades} trades · DD{" "}
                    <span className={pctTone(agent.max_drawdown_pct)}>
                      {formatPct(agent.max_drawdown_pct)}
                    </span>
                  </p>
                </div>
                <div className="text-right">
                  <p className="font-mono text-sm text-ink">{formatMoney(agent.equity)}</p>
                  <p className={`font-mono text-xs ${pctTone(agent.return_pct)}`}>
                    {formatPct(agent.return_pct)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {featured && (
        <section className="panel fade-up p-5">
          <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="font-display text-2xl text-ink">
                {featured.symbol} · {featured.name}
              </h2>
              <p className="text-sm text-slateui">
                Trend{" "}
                <span
                  className={
                    featured.trend_label === "bullish"
                      ? "text-mint"
                      : featured.trend_label === "bearish"
                        ? "text-crimson"
                        : "text-slateui"
                  }
                >
                  {featured.trend_label}
                </span>{" "}
                · RSI {featured.rsi_14?.toFixed(1) ?? "—"} · ADX {featured.adx_14?.toFixed(1) ?? "—"}{" "}
                · source {featured.source}
              </p>
            </div>
            <Link
              href={`/markets?symbol=${featured.symbol}`}
              className="text-sm text-mint hover:underline"
            >
              Open market analysis
            </Link>
          </div>
          <CandleChart candles={featured.candles} />
        </section>
      )}

      <section className="grid gap-4 md:grid-cols-3">
        <div className="panel p-5">
          <h3 className="font-display text-xl text-ink">Recent decisions</h3>
          <ul className="mt-3 space-y-2 text-sm">
            {snapshot.trades.slice(0, 6).map((trade) => (
              <li key={`${trade.id}-${trade.ts}`} className="border-b border-white/10 pb-2">
                <span
                  className={`font-mono text-xs ${
                    trade.side === "BUY" ? "text-mint" : "text-crimson"
                  }`}
                >
                  {trade.side}
                </span>{" "}
                <span className="text-ink">{trade.symbol}</span> ·{" "}
                <span className="text-slateui">{trade.agent_id}</span>
                <p className="text-xs text-slateui-dim">{trade.reason}</p>
              </li>
            ))}
          </ul>
        </div>
        <div className="panel p-5">
          <h3 className="font-display text-xl text-ink">Historical research</h3>
          <p className="mt-3 text-sm text-slateui">
            Bullish matches: <span className="text-mint">{snapshot.research.bullish.matches}</span>
            {snapshot.research.bullish.hit_rate != null &&
              ` · hit rate ${(snapshot.research.bullish.hit_rate * 100).toFixed(0)}%`}
          </p>
          <p className="mt-2 text-sm text-slateui">
            Bearish matches: <span className="text-crimson">{snapshot.research.bearish.matches}</span>
            {snapshot.research.bearish.hit_rate != null &&
              ` · hit rate ${(snapshot.research.bearish.hit_rate * 100).toFixed(0)}%`}
          </p>
          <Link href="/research" className="mt-4 inline-block text-sm text-mint hover:underline">
            Explore patterns
          </Link>
        </div>
        <div className="panel p-5">
          <h3 className="font-display text-xl text-ink">System status</h3>
          <ul className="mt-3 space-y-2 text-sm text-slateui">
            <li>Budget: {snapshot.status.budget}</li>
            <li>Stack: {snapshot.status.deployment}</li>
            <li>
              Real-money trading:{" "}
              <span className="text-mint">
                {snapshot.status.real_money_trading ? "enabled" : "disabled"}
              </span>
            </li>
            <li>
              Last collection:{" "}
              {snapshot.last_collection
                ? new Date(snapshot.last_collection).toLocaleString()
                : "—"}
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
