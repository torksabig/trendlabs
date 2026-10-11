import Link from "next/link";

import { EquityChart } from "@/components/charts/equity-chart";
import { CandleChart } from "@/components/charts/candle-chart";
import { loadSnapshot } from "@/lib/snapshot";
import { formatMoney, formatPct } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const snapshot = await loadSnapshot();

  if (!snapshot) {
    return (
      <section className="panel fade-up p-8">
        <h1 className="font-display text-5xl text-ink">TrendLab</h1>
        <p className="mt-4 max-w-2xl text-ink/70">
          No market snapshot yet. Collect candles and run the four paper agents, then refresh.
        </p>
        <pre className="mt-6 overflow-x-auto rounded-xl bg-ink px-4 py-3 font-mono text-sm text-paper">
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
          <p className="text-xs uppercase tracking-[0.24em] text-teal-800">Multi-agent research</p>
          <h1 className="mt-2 font-display text-5xl leading-none text-ink sm:text-6xl">
            TrendLab
          </h1>
          <p className="mt-4 max-w-xl text-lg text-ink/70">
            Twelve instruments. Four competing trend agents. Simulated capital only — built for a
            $0/month research loop on Vercel, GitHub Actions, and free market data.
          </p>
          <div className="mt-6 flex flex-wrap gap-3 text-sm">
            <span className="rounded-full border border-ink/10 bg-white/70 px-3 py-1">
              Source: {snapshot.data_source}
            </span>
            <span className="rounded-full border border-ink/10 bg-white/70 px-3 py-1">
              TF: {snapshot.timeframe}
            </span>
            <span className="rounded-full border border-ink/10 bg-white/70 px-3 py-1">
              Updated: {new Date(snapshot.generated_at).toLocaleString()}
            </span>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="metric">
            <p className="text-xs uppercase tracking-wider text-ink/45">Virtual capital</p>
            <p className="mt-2 font-mono text-2xl">{formatMoney(snapshot.totals.total_equity)}</p>
          </div>
          <div className="metric">
            <p className="text-xs uppercase tracking-wider text-ink/45">Avg return</p>
            <p className="mt-2 font-mono text-2xl">{formatPct(snapshot.totals.avg_return_pct)}</p>
          </div>
          <div className="metric">
            <p className="text-xs uppercase tracking-wider text-ink/45">Agents</p>
            <p className="mt-2 font-mono text-2xl">{snapshot.totals.agents}</p>
          </div>
          <div className="metric">
            <p className="text-xs uppercase tracking-wider text-ink/45">Instruments</p>
            <p className="mt-2 font-mono text-2xl">{snapshot.totals.instruments}</p>
          </div>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
        <div className="panel fade-up p-5">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-display text-2xl">Agent equity</h2>
            <Link href="/agents" className="text-sm text-teal-800 hover:underline">
              Leaderboard
            </Link>
          </div>
          <EquityChart curves={snapshot.equity_curves} labels={labels} />
        </div>
        <div className="panel fade-up p-5">
          <h2 className="font-display text-2xl">Leaderboard</h2>
          <div className="mt-4 space-y-3">
            {snapshot.leaderboard.map((agent, idx) => (
              <div
                key={agent.id}
                className="flex items-center justify-between rounded-xl border border-ink/8 bg-white/50 px-3 py-3"
              >
                <div>
                  <p className="text-sm font-medium">
                    #{idx + 1} {agent.name}
                  </p>
                  <p className="text-xs text-ink/50">
                    {agent.open_positions} open · {agent.trades} trades · DD{" "}
                    {formatPct(agent.max_drawdown_pct)}
                  </p>
                </div>
                <div className="text-right">
                  <p className="font-mono text-sm">{formatMoney(agent.equity)}</p>
                  <p className="font-mono text-xs text-ink/60">{formatPct(agent.return_pct)}</p>
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
              <h2 className="font-display text-2xl">
                {featured.symbol} · {featured.name}
              </h2>
              <p className="text-sm text-ink/55">
                Trend {featured.trend_label} · RSI {featured.rsi_14?.toFixed(1) ?? "—"} · ADX{" "}
                {featured.adx_14?.toFixed(1) ?? "—"} · source {featured.source}
              </p>
            </div>
            <Link href={`/markets?symbol=${featured.symbol}`} className="text-sm text-teal-800 hover:underline">
              Open market analysis
            </Link>
          </div>
          <CandleChart candles={featured.candles} />
        </section>
      )}

      <section className="grid gap-4 md:grid-cols-3">
        <div className="panel p-5">
          <h3 className="font-display text-xl">Recent decisions</h3>
          <ul className="mt-3 space-y-2 text-sm">
            {snapshot.trades.slice(0, 6).map((trade) => (
              <li key={`${trade.id}-${trade.ts}`} className="border-b border-ink/8 pb-2">
                <span className="font-mono text-xs text-ink/45">{trade.side}</span> {trade.symbol} ·{" "}
                {trade.agent_id}
                <p className="text-xs text-ink/55">{trade.reason}</p>
              </li>
            ))}
          </ul>
        </div>
        <div className="panel p-5">
          <h3 className="font-display text-xl">Historical research</h3>
          <p className="mt-3 text-sm text-ink/70">
            Bullish matches: {snapshot.research.bullish.matches}
            {snapshot.research.bullish.hit_rate != null &&
              ` · hit rate ${(snapshot.research.bullish.hit_rate * 100).toFixed(0)}%`}
          </p>
          <p className="mt-2 text-sm text-ink/70">
            Bearish matches: {snapshot.research.bearish.matches}
            {snapshot.research.bearish.hit_rate != null &&
              ` · hit rate ${(snapshot.research.bearish.hit_rate * 100).toFixed(0)}%`}
          </p>
          <Link href="/research" className="mt-4 inline-block text-sm text-teal-800 hover:underline">
            Explore patterns
          </Link>
        </div>
        <div className="panel p-5">
          <h3 className="font-display text-xl">System status</h3>
          <ul className="mt-3 space-y-2 text-sm text-ink/70">
            <li>Budget: {snapshot.status.budget}</li>
            <li>Stack: {snapshot.status.deployment}</li>
            <li>Real-money trading: {snapshot.status.real_money_trading ? "enabled" : "disabled"}</li>
            <li>Last collection: {snapshot.last_collection ? new Date(snapshot.last_collection).toLocaleString() : "—"}</li>
          </ul>
        </div>
      </section>
    </div>
  );
}
