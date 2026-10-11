"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { CandleChart } from "@/components/charts/candle-chart";
import { EquityChart } from "@/components/charts/equity-chart";
import type { Snapshot } from "@/lib/types";
import { formatMoney, formatPct, pctTone } from "@/lib/utils";

const INTERVALS = ["15m", "1h", "4h", "1d"];

type Props = {
  snapshot: Snapshot;
};

export function ArenaDashboard({ snapshot }: Props) {
  const router = useRouter();
  const [symbol, setSymbol] = useState(
    snapshot.markets.find((m) => m.symbol === "AAPL")?.symbol ??
      snapshot.markets[0]?.symbol ??
      "AAPL",
  );
  const [interval, setInterval] = useState(snapshot.timeframe || "1h");
  const [message, setMessage] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const market = useMemo(
    () => snapshot.markets.find((m) => m.symbol === symbol) ?? snapshot.markets[0],
    [snapshot.markets, symbol],
  );
  const labels = useMemo(
    () => Object.fromEntries(snapshot.leaderboard.map((a) => [a.id, shortName(a.name)])),
    [snapshot.leaderboard],
  );
  const latestReport = snapshot.research.ai_reports?.[0];

  async function runJob(job: string) {
    setMessage(null);
    setPending(true);
    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job, symbol, timeframe: interval }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        setMessage(data.error || data.output || "Job failed");
        return;
      }
      setMessage(`${job} finished`);
      router.refresh();
    } catch (err) {
      setMessage(String(err));
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="fade-up grid gap-6 lg:grid-cols-[260px_minmax(0,1fr)]">
      {/* Sidebar — Trend Arena control rail */}
      <aside className="panel h-fit p-4 lg:sticky lg:top-4">
        <h2 className="text-xs uppercase tracking-[0.22em] text-slateui-dim">Controls</h2>

        <label className="mt-4 block text-xs text-slateui">Symbol</label>
        <select
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          className="mt-1 w-full rounded-md border border-white/10 bg-charcoal px-3 py-2 text-sm text-ink outline-none focus:border-mint"
        >
          {snapshot.markets.map((m) => (
            <option key={m.symbol} value={m.symbol}>
              {m.symbol}
            </option>
          ))}
        </select>

        <label className="mt-4 block text-xs text-slateui">Interval</label>
        <select
          value={interval}
          onChange={(e) => setInterval(e.target.value)}
          className="mt-1 w-full rounded-md border border-white/10 bg-charcoal px-3 py-2 text-sm text-ink outline-none focus:border-mint"
        >
          {INTERVALS.map((tf) => (
            <option key={tf} value={tf}>
              {tf}
            </option>
          ))}
        </select>

        <div className="mt-5 space-y-2">
          <button
            type="button"
            disabled={pending}
            onClick={() => runJob("seed")}
            className="w-full rounded-md bg-crimson px-3 py-2.5 text-sm font-medium text-ink transition hover:bg-crimson-dim disabled:opacity-50"
          >
            {pending ? "Running…" : "Run full backtest"}
          </button>
          <button
            type="button"
            disabled={pending}
            onClick={() => runJob("collect")}
            className="w-full rounded-md border border-white/15 bg-charcoal-2 px-3 py-2 text-sm text-ink hover:border-white/25 disabled:opacity-50"
          >
            Fetch latest candles
          </button>
          <button
            type="button"
            disabled={pending}
            onClick={() => runJob("research")}
            className="w-full rounded-md border border-white/15 bg-charcoal-2 px-3 py-2 text-sm text-ink hover:border-white/25 disabled:opacity-50"
          >
            Daily research report
          </button>
          <button
            type="button"
            disabled={pending}
            onClick={() => runJob("simulate")}
            className="w-full rounded-md border border-white/15 bg-charcoal-2 px-3 py-2 text-sm text-ink hover:border-white/25 disabled:opacity-50"
          >
            Reset & resimulate books
          </button>
        </div>

        {message && (
          <p className="mt-3 rounded-md border border-white/10 bg-black/40 px-3 py-2 text-xs text-slateui">
            {message}
          </p>
        )}

        <div className="mt-6 border-t border-white/10 pt-4">
          <h3 className="text-xs uppercase tracking-[0.22em] text-slateui-dim">Cost posture</h3>
          <p className="mt-2 text-sm text-slateui">
            Market data · agents · SQLite · dashboard →{" "}
            <span className="text-mint">$0</span>
          </p>
          <p className="mt-2 text-sm text-slateui">
            Optional LLM research →{" "}
            {snapshot.research.research_agent?.model?.includes("qwen")
              ? "local Ollama / Qwen"
              : "rule-based fallback (no key)"}
          </p>
        </div>
      </aside>

      {/* Main arena */}
      <div className="space-y-6">
        <header>
          <h1 className="font-display text-4xl text-ink sm:text-5xl">TrendLab Arena</h1>
        </header>

        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {snapshot.leaderboard.map((agent) => (
            <div key={agent.id} className="metric">
              <p className="text-xs uppercase tracking-wider text-slateui-dim">
                {shortName(agent.name)}
              </p>
              <p className="mt-2 font-mono text-2xl text-ink">{formatMoney(agent.equity)}</p>
              <p className={`mt-1 font-mono text-sm ${pctTone(agent.return_pct)}`}>
                {formatPct(agent.return_pct)}
              </p>
            </div>
          ))}
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.35fr_1fr]">
          <div className="panel p-5">
            <h2 className="font-display text-2xl text-ink">Equity curves</h2>
            <div className="mt-3">
              <EquityChart curves={snapshot.equity_curves} labels={labels} height={360} />
            </div>
          </div>

          <div className="space-y-6">
            <div className="panel p-5">
              <h2 className="font-display text-2xl text-ink">Leaderboard</h2>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full min-w-[420px] text-left text-sm">
                  <thead className="text-xs uppercase tracking-wide text-slateui-dim">
                    <tr>
                      <th className="pb-2">Agent</th>
                      <th className="pb-2">Equity</th>
                      <th className="pb-2">Return %</th>
                      <th className="pb-2">Side</th>
                      <th className="pb-2">Positions</th>
                      <th className="pb-2">Cash</th>
                    </tr>
                  </thead>
                  <tbody>
                    {snapshot.leaderboard.map((agent) => (
                      <tr key={agent.id} className="border-t border-white/10">
                        <td className="py-2 text-ink">{shortName(agent.name)}</td>
                        <td className="py-2 font-mono text-ink">{formatMoney(agent.equity)}</td>
                        <td className={`py-2 font-mono ${pctTone(agent.return_pct)}`}>
                          {agent.return_pct.toFixed(2)}
                        </td>
                        <td className="py-2">
                          <span
                            className={
                              agent.open_positions > 0 ? "text-mint" : "text-slateui"
                            }
                          >
                            {agent.open_positions > 0 ? "LONG" : "FLAT"}
                          </span>
                        </td>
                        <td className="py-2 font-mono text-slateui">{agent.open_positions}</td>
                        <td className="py-2 font-mono text-slateui">{formatMoney(agent.cash)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="panel p-5">
              <h2 className="font-display text-2xl text-ink">Strategies</h2>
              <ul className="mt-3 space-y-3 text-sm">
                {snapshot.leaderboard.map((agent) => (
                  <li key={agent.id}>
                    <p className="font-medium text-ink">{shortName(agent.name)}</p>
                    <p className="text-slateui">{agent.description}</p>
                  </li>
                ))}
              </ul>
            </div>

            <div className="panel p-5">
              <div className="flex items-center justify-between gap-3">
                <h2 className="font-display text-2xl text-ink">Research notes</h2>
                <Link href="/research" className="text-sm text-mint hover:underline">
                  Open desk
                </Link>
              </div>
              {latestReport ? (
                <details open className="mt-3 rounded-lg border border-white/10 bg-charcoal p-3">
                  <summary className="cursor-pointer font-mono text-xs text-slateui">
                    {latestReport.created_at.slice(0, 19)} · {latestReport.source}
                  </summary>
                  <p className="mt-2 text-sm text-ink">
                    {latestReport.symbol} · {latestReport.report.regime}
                  </p>
                  <p className="mt-2 text-sm text-slateui">{latestReport.report.summary}</p>
                </details>
              ) : (
                <p className="mt-3 text-sm text-slateui">
                  No research notes yet. Use <span className="text-mint">Daily research report</span>.
                </p>
              )}
            </div>
          </div>
        </section>

        {market && (
          <section className="panel p-5">
            <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
              <div>
                <h2 className="font-display text-2xl text-ink">Price · {market.symbol}</h2>
                <p className="text-sm text-slateui">
                  {market.name} ·{" "}
                  <span
                    className={
                      market.trend_label === "bullish"
                        ? "text-mint"
                        : market.trend_label === "bearish"
                          ? "text-crimson"
                          : "text-slateui"
                    }
                  >
                    {market.trend_label}
                  </span>{" "}
                  · {formatMoney(market.last_close)} · source {market.source}
                </p>
              </div>
              <Link href={`/markets?symbol=${market.symbol}`} className="text-sm text-mint hover:underline">
                Market analysis
              </Link>
            </div>
            <CandleChart candles={market.candles} height={300} />
          </section>
        )}

        <section className="panel overflow-x-auto p-5">
          <h2 className="font-display text-2xl text-ink">Recent fills</h2>
          <table className="mt-3 w-full min-w-[720px] text-left text-sm">
            <thead className="text-xs uppercase tracking-wide text-slateui-dim">
              <tr>
                <th className="pb-2">Time</th>
                <th className="pb-2">Agent</th>
                <th className="pb-2">Side</th>
                <th className="pb-2">Symbol</th>
                <th className="pb-2">Price</th>
                <th className="pb-2">Reason</th>
              </tr>
            </thead>
            <tbody>
              {snapshot.trades.slice(0, 12).map((trade) => (
                <tr
                  key={`${trade.id}-${trade.ts}-${trade.agent_id}`}
                  className="border-t border-white/10"
                >
                  <td className="py-2 font-mono text-xs text-slateui">
                    {new Date(trade.ts).toLocaleString()}
                  </td>
                  <td className="py-2 text-slateui">{trade.agent_id}</td>
                  <td
                    className={`py-2 font-mono ${
                      trade.side === "BUY" ? "text-mint" : "text-crimson"
                    }`}
                  >
                    {trade.side}
                  </td>
                  <td className="py-2 text-ink">{trade.symbol}</td>
                  <td className="py-2 font-mono text-ink">{formatMoney(trade.price)}</td>
                  <td className="py-2 text-slateui">{trade.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}

function shortName(name: string) {
  return name
    .replace(/^Agent [A-D] — /, "")
    .replace(/^Agent [A-D] - /, "")
    .trim();
}
