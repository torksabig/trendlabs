import Link from "next/link";
import { promises as fs } from "fs";
import path from "path";

export const dynamic = "force-dynamic";

type Metrics = {
  total_return_pct: number;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown_pct: number;
  trade_count: number;
  win_rate: number | null;
  profit_factor: number | null;
  alpha_pct: number | null;
  exposure_pct: number | null;
  cagr_pct: number | null;
};

type AgentRow = {
  agent_id: string;
  name: string;
  metrics: Metrics;
};

async function loadBacktest() {
  const file = path.join(process.cwd(), "public", "data", "backtests", "latest.json");
  try {
    const raw = await fs.readFile(file, "utf8");
    return JSON.parse(raw) as {
      run_id: string;
      created_at: string;
      timeframe: string;
      start_ts: string;
      end_ts: string;
      bars_tested: number;
      benchmark_symbol: string;
      execution_model: string;
      agents: AgentRow[];
    };
  } catch {
    return null;
  }
}

function fmt(n: number | null | undefined, digits = 2) {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

function tone(n: number | null | undefined) {
  if (n == null) return "text-slateui";
  if (n > 0) return "text-mint";
  if (n < 0) return "text-crimson";
  return "text-slateui";
}

export default async function BacktestsPage() {
  const bt = await loadBacktest();

  if (!bt) {
    return (
      <div className="panel p-6">
        <h1 className="font-display text-4xl text-ink">Backtests</h1>
        <p className="mt-3 text-slateui">
          No deterministic backtest yet. Run{" "}
          <code className="text-mint">npm run backtest</code> (does not reset paper books).
        </p>
      </div>
    );
  }

  const ranked = [...bt.agents].sort((a, b) => {
    const sa = a.metrics.sharpe ?? -999;
    const sb = b.metrics.sharpe ?? -999;
    return sb - sa;
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl text-ink">Backtests</h1>
        <p className="mt-2 text-slateui">
          Risk-adjusted historical replay · next-bar open fills · SPY benchmark · separate from
          forward paper trading.
        </p>
      </div>

      <section className="panel p-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-wide text-slateui-dim">Latest run</p>
            <p className="font-mono text-sm text-ink">{bt.run_id}</p>
          </div>
          <div className="text-sm text-slateui">
            {bt.timeframe} · {bt.bars_tested} bars · {bt.start_ts} → {bt.end_ts}
          </div>
        </div>
        <p className="mt-2 text-xs text-slateui-dim">
          Model: {bt.execution_model} · Benchmark: {bt.benchmark_symbol} · Created{" "}
          {new Date(bt.created_at).toLocaleString()}
        </p>
      </section>

      <section className="panel overflow-x-auto p-5">
        <h2 className="font-display text-2xl text-ink">Leaderboard (Sharpe)</h2>
        <table className="mt-4 w-full min-w-[860px] text-left text-sm">
          <thead className="text-xs uppercase tracking-wide text-slateui-dim">
            <tr>
              <th className="pb-2">Agent</th>
              <th className="pb-2">Return</th>
              <th className="pb-2">CAGR</th>
              <th className="pb-2">Sharpe</th>
              <th className="pb-2">Sortino</th>
              <th className="pb-2">Max DD</th>
              <th className="pb-2">Win%</th>
              <th className="pb-2">PF</th>
              <th className="pb-2">α vs {bt.benchmark_symbol}</th>
              <th className="pb-2">Trades</th>
              <th className="pb-2">Exposure</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((a) => {
              const m = a.metrics;
              return (
                <tr key={a.agent_id} className="border-t border-white/10">
                  <td className="py-2 text-ink">{a.name.replace(/^Agent [A-D] — /, "")}</td>
                  <td className={`py-2 font-mono ${tone(m.total_return_pct)}`}>
                    {fmt(m.total_return_pct)}%
                  </td>
                  <td className={`py-2 font-mono ${tone(m.cagr_pct)}`}>{fmt(m.cagr_pct)}%</td>
                  <td className={`py-2 font-mono ${tone(m.sharpe)}`}>{fmt(m.sharpe, 3)}</td>
                  <td className={`py-2 font-mono ${tone(m.sortino)}`}>{fmt(m.sortino, 3)}</td>
                  <td className="py-2 font-mono text-crimson">{fmt(m.max_drawdown_pct)}%</td>
                  <td className="py-2 font-mono text-slateui">
                    {m.win_rate == null ? "—" : `${(m.win_rate * 100).toFixed(0)}%`}
                  </td>
                  <td className="py-2 font-mono text-slateui">{fmt(m.profit_factor, 2)}</td>
                  <td className={`py-2 font-mono ${tone(m.alpha_pct)}`}>{fmt(m.alpha_pct)}%</td>
                  <td className="py-2 font-mono text-slateui">{m.trade_count}</td>
                  <td className="py-2 font-mono text-slateui">
                    {m.exposure_pct == null ? "—" : `${(m.exposure_pct * 100).toFixed(0)}%`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <p className="text-sm text-slateui">
        Paper Arena remains separate.{" "}
        <Link href="/" className="text-mint hover:underline">
          Back to Arena
        </Link>
      </p>
    </div>
  );
}
