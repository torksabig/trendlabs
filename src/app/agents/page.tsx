import { EquityChart } from "@/components/charts/equity-chart";
import { loadSnapshot } from "@/lib/snapshot";
import { formatMoney, formatPct } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function AgentsPage() {
  const snapshot = await loadSnapshot();
  if (!snapshot) {
    return <p className="panel p-6">No agent data. Run `npm run seed`.</p>;
  }

  const labels = Object.fromEntries(snapshot.leaderboard.map((a) => [a.id, a.name]));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl">Agent leaderboard</h1>
        <p className="mt-2 max-w-2xl text-ink/65">
          Four isolated $10,000 paper books. Long-only, max three positions, 1% risk budget, fees
          and slippage included. No real-money order path.
        </p>
      </div>

      <div className="panel p-5">
        <EquityChart curves={snapshot.equity_curves} labels={labels} height={340} />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {snapshot.leaderboard.map((agent, idx) => (
          <article key={agent.id} className="panel p-5">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-ink/45">Rank #{idx + 1}</p>
                <h2 className="font-display text-2xl">{agent.name}</h2>
                <p className="mt-1 text-sm text-ink/60">{agent.description}</p>
              </div>
              <div className="text-right">
                <p className="font-mono text-xl">{formatMoney(agent.equity)}</p>
                <p className="font-mono text-sm text-ink/60">{formatPct(agent.return_pct)}</p>
              </div>
            </div>
            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <div>
                <dt className="text-ink/45">Max drawdown</dt>
                <dd className="font-mono">{formatPct(agent.max_drawdown_pct)}</dd>
              </div>
              <div>
                <dt className="text-ink/45">Trades</dt>
                <dd className="font-mono">{agent.trades}</dd>
              </div>
              <div>
                <dt className="text-ink/45">Cash</dt>
                <dd className="font-mono">{formatMoney(agent.cash)}</dd>
              </div>
              <div>
                <dt className="text-ink/45">Open positions</dt>
                <dd className="font-mono">{agent.open_positions}</dd>
              </div>
            </dl>
            {agent.positions.length > 0 && (
              <ul className="mt-4 space-y-1 border-t border-ink/10 pt-3 text-sm">
                {agent.positions.map((p) => (
                  <li key={`${agent.id}-${p.symbol}`} className="flex justify-between">
                    <span>{p.symbol}</span>
                    <span className="font-mono text-ink/70">
                      {p.qty.toFixed(4)} @ {formatMoney(p.avg_entry)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}
