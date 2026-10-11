import { EquityChart } from "@/components/charts/equity-chart";
import { loadSnapshot } from "@/lib/snapshot";
import { formatMoney, formatPct, pctTone } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function AgentsPage() {
  const snapshot = await loadSnapshot();
  if (!snapshot) {
    return <p className="panel p-6 text-slateui">No agent data. Run `npm run seed`.</p>;
  }

  const labels = Object.fromEntries(snapshot.leaderboard.map((a) => [a.id, a.name]));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl text-ink">Agent leaderboard</h1>
        <p className="mt-2 max-w-2xl text-slateui">
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
                <p className="text-xs uppercase tracking-[0.18em] text-slateui-dim">
                  Rank #{idx + 1}
                </p>
                <h2 className="font-display text-2xl text-ink">{agent.name}</h2>
                <p className="mt-1 text-sm text-slateui">{agent.description}</p>
              </div>
              <div className="text-right">
                <p className="font-mono text-xl text-ink">{formatMoney(agent.equity)}</p>
                <p className={`font-mono text-sm ${pctTone(agent.return_pct)}`}>
                  {formatPct(agent.return_pct)}
                </p>
              </div>
            </div>
            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <div>
                <dt className="text-slateui-dim">Max drawdown</dt>
                <dd className={`font-mono ${pctTone(agent.max_drawdown_pct)}`}>
                  {formatPct(agent.max_drawdown_pct)}
                </dd>
              </div>
              <div>
                <dt className="text-slateui-dim">Trades</dt>
                <dd className="font-mono text-ink">{agent.trades}</dd>
              </div>
              <div>
                <dt className="text-slateui-dim">Cash</dt>
                <dd className="font-mono text-ink">{formatMoney(agent.cash)}</dd>
              </div>
              <div>
                <dt className="text-slateui-dim">Open positions</dt>
                <dd className="font-mono text-ink">{agent.open_positions}</dd>
              </div>
            </dl>
            {agent.positions.length > 0 && (
              <ul className="mt-4 space-y-1 border-t border-white/10 pt-3 text-sm">
                {agent.positions.map((p) => (
                  <li key={`${agent.id}-${p.symbol}`} className="flex justify-between text-ink">
                    <span>{p.symbol}</span>
                    <span className="font-mono text-slateui">
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
