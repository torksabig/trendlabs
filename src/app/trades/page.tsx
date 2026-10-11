import { loadSnapshot } from "@/lib/snapshot";
import { formatMoney } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function TradesPage() {
  const snapshot = await loadSnapshot();
  if (!snapshot) {
    return <p className="panel p-6 text-slateui">No trades yet. Run `npm run seed`.</p>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl text-ink">Trade history</h1>
        <p className="mt-2 text-slateui">
          Every simulated fill includes fees, slippage, stop level, and the strategy reason.
        </p>
      </div>
      <div className="panel overflow-x-auto p-5">
        <table className="w-full min-w-[900px] text-left text-sm">
          <thead className="text-xs uppercase tracking-wide text-slateui-dim">
            <tr>
              <th className="pb-2">Time</th>
              <th className="pb-2">Agent</th>
              <th className="pb-2">Side</th>
              <th className="pb-2">Symbol</th>
              <th className="pb-2">Qty</th>
              <th className="pb-2">Price</th>
              <th className="pb-2">Fee</th>
              <th className="pb-2">Stop</th>
              <th className="pb-2">Reason</th>
            </tr>
          </thead>
          <tbody>
            {snapshot.trades.map((trade) => (
              <tr
                key={`${trade.id}-${trade.ts}-${trade.agent_id}-${trade.side}`}
                className="border-t border-white/10 align-top text-ink"
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
                <td className="py-2">{trade.symbol}</td>
                <td className="py-2 font-mono">{trade.qty.toFixed(4)}</td>
                <td className="py-2 font-mono">{formatMoney(trade.price)}</td>
                <td className="py-2 font-mono text-slateui">{formatMoney(trade.fee)}</td>
                <td className="py-2 font-mono text-slateui">
                  {trade.stop_loss == null ? "—" : formatMoney(trade.stop_loss)}
                </td>
                <td className="py-2 text-slateui">{trade.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
