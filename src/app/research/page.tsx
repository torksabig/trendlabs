import { loadSnapshot } from "@/lib/snapshot";
import { formatPct, pctTone } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function ResearchPage() {
  const snapshot = await loadSnapshot();
  if (!snapshot) {
    return <p className="panel p-6 text-slateui">No research snapshot. Run `npm run seed`.</p>;
  }

  const events = snapshot.research.recent_events;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl text-ink">Historical research</h1>
        <p className="mt-2 max-w-2xl text-slateui">
          Trend events are detected on completed candles. Forward returns are attached only after
          the outcome window has elapsed to avoid look-ahead bias.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {(["bullish", "bearish"] as const).map((side) => {
          const stats = snapshot.research[side];
          return (
            <div key={side} className="panel p-5">
              <h2
                className={`font-display text-2xl capitalize ${
                  side === "bullish" ? "text-mint" : "text-crimson"
                }`}
              >
                {side} setups
              </h2>
              <p className="mt-3 font-mono text-3xl text-ink">{stats.matches}</p>
              <p className="text-sm text-slateui">historical matches with 10-bar outcomes</p>
              <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div>
                  <dt className="text-slateui-dim">Hit rate</dt>
                  <dd className="font-mono text-ink">
                    {stats.hit_rate == null ? "—" : `${(stats.hit_rate * 100).toFixed(1)}%`}
                  </dd>
                </div>
                <div>
                  <dt className="text-slateui-dim">Avg forward 10</dt>
                  <dd
                    className={`font-mono ${
                      stats.avg_forward_10 == null ? "text-slateui" : pctTone(stats.avg_forward_10)
                    }`}
                  >
                    {stats.avg_forward_10 == null ? "—" : formatPct(stats.avg_forward_10)}
                  </dd>
                </div>
              </dl>
            </div>
          );
        })}
      </div>

      <div className="panel overflow-x-auto p-5">
        <h2 className="font-display text-2xl text-ink">Recent trend events</h2>
        <table className="mt-4 w-full min-w-[720px] text-left text-sm">
          <thead className="text-xs uppercase tracking-wide text-slateui-dim">
            <tr>
              <th className="pb-2">Detected</th>
              <th className="pb-2">Symbol</th>
              <th className="pb-2">Direction</th>
              <th className="pb-2">Strength</th>
              <th className="pb-2">Fwd 5</th>
              <th className="pb-2">Fwd 10</th>
              <th className="pb-2">Fwd 20</th>
            </tr>
          </thead>
          <tbody>
            {events.map((event) => (
              <tr key={String(event.id)} className="border-t border-white/10 text-ink">
                <td className="py-2 font-mono text-xs text-slateui">
                  {new Date(String(event.detected_at)).toLocaleString()}
                </td>
                <td className="py-2">{String(event.symbol)}</td>
                <td
                  className={`py-2 capitalize ${
                    event.direction === "bullish"
                      ? "text-mint"
                      : event.direction === "bearish"
                        ? "text-crimson"
                        : "text-slateui"
                  }`}
                >
                  {String(event.direction)}
                </td>
                <td className="py-2 font-mono text-slateui">
                  {event.strength == null ? "—" : Number(event.strength).toFixed(1)}
                </td>
                <td
                  className={`py-2 font-mono ${
                    event.forward_return_5 == null
                      ? "text-slateui"
                      : pctTone(Number(event.forward_return_5))
                  }`}
                >
                  {event.forward_return_5 == null
                    ? "—"
                    : formatPct(Number(event.forward_return_5))}
                </td>
                <td
                  className={`py-2 font-mono ${
                    event.forward_return_10 == null
                      ? "text-slateui"
                      : pctTone(Number(event.forward_return_10))
                  }`}
                >
                  {event.forward_return_10 == null
                    ? "—"
                    : formatPct(Number(event.forward_return_10))}
                </td>
                <td
                  className={`py-2 font-mono ${
                    event.forward_return_20 == null
                      ? "text-slateui"
                      : pctTone(Number(event.forward_return_20))
                  }`}
                >
                  {event.forward_return_20 == null
                    ? "—"
                    : formatPct(Number(event.forward_return_20))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
