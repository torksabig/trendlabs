import { loadSnapshot } from "@/lib/snapshot";
import { formatPct } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function ResearchPage() {
  const snapshot = await loadSnapshot();
  if (!snapshot) {
    return <p className="panel p-6">No research snapshot. Run `npm run seed`.</p>;
  }

  const events = snapshot.research.recent_events;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl">Historical research</h1>
        <p className="mt-2 max-w-2xl text-ink/65">
          Trend events are detected on completed candles. Forward returns are attached only after
          the outcome window has elapsed to avoid look-ahead bias.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {(["bullish", "bearish"] as const).map((side) => {
          const stats = snapshot.research[side];
          return (
            <div key={side} className="panel p-5">
              <h2 className="font-display text-2xl capitalize">{side} setups</h2>
              <p className="mt-3 font-mono text-3xl">{stats.matches}</p>
              <p className="text-sm text-ink/60">historical matches with 10-bar outcomes</p>
              <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div>
                  <dt className="text-ink/45">Hit rate</dt>
                  <dd className="font-mono">
                    {stats.hit_rate == null ? "—" : `${(stats.hit_rate * 100).toFixed(1)}%`}
                  </dd>
                </div>
                <div>
                  <dt className="text-ink/45">Avg forward 10</dt>
                  <dd className="font-mono">
                    {stats.avg_forward_10 == null ? "—" : formatPct(stats.avg_forward_10)}
                  </dd>
                </div>
              </dl>
            </div>
          );
        })}
      </div>

      <div className="panel overflow-x-auto p-5">
        <h2 className="font-display text-2xl">Recent trend events</h2>
        <table className="mt-4 w-full min-w-[720px] text-left text-sm">
          <thead className="text-xs uppercase tracking-wide text-ink/45">
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
              <tr key={String(event.id)} className="border-t border-ink/8">
                <td className="py-2 font-mono text-xs">
                  {new Date(String(event.detected_at)).toLocaleString()}
                </td>
                <td className="py-2">{String(event.symbol)}</td>
                <td className="py-2 capitalize">{String(event.direction)}</td>
                <td className="py-2 font-mono">
                  {event.strength == null ? "—" : Number(event.strength).toFixed(1)}
                </td>
                <td className="py-2 font-mono">
                  {event.forward_return_5 == null
                    ? "—"
                    : formatPct(Number(event.forward_return_5))}
                </td>
                <td className="py-2 font-mono">
                  {event.forward_return_10 == null
                    ? "—"
                    : formatPct(Number(event.forward_return_10))}
                </td>
                <td className="py-2 font-mono">
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
