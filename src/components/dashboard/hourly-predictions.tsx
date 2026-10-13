import type { Prediction, Snapshot } from "@/lib/types";
import { directionTone, formatMoney, formatWhen } from "@/lib/utils";

type Props = {
  snapshot: Snapshot;
  focusSymbol?: string;
};

function DirectionBadge({ direction }: { direction: string }) {
  const tone = directionTone(direction);
  return (
    <span className={`font-mono text-sm uppercase ${tone}`}>{direction}</span>
  );
}

export function HourlyPredictions({ snapshot, focusSymbol }: Props) {
  const latest = snapshot.predictions?.latest ?? [];
  const history = snapshot.predictions?.history ?? [];
  const focus =
    latest.find((p) => p.symbol === focusSymbol) ??
    latest[0] ??
    null;

  const sorted = [...latest].sort((a, b) => {
    if (focusSymbol && a.symbol === focusSymbol) return -1;
    if (focusSymbol && b.symbol === focusSymbol) return 1;
    return b.confidence - a.confidence;
  });

  return (
    <section className="panel p-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="font-display text-2xl text-ink">Hourly predictions</h2>
          <p className="mt-1 text-sm text-slateui">
            Rule-based 1h outlook from collected candles · model{" "}
            {snapshot.predictions?.model ?? "indicator_score_v1"} · never trades
          </p>
        </div>
        <p className="font-mono text-xs text-slateui-dim">
          Last run {formatWhen(snapshot.last_prediction)}
        </p>
      </div>

      {focus ? (
        <div className="mt-4 rounded-lg border border-mint/30 bg-mint/5 px-4 py-3">
          <p className="text-xs uppercase tracking-wide text-slateui-dim">
            Focus · {focus.symbol}
          </p>
          <div className="mt-1 flex flex-wrap items-baseline gap-3">
            <DirectionBadge direction={focus.direction} />
            <span className="font-mono text-ink">
              {(focus.confidence * 100).toFixed(0)}% conf
            </span>
            <span className="font-mono text-slateui">{formatMoney(focus.last_close)}</span>
          </div>
          <p className="mt-2 text-sm text-slateui">{focus.summary}</p>
        </div>
      ) : (
        <p className="mt-4 text-sm text-slateui">
          No predictions yet. Wait for the hourly job, or run{" "}
          <span className="text-mint">npm run predict</span> / Arena predict.
        </p>
      )}

      {sorted.length > 0 ? (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead className="text-xs uppercase tracking-wide text-slateui-dim">
              <tr>
                <th className="pb-2">Symbol</th>
                <th className="pb-2">Direction</th>
                <th className="pb-2">Conf</th>
                <th className="pb-2">Price</th>
                <th className="pb-2">As of</th>
                <th className="pb-2">Next update</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((p: Prediction) => (
                <tr
                  key={`${p.symbol}-${p.created_at}`}
                  className={`border-t border-white/10 ${
                    p.symbol === focusSymbol ? "bg-white/[0.03]" : ""
                  }`}
                >
                  <td className="py-2 font-medium text-ink">{p.symbol}</td>
                  <td className="py-2">
                    <DirectionBadge direction={p.direction} />
                  </td>
                  <td className="py-2 font-mono text-ink">
                    {(p.confidence * 100).toFixed(0)}%
                  </td>
                  <td className="py-2 font-mono text-slateui">
                    {formatMoney(p.last_close)}
                  </td>
                  <td className="py-2 font-mono text-xs text-slateui">
                    {formatWhen(p.created_at)}
                  </td>
                  <td className="py-2 font-mono text-xs text-slateui">
                    {formatWhen(p.next_update_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {history.length > 0 ? (
        <details className="mt-4">
          <summary className="cursor-pointer text-sm text-mint">
            Recent prediction history ({Math.min(history.length, 12)})
          </summary>
          <ul className="mt-2 space-y-2 text-sm text-slateui">
            {history.slice(0, 12).map((p) => (
              <li key={`${p.id ?? p.symbol}-${p.created_at}`} className="font-mono text-xs">
                {formatWhen(p.created_at)} · {p.symbol} ·{" "}
                <span className={directionTone(p.direction)}>{p.direction}</span> ·{" "}
                {(p.confidence * 100).toFixed(0)}%
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}
