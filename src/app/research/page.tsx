import { loadSnapshot } from "@/lib/snapshot";
import { formatPct, pctTone } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function ResearchPage() {
  const snapshot = await loadSnapshot();
  if (!snapshot) {
    return <p className="panel p-6 text-slateui">No research snapshot. Run `npm run seed`.</p>;
  }

  const events = snapshot.research.recent_events;
  const aiReports = snapshot.research.ai_reports ?? [];
  const agentMeta = snapshot.research.research_agent;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl text-ink">Historical research</h1>
        <p className="mt-2 max-w-2xl text-slateui">
          Python owns indicators and historical matches. A shared local research agent (Qwen3 14B via
          Ollama on your Mac) interprets those facts and proposes experiments — it never executes
          trades.
        </p>
      </div>

      <section className="panel p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-mint">Shared research agent</p>
            <h2 className="mt-1 font-display text-2xl text-ink">
              {agentMeta?.name ?? "Qwen3 research agent"}
            </h2>
            <p className="mt-1 text-sm text-slateui">
              {agentMeta?.role ?? "Interprets verified analytics; never executes trades"}
            </p>
          </div>
          <div className="text-right text-sm text-slateui">
            <p>
              Model:{" "}
              <span className="font-mono text-ink">{agentMeta?.model ?? "not run yet"}</span>
            </p>
            <p>
              Last run:{" "}
              {agentMeta?.last_research
                ? new Date(agentMeta.last_research).toLocaleString()
                : "—"}
            </p>
            <p className="text-mint">LLM trade execution: disabled</p>
          </div>
        </div>

        {aiReports.length === 0 ? (
          <pre className="mt-5 overflow-x-auto rounded-xl bg-charcoal px-4 py-3 font-mono text-sm text-mint">
            {`# On your Mac with Ollama + qwen3:14b-q4_K_M
ollama pull qwen3:14b-q4_K_M
npm run research -- --symbol AAPL --timeframe 1h

# Cloud/CI without Ollama
npm run research -- --symbol AAPL --no-ollama`}
          </pre>
        ) : (
          <div className="mt-5 space-y-4">
            {aiReports.map((item) => (
              <article
                key={item.id}
                className="rounded-xl border border-white/10 bg-charcoal p-4"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="font-display text-xl text-ink">
                    {item.symbol} · {item.timeframe}{" "}
                    <span
                      className={
                        item.report.regime === "bullish"
                          ? "text-mint"
                          : item.report.regime === "bearish"
                            ? "text-crimson"
                            : "text-slateui"
                      }
                    >
                      {item.report.regime}
                    </span>
                  </h3>
                  <p className="font-mono text-xs text-slateui">
                    {item.source} · {item.model} · confidence{" "}
                    {(item.report.confidence * 100).toFixed(0)}%
                  </p>
                </div>
                <p className="mt-3 text-sm leading-relaxed text-slateui">{item.report.summary}</p>
                <p className="mt-3 text-sm text-ink">{item.report.historical_context}</p>

                <div className="mt-4 grid gap-4 md:grid-cols-2">
                  <div>
                    <h4 className="text-xs uppercase tracking-wide text-crimson">Key risks</h4>
                    <ul className="mt-2 space-y-1 text-sm text-slateui">
                      {item.report.key_risks.map((risk) => (
                        <li key={risk}>• {risk}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-xs uppercase tracking-wide text-mint">
                      Strategy proposals
                    </h4>
                    <ul className="mt-2 space-y-2 text-sm">
                      {item.report.proposals.map((p) => (
                        <li key={p.title} className="rounded-lg border border-white/10 p-2">
                          <p className="font-medium text-ink">{p.title}</p>
                          <p className="text-slateui">{p.hypothesis}</p>
                          <p className="mt-1 text-xs text-slateui-dim">Test: {p.test_plan}</p>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {item.report.agent_notes?.length > 0 && (
                  <div className="mt-4">
                    <h4 className="text-xs uppercase tracking-wide text-slateui-dim">
                      Notes on trading agents
                    </h4>
                    <ul className="mt-2 space-y-1 text-sm text-slateui">
                      {item.report.agent_notes.map((note) => (
                        <li key={`${item.id}-${note.agent_id}`}>
                          <span className="text-ink">{note.agent_id}</span> — {note.observation}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </article>
            ))}
          </div>
        )}
      </section>

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
