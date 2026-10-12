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
          Python owns indicators and historical matches. A shared research agent (switchable free-tier
          cloud LLMs, local Ollama, or rule-based fallback) interprets those facts and proposes
          experiments — it never executes trades.
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
            {`# Pick a provider from the Arena sidebar, or CLI:\nnpm run research -- --symbol AAPL --timeframe 1h --provider google\nnpm run research -- --symbol AAPL --provider groq\nnpm run research -- --symbol AAPL --provider openrouter\n\n# Local Mac Ollama\nollama pull qwen3:14b-q4_K_M\nnpm run research -- --symbol AAPL --provider ollama\n\n# No keys → deterministic fallback\nnpm run research -- --symbol AAPL --provider fallback`}
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
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
