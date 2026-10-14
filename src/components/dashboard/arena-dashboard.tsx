"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { LivePriceChart } from "@/components/charts/live-price-chart";
import { EquityChart } from "@/components/charts/equity-chart";
import { HourlyPredictions } from "@/components/dashboard/hourly-predictions";
import { UpdateTimeline } from "@/components/dashboard/update-timeline";
import type { RiskBoardPayload, RiskRow } from "@/lib/evaluation";
import { INSTRUMENTS, TIMEFRAMES } from "@/lib/instruments";
import type { Snapshot } from "@/lib/types";
import { formatMoney, formatPct, pctTone } from "@/lib/utils";

type Props = {
  snapshot: Snapshot;
  riskBoard?: RiskBoardPayload | null;
};

type JobResponse = {
  ok?: boolean;
  mode?: "github_actions" | "local";
  job?: string;
  error?: string;
  output?: string;
  message?: string;
  run_url?: string;
};

type LlmProviderRow = {
  id: string;
  label: string;
  configured: boolean;
  default_model: string;
};

type LlmStatus = {
  ok?: boolean;
  default_provider?: string;
  any_cloud_configured?: boolean;
  providers?: LlmProviderRow[];
};

const STORAGE_KEY = "trendlabs.llm.provider";

function regimeLabel(raw: string | undefined) {
  if (!raw) return "Unknown";
  return raw.replace(/_/g, " ");
}

export function ArenaDashboard({ snapshot, riskBoard: initialRisk }: Props) {
  const router = useRouter();
  const [symbol, setSymbol] = useState("AAPL");
  const [interval, setInterval] = useState(snapshot.timeframe || "1h");
  const [chartRefresh, setChartRefresh] = useState(0);
  const [provider, setProvider] = useState("auto");
  const [llmStatus, setLlmStatus] = useState<LlmStatus | null>(null);
  const [riskBoard, setRiskBoard] = useState<RiskRow[] | null>(initialRisk?.rows ?? null);
  const [riskSource, setRiskSource] = useState<string | null>(initialRisk?.source ?? null);
  const [message, setMessage] = useState<string | null>(null);
  const [runUrl, setRunUrl] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [polling, setPolling] = useState(false);
  const baselineGeneratedAt = useRef(snapshot.generated_at);

  const market = useMemo(
    () => snapshot.markets.find((m) => m.symbol === symbol) ?? snapshot.markets[0],
    [snapshot.markets, symbol],
  );
  const labels = useMemo(
    () => Object.fromEntries(snapshot.leaderboard.map((a) => [a.id, shortName(a.name)])),
    [snapshot.leaderboard],
  );
  const latestReport = snapshot.research.ai_reports?.[0];

  useEffect(() => {
    if (initialRisk?.rows?.length) {
      setRiskBoard(initialRisk.rows);
      setRiskSource(initialRisk.source);
    }
  }, [initialRisk]);

  useEffect(() => {
    let cancelled = false;
    async function loadLlm() {
      try {
        const res = await fetch("/api/llm", { cache: "no-store" });
        const data = (await res.json()) as LlmStatus;
        if (cancelled || !data.ok) return;
        setLlmStatus(data);
        const saved =
          typeof window !== "undefined" ? window.localStorage.getItem(STORAGE_KEY) : null;
        const preferred = saved || data.default_provider || "auto";
        const known = data.providers?.some((p) => p.id === preferred);
        setProvider(known ? preferred : "auto");
      } catch {
        // selector still works with static auto/fallback options
      }
    }
    async function loadRisk() {
      try {
        const wfRes = await fetch("/api/walk-forward", { cache: "no-store" });
        if (wfRes.ok) {
          const data = (await wfRes.json()) as {
            ok?: boolean;
            fold_count?: number;
            leaderboard?: RiskRow[];
          };
          if (!cancelled && data.ok && data.leaderboard?.length) {
            setRiskBoard(data.leaderboard);
            setRiskSource(`Walk-forward OOS · ${data.fold_count ?? "?"} folds`);
            return;
          }
        }
        const btRes = await fetch("/api/backtests", { cache: "no-store" });
        if (btRes.ok) {
          const data = (await btRes.json()) as {
            ok?: boolean;
            leaderboard?: RiskRow[];
          };
          if (!cancelled && data.ok && data.leaderboard?.length) {
            setRiskBoard(data.leaderboard);
            setRiskSource("Full-period backtest");
          }
        }
      } catch {
        // paper leaderboard still shown
      }
    }
    void loadLlm();
    void loadRisk();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!polling) return;

    let cancelled = false;
    const started = Date.now();
    const maxMs = 12 * 60 * 1000;

    async function tick() {
      try {
        const res = await fetch("/api/status", { cache: "no-store" });
        const data = (await res.json()) as { ok?: boolean; generated_at?: string };
        if (cancelled) return;
        if (data.ok && data.generated_at && data.generated_at !== baselineGeneratedAt.current) {
          setPolling(false);
          setMessage("Snapshot updated — refreshing dashboard…");
          router.refresh();
          return;
        }
      } catch {
        // keep polling; network blips are expected during redeploy
      }
      if (Date.now() - started > maxMs) {
        if (!cancelled) {
          setPolling(false);
          setMessage(
            "Job was queued. Snapshot has not changed yet — check Actions or refresh in a few minutes.",
          );
        }
      }
    }

    void tick();
    const id = window.setInterval(() => void tick(), 15000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [polling, router]);

  async function runJob(job: string) {
    setMessage(null);
    setRunUrl(null);
    setPolling(false);
    setPending(true);
    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job,
          symbol,
          timeframe: interval,
          provider: job === "research" || job === "seed" ? provider : undefined,
        }),
      });
      const data = (await res.json()) as JobResponse;
      if (!res.ok || !data.ok) {
        setMessage(data.error || data.output || "Job failed");
        return;
      }
      if (data.mode === "github_actions") {
        baselineGeneratedAt.current = snapshot.generated_at;
        setRunUrl(data.run_url || null);
        setMessage(
          data.message ||
            "Job queued on GitHub Actions — dashboard will refresh after the snapshot updates / redeploy.",
        );
        setPolling(true);
        return;
      }
      setMessage(
        job === "research" || job === "seed"
          ? `${job} finished (${provider})`
          : `${job} finished`,
      );
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
          {INSTRUMENTS.map((m) => (
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
          {TIMEFRAMES.map((tf) => (
            <option key={tf} value={tf}>
              {tf}
            </option>
          ))}
        </select>

        <label className="mt-4 block text-xs text-slateui">Research LLM</label>
        <select
          value={provider}
          onChange={(e) => {
            const next = e.target.value;
            setProvider(next);
            try {
              window.localStorage.setItem(STORAGE_KEY, next);
            } catch {
              // ignore quota / private mode
            }
          }}
          className="mt-1 w-full rounded-md border border-white/10 bg-charcoal px-3 py-2 text-sm text-ink outline-none focus:border-mint"
        >
          {(llmStatus?.providers ?? DEFAULT_PROVIDERS).map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
              {p.id !== "auto" && p.id !== "fallback"
                ? p.configured
                  ? " · ready"
                  : " · needs key"
                : ""}
            </option>
          ))}
        </select>
        <p className="mt-1 text-[11px] leading-snug text-slateui-dim">
          Used for research / seed. Keys on Vercel or Actions secrets; never shown here.
        </p>

        <div className="mt-5 space-y-2">
          <button
            type="button"
            disabled={pending}
            onClick={() => runJob("seed")}
            className="w-full rounded-md bg-crimson px-3 py-2.5 text-sm font-medium text-ink transition hover:bg-crimson-dim disabled:opacity-50"
          >
            {pending ? "Queuing…" : "Run full backtest"}
          </button>
          <button
            type="button"
            disabled={pending}
            onClick={() => {
              setChartRefresh((n) => n + 1);
              void runJob("collect");
            }}
            className="w-full rounded-md border border-white/15 bg-charcoal-2 px-3 py-2 text-sm text-ink hover:border-white/25 disabled:opacity-50"
          >
            Fetch latest candles
          </button>
          <button
            type="button"
            disabled={pending}
            onClick={() => setChartRefresh((n) => n + 1)}
            className="w-full rounded-md border border-mint/40 bg-mint/10 px-3 py-2 text-sm text-mint hover:bg-mint/15 disabled:opacity-50"
          >
            Refresh chart · {symbol} {interval}
          </button>
          <button
            type="button"
            disabled={pending}
            onClick={() => runJob("predict")}
            className="w-full rounded-md border border-mint/40 bg-mint/10 px-3 py-2 text-sm text-mint hover:bg-mint/15 disabled:opacity-50"
          >
            Run hourly predictions
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

        {(message || runUrl || polling) && (
          <div className="mt-3 space-y-2 rounded-md border border-white/10 bg-black/40 px-3 py-2 text-xs text-slateui">
            {message && <p>{message}</p>}
            {polling && (
              <p className="text-mint">Waiting for snapshot update…</p>
            )}
            {runUrl && (
              <a
                href={runUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-block text-mint hover:underline"
              >
                View GitHub Actions runs
              </a>
            )}
          </div>
        )}

        <div className="mt-6 border-t border-white/10 pt-4">
          <h3 className="text-xs uppercase tracking-[0.22em] text-slateui-dim">Cost posture</h3>
          <p className="mt-2 text-sm text-slateui">
            Market data · agents · SQLite · dashboard →{" "}
            <span className="text-mint">$0</span>
          </p>
          <p className="mt-2 text-sm text-slateui">
            Optional LLM research →{" "}
            <span className="text-mint">{providerLabel(provider, llmStatus)}</span>
            {llmStatus?.any_cloud_configured ? " (cloud key present)" : " (free-tier / fallback)"}
          </p>
        </div>
      </aside>

      {/* Main arena */}
      <div className="space-y-6">
        <header>
          <p className="text-xs uppercase tracking-[0.22em] text-slateui-dim">
            Market experiment terminal
          </p>
          <h1 className="font-display text-4xl text-ink sm:text-5xl">TrendLab Arena</h1>
          <p className="mt-2 font-mono text-sm text-slateui">
            {symbol} · {interval} · Market regime:{" "}
            <span className="text-mint">
              {regimeLabel(market?.market_regime || market?.trend_label)}
            </span>
            {market?.adx_14 != null ? (
              <span className="text-slateui-dim"> · ADX {market.adx_14.toFixed(1)}</span>
            ) : null}
          </p>
        </header>

        <UpdateTimeline snapshot={snapshot} />

        <HourlyPredictions snapshot={snapshot} focusSymbol={symbol} />

        {riskBoard && riskBoard.length > 0 ? (
          <section className="panel overflow-x-auto p-5">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <h2 className="font-display text-2xl text-ink">Agent arena · risk-adjusted</h2>
              <Link href="/backtests" className="text-sm text-mint hover:underline">
                {riskSource || "Evaluation lab"}
              </Link>
            </div>
            <table className="mt-4 w-full min-w-[640px] text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-slateui-dim">
                <tr>
                  <th className="pb-2">Agent</th>
                  <th className="pb-2">Return</th>
                  <th className="pb-2">Sharpe</th>
                  <th className="pb-2">Drawdown</th>
                  <th className="pb-2">Trades</th>
                  <th className="pb-2">Paper status</th>
                </tr>
              </thead>
              <tbody>
                {[...riskBoard]
                  .sort(
                    (a, b) =>
                      (b.metrics.mean_fold_sharpe ?? b.metrics.sharpe ?? -999) -
                      (a.metrics.mean_fold_sharpe ?? a.metrics.sharpe ?? -999),
                  )
                  .map((row) => {
                    const paper = snapshot.leaderboard.find((a) => a.id === row.agent_id);
                    const sharpe = row.metrics.mean_fold_sharpe ?? row.metrics.sharpe;
                    return (
                      <tr key={row.agent_id} className="border-t border-white/10">
                        <td className="py-2 text-ink">{shortName(row.name)}</td>
                        <td
                          className={`py-2 font-mono ${pctTone(row.metrics.total_return_pct ?? 0)}`}
                        >
                          {row.metrics.total_return_pct == null
                            ? "—"
                            : formatPct(row.metrics.total_return_pct)}
                        </td>
                        <td className="py-2 font-mono text-ink">
                          {sharpe == null ? "—" : sharpe.toFixed(2)}
                        </td>
                        <td className="py-2 font-mono text-crimson">
                          {row.metrics.max_drawdown_pct == null
                            ? "—"
                            : `${row.metrics.max_drawdown_pct.toFixed(1)}%`}
                        </td>
                        <td className="py-2 font-mono text-slateui">
                          {row.metrics.trade_count ?? "—"}
                        </td>
                        <td className="py-2">
                          <span
                            className={
                              (paper?.open_positions ?? 0) > 0 ? "text-mint" : "text-slateui"
                            }
                          >
                            {(paper?.open_positions ?? 0) > 0 ? "LONG" : "FLAT"}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </section>
        ) : (
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
        )}

        <section className="grid gap-6 xl:grid-cols-[1.35fr_1fr]">
          <div className="panel p-5">
            <h2 className="font-display text-2xl text-ink">Equity curves</h2>
            <div className="mt-3">
              <EquityChart curves={snapshot.equity_curves} labels={labels} height={360} />
            </div>
          </div>

          <div className="space-y-6">
            <div className="panel p-5">
              <h2 className="font-display text-2xl text-ink">Paper books</h2>
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

        <section className="panel p-5">
          <div className="mb-2 flex justify-end">
            <Link href={`/markets?symbol=${symbol}`} className="text-sm text-mint hover:underline">
              Market analysis
            </Link>
          </div>
          <LivePriceChart
            symbol={symbol}
            timeframe={interval}
            fallback={market && market.symbol === symbol ? market : null}
            height={300}
            refreshToken={chartRefresh}
          />
        </section>

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

const DEFAULT_PROVIDERS: LlmProviderRow[] = [
  { id: "auto", label: "Auto (free cloud → Ollama → fallback)", configured: true, default_model: "" },
  { id: "google", label: "Google AI Studio (Gemini)", configured: false, default_model: "gemini-2.0-flash" },
  { id: "groq", label: "Groq", configured: false, default_model: "llama-3.3-70b-versatile" },
  { id: "openrouter", label: "OpenRouter", configured: false, default_model: "openai/gpt-6-luna" },
  { id: "cloudflare", label: "Cloudflare Workers AI", configured: false, default_model: "@cf/meta/llama-3.1-8b-instruct" },
  { id: "cerebras", label: "Cerebras", configured: false, default_model: "llama3.1-8b" },
  { id: "mistral", label: "Mistral", configured: false, default_model: "mistral-small-latest" },
  { id: "ollama", label: "Ollama (local)", configured: false, default_model: "qwen3:14b-q4_K_M" },
  { id: "fallback", label: "Rule-based fallback", configured: true, default_model: "rule_based_fallback" },
];

function providerLabel(id: string, status: LlmStatus | null): string {
  const row = status?.providers?.find((p) => p.id === id) ?? DEFAULT_PROVIDERS.find((p) => p.id === id);
  return row?.label ?? id;
}
