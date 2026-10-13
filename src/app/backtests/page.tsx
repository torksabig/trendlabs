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
  mean_fold_sharpe?: number | null;
};

type AgentRow = {
  agent_id: string;
  name: string;
  metrics: Metrics;
  regime_performance?: Record<
    string,
    {
      trade_count: number;
      win_rate: number;
      avg_pnl?: number;
      total_pnl?: number;
      profit_factor?: number | null;
    }
  >;
  regime_fit?: Record<string, string>;
};

type BacktestPayload = {
  run_id: string;
  created_at: string;
  timeframe: string;
  start_ts: string;
  end_ts: string;
  bars_tested: number;
  benchmark_symbol: string;
  execution_model: string;
  market_regime_distribution?: Record<string, number>;
  agents: AgentRow[];
};

type WalkForwardPayload = {
  run_id: string;
  created_at: string;
  timeframe: string;
  fold_count: number;
  warm_up: number;
  calibrate_bars: number;
  test_bars: number;
  folds: Array<{
    fold_id: number;
    calibrate_start_ts: string;
    calibrate_end_ts: string;
    test_start_ts: string;
    test_end_ts: string;
  }>;
  fold_details?: Array<{
    fold_id: number;
    test_start_ts: string;
    test_end_ts: string;
    market_regime_distribution?: Record<string, number>;
    leaderboard: Array<{
      agent_id: string;
      sharpe: number | null;
      total_return_pct: number;
      max_drawdown_pct: number;
      trade_count: number;
    }>;
  }>;
  agents: AgentRow[];
};

async function loadJson<T>(rel: string): Promise<T | null> {
  const file = path.join(process.cwd(), "public", "data", rel);
  try {
    const raw = await fs.readFile(file, "utf8");
    return JSON.parse(raw) as T;
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

function shortName(name: string) {
  return name.replace(/^Agent [A-D] — /, "").replace(/^Agent [A-D] - /, "");
}

function RegimeTable({ agents }: { agents: AgentRow[] }) {
  const regimes = Array.from(
    new Set(agents.flatMap((a) => Object.keys(a.regime_performance || {}))),
  ).sort();
  if (!regimes.length) return null;
  return (
    <section className="panel overflow-x-auto p-5">
      <h2 className="font-display text-2xl text-ink">Regime performance</h2>
      <p className="mt-1 text-sm text-slateui">
        Round-trips attributed to market regime at entry — not raw equity.
      </p>
      <table className="mt-4 w-full min-w-[720px] text-left text-sm">
        <thead className="text-xs uppercase tracking-wide text-slateui-dim">
          <tr>
            <th className="pb-2">Agent</th>
            {regimes.map((r) => (
              <th key={r} className="pb-2">
                {r.replace(/_/g, " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {agents.map((a) => (
            <tr key={a.agent_id} className="border-t border-white/10">
              <td className="py-2 text-ink">{shortName(a.name)}</td>
              {regimes.map((r) => {
                const s = a.regime_performance?.[r];
                const fit = a.regime_fit?.[r];
                if (!s) {
                  return (
                    <td key={r} className="py-2 font-mono text-slateui-dim">
                      —
                    </td>
                  );
                }
                return (
                  <td key={r} className="py-2 font-mono text-slateui">
                    <span className={tone(s.avg_pnl ?? s.total_pnl)}>
                      {s.trade_count}t · {((s.win_rate || 0) * 100).toFixed(0)}%
                    </span>
                    {fit ? (
                      <span className="mt-0.5 block text-[11px] text-slateui-dim">{fit}</span>
                    ) : null}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export default async function BacktestsPage() {
  const bt = await loadJson<BacktestPayload>("backtests/latest.json");
  const wf = await loadJson<WalkForwardPayload>("walk_forward/latest.json");

  if (!bt && !wf) {
    return (
      <div className="panel p-6">
        <h1 className="font-display text-4xl text-ink">Backtests</h1>
        <p className="mt-3 text-slateui">
          No evaluation runs yet. Prefer{" "}
          <code className="text-mint">npm run walk-forward</code> (OOS folds + full
          backtest). Does not reset paper books.
        </p>
      </div>
    );
  }

  const rankedBt = bt
    ? [...bt.agents].sort((a, b) => (b.metrics.sharpe ?? -999) - (a.metrics.sharpe ?? -999))
    : [];
  const rankedWf = wf
    ? [...wf.agents].sort(
        (a, b) =>
          (b.metrics.mean_fold_sharpe ?? b.metrics.sharpe ?? -999) -
          (a.metrics.mean_fold_sharpe ?? a.metrics.sharpe ?? -999),
      )
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl text-ink">Evaluation lab</h1>
        <p className="mt-2 text-slateui">
          Risk-adjusted backtests and walk-forward out-of-sample folds · next-bar fills ·
          SPY benchmark · separate from forward paper trading.
        </p>
      </div>

      {wf && (
        <section className="panel overflow-x-auto p-5">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-wide text-slateui-dim">
                Walk-forward (OOS only)
              </p>
              <p className="font-mono text-sm text-ink">{wf.run_id}</p>
            </div>
            <div className="text-sm text-slateui">
              {wf.fold_count} folds · warm-up {wf.warm_up} · calibrate {wf.calibrate_bars} ·
              test {wf.test_bars} · {wf.timeframe}
            </div>
          </div>
          <h2 className="mt-4 font-display text-2xl text-ink">OOS leaderboard</h2>
          <table className="mt-4 w-full min-w-[860px] text-left text-sm">
            <thead className="text-xs uppercase tracking-wide text-slateui-dim">
              <tr>
                <th className="pb-2">Agent</th>
                <th className="pb-2">OOS Return</th>
                <th className="pb-2">OOS Sharpe</th>
                <th className="pb-2">Mean fold Sharpe</th>
                <th className="pb-2">Max DD</th>
                <th className="pb-2">α</th>
                <th className="pb-2">Trades</th>
              </tr>
            </thead>
            <tbody>
              {rankedWf.map((a) => {
                const m = a.metrics;
                return (
                  <tr key={a.agent_id} className="border-t border-white/10">
                    <td className="py-2 text-ink">{shortName(a.name)}</td>
                    <td className={`py-2 font-mono ${tone(m.total_return_pct)}`}>
                      {fmt(m.total_return_pct)}%
                    </td>
                    <td className={`py-2 font-mono ${tone(m.sharpe)}`}>{fmt(m.sharpe, 3)}</td>
                    <td className={`py-2 font-mono ${tone(m.mean_fold_sharpe)}`}>
                      {fmt(m.mean_fold_sharpe, 3)}
                    </td>
                    <td className="py-2 font-mono text-crimson">{fmt(m.max_drawdown_pct)}%</td>
                    <td className={`py-2 font-mono ${tone(m.alpha_pct)}`}>{fmt(m.alpha_pct)}%</td>
                    <td className="py-2 font-mono text-slateui">{m.trade_count}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {wf.folds?.length ? (
            <div className="mt-5">
              <h3 className="text-sm uppercase tracking-wide text-slateui-dim">Folds</h3>
              <ul className="mt-2 space-y-1 font-mono text-xs text-slateui">
                {wf.folds.map((f) => (
                  <li key={f.fold_id}>
                    #{f.fold_id} calibrate {f.calibrate_start_ts.slice(0, 16)} →{" "}
                    {f.calibrate_end_ts.slice(0, 16)} · test {f.test_start_ts.slice(0, 16)} →{" "}
                    {f.test_end_ts.slice(0, 16)}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </section>
      )}

      {wf ? <RegimeTable agents={rankedWf} /> : null}

      {bt && (
        <>
          <section className="panel p-5">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-wide text-slateui-dim">
                  Full-period backtest
                </p>
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
            {bt.market_regime_distribution && (
              <p className="mt-2 text-xs text-slateui">
                Regimes:{" "}
                {Object.entries(bt.market_regime_distribution)
                  .map(([k, v]) => `${k.replace(/_/g, " ")} ${(v * 100).toFixed(0)}%`)
                  .join(" · ")}
              </p>
            )}
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
                {rankedBt.map((a) => {
                  const m = a.metrics;
                  return (
                    <tr key={a.agent_id} className="border-t border-white/10">
                      <td className="py-2 text-ink">{shortName(a.name)}</td>
                      <td className={`py-2 font-mono ${tone(m.total_return_pct)}`}>
                        {fmt(m.total_return_pct)}%
                      </td>
                      <td className={`py-2 font-mono ${tone(m.cagr_pct)}`}>{fmt(m.cagr_pct)}%</td>
                      <td className={`py-2 font-mono ${tone(m.sharpe)}`}>{fmt(m.sharpe, 3)}</td>
                      <td className={`py-2 font-mono ${tone(m.sortino)}`}>
                        {fmt(m.sortino, 3)}
                      </td>
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

          <RegimeTable agents={rankedBt} />
        </>
      )}

      <p className="text-sm text-slateui">
        Paper Arena remains separate.{" "}
        <Link href="/" className="text-mint hover:underline">
          Back to Arena
        </Link>
      </p>
    </div>
  );
}
