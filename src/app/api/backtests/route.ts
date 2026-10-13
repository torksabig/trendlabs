import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

async function readLatest() {
  const candidates = [
    path.join(process.cwd(), "public", "data", "backtests", "latest.json"),
    path.join(process.cwd(), "data", "backtests", "latest.json"),
  ];
  for (const file of candidates) {
    try {
      const raw = await fs.readFile(file, "utf8");
      return JSON.parse(raw) as Record<string, unknown>;
    } catch {
      // try next
    }
  }
  return null;
}

export async function GET() {
  const payload = await readLatest();
  if (!payload) {
    return NextResponse.json(
      {
        ok: false,
        error: "No backtest results yet. Run: npm run backtest",
      },
      { status: 404 },
    );
  }
  const agents = (payload.agents as Array<Record<string, unknown>>) || [];
  return NextResponse.json({
    ok: true,
    run_id: payload.run_id,
    created_at: payload.created_at,
    timeframe: payload.timeframe,
    start_ts: payload.start_ts,
    end_ts: payload.end_ts,
    bars_tested: payload.bars_tested,
    benchmark_symbol: payload.benchmark_symbol,
    execution_model: payload.execution_model,
    leaderboard: agents.map((a) => ({
      agent_id: a.agent_id,
      name: a.name,
      metrics: a.metrics,
    })),
  });
}
