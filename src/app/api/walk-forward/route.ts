import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

async function readLatest() {
  const candidates = [
    path.join(process.cwd(), "public", "data", "walk_forward", "latest.json"),
    path.join(process.cwd(), "data", "walk_forward", "latest.json"),
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
        error: "No walk-forward results yet. Run: npm run walk-forward",
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
    fold_count: payload.fold_count,
    warm_up: payload.warm_up,
    calibrate_bars: payload.calibrate_bars,
    test_bars: payload.test_bars,
    folds: payload.folds,
    fold_details: payload.fold_details,
    out_of_sample_only: payload.out_of_sample_only,
    leaderboard: agents.map((a) => ({
      agent_id: a.agent_id,
      name: a.name,
      metrics: a.metrics,
      regime_performance: a.regime_performance,
    })),
  });
}
