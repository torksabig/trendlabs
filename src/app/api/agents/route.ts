import { NextResponse } from "next/server";

import { loadSnapshot } from "@/lib/snapshot";

export const dynamic = "force-dynamic";

export async function GET() {
  const snapshot = await loadSnapshot();
  if (!snapshot) {
    return NextResponse.json({ error: "Snapshot missing" }, { status: 404 });
  }
  return NextResponse.json({
    leaderboard: snapshot.leaderboard,
    equity_curves: snapshot.equity_curves,
    trades: snapshot.trades.slice(0, 50),
  });
}
