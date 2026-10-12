import { NextResponse } from "next/server";

import { loadSnapshot } from "@/lib/snapshot";

export const dynamic = "force-dynamic";

export async function GET() {
  const snapshot = await loadSnapshot();
  if (!snapshot) {
    return NextResponse.json({ error: "Snapshot missing" }, { status: 404 });
  }
  return NextResponse.json({
    research_agent: snapshot.research.research_agent ?? null,
    ai_reports: snapshot.research.ai_reports ?? [],
    historical: {
      bullish: snapshot.research.bullish,
      bearish: snapshot.research.bearish,
    },
    llm_executes_trades: snapshot.status.llm_executes_trades ?? false,
  });
}
