import { NextResponse } from "next/server";

import { loadSnapshot } from "@/lib/snapshot";

export const dynamic = "force-dynamic";

export async function GET() {
  const snapshot = await loadSnapshot();
  if (!snapshot) {
    return NextResponse.json(
      {
        ok: false,
        message: "No snapshot yet. Run npm run seed to collect data and simulate agents.",
      },
      { status: 404 },
    );
  }
  return NextResponse.json({
    ok: true,
    generated_at: snapshot.generated_at,
    data_source: snapshot.data_source,
    last_collection: snapshot.last_collection,
    last_simulation: snapshot.last_simulation,
    last_research: snapshot.last_research ?? null,
    last_prediction: snapshot.last_prediction ?? null,
    next_prediction_at: snapshot.next_prediction_at ?? null,
    updates: snapshot.updates ?? null,
    prediction_count: snapshot.predictions?.latest?.length ?? 0,
    instruments: snapshot.totals.instruments,
    agents: snapshot.totals.agents,
    real_money_trading: snapshot.status.real_money_trading,
  });
}
