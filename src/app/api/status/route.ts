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
    instruments: snapshot.totals.instruments,
    agents: snapshot.totals.agents,
    real_money_trading: snapshot.status.real_money_trading,
  });
}
