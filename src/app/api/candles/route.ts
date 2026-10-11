import { NextRequest, NextResponse } from "next/server";

import { loadSnapshot } from "@/lib/snapshot";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const snapshot = await loadSnapshot();
  if (!snapshot) {
    return NextResponse.json({ error: "Snapshot missing" }, { status: 404 });
  }
  const symbol = (request.nextUrl.searchParams.get("symbol") || "AAPL").toUpperCase();
  const market = snapshot.markets.find((m) => m.symbol === symbol);
  if (!market) {
    return NextResponse.json({ error: `No candles for ${symbol}` }, { status: 404 });
  }
  return NextResponse.json(market);
}
