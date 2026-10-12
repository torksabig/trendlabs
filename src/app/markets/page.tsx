import { MarketsView } from "@/components/markets/markets-view";
import { loadSnapshot } from "@/lib/snapshot";

export const dynamic = "force-dynamic";

export default async function MarketsPage({
  searchParams,
}: {
  searchParams: Promise<{ symbol?: string }>;
}) {
  const snapshot = await loadSnapshot();
  const params = await searchParams;
  const symbol = (params.symbol || "AAPL").toUpperCase();

  return <MarketsView snapshot={snapshot} initialSymbol={symbol} />;
}
