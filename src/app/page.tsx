import { ArenaDashboard } from "@/components/dashboard/arena-dashboard";
import { loadSnapshot } from "@/lib/snapshot";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const snapshot = await loadSnapshot();

  if (!snapshot) {
    return (
      <section className="panel fade-up p-8">
        <h1 className="font-display text-5xl text-ink">TrendLab Arena</h1>
        <p className="mt-4 max-w-2xl text-slateui">
          No market snapshot yet. Collect candles and run the four paper agents, then refresh.
        </p>
        <pre className="mt-6 overflow-x-auto rounded-xl bg-charcoal-2 px-4 py-3 font-mono text-sm text-mint">
          npm run seed
        </pre>
      </section>
    );
  }

  return <ArenaDashboard snapshot={snapshot} />;
}
