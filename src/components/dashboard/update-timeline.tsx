import type { Snapshot } from "@/lib/types";
import { formatAge, formatUntil, formatWhen } from "@/lib/utils";

type Props = {
  snapshot: Snapshot;
};

const ROWS: Array<{ key: keyof NonNullable<Snapshot["updates"]>; label: string }> = [
  { key: "collection", label: "Market data collected" },
  { key: "simulation", label: "Paper agents simulated" },
  { key: "prediction", label: "Hourly predictions" },
  { key: "research", label: "Research report" },
  { key: "snapshot", label: "Dashboard snapshot" },
  { key: "next_prediction", label: "Next prediction run" },
];

export function UpdateTimeline({ snapshot }: Props) {
  const updates = snapshot.updates ?? {
    snapshot: snapshot.generated_at,
    collection: snapshot.last_collection,
    simulation: snapshot.last_simulation,
    research: snapshot.last_research ?? snapshot.research.research_agent?.last_research,
    prediction: snapshot.last_prediction,
    next_prediction: snapshot.next_prediction_at,
    schedule: "Hourly GitHub Actions at :15 UTC — collect → simulate → predict",
  };

  return (
    <section className="panel p-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="font-display text-2xl text-ink">Update times</h2>
          <p className="mt-1 text-sm text-slateui">
            {updates.schedule ?? "Hourly collect → simulate → predict"}
          </p>
        </div>
        <p className="font-mono text-xs text-slateui-dim" suppressHydrationWarning>
          Snapshot {formatAge(snapshot.generated_at)}
        </p>
      </div>
      <dl className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {ROWS.map(({ key, label }) => {
          const value = updates[key] ?? null;
          const isFuture = key === "next_prediction";
          return (
            <div
              key={key}
              className="rounded-lg border border-white/10 bg-charcoal px-3 py-3"
            >
              <dt className="text-xs uppercase tracking-wide text-slateui-dim">{label}</dt>
              <dd className="mt-1 font-mono text-sm text-ink">{formatWhen(value)}</dd>
              <dd className="mt-1 text-xs text-slateui" suppressHydrationWarning>
                {isFuture ? formatUntil(value) : formatAge(value)}
              </dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}
