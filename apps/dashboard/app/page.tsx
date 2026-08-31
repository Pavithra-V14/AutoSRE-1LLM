import Link from "next/link";
import { listIncidents } from "@/lib/supabase";

export const revalidate = 0; // always fetch fresh -- this is an ops dashboard, not a marketing page

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ page?: string }>;
}) {
  const { page: pageParam } = await searchParams;
  const page = Math.max(0, parseInt(pageParam ?? "0", 10) || 0);
  const { incidents, hasMore } = await listIncidents(page);

  return (
    <div className="min-h-screen px-6 py-10">
      <div className="mx-auto max-w-4xl">
        <h1 className="text-2xl font-semibold">Incidents</h1>
        <p className="mt-1 text-sm text-[var(--muted)]">
          Autonomous Incident Commander — reasoning trail, causal graph, and cost per incident.
        </p>

        {incidents.length === 0 ? (
          <div className="mt-10 rounded-lg border border-dashed border-[var(--border)] p-10 text-center text-sm text-[var(--muted)]">
            {page === 0 ? (
              <>
                No incidents recorded yet. Run{" "}
                <code className="rounded bg-[var(--surface-raised)] px-1.5 py-0.5 font-mono text-xs">
                  python -m services.agent.graph --scenario db_timeout
                </code>{" "}
                with Supabase configured to see one here.
              </>
            ) : (
              "No more incidents on this page."
            )}
          </div>
        ) : (
          <>
            <div className="mt-8 divide-y divide-[var(--border)] overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]">
              {incidents.map((incident) => (
                <Link
                  key={incident.incident_id}
                  href={`/incident/${incident.incident_id}`}
                  className="flex items-center justify-between gap-4 px-5 py-4 transition-colors hover:bg-[var(--surface-raised)]"
                >
                  <div>
                    <div className="font-mono text-sm">{incident.incident_id}</div>
                    <div className="mt-1 flex gap-3 text-xs text-[var(--muted)]">
                      <span>{incident.plan_steps} plan step(s)</span>
                      <span>
                        {incident.total_input_tokens + incident.total_output_tokens} tokens
                      </span>
                      {incident.latest_status && <span>latest: {incident.latest_status}</span>}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm">
                      ${incident.total_cost_usd.toFixed(6)}
                    </div>
                    <div className="text-xs text-[var(--muted)]">equivalent cost</div>
                  </div>
                </Link>
              ))}
            </div>

            <div className="mt-4 flex items-center justify-between text-sm">
              <Link
                href={`/?page=${Math.max(0, page - 1)}`}
                aria-disabled={page === 0}
                className={`rounded-md border border-[var(--border)] px-3 py-1.5 ${
                  page === 0
                    ? "pointer-events-none text-[var(--muted)] opacity-40"
                    : "text-[var(--foreground)] hover:bg-[var(--surface-raised)]"
                }`}
              >
                ← Newer
              </Link>
              <span className="text-[var(--muted)]">Page {page + 1}</span>
              <Link
                href={`/?page=${page + 1}`}
                aria-disabled={!hasMore}
                className={`rounded-md border border-[var(--border)] px-3 py-1.5 ${
                  !hasMore
                    ? "pointer-events-none text-[var(--muted)] opacity-40"
                    : "text-[var(--foreground)] hover:bg-[var(--surface-raised)]"
                }`}
              >
                Older →
              </Link>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
