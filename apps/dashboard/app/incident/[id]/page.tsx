import Link from "next/link";
import { getIncidentTrace } from "@/lib/supabase";
import { TraceViewer } from "@/components/TraceViewer";
import { CausalGraph } from "@/components/CausalGraph";

export const revalidate = 0;

export default async function IncidentPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const steps = await getIncidentTrace(id);

  const synthesisStep = steps.find((s) => s.step_type === "synthesis");
  const causalGraph = (synthesisStep?.content ?? { edges: [] }) as {
    edges: { id: string; from: string; to: string; evidence_tool_call_id?: string }[];
    confidence?: number;
    confidence_rationale?: string;
    validation_error?: string;
  };

  const totalCost = steps.reduce((sum, s) => sum + (s.cost_usd ?? 0), 0);
  const totalTokens = steps.reduce(
    (sum, s) => sum + (s.input_tokens ?? 0) + (s.output_tokens ?? 0),
    0
  );
  const totalLatency = steps.reduce((sum, s) => sum + (s.latency_ms ?? 0), 0);
  const requiredDisclosures = steps.filter(
    (s) => s.tool_role === "required" && (s.content as Record<string, unknown>)?.disclosure
  ).length;

  return (
    <div className="min-h-screen px-6 py-10">
      <div className="mx-auto max-w-4xl">
        <Link href="/" className="text-sm text-[var(--muted)] hover:text-[var(--foreground)]">
          ← All incidents
        </Link>
        <h1 className="mt-2 font-mono text-lg font-semibold">{id}</h1>

        {/* Cost panel */}
        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="Equivalent cost" value={`$${totalCost.toFixed(6)}`} />
          <StatCard label="Tokens" value={totalTokens.toLocaleString()} />
          <StatCard label="Wall-clock latency" value={`${totalLatency}ms`} />
          <StatCard
            label="Fail-closed signals"
            value={String(requiredDisclosures)}
            warn={requiredDisclosures > 0}
          />
        </div>

        {/* Causal graph */}
        <section className="mt-8">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            Causal graph
          </h2>
          <CausalGraph graph={causalGraph} />
        </section>

        {/* Reasoning trail */}
        <section className="mt-8">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            Reasoning trail
          </h2>
          <TraceViewer steps={steps} />
        </section>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  warn = false,
}: {
  label: string;
  value: string;
  warn?: boolean;
}) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="text-xs text-[var(--muted)]">{label}</div>
      <div className={`mt-1 font-mono text-lg ${warn ? "text-[var(--accent-rose)]" : "text-[var(--foreground)]"}`}>
        {value}
      </div>
    </div>
  );
}
