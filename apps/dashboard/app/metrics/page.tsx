"use client";

import { useSystemMetrics } from "@/lib/useSystemMetrics";

function StatCard({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "amber" | "teal" | "rose";
}) {
  const toneClass =
    tone === "amber"
      ? "text-[var(--accent-amber)]"
      : tone === "teal"
      ? "text-[var(--accent-teal)]"
      : tone === "rose"
      ? "text-[var(--accent-rose)]"
      : "text-[var(--foreground)]";
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
      <div className="text-xs uppercase tracking-wide text-[var(--muted)]">{label}</div>
      <div className={`mt-2 font-mono text-2xl ${toneClass}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-[var(--muted)]">{sub}</div>}
    </div>
  );
}

function Segment({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-10">
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className="mt-1 text-sm text-[var(--muted)]">{description}</p>
      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3">{children}</div>
    </section>
  );
}

function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(0)}%`;
}

function TrustBar({ score, samples }: { score: number; samples: number }) {
  const p = Math.round(score * 100);
  const color = score >= 0.9 ? "bg-[var(--accent-teal)]" : score >= 0.5 ? "bg-[var(--accent-amber)]" : "bg-[var(--accent-rose)]";
  return (
    <div className="flex items-center gap-3">
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-[var(--surface-raised)]">
        <div className={`h-full ${color}`} style={{ width: `${p}%` }} />
      </div>
      <span className="w-20 shrink-0 font-mono text-xs text-[var(--muted)]">
        {p}% ({samples})
      </span>
    </div>
  );
}

export default function MetricsPage() {
  const metrics = useSystemMetrics();

  if (metrics === null) {
    return <div className="px-6 py-10 text-sm text-[var(--muted)]">Loading…</div>;
  }

  if (!metrics.configured) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-10">
        <h1 className="text-2xl font-semibold">Evaluation metrics</h1>
        <div className="mt-6 rounded-lg border border-dashed border-[var(--border)] p-10 text-center text-sm text-[var(--muted)]">
          Control API can&apos;t reach Supabase, or no data has been recorded yet.
        </div>
      </div>
    );
  }

  const decisionTotal = (metrics.approved_count ?? 0) + (metrics.denied_count ?? 0);

  return (
    <div className="min-h-screen px-6 py-10">
      <div className="mx-auto max-w-4xl">
        <h1 className="text-2xl font-semibold">Evaluation metrics</h1>
        <p className="mt-1 text-sm text-[var(--muted)]">
          Live, aggregated across every incident this system has run — agent, tools, and
          overall system, each measuring a different question.
        </p>

        {/* AGENT PERFORMANCE */}
        <Segment
          title="Agent performance"
          description="Is the agent's reasoning actually working — does it reach conclusions, recover from partial failures, and stay affordable when it does?"
        >
          <StatCard
            label="Success rate (benchmark)"
            value={metrics.benchmark_success_rate != null ? pct(metrics.benchmark_success_rate) : "—"}
            sub={metrics.benchmark_runs_count ? `${metrics.benchmark_runs_count} scored runs` : "no runs yet"}
            tone={metrics.benchmark_success_rate != null && metrics.benchmark_success_rate >= 0.7 ? "teal" : "amber"}
          />
          <StatCard
            label="Resolution rate (live)"
            value={pct(metrics.resolution_rate)}
            sub="reached a non-empty causal graph, any confidence"
          />
          <StatCard
            label="Recovery rate"
            value={pct(metrics.recovery_rate)}
            sub="resolved despite a degraded tool call"
            tone={metrics.recovery_rate != null && metrics.recovery_rate < 0.5 ? "rose" : undefined}
          />
          <StatCard
            label="Human requests"
            value={String(metrics.human_requests_total ?? 0)}
            sub={
              metrics.human_requests_per_incident != null
                ? `${metrics.human_requests_per_incident.toFixed(2)} per incident`
                : undefined
            }
            tone={metrics.pending_approvals ? "amber" : undefined}
          />
          <StatCard
            label="Cost per completed task"
            value={metrics.cost_per_completed_task != null ? `$${metrics.cost_per_completed_task.toFixed(6)}` : "—"}
            sub="equivalent paid-tier rate"
          />
          <StatCard label="Total incidents" value={String(metrics.total_incidents ?? 0)} />
        </Segment>

        {/* TOOL PERFORMANCE */}
        <Segment
          title="Tool performance"
          description="Is the planner picking the right tools, and how often does the tool layer itself degrade?"
        >
          <StatCard
            label="Tool selection accuracy"
            value={pct(metrics.tool_selection_accuracy)}
            sub="valid, non-repeated calls / total plan attempts"
            tone={metrics.tool_selection_accuracy != null && metrics.tool_selection_accuracy < 0.8 ? "amber" : "teal"}
          />
          <StatCard
            label="Tool failure rate"
            value={pct(metrics.tool_failure_rate)}
            sub={`${metrics.total_tool_calls ?? 0} real tool calls total`}
            tone={metrics.tool_failure_rate != null && metrics.tool_failure_rate > 0.2 ? "amber" : undefined}
          />
          <StatCard
            label="Fail-closed / fail-open"
            value={`${metrics.fail_closed_count ?? 0} / ${metrics.fail_open_count ?? 0}`}
            sub="required-tool vs preferred-tool degradations"
          />
        </Segment>

        {/* SYSTEM OVERVIEW */}
        <Segment
          title="System overview"
          description="Everything else — total spend, where it goes, and approval throughput."
        >
          <StatCard label="Total cost" value={`$${(metrics.total_cost_usd ?? 0).toFixed(4)}`} />
          <StatCard
            label="Pending approvals"
            value={String(metrics.pending_approvals ?? 0)}
            tone={metrics.pending_approvals ? "amber" : undefined}
          />
          <StatCard label="Approved / Denied" value={`${metrics.approved_count ?? 0} / ${metrics.denied_count ?? 0}`} />
        </Segment>

        {/* Approval outcomes bar */}
        <section className="mt-10">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            Approval decisions ({decisionTotal} total)
          </h2>
          <div className="flex h-3 overflow-hidden rounded-full bg-[var(--surface-raised)]">
            {decisionTotal > 0 && (
              <>
                <div
                  className="bg-[var(--accent-teal)]"
                  style={{ width: `${((metrics.approved_count ?? 0) / decisionTotal) * 100}%` }}
                />
                <div
                  className="bg-[var(--accent-rose)]"
                  style={{ width: `${((metrics.denied_count ?? 0) / decisionTotal) * 100}%` }}
                />
              </>
            )}
          </div>
        </section>

        {/* Latency & cost percentiles -- real observability, not averages.
            p50 is the typical case; p95/p99 show what the slow tail
            actually looks like, which an average hides completely. */}
        <section className="mt-10">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            LLM call latency (ms)
            {metrics.latency_percentiles_ms && (
              <span className="ml-2 normal-case text-[var(--muted)]">
                — {metrics.latency_percentiles_ms.sample_size} calls
              </span>
            )}
          </h2>
          <div className="grid grid-cols-3 gap-4">
            <StatCard label="p50 (typical)" value={`${metrics.latency_percentiles_ms?.p50 ?? 0}ms`} />
            <StatCard label="p95" value={`${metrics.latency_percentiles_ms?.p95 ?? 0}ms`} tone="amber" />
            <StatCard label="p99 (worst case)" value={`${metrics.latency_percentiles_ms?.p99 ?? 0}ms`} tone="rose" />
          </div>

          {metrics.latency_percentiles_ms_by_provider && Object.keys(metrics.latency_percentiles_ms_by_provider).length > 0 && (
            <div className="mt-3 overflow-hidden rounded-lg border border-[var(--border)]">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[var(--border)] bg-[var(--surface-raised)] text-left text-xs text-[var(--muted)]">
                    <th className="px-4 py-2 font-medium">Provider</th>
                    <th className="px-4 py-2 font-medium">p50</th>
                    <th className="px-4 py-2 font-medium">p95</th>
                    <th className="px-4 py-2 font-medium">p99</th>
                    <th className="px-4 py-2 font-medium">Calls</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(metrics.latency_percentiles_ms_by_provider).map(([provider, p]) => (
                    <tr key={provider} className="border-b border-[var(--border)] last:border-0">
                      <td className="px-4 py-2 font-mono">{provider}</td>
                      <td className="px-4 py-2 font-mono">{p.p50}ms</td>
                      <td className="px-4 py-2 font-mono text-[var(--accent-amber)]">{p.p95}ms</td>
                      <td className="px-4 py-2 font-mono text-[var(--accent-rose)]">{p.p99}ms</td>
                      <td className="px-4 py-2 font-mono text-[var(--muted)]">{p.sample_size}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="border-t border-[var(--border)] bg-[var(--surface)] px-4 py-2 text-xs text-[var(--muted)]">
                Combined p50 above can mislead when providers differ a lot in speed — a
                blend of a fast and a slow provider looks moderate on average while
                neither provider is actually moderate. This breaks that out per provider.
              </p>
            </div>
          )}
        </section>

        <section className="mt-6">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            Cost per incident (equivalent paid-tier rate)
            {metrics.cost_per_incident_percentiles && (
              <span className="ml-2 normal-case text-[var(--muted)]">
                — {metrics.cost_per_incident_percentiles.sample_size} incidents
              </span>
            )}
          </h2>
          <div className="grid grid-cols-3 gap-4">
            <StatCard label="p50 (typical)" value={`$${(metrics.cost_per_incident_percentiles?.p50 ?? 0).toFixed(6)}`} />
            <StatCard label="p95" value={`$${(metrics.cost_per_incident_percentiles?.p95 ?? 0).toFixed(6)}`} tone="amber" />
            <StatCard label="p99 (worst case)" value={`$${(metrics.cost_per_incident_percentiles?.p99 ?? 0).toFixed(6)}`} tone="rose" />
          </div>
          <p className="mt-2 text-xs text-[var(--muted)]">
            Real spend on the free tier is $0 — this states what the same token volume
            would cost on comparable paid models (see EQUIVALENT_PAID_RATES), so this
            number is meaningful even though nothing is actually being billed.
          </p>
        </section>

        {/* Cost by provider */}
        <section className="mt-10">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">Cost by LLM provider</h2>
          {metrics.cost_by_provider && Object.keys(metrics.cost_by_provider).length > 0 ? (
            <div className="space-y-2">
              {Object.entries(metrics.cost_by_provider).map(([provider, cost]) => (
                <div key={provider} className="flex items-center justify-between rounded-md border border-[var(--border)] bg-[var(--surface)] px-4 py-2">
                  <span className="font-mono text-sm">{provider}</span>
                  <span className="font-mono text-sm text-[var(--muted)]">${cost.toFixed(6)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-[var(--muted)]">No cost data yet.</p>
          )}
        </section>

        {/* Per-pattern trust */}
        <section className="mt-10">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
            Per-pattern trust (graduated autonomy)
          </h2>
          {metrics.trust_patterns && metrics.trust_patterns.length > 0 ? (
            <div className="space-y-3">
              {metrics.trust_patterns.map((p) => (
                <div key={p.pattern_key} className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3">
                  <div className="mb-2 font-mono text-xs text-[var(--foreground)]">{p.pattern_key}</div>
                  <TrustBar score={p.trust_score} samples={p.total_events} />
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-[var(--muted)]">
              No pattern history yet -- trust scores build up as incidents of the same
              pattern get approved, denied, or benchmarked.
            </p>
          )}
        </section>
      </div>
    </div>
  );
}