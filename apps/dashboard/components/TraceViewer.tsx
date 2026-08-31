"use client";

import type { TraceStep } from "@/lib/supabase";

const STEP_TYPE_LABEL: Record<string, string> = {
  plan: "Plan",
  observation: "Observation",
  synthesis: "Synthesis",
};

function StepBadge({ step }: { step: TraceStep }) {
  const label = STEP_TYPE_LABEL[step.step_type] ?? step.step_type;
  const color =
    step.step_type === "plan"
      ? "bg-blue-500/15 text-blue-300"
      : step.step_type === "observation"
      ? "bg-violet-500/15 text-violet-300"
      : "bg-[var(--accent-teal-dim)] text-[var(--accent-teal)]";
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${color}`}>{label}</span>
  );
}

function ToolRoleBadge({ role }: { role: "required" | "preferred" }) {
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
        role === "required"
          ? "bg-[var(--accent-rose-dim)] text-[var(--accent-rose)]"
          : "bg-[var(--surface-raised)] text-[var(--muted)]"
      }`}
    >
      {role}
    </span>
  );
}

function DisclosureCallout({ disclosure, role }: { disclosure: string; role?: string }) {
  const isFailClosed = role === "required";
  return (
    <div
      className={`mt-2 rounded-md border-l-4 p-3 text-sm ${
        isFailClosed
          ? "border-[var(--accent-rose)] bg-[var(--accent-rose-dim)] text-rose-200"
          : "border-[var(--accent-amber)] bg-[var(--accent-amber-dim)] text-amber-200"
      }`}
    >
      <span className="font-semibold">
        {isFailClosed ? "Fail-closed: " : "Fail-open: "}
      </span>
      {disclosure}
    </div>
  );
}

function StepCard({ step }: { step: TraceStep }) {
  const content = step.content as Record<string, unknown>;
  const disclosure = typeof content?.disclosure === "string" ? content.disclosure : null;

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="flex flex-wrap items-center gap-2">
        <StepBadge step={step} />
        {step.tool && (
          <span className="rounded bg-[var(--surface-raised)] px-2 py-0.5 font-mono text-xs text-[var(--foreground)]">
            {step.tool}
          </span>
        )}
        {step.tool_role && <ToolRoleBadge role={step.tool_role} />}
        {step.status && (
          <span className="text-xs text-[var(--muted)]">status: {step.status}</span>
        )}
        {step.provider && (
          <span className="ml-auto text-xs text-[var(--muted)]">via {step.provider}</span>
        )}
      </div>

      {disclosure && <DisclosureCallout disclosure={disclosure} role={step.tool_role ?? undefined} />}

      <details className="mt-2">
        <summary className="cursor-pointer text-xs text-[var(--muted)] hover:text-[var(--foreground)]">
          raw step content
        </summary>
        <pre className="mt-1 overflow-x-auto rounded bg-[var(--surface-raised)] p-2 text-[11px] text-[var(--muted)]">
          {JSON.stringify(step.content, null, 2)}
        </pre>
      </details>

      <div className="mt-2 flex gap-4 text-[11px] text-[var(--muted)]">
        {(step.input_tokens > 0 || step.output_tokens > 0) && (
          <span>
            {step.input_tokens}in / {step.output_tokens}out tokens
          </span>
        )}
        {step.latency_ms != null && <span>{step.latency_ms}ms</span>}
        {step.cost_usd > 0 && <span>${step.cost_usd.toFixed(6)}</span>}
      </div>
    </div>
  );
}

export function TraceViewer({ steps }: { steps: TraceStep[] }) {
  if (steps.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border)] p-6 text-center text-sm text-[var(--muted)]">
        No trace steps recorded for this incident.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {steps.map((step) => (
        <StepCard key={step.id} step={step} />
      ))}
    </div>
  );
}
