"use client";

import { useEffect, useState } from "react";
import {
  decideApproval,
  fetchPendingApprovals,
  ControlApiError,
  type PendingApproval,
} from "@/lib/controlApi";

function timeAgo(iso: string): string {
  const minutes = Math.floor((Date.now() - new Date(iso).getTime()) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m ago`;
}

const BLAST_STYLE: Record<string, string> = {
  low: "bg-[var(--accent-teal-dim)] text-[var(--accent-teal)]",
  medium: "bg-[var(--accent-amber-dim)] text-[var(--accent-amber)]",
  high: "bg-[var(--accent-rose-dim)] text-[var(--accent-rose)]",
};

function ApprovalCard({
  approval,
  onDecided,
}: {
  approval: PendingApproval;
  onDecided: (incidentId: string) => void;
}) {
  const [decidedBy, setDecidedBy] = useState("");
  const [reason, setReason] = useState("");
  const [showDenyReason, setShowDenyReason] = useState(false);
  const [busy, setBusy] = useState<"approve" | "deny" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resultMessage, setResultMessage] = useState<string | null>(null);

  const isEscalated = Boolean(approval.escalated_at);

  async function handleDecide(decision: "approve" | "deny") {
    if (!decidedBy.trim()) {
      setError("Enter your name before deciding.");
      return;
    }
    if (decision === "deny" && !showDenyReason) {
      setShowDenyReason(true);
      return;
    }

    setBusy(decision);
    setError(null);
    try {
      const result = await decideApproval(approval.incident_id, decision, decidedBy, reason || undefined);
      if (decision === "approve" && result.execution) {
        setResultMessage(
          result.execution.executed
            ? `Executed: ${JSON.stringify(result.execution.result)}`
            : `Approved, but not executed: ${result.execution.error}`
        );
      } else {
        setResultMessage(`Denied.`);
      }
      setTimeout(() => onDecided(approval.incident_id), 1200);
    } catch (e) {
      setError(e instanceof ControlApiError ? e.message : "Request failed -- is the control API running?");
    } finally {
      setBusy(null);
    }
  }

  const causalSummary = approval.evidence?.edges
    ?.map((e) => `${e.from} → ${e.to}`)
    .join("; ");

  return (
    <div
      className={`rounded-lg border bg-[var(--surface)] p-5 ${
        isEscalated ? "border-[var(--accent-rose)]" : "border-[var(--border)]"
      }`}
    >
      {isEscalated && (
        <div className="mb-3 rounded-md border border-[var(--accent-rose)] bg-[var(--accent-rose-dim)] px-3 py-1.5 text-xs font-semibold text-[var(--accent-rose)]">
          ESCALATED — pending longer than the SLA threshold
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm text-[var(--foreground)]">{approval.incident_id}</span>
        <span className={`rounded px-2 py-0.5 text-xs font-medium ${BLAST_STYLE[approval.blast_radius]}`}>
          {approval.blast_radius}
        </span>
        <span className="rounded bg-[var(--surface-raised)] px-2 py-0.5 font-mono text-xs text-[var(--muted)]">
          {approval.action}
        </span>
        {approval.service && (
          <span className="text-xs text-[var(--muted)]">on {approval.service}</span>
        )}
        <span className="ml-auto text-xs text-[var(--muted)]">{timeAgo(approval.created_at)}</span>
      </div>

      <div className="mt-3 text-sm text-[var(--foreground)]">
        Confidence: <span className="font-mono">{(approval.confidence * 100).toFixed(0)}%</span>
      </div>

      {approval.confidence_rationale && (
        <p className="mt-1 text-sm text-[var(--muted)]">{approval.confidence_rationale}</p>
      )}

      {causalSummary && (
        <p className="mt-2 font-mono text-xs text-[var(--muted)]">{causalSummary}</p>
      )}

      <div className="mt-4 border-t border-[var(--border)] pt-4">
        <input
          type="text"
          placeholder="Your name"
          aria-label="Your name"
          value={decidedBy}
          onChange={(e) => setDecidedBy(e.target.value)}
          className="w-full rounded-md border border-[var(--border)] bg-[var(--surface-raised)] px-3 py-1.5 text-sm text-[var(--foreground)] placeholder:text-[var(--muted)] focus:border-[var(--accent-teal)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-teal)]/40"
        />

        {showDenyReason && (
          <input
            type="text"
            placeholder="Reason for denial (shown on future approvals for this pattern)"
            aria-label="Reason for denial"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            className="mt-2 w-full rounded-md border border-[var(--border)] bg-[var(--surface-raised)] px-3 py-1.5 text-sm text-[var(--foreground)] placeholder:text-[var(--muted)] focus:border-[var(--accent-rose)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-rose)]/40"
          />
        )}

        <div role="status" aria-live="polite">
          {error && <p className="mt-2 text-xs text-[var(--accent-rose)]">{error}</p>}
          {resultMessage && <p className="mt-2 text-xs text-[var(--accent-teal)]">{resultMessage}</p>}
        </div>

        <div className="mt-3 flex gap-2">
          <button
            onClick={() => handleDecide("approve")}
            disabled={busy !== null || Boolean(resultMessage)}
            aria-label={`Approve action for incident ${approval.incident_id}`}
            className="rounded-md bg-[var(--accent-teal)] px-4 py-1.5 text-sm font-medium text-[var(--background)] transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-[var(--accent-teal)] focus:ring-offset-2 focus:ring-offset-[var(--surface)] disabled:opacity-40"
          >
            {busy === "approve" ? "Approving…" : "Approve"}
          </button>
          <button
            onClick={() => handleDecide("deny")}
            disabled={busy !== null || Boolean(resultMessage)}
            aria-label={`Deny action for incident ${approval.incident_id}`}
            className="rounded-md border border-[var(--accent-rose)] px-4 py-1.5 text-sm font-medium text-[var(--accent-rose)] transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-[var(--accent-rose)] focus:ring-offset-2 focus:ring-offset-[var(--surface)] disabled:opacity-40"
          >
            {busy === "deny" ? "Denying…" : showDenyReason ? "Confirm deny" : "Deny"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function ApprovalsPage() {
  const [approvals, setApprovals] = useState<PendingApproval[] | null>(null);

  useEffect(() => {
    fetchPendingApprovals().then(setApprovals);
    const interval = setInterval(() => {
      fetchPendingApprovals().then(setApprovals);
    }, 15_000);
    return () => clearInterval(interval);
  }, []);

  function handleDecided(incidentId: string) {
    setApprovals((prev) => (prev ? prev.filter((a) => a.incident_id !== incidentId) : prev));
  }

  return (
    <div className="min-h-screen px-6 py-10">
      <div className="mx-auto max-w-3xl">
        <h1 className="text-2xl font-semibold">Pending approvals</h1>
        <p className="mt-1 text-sm text-[var(--muted)]">
          Actions the agent flagged as irreversible, high blast-radius, or low-confidence.
          Approving executes the action for real via a short-lived approver token.
        </p>

        {approvals === null ? (
          <div className="mt-10 text-sm text-[var(--muted)]">Loading…</div>
        ) : approvals.length === 0 ? (
          <div className="mt-10 rounded-lg border border-dashed border-[var(--border)] p-10 text-center text-sm text-[var(--muted)]">
            No pending approvals. Nothing is waiting on you right now.
          </div>
        ) : (
          <div className="mt-8 space-y-4">
            {approvals.map((a) => (
              <ApprovalCard key={a.id} approval={a} onDecided={handleDecided} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
