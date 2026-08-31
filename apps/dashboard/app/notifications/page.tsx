"use client";

import Link from "next/link";
import { useSystemMetrics } from "@/lib/useSystemMetrics";

function timeAgo(iso: string): string {
  const minutes = Math.floor((Date.now() - new Date(iso).getTime()) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`;
  return `${Math.floor(minutes / 1440)}d ago`;
}

export default function NotificationsPage() {
  const metrics = useSystemMetrics();

  const notifications = metrics?.recent_notifications ?? [];

  return (
    <div className="min-h-screen px-6 py-10">
      <div className="mx-auto max-w-3xl">
        <h1 className="text-2xl font-semibold">Notifications</h1>
        <p className="mt-1 text-sm text-[var(--muted)]">
          Audit trail for everything that executed without needing your approval — low
          blast-radius actions (e.g. <code className="font-mono text-xs">restart_service</code>)
          and trust-based auto-approvals. Visibility without gating everything.
        </p>

        {metrics === null ? (
          <div className="mt-10 text-sm text-[var(--muted)]">Loading…</div>
        ) : !metrics.configured ? (
          <div className="mt-10 rounded-lg border border-dashed border-[var(--border)] p-10 text-center text-sm text-[var(--muted)]">
            Control API can&apos;t reach Supabase, or no data has been recorded yet.
          </div>
        ) : notifications.length === 0 ? (
          <div className="mt-10 rounded-lg border border-dashed border-[var(--border)] p-10 text-center text-sm text-[var(--muted)]">
            Nothing has auto-executed yet. This fills in as low-risk actions run, or once a
            pattern earns enough trust (see{" "}
            <Link href="/metrics" className="underline hover:text-[var(--foreground)]">
              Metrics
            </Link>
            ) to auto-approve.
          </div>
        ) : (
          <div className="mt-8 space-y-2">
            {notifications.map((n) => (
              <div
                key={n.id}
                className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-4 py-3"
              >
                <div className="flex items-center gap-2">
                  <Link
                    href={`/incident/${n.incident_id}`}
                    className="font-mono text-xs text-[var(--accent-teal)] hover:underline"
                  >
                    {n.incident_id}
                  </Link>
                  <span className="rounded bg-[var(--surface-raised)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--muted)]">
                    {n.action}
                  </span>
                  <span className="ml-auto text-xs text-[var(--muted)]">{timeAgo(n.created_at)}</span>
                </div>
                <p className="mt-1 text-sm text-[var(--foreground)]">{n.message}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
