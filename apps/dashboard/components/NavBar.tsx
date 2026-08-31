"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSystemMetrics } from "@/lib/useSystemMetrics";

const LINKS = [
  { href: "/", label: "Incidents" },
  { href: "/approvals", label: "Approvals" },
  { href: "/notifications", label: "Notifications" },
  { href: "/metrics", label: "Metrics" },
  { href: "/status", label: "Status" },
];

export function NavBar() {
  const pathname = usePathname();
  const metrics = useSystemMetrics();

  return (
    <div className="sticky top-0 z-10 border-b border-[var(--border)] bg-[var(--surface)]/95 backdrop-blur">
      {/* Signature element: live system pulse, real numbers*/}
      <div className="flex items-center gap-6 border-b border-[var(--border)] px-6 py-1.5 font-mono text-[11px] text-[var(--muted)]">
        <span className="flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent-teal)]" />
          SYSTEM LIVE
        </span>
        {metrics?.configured ? (
          <>
            <span>
              incidents: <span className="text-[var(--foreground)]">{metrics.total_incidents ?? 0}</span>
            </span>
            <span className={metrics.pending_approvals ? "text-[var(--accent-amber)]" : ""}>
              pending: <span className="font-semibold">{metrics.pending_approvals ?? 0}</span>
            </span>
            <span>
              cost: <span className="text-[var(--foreground)]">${(metrics.total_cost_usd ?? 0).toFixed(4)}</span>
            </span>
          </>
        ) : (
          <span>Supabase not configured</span>
        )}
      </div>

      <nav className="flex items-center gap-1 px-6 py-3">
        <span className="mr-6 font-mono text-sm font-semibold tracking-tight text-[var(--foreground)]">
          genesis-kit
        </span>
        {LINKS.map((link) => {
          const active = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
                active
                  ? "bg-[var(--surface-raised)] text-[var(--foreground)]"
                  : "text-[var(--muted)] hover:text-[var(--foreground)]"
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
