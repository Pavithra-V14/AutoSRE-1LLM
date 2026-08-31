"use client";

import { useEffect, useState } from "react";
import { fetchControlApiHealth, wipeAllData, ControlApiError, type WipeDataResult } from "@/lib/controlApi";
import { fetchSimulatorHealth } from "@/lib/simulatorHealth";
import { checkSupabaseHealth } from "@/lib/supabase";

type ServiceStatus = "checking" | "up" | "down";

function StatusDot({ status }: { status: ServiceStatus }) {
  const color =
    status === "up"
      ? "bg-[var(--accent-teal)]"
      : status === "down"
      ? "bg-[var(--accent-rose)]"
      : "bg-[var(--muted)] animate-pulse";
  return <span className={`inline-block h-2.5 w-2.5 rounded-full ${color}`} />;
}

function ServiceRow({
  name,
  description,
  status,
  detail,
}: {
  name: string;
  description: string;
  status: ServiceStatus;
  detail?: string;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-[var(--border)] bg-[var(--surface)] px-5 py-4">
      <div>
        <div className="flex items-center gap-2">
          <StatusDot status={status} />
          <span className="font-medium">{name}</span>
        </div>
        <p className="mt-1 text-sm text-[var(--muted)]">{description}</p>
      </div>
      <div className="text-right">
        <div
          className={`font-mono text-sm ${
            status === "up"
              ? "text-[var(--accent-teal)]"
              : status === "down"
              ? "text-[var(--accent-rose)]"
              : "text-[var(--muted)]"
          }`}
        >
          {status === "checking" ? "checking…" : status.toUpperCase()}
        </div>
        {detail && <div className="mt-0.5 text-xs text-[var(--muted)]">{detail}</div>}
      </div>
    </div>
  );
}

const CONFIRMATION_PHRASE = "DELETE ALL DATA";

function DangerZone() {
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<WipeDataResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canDelete = typed === CONFIRMATION_PHRASE;

  async function handleDelete() {
    if (!canDelete) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await wipeAllData(typed);
      setResult(res);
      setTyped("");
    } catch (e) {
      setError(e instanceof ControlApiError ? e.message : "Request failed -- is the control API running?");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-10 rounded-lg border border-[var(--accent-rose)] bg-[var(--accent-rose-dim)] p-5">
      <h2 className="font-semibold text-[var(--accent-rose)]">Danger zone</h2>
      <p className="mt-1 text-sm text-rose-200">
        Permanently deletes every row in every table this project writes to --
        traces, approvals, trust history, notifications, dedup records, and
        benchmark runs. <strong>This cannot be undone.</strong> Mainly useful for
        resetting a demo/dev environment back to empty, not something you want
        to run against data you care about.
      </p>

      <label htmlFor="wipe-confirmation" className="mt-4 block text-xs text-rose-200">
        Type <code className="rounded bg-black/30 px-1 py-0.5 font-mono">{CONFIRMATION_PHRASE}</code> to enable the button:
      </label>
      <input
        id="wipe-confirmation"
        type="text"
        value={typed}
        onChange={(e) => setTyped(e.target.value)}
        aria-label={`Type ${CONFIRMATION_PHRASE} to confirm`}
        className="mt-1 w-full rounded-md border border-[var(--accent-rose)]/50 bg-black/20 px-3 py-1.5 font-mono text-sm text-white placeholder:text-rose-300/50 focus:outline-none focus:ring-2 focus:ring-[var(--accent-rose)]"
        placeholder={CONFIRMATION_PHRASE}
      />

      <button
        onClick={handleDelete}
        disabled={!canDelete || busy}
        aria-label="Permanently delete all data"
        className="mt-3 rounded-md bg-[var(--accent-rose)] px-4 py-1.5 text-sm font-semibold text-white transition-opacity hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-[var(--accent-rose-dim)] disabled:cursor-not-allowed disabled:opacity-30"
      >
        {busy ? "Deleting…" : "Permanently delete all data"}
      </button>

      <div role="status" aria-live="polite">
        {error && <p className="mt-3 text-sm text-rose-200">{error}</p>}
        {result && (
          <div className="mt-3 rounded-md bg-black/20 p-3 text-xs text-rose-100">
            {result.configured ? (
              <>
                <p className="font-semibold text-white">Deleted:</p>
                <ul className="mt-1 space-y-0.5 font-mono">
                  {Object.entries(result.tables).map(([table, r]) => (
                    <li key={table}>
                      {table}: {r.error ? `ERROR (${r.error})` : `${r.deleted} row(s)`}
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p>Control API isn&apos;t connected to Supabase -- nothing to delete.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function StatusPage() {
  const [simulator, setSimulator] = useState<ServiceStatus>("checking");
  const [controlApi, setControlApi] = useState<ServiceStatus>("checking");
  const [controlApiDetail, setControlApiDetail] = useState<string | undefined>();
  const [supabaseUp, setSupabaseUp] = useState<ServiceStatus>("checking");
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  useEffect(() => {
    function runCheck() {
      Promise.all([fetchSimulatorHealth(), fetchControlApiHealth(), checkSupabaseHealth()]).then(
        ([simRes, apiRes, dbRes]) => {
          setSimulator(simRes ? "up" : "down");
          setControlApi(apiRes ? "up" : "down");
          setControlApiDetail(
            apiRes ? (apiRes.auth_configured ? "auth enabled" : "auth NOT configured") : undefined
          );
          setSupabaseUp(dbRes ? "up" : "down");
          setLastChecked(new Date());
        }
      );
    }
    runCheck();
    const interval = setInterval(runCheck, 15_000);
    return () => clearInterval(interval);
  }, []);

  const allUp = simulator === "up" && controlApi === "up" && supabaseUp === "up";
  const anyChecked = simulator !== "checking";

  return (
    <div className="min-h-screen px-6 py-10">
      <div className="mx-auto max-w-2xl">
        <h1 className="text-2xl font-semibold">System status</h1>
        <p className="mt-1 text-sm text-[var(--muted)]">
          Live reachability of every backend service this dashboard depends on.
        </p>

        {anyChecked && (
          <div
            className={`mt-6 rounded-lg border px-4 py-3 text-sm font-medium ${
              allUp
                ? "border-[var(--accent-teal)] bg-[var(--accent-teal-dim)] text-[var(--accent-teal)]"
                : "border-[var(--accent-rose)] bg-[var(--accent-rose-dim)] text-[var(--accent-rose)]"
            }`}
          >
            {allUp ? "All systems operational" : "One or more services are unreachable"}
          </div>
        )}

        <div className="mt-6 space-y-3">
          <ServiceRow
            name="Incident simulator"
            description="The synthetic production environment the agent investigates."
            status={simulator}
          />
          <ServiceRow
            name="Control API"
            description="Backend for approve/deny actions and system metrics."
            status={controlApi}
            detail={controlApiDetail}
          />
          <ServiceRow
            name="Supabase"
            description="Traces, approvals, trust events -- the source of truth for everything else on this dashboard."
            status={supabaseUp}
          />
        </div>

        {lastChecked && (
          <p className="mt-6 text-xs text-[var(--muted)]">
            Last checked {lastChecked.toLocaleTimeString()} · rechecks every 15s
          </p>
        )}

        <div className="mt-8 rounded-lg border border-dashed border-[var(--border)] p-4 text-xs text-[var(--muted)]">
          A service showing DOWN doesn&apos;t necessarily mean it crashed -- it may
          not be running yet. See the README &quot;Setup&quot; section for the commands
          to start each one.
        </div>

        <DangerZone />
      </div>
    </div>
  );
}
