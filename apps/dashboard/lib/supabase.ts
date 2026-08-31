import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "https://placeholder.supabase.co";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "placeholder-anon-key";

const SUPABASE_FETCH_TIMEOUT_MS = Number(process.env.NEXT_PUBLIC_SUPABASE_FETCH_TIMEOUT_MS) || 20_000;

function fetchWithTimeout(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), SUPABASE_FETCH_TIMEOUT_MS);

  return fetch(input, { ...init, signal: controller.signal }).finally(() => {
    clearTimeout(timeoutId);
  });
}

function logSupabaseIssue(context: string, error: { message?: string; code?: string } | null): void {
  const hasRealErrorCode = Boolean(error?.code);
  if (hasRealErrorCode) {
    console.error(`${context}:`, error?.message, `(code: ${error?.code})`);
  } else {
    console.warn(
      `${context}: request failed (${error?.message || "unknown"}) -- degrading to empty result. ` +
      `If this happens often, your network path to Supabase may be slow or misconfigured; check ` +
      `.env.local and consider raising NEXT_PUBLIC_SUPABASE_FETCH_TIMEOUT_MS.`
    );
  }
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  global: { fetch: fetchWithTimeout },
});

export type TraceStep = {
  id: number;
  incident_id: string;
  step_index: number;
  step_type: "plan" | "observation" | "synthesis";
  tool: string | null;
  tool_role: "required" | "preferred" | null;
  status: string | null;
  provider: string | null;
  content: Record<string, unknown>;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  latency_ms: number | null;
  created_at: string;
};

export type IncidentSummary = {
  incident_id: string;
  total_cost_usd: number;
  total_input_tokens: number;
  total_output_tokens: number;
  plan_steps: number;
  last_activity_at: string | null;
  latest_step_type: string;
  latest_status: string | null;
};

const PAGE_SIZE = 20;

export async function listIncidents(page: number = 0): Promise<{ incidents: IncidentSummary[]; hasMore: boolean }> {
 
  const from = page * PAGE_SIZE;
  const to = from + PAGE_SIZE;

  const { data: summaryRows, error } = await supabase
    .from("incident_cost_summary")
    .select("*")
    .order("last_activity_at", { ascending: false })
    .range(from, to);

  if (error) {
    logSupabaseIssue("listIncidents", error);
    return { incidents: [], hasMore: false };
  }
  if (!summaryRows || summaryRows.length === 0) {
    return { incidents: [], hasMore: false };
  }

  const hasMore = summaryRows.length > PAGE_SIZE;
  const pageRows = summaryRows.slice(0, PAGE_SIZE);
  const incidentIds = pageRows.map((r) => r.incident_id);

  const { data: recentSteps } = await supabase
    .from("traces")
    .select("incident_id, step_type, status, created_at")
    .in("incident_id", incidentIds)
    .order("created_at", { ascending: false });

  const latestByIncident = new Map<string, { step_type: string; status: string | null }>();
  for (const row of recentSteps ?? []) {
    if (!latestByIncident.has(row.incident_id)) {
      latestByIncident.set(row.incident_id, { step_type: row.step_type, status: row.status });
    }
  }

  const incidents: IncidentSummary[] = pageRows.map((r) => ({
    incident_id: r.incident_id,
    total_cost_usd: r.total_cost_usd ?? 0,
    total_input_tokens: r.total_input_tokens ?? 0,
    total_output_tokens: r.total_output_tokens ?? 0,
    plan_steps: r.plan_steps ?? 0,
    last_activity_at: r.last_activity_at ?? null,
    latest_step_type: latestByIncident.get(r.incident_id)?.step_type ?? "unknown",
    latest_status: latestByIncident.get(r.incident_id)?.status ?? null,
  }));

  return { incidents, hasMore };
}

export async function getIncidentTrace(incidentId: string): Promise<TraceStep[]> {
  const { data, error } = await supabase
    .from("traces")
    .select("*")
    .eq("incident_id", incidentId)
    .order("step_index", { ascending: true });

  if (error) {
    logSupabaseIssue("getIncidentTrace", error);
    return [];
  }
  return (data as TraceStep[]) ?? [];
}

export async function checkSupabaseHealth(): Promise<boolean> {

  try {
    const { error } = await supabase.from("traces").select("id", { count: "exact", head: true });
    return !error;
  } catch {
    return false;
  }
}
