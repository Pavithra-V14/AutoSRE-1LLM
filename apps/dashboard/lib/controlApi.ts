const CONTROL_API_URL = process.env.NEXT_PUBLIC_CONTROL_API_URL || "http://localhost:8101";
const FETCH_TIMEOUT_MS = 15_000; 

async function fetchWithTimeout(url: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeoutId);
  }
}

export type PendingApproval = {
  id: number;
  incident_id: string;
  action: string;
  service: string | null;
  blast_radius: "low" | "medium" | "high";
  confidence: number;
  confidence_rationale: string | null;
  pattern_key: string | null;
  evidence: { edges?: { from: string; to: string }[] } | null;
  created_at: string;
  escalated_at: string | null;
};

export type DecisionResult = {
  decision: { status: string };
  execution: { executed: boolean; result: Record<string, unknown> | null; error: string | null } | null;
};

export class ControlApiError extends Error {}

export async function fetchPendingApprovals(): Promise<PendingApproval[]> {
  try {
    const res = await fetchWithTimeout(`${CONTROL_API_URL}/approvals/pending`);
    if (!res.ok) throw new ControlApiError(`control API returned ${res.status}`);
    return await res.json();
  } catch (e) {
    console.error("fetchPendingApprovals failed:", e);
    return [];
  }
}

export async function decideApproval(
  incidentId: string,
  decision: "approve" | "deny",
  decidedBy: string,
  reason?: string
): Promise<DecisionResult> {
  const res = await fetchWithTimeout(`/api/approvals/${incidentId}/decide`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision, decided_by: decidedBy, reason }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ControlApiError(body.detail || `request failed (${res.status})`);
  }
  return res.json();
}

export async function fetchControlApiHealth(): Promise<{ status: string; auth_configured: boolean } | null> {
  try {
    const res = await fetchWithTimeout(`${CONTROL_API_URL}/health`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export type SystemMetrics = {
  configured: boolean;
  total_incidents?: number;
  total_cost_usd?: number;
  cost_by_provider?: Record<string, number>;
  pending_approvals?: number;
  approved_count?: number;
  denied_count?: number;
  benchmark_success_rate?: number | null;
  benchmark_runs_count?: number;
  trust_patterns?: {
    pattern_key: string;
    positive_events: number;
    negative_events: number;
    total_events: number;
    trust_score: number;
  }[];
  recent_notifications?: {
    id: number;
    incident_id: string;
    action: string;
    message: string;
    created_at: string;
  }[];
  latency_percentiles_ms?: { p50: number; p95: number; p99: number; sample_size: number };
  latency_percentiles_ms_by_provider?: Record<string, { p50: number; p95: number; p99: number; sample_size: number }>;
  cost_per_incident_percentiles?: { p50: number; p95: number; p99: number; sample_size: number };
  // Agent segment
  resolution_rate?: number | null;
  recovery_rate?: number | null;
  human_requests_total?: number;
  human_requests_per_incident?: number | null;
  cost_per_completed_task?: number | null;
  // Tools segment
  tool_selection_accuracy?: number | null;
  tool_failure_rate?: number | null;
  fail_closed_count?: number;
  fail_open_count?: number;
  total_tool_calls?: number;
};

export async function fetchSystemMetrics(): Promise<SystemMetrics> {
  try {
    const res = await fetchWithTimeout(`${CONTROL_API_URL}/metrics/summary`);
    if (!res.ok) throw new ControlApiError(`control API returned ${res.status}`);
    return await res.json();
  } catch (e) {
    console.error("fetchSystemMetrics failed:", e);
    return { configured: false };
  }
}

export type WipeDataResult = {
  configured: boolean;
  tables: Record<string, { deleted: number; error: string | null }>;
};

export async function wipeAllData(confirmation: string): Promise<WipeDataResult> {
  const res = await fetchWithTimeout("/api/admin/wipe-data", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmation }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new ControlApiError(data.detail || `request failed (${res.status})`);
  }
  return data;
}