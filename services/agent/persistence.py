from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, TypeVar

from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TIMEOUT_SECONDS = float(os.getenv("SUPABASE_TIMEOUT_SECONDS", "10"))
CORRELATION_QUERY_LIMIT = int(os.getenv("CORRELATION_QUERY_LIMIT", "500"))

def _percentile(sorted_values: list[float], pct: float) -> float:
    """Standard linear-interpolation percentile (same method most
    observability tools use, e.g. Datadog/Prometheus 'linear
    interpolation'). Takes an already-sorted list -- callers sort once,
    not once per percentile requested."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)

T = TypeVar("T")

PROVIDER_RATES: dict[str, tuple[float, float]] = {
    "groq": (0.0, 0.0),        # free tier
    "gemini": (0.0, 0.0),      # free tier
    "openrouter": (0.0, 0.0),  # free-model routes
}
EQUIVALENT_PAID_RATES: dict[str, tuple[float, float]] = {
    "groq": (0.59, 0.79),        
    "gemini": (0.30, 2.50),     
    "openrouter": (0.59, 0.79),  
}

def compute_equivalent_cost_usd(provider: str, input_tokens: int, output_tokens: int) -> float:
    """For benchmark/CPSO reporting only -- NOT what you're actually billed."""
    in_rate, out_rate = EQUIVALENT_PAID_RATES.get(provider, (0.0, 0.0))
    return round(
        (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate, 6
    )

def _get_client():
    if not (SUPABASE_URL and SUPABASE_KEY):
        return None
    from supabase import ClientOptions, create_client

    try:
        options = ClientOptions(postgrest_client_timeout=SUPABASE_TIMEOUT_SECONDS)
        return create_client(SUPABASE_URL, SUPABASE_KEY, options=options)
    except TypeError:
        return create_client(SUPABASE_URL, SUPABASE_KEY)

_client = None

def _client_or_none():
    global _client
    if _client is None and SUPABASE_URL and SUPABASE_KEY:
        _client = _get_client()
    return _client


def _safe(label: str, call: Callable[[], T], default: T) -> T:
    """Runs a Supabase call, catching ANY exception (timeout, connection
    error, malformed response, whatever) and returning `default` instead
    of propagating. This is the actual fix for the crash: previously each
    function called client.table(...).execute() completely unguarded."""
    try:
        return call()
    except Exception as e:
        print(f"[persistence:warn] {label} failed ({type(e).__name__}: {e}) -- continuing without it")
        return default


def compute_cost_usd(provider: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = EQUIVALENT_PAID_RATES.get(provider, (0.0, 0.0))
    return round((input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate, 6)

_warned_missing_scoring_columns = False
_write_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="trace-writer")
_pending_writes: list[Future] = []


def flush_pending_writes(timeout_seconds: float = 15.0) -> None:
    """Waits for every trace write submitted so far to actually complete.
    Called once at the end of run_incident() -- durability is still
    guaranteed by the time the caller sees a result; the loop just isn't
    blocked on each individual write while the investigation is running."""
    global _pending_writes
    if not _pending_writes:
        return
    for future in _pending_writes:
        try:
            future.result(timeout=timeout_seconds)
        except Exception as e:
            print(f"[persistence:warn] a background trace write failed or timed out: {e}")
    _pending_writes = []


def insert_trace_step(
    incident_id: str,
    step_index: int,
    step_type: str,
    content: dict[str, Any],
    tool: Optional[str] = None,
    tool_role: Optional[str] = None,
    status: Optional[str] = None,
    provider: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: Optional[int] = None,
) -> None:
    cost_usd = compute_equivalent_cost_usd(provider or "", input_tokens, output_tokens)
    row = {
        "incident_id": incident_id,
        "step_index": step_index,
        "step_type": step_type,
        "tool": tool,
        "tool_role": tool_role,
        "status": status,
        "provider": provider,
        "content": content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
    }

    client = _client_or_none()
    if client is None:
        print(f"[persistence:noop] trace step not saved (no Supabase configured): {row}")
        return

    future = _write_executor.submit(_safe, "insert_trace_step", lambda: client.table("traces").insert(row).execute(), None)
    _pending_writes.append(future)


def insert_approval_request(
    incident_id: str,
    action: str,
    blast_radius: str,
    confidence: float,
    evidence: dict[str, Any],
    alternatives_considered: Optional[list[dict[str, Any]]] = None,
    confidence_rationale: Optional[str] = None,
    pattern_key: Optional[str] = None,
    service: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    row = {
        "incident_id": incident_id,
        "action": action,
        "blast_radius": blast_radius,
        "confidence": confidence,
        "confidence_rationale": confidence_rationale,
        "pattern_key": pattern_key,
        "service": service,
        "evidence": evidence,
        "alternatives_considered": alternatives_considered or [],
        "status": "pending",
    }

    client = _client_or_none()
    if client is None:
        print(f"[persistence:noop] approval request not saved: {row}")
        return None

    result = _safe("insert_approval_request", lambda: client.table("approval_queue").insert(row).execute(), None)
    return result.data[0] if result and result.data else None


def get_pending_approvals(incident_id: Optional[str] = None) -> list[dict[str, Any]]:
    client = _client_or_none()
    if client is None:
        print("[persistence:noop] no Supabase configured, cannot list pending approvals")
        return []

    def _call():
        query = client.table("approval_queue").select("*").eq("status", "pending")
        if incident_id:
            query = query.eq("incident_id", incident_id)
        return query.order("created_at", desc=True).execute()

    result = _safe("get_pending_approvals", _call, None)
    return (result.data or []) if result else []


def update_approval_decision(
    incident_id: str, decision: str, decided_by: str, reason: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Records a human's approve/deny decision against the most recent
    pending approval row for this incident. Item 6: `reason` (typically
    only given on a denial) is stored so future approval requests for the
    same pattern can show a human reviewer what happened last time --
    see get_denial_notes_for_pattern below."""
    if decision not in ("approved", "denied"):
        raise ValueError("decision must be 'approved' or 'denied'")

    client = _client_or_none()
    if client is None:
        print(
            f"[persistence:noop] decision not saved: incident={incident_id} "
            f"decision={decision} by={decided_by} reason={reason}"
        )
        return None

    pending = get_pending_approvals(incident_id)
    if not pending:
        print(f"[persistence] no pending approval found for incident {incident_id}")
        return None

    row_id = pending[0]["id"]
    update_fields = {
        "status": decision,
        "decided_by": decided_by,
        "decided_at": datetime.utcnow().isoformat(),
    }
    if reason:
        update_fields["denial_reason"] = reason

    result = _safe(
        "update_approval_decision",
        lambda: client.table("approval_queue").update(update_fields).eq("id", row_id).execute(),
        None,
    )
    return result.data[0] if result and result.data else None


def get_denial_notes_for_pattern(pattern_key: str, limit: int = 3) -> list[dict[str, Any]]:
    """Item 6: the actual feedback-loop read path. Returns the most recent
    denials (with their reasons, if given) for this pattern, so a new
    approval request for the same class of incident can show "last time,
    a human rejected this because X" instead of denials just being logged
    and forgotten."""
    client = _client_or_none()
    if client is None:
        return []

    def _call():
        return (
            client.table("approval_queue")
            .select("incident_id, decided_by, decided_at, denial_reason")
            .eq("pattern_key", pattern_key)
            .eq("status", "denied")
            .order("decided_at", desc=True)
            .limit(limit)
            .execute()
        )

    result = _safe("get_denial_notes_for_pattern", _call, None)
    return (result.data or []) if result else []


def insert_benchmark_run(
    run_id: str,
    scenario: str,
    success: bool,
    ground_truth_match: bool,
    total_cost_usd: float,
    cpso: float,
    steps_taken: int,
    tool_selection_f1: Optional[float] = None,
    needed_approval: Optional[bool] = None,
    had_degradation: Optional[bool] = None,
    recovered: Optional[bool] = None,
    score_method: Optional[str] = None,
    score: Optional[float] = None,
    score_rationale: Optional[str] = None,
    run_index: Optional[int] = None,
) -> None:
    """The four trailing fields (why a run passed or failed) need the columns
    added by infra/db/migrations/002_benchmark_scoring.sql. Until that
    migration is applied the insert is retried without them, so an
    out-of-date database never loses the pass/fail record itself."""
    row = {
        "run_id": run_id,
        "scenario": scenario,
        "success": success,
        "ground_truth_match": ground_truth_match,
        "total_cost_usd": total_cost_usd,
        "cpso": cpso,
        "steps_taken": steps_taken,
        "tool_selection_f1": tool_selection_f1,
        "needed_approval": needed_approval,
        "had_degradation": had_degradation,
        "recovered": recovered,
    }

    scoring_columns = {
        "score_method": score_method,
        "score": score,
        "score_rationale": (score_rationale or None) and score_rationale[:1000],
        "run_index": run_index,
    }
    client = _client_or_none()
    if client is None:
        print(f"[persistence:noop] benchmark run not saved: {row | scoring_columns}")
        return

    def _insert() -> None:
        try:
            client.table("benchmark_runs").insert({**row, **scoring_columns}).execute()
        except Exception as e:
            text = str(e).lower()
            if "column" not in text or not any(name in text for name in scoring_columns):
                raise
            global _warned_missing_scoring_columns
            if not _warned_missing_scoring_columns:
                _warned_missing_scoring_columns = True
                print(
                    "[persistence:warn] benchmark_runs has no scoring columns -- apply "
                    "infra/db/migrations/002_benchmark_scoring.sql to keep the judge's verdict; "
                    "saving pass/fail only"
                )
            client.table("benchmark_runs").insert(row).execute()

    _safe("insert_benchmark_run", _insert, None)


def get_incident_cost_summary(incident_id: str) -> Optional[dict[str, Any]]:
    """Reads the incident_cost_summary view (see db/schema.sql) for one
    incident -- used by the CLI and, later, the dashboard."""
    client = _client_or_none()
    if client is None:
        return None

    result = _safe(
        "get_incident_cost_summary",
        lambda: client.table("incident_cost_summary").select("*").eq("incident_id", incident_id).execute(),
        None,
    )
    return (result.data[0] if result.data else None) if result else None

def record_trust_event(pattern_key: str, event_type: str, incident_id: Optional[str] = None) -> None:
    row = {"pattern_key": pattern_key, "event_type": event_type, "incident_id": incident_id}

    client = _client_or_none()
    if client is None:
        print(f"[persistence:noop] trust event not saved: {row}")
        return

    _safe("record_trust_event", lambda: client.table("pattern_trust_events").insert(row).execute(), None)


def get_pattern_trust_stats(pattern_key: str) -> dict[str, Any]:
    """Aggregates trust events for a pattern client-side. Fine at this
    project's scale (a few hundred events per pattern at most); a
    high-volume production version would use the pattern_trust_summary
    SQL view (see db/schema.sql) or a server-side RPC instead of pulling
    every row over the wire."""
    default = {"total": 0, "positive": 0, "negative": 0}
    client = _client_or_none()
    if client is None:
        return default

    result = _safe(
        "get_pattern_trust_stats",
        lambda: client.table("pattern_trust_events").select("event_type").eq("pattern_key", pattern_key).execute(),
        None,
    )
    if not result:
        return default

    events = result.data or []
    positive = sum(
        1 for e in events if e["event_type"] in ("approved", "benchmark_success", "auto_approved")
    )
    negative = sum(1 for e in events if e["event_type"] in ("denied", "benchmark_fail"))
    return {"total": len(events), "positive": positive, "negative": negative}


def get_pattern_key_for_incident(incident_id: str) -> Optional[str]:
    """Looks up the pattern_key stored on an incident's approval_queue row
    -- used when a human decision comes in later and needs to be recorded
    against the right pattern (approval.record_decision doesn't otherwise
    know the pattern_key at that point, only the incident_id)."""
    pending_or_recent = get_pending_approvals(incident_id)
    if pending_or_recent:
        return pending_or_recent[0].get("pattern_key")

    client = _client_or_none()
    if client is None:
        return None

    result = _safe(
        "get_pattern_key_for_incident",
        lambda: (
            client.table("approval_queue")
            .select("pattern_key")
            .eq("incident_id", incident_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ),
        None,
    )
    return result.data[0]["pattern_key"] if result and result.data else None

def get_recent_incident_services(within_minutes: int) -> dict[str, set[str]]:
    """Returns {incident_id: {service_names}} for every incident with a
    successful get_metrics observation in the last `within_minutes` --
    agent/correlation.py uses this to check whether a related service has
    an active/recent incident before finalizing a causal graph.

    Bounded to CORRELATION_QUERY_LIMIT rows -- this query previously had
    no limit at all, and gets called on every single synthesize_step, so
    it would scan the entire traces table for every incident once that
    table grows. within_minutes already bounds it by time; the limit is a
    second, defense-in-depth bound in case a busy period produces more
    get_metrics calls than expected within that window."""
    client = _client_or_none()
    if client is None:
        return {}

    cutoff = (datetime.utcnow() - timedelta(minutes=within_minutes)).isoformat()
    result = _safe(
        "get_recent_incident_services",
        lambda: (
            client.table("traces")
            .select("incident_id, content")
            .eq("tool", "get_metrics")
            .eq("status", "ok")
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(CORRELATION_QUERY_LIMIT)
            .execute()
        ),
        None,
    )
    if not result:
        return {}

    services_by_incident: dict[str, set[str]] = {}
    for row in result.data or []:
        data = (row.get("content") or {}).get("data") or {}
        if not data:
            continue
        services_by_incident.setdefault(row["incident_id"], set()).update(data.keys())

    return services_by_incident

def find_active_incident(fingerprint: str, window_seconds: int) -> Optional[str]:
    """Returns the incident_id of a still-active duplicate alert, if one
    exists within window_seconds. Cross-process fallback for
    agent/dedup.py's in-memory fast path."""
    client = _client_or_none()
    if client is None:
        return None

    cutoff = (datetime.utcnow() - timedelta(seconds=window_seconds)).isoformat()
    result = _safe(
        "find_active_incident",
        lambda: (
            client.table("active_incidents")
            .select("incident_id, created_at")
            .eq("fingerprint", fingerprint)
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ),
        None,
    )
    return result.data[0]["incident_id"] if result and result.data else None


def register_active_incident(fingerprint: str, incident_id: str, ttl_seconds: int) -> None:
    row = {"fingerprint": fingerprint, "incident_id": incident_id}

    client = _client_or_none()
    if client is None:
        print(f"[persistence:noop] active incident not registered: {row}")
        return

    _safe("register_active_incident", lambda: client.table("active_incidents").insert(row).execute(), None)


def insert_notification(incident_id: str, action: str, message: str) -> None:
    row = {"incident_id": incident_id, "action": action, "message": message}

    client = _client_or_none()
    if client is None:
        print(f"[persistence:noop] notification not saved: {row}")
        return

    _safe("insert_notification", lambda: client.table("notifications").insert(row).execute(), None)

def get_stale_pending_approvals(threshold_minutes: int) -> list[dict[str, Any]]:
    """Pending approvals older than threshold_minutes that haven't already
    been escalated -- what agent/escalation.py polls."""
    client = _client_or_none()
    if client is None:
        return []

    cutoff = (datetime.utcnow() - timedelta(minutes=threshold_minutes)).isoformat()
    result = _safe(
        "get_stale_pending_approvals",
        lambda: (
            client.table("approval_queue")
            .select("*")
            .eq("status", "pending")
            .is_("escalated_at", "null")
            .lt("created_at", cutoff)
            .execute()
        ),
        None,
    )
    return (result.data or []) if result else []


def mark_approval_escalated(approval_id: int) -> None:
    client = _client_or_none()
    if client is None:
        return
    _safe(
        "mark_approval_escalated",
        lambda: (
            client.table("approval_queue")
            .update({"escalated_at": datetime.utcnow().isoformat()})
            .eq("id", approval_id)
            .execute()
        ),
        None,
    )

def get_system_metrics() -> dict[str, Any]:
    """One aggregate read backing the dashboard's system metrics page.
    Deliberately a handful of separate small queries rather than one
    giant join -- easier to reason about, and any single piece failing
    (see _safe()) degrades that one number to a default instead of
    blanking the whole page."""
    client = _client_or_none()
    if client is None:
        return {"configured": False}

    def _count(table: str, **filters: str) -> int:
        result = _safe(
            f"get_system_metrics:{table}",
            lambda: _apply_filters(client.table(table).select("id", count="exact"), filters).execute(),
            None,
        )
        return result.count if result and result.count is not None else 0

    def _apply_filters(query, filters: dict[str, str]):
        for k, v in filters.items():
            query = query.eq(k, v)
        return query

    total_incidents_result = _safe(
        "get_system_metrics:distinct_incidents",
        lambda: client.table("traces").select("incident_id").execute(),
        None,
    )
    distinct_incidents = (
        len({row["incident_id"] for row in (total_incidents_result.data or [])})
        if total_incidents_result
        else 0
    )

    cost_result = _safe(
        "get_system_metrics:cost",
        lambda: client.table("traces").select("cost_usd, provider").execute(),
        None,
    )
    cost_rows = (cost_result.data or []) if cost_result else []
    total_cost = round(sum(r.get("cost_usd") or 0 for r in cost_rows), 6)
    cost_by_provider: dict[str, float] = {}
    for r in cost_rows:
        provider = r.get("provider") or "unknown"
        cost_by_provider[provider] = round(cost_by_provider.get(provider, 0) + (r.get("cost_usd") or 0), 6)

    latency_result = _safe(
        "get_system_metrics:latency",
        lambda: client.table("traces").select("latency_ms, provider").not_.is_("latency_ms", "null").limit(5000).execute(),
        None,
    )
    latency_rows = (latency_result.data or []) if latency_result else []
    latencies = sorted(r["latency_ms"] for r in latency_rows if r.get("latency_ms") is not None)

    latencies_by_provider: dict[str, list[float]] = {}
    for r in latency_rows:
        provider = r.get("provider") or "unknown"
        if r.get("latency_ms") is not None:
            latencies_by_provider.setdefault(provider, []).append(r["latency_ms"])
    latency_percentiles_by_provider = {
        provider: {
            "p50": round(_percentile(sorted(vals), 50), 1),
            "p95": round(_percentile(sorted(vals), 95), 1),
            "p99": round(_percentile(sorted(vals), 99), 1),
            "sample_size": len(vals),
        }
        for provider, vals in latencies_by_provider.items()
    }

    per_incident_cost: dict[str, float] = {}
    incident_cost_result = _safe(
        "get_system_metrics:incident_cost",
        lambda: client.table("traces").select("incident_id, cost_usd").limit(5000).execute(),
        None,
    )
    for r in (incident_cost_result.data or []) if incident_cost_result else []:
        per_incident_cost[r["incident_id"]] = per_incident_cost.get(r["incident_id"], 0.0) + (r.get("cost_usd") or 0.0)
    incident_costs = sorted(per_incident_cost.values())

    latency_percentiles = {
        "p50": round(_percentile(latencies, 50), 1),
        "p95": round(_percentile(latencies, 95), 1),
        "p99": round(_percentile(latencies, 99), 1),
        "sample_size": len(latencies),
    }
    cost_percentiles = {
        "p50": round(_percentile(incident_costs, 50), 6),
        "p95": round(_percentile(incident_costs, 95), 6),
        "p99": round(_percentile(incident_costs, 99), 6),
        "sample_size": len(incident_costs),
    }

    benchmark_result = _safe(
        "get_system_metrics:benchmark",
        lambda: client.table("benchmark_runs").select("success, cpso, scenario").execute(),
        None,
    )
    benchmark_rows = (benchmark_result.data or []) if benchmark_result else []
    benchmark_success_rate = (
        round(sum(1 for r in benchmark_rows if r.get("success")) / len(benchmark_rows), 3)
        if benchmark_rows
        else None
    )

    trust_result = _safe(
        "get_system_metrics:trust",
        lambda: client.table("pattern_trust_summary").select("*").execute(),
        None,
    )
    trust_rows = (trust_result.data or []) if trust_result else []

    notifications_result = _safe(
        "get_system_metrics:notifications",
        lambda: (
            client.table("notifications")
            .select("*")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        ),
        None,
    )
    recent_notifications = (notifications_result.data or []) if notifications_result else []

    plan_result = _safe(
        "get_system_metrics:plan_steps",
        lambda: client.table("traces").select("id").eq("step_type", "plan").execute(),
        None,
    )
    total_plan_attempts = len(plan_result.data or []) if plan_result else 0

    obs_result = _safe(
        "get_system_metrics:observations",
        lambda: client.table("traces").select("incident_id, status, content").eq("step_type", "observation").execute(),
        None,
    )
    obs_rows = (obs_result.data or []) if obs_result else []

    BAD_PLAN_STATUSES = {"invalid_plan", "duplicate_call_skipped"}
    REAL_TOOL_STATUSES = {"ok", "empty", "unreachable", "denied"}
    DEGRADED_STATUSES = {"unreachable", "empty", "denied"}

    bad_plan_attempts = sum(1 for r in obs_rows if r.get("status") in BAD_PLAN_STATUSES)
    real_tool_calls = [r for r in obs_rows if r.get("status") in REAL_TOOL_STATUSES]
    degraded_calls = [r for r in real_tool_calls if r.get("status") in DEGRADED_STATUSES]
    tool_selection_accuracy = (
        round((total_plan_attempts - bad_plan_attempts) / total_plan_attempts, 3)
        if total_plan_attempts > 0 else None
    )
    tool_failure_rate = (
        round(len(degraded_calls) / len(real_tool_calls), 3) if real_tool_calls else None
    )

    fail_closed_count = 0
    fail_open_count = 0
    incidents_with_degradation: set[str] = set()
    for r in degraded_calls:
        disclosure = ((r.get("content") or {}).get("disclosure") or "").lower()
        if "cannot confirm" in disclosure:
            fail_closed_count += 1
        elif "continuing without" in disclosure:
            fail_open_count += 1
        incidents_with_degradation.add(r["incident_id"])

    synth_result = _safe(
        "get_system_metrics:synthesis",
        lambda: client.table("traces").select("incident_id, content").eq("step_type", "synthesis").execute(),
        None,
    )
    synth_rows = (synth_result.data or []) if synth_result else []

    incidents_resolved: set[str] = set()
    for r in synth_rows:
        edges = (r.get("content") or {}).get("edges") or []
        if edges:
            incidents_resolved.add(r["incident_id"])

    resolution_rate = (
        round(len(incidents_resolved) / distinct_incidents, 3) if distinct_incidents else None
    )

    recovered = incidents_with_degradation & incidents_resolved
    recovery_rate = (
        round(len(recovered) / len(incidents_with_degradation), 3)
        if incidents_with_degradation else None
    )
    
    cost_per_completed_task = (
        round(total_cost / len(incidents_resolved), 6) if incidents_resolved else None
    )

    pending_count = _count("approval_queue", status="pending")
    approved_count = _count("approval_queue", status="approved")
    denied_count = _count("approval_queue", status="denied")
    human_requests_total = pending_count + approved_count + denied_count
    human_requests_per_incident = (
        round(human_requests_total / distinct_incidents, 3) if distinct_incidents else None
    )

    return {
        "configured": True,
        "total_incidents": distinct_incidents,
        "total_cost_usd": total_cost,
        "cost_by_provider": cost_by_provider,
        "pending_approvals": pending_count,
        "approved_count": approved_count,
        "denied_count": denied_count,
        "benchmark_success_rate": benchmark_success_rate,
        "benchmark_runs_count": len(benchmark_rows),
        "trust_patterns": trust_rows,
        "recent_notifications": recent_notifications,
        # Agent segment
        "resolution_rate": resolution_rate,
        "recovery_rate": recovery_rate,
        "human_requests_total": human_requests_total,
        "human_requests_per_incident": human_requests_per_incident,
        "cost_per_completed_task": cost_per_completed_task,
        # Tools segment
        "tool_selection_accuracy": tool_selection_accuracy,
        "tool_failure_rate": tool_failure_rate,
        "fail_closed_count": fail_closed_count,
        "fail_open_count": fail_open_count,
        "total_tool_calls": len(real_tool_calls),
        # Latency/cost percentiles (system segment)
        "latency_percentiles_ms": latency_percentiles,
        "latency_percentiles_ms_by_provider": latency_percentiles_by_provider,
        "cost_per_incident_percentiles": cost_percentiles,
    }

ALL_TABLES = [
    "traces",
    "approval_queue",
    "pattern_trust_events",
    "notifications",
    "active_incidents",
    "benchmark_runs",
]


def wipe_all_data() -> dict[str, Any]:
    """Deletes every row from every table this project writes to.
    Genuinely destructive and irreversible -- see services/api/main.py's
    /admin/wipe-data endpoint for the auth gate, and the dashboard's
    typed-confirmation UI. Returns a per-table count of rows deleted (or
    an error string) so the caller can show exactly what happened,
    including partial failures -- one table's delete failing shouldn't
    hide that the others succeeded."""
    client = _client_or_none()
    if client is None:
        return {"configured": False, "tables": {}}

    results: dict[str, Any] = {}
    for table in ALL_TABLES:
        result = _safe(
            f"wipe_all_data:{table}",
            lambda t=table: client.table(t).delete().gt("id", 0).execute(),
            None,
        )
        if result is None:
            results[table] = {"deleted": 0, "error": "delete failed, see server logs"}
        else:
            results[table] = {"deleted": len(result.data or []), "error": None}

    return {"configured": True, "tables": results}