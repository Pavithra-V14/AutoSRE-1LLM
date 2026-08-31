from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.agent.approval import execute_approved_action, record_decision
from services.agent.escalation import check_and_escalate
from services.agent.persistence import get_pending_approvals, get_system_metrics, wipe_all_data

CONTROL_API_KEY = os.getenv("CONTROL_API_KEY")

if not CONTROL_API_KEY:
    print(
        "\n*** WARNING: CONTROL_API_KEY is not set -- the control API is running "
        "UNAUTHENTICATED. Anyone who can reach this service can approve/deny "
        "incidents. Fine for local dev; set CONTROL_API_KEY before deploying "
        "anywhere reachable. ***\n"
    )

app = FastAPI(title="Genesis Kit Control API")
DASHBOARD_ORIGIN = os.getenv("DASHBOARD_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[DASHBOARD_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """FastAPI dependency, applied only to mutating endpoints. A no-op if
    CONTROL_API_KEY isn't configured (see the startup warning above)."""
    if not CONTROL_API_KEY:
        return
    if x_api_key != CONTROL_API_KEY:
        raise HTTPException(401, "missing or invalid X-API-Key header")


class DecisionRequest(BaseModel):
    decision: str  # "approve" | "deny"
    decided_by: str
    reason: Optional[str] = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "auth_configured": bool(CONTROL_API_KEY)}


@app.get("/approvals/pending")
def list_pending_approvals(incident_id: Optional[str] = None) -> list[dict[str, Any]]:
    return get_pending_approvals(incident_id)


@app.post("/approvals/{incident_id}/decide", dependencies=[Depends(require_api_key)])
def decide_approval(incident_id: str, req: DecisionRequest) -> dict[str, Any]:
    """The endpoint the dashboard's Approve/Deny buttons actually call
    (via the Next.js server-side proxy, not directly from the browser).
    Mirrors tools/handle_slack_response.py exactly (both now call the
    same agent.approval functions) -- the CLI and the browser are two
    front ends for the identical decision path, not two separate
    implementations that could drift apart."""
    if req.decision not in ("approve", "deny"):
        raise HTTPException(400, "decision must be 'approve' or 'deny'")

    pending = get_pending_approvals(incident_id)
    if not pending:
        raise HTTPException(404, f"no pending approval found for incident {incident_id}")

    action = pending[0]["action"]
    service = pending[0].get("service")

    result = record_decision(incident_id, req.decision, req.decided_by, reason=req.reason)
    if not result:
        raise HTTPException(500, "failed to record decision (Supabase write failed)")

    execution: Optional[dict[str, Any]] = None
    if req.decision == "approve":
        execution = execute_approved_action(incident_id, action, service)

    return {"decision": result, "execution": execution}


@app.get("/metrics/summary")
def metrics_summary() -> dict[str, Any]:
    """Backs the dashboard's system-wide metrics page."""
    return get_system_metrics()


@app.post("/escalations/check", dependencies=[Depends(require_api_key)])
def trigger_escalation_check() -> dict[str, int]:
    """Manual trigger for the escalation check (the dashboard can call
    this on load, in addition to tools/escalation_monitor.py running on
    its own schedule) -- belt-and-suspenders so a stale approval doesn't
    depend on the background monitor being up."""
    n = check_and_escalate()
    return {"escalated": n}


class WipeDataRequest(BaseModel):
    confirmation: str


@app.post("/admin/wipe-data", dependencies=[Depends(require_api_key)])
def admin_wipe_data(req: WipeDataRequest) -> dict[str, Any]:
    """Deletes EVERY row in EVERY table this project writes to --
    traces, approval_queue, pattern_trust_events, notifications,
    active_incidents, benchmark_runs. Irreversible. Gated by both
    CONTROL_API_KEY (via require_api_key) and this explicit confirmation
    phrase, checked here rather than trusting the frontend's own
    confirmation UI -- a UI-only safeguard can be bypassed by anyone
    calling this endpoint directly."""
    if req.confirmation != "DELETE ALL DATA":
        raise HTTPException(
            400,
            "confirmation must be exactly 'DELETE ALL DATA' (case-sensitive) -- nothing was deleted",
        )
    result = wipe_all_data()
    return result

