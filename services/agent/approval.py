from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from services.agent.http_utils import post_with_retry
from services.agent.logging_utils import log_stage
from services.agent.persistence import (
    insert_approval_request,
    insert_notification,
    update_approval_decision,
)
from services.agent.tokens import TokenIssuanceError, get_or_issue_token

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
SIMULATOR_BASE_URL = os.getenv("SIMULATOR_BASE_URL", "http://localhost:8100")


def _format_slack_message(
    incident_id: str,
    action: str,
    blast_radius: str,
    confidence: float,
    evidence: dict[str, Any],
    alternatives_considered: list[dict[str, Any]],
    confidence_rationale: str = "",
) -> dict[str, Any]:
    edges = evidence.get("edges", [])
    causal_summary = (
        " -> ".join(f"{e.get('from', '?')} => {e.get('to', '?')}" for e in edges)
        if edges
        else "(no causal chain established)"
    )

    alt_summary = "\n".join(
        f"  - considered: `{a.get('tool', '?')}` ({a.get('role', '?')})"
        for a in alternatives_considered[:5]  # keep the message readable
    ) or "  (none logged)"

    rationale_line = f"*Why this confidence:* {confidence_rationale}\n" if confidence_rationale else ""

    text = (
        f"*Incident approval needed* — `{incident_id}`\n"
        f"*Action:* `{action}`  *Blast radius:* `{blast_radius}`  "
        f"*Confidence:* {confidence:.0%}\n"
        f"{rationale_line}"
        f"*Causal hypothesis:* {causal_summary}\n"
        f"*Alternatives investigated:*\n{alt_summary}\n\n"
        f"Respond with:\n"
        f"`python scripts/handle_slack_response.py --incident {incident_id} "
        f"--decision approve --by <your name>`\n"
        f"or `--decision deny`"
    )
    return {"text": text}


def post_approval_request(
    incident_id: str,
    action: str,
    blast_radius: str,
    confidence: float,
    evidence: dict[str, Any],
    alternatives_considered: Optional[list[dict[str, Any]]] = None,
    confidence_rationale: str = "",
    pattern_key: Optional[str] = None,
    service: Optional[str] = None,
) -> bool:
    """Posts to Slack (if configured) and always logs the approval row.
    Returns True if the Slack post succeeded (or was skipped because
    Slack isn't configured -- the DB row is the source of truth either way)."""
    alternatives_considered = alternatives_considered or []

    insert_approval_request(
        incident_id=incident_id,
        action=action,
        blast_radius=blast_radius,
        confidence=confidence,
        evidence=evidence,
        alternatives_considered=alternatives_considered,
        confidence_rationale=confidence_rationale,
        pattern_key=pattern_key,
        service=service,
    )

    if not SLACK_WEBHOOK_URL:
        log_stage("slack_post", incident_id, "SKIP", reason="SLACK_WEBHOOK_URL not configured", saved_to="db only")
        return True

    payload = _format_slack_message(
        incident_id, action, blast_radius, confidence, evidence,
        alternatives_considered, confidence_rationale,
    )
    try:
        post_with_retry(lambda: httpx.post(SLACK_WEBHOOK_URL, json=payload, timeout=10.0))
        log_stage("slack_post", incident_id, "OK", type="approval_request")
        return True
    except httpx.HTTPError as e:
        log_stage("slack_post", incident_id, "FAIL", type="approval_request", error=str(e)[:80], note="retries exhausted")
        return False


def post_notification(incident_id: str, action: str, message: str) -> bool:
    """Item 4: a non-blocking notification for an action that executed
    without requiring approval. Always logged to the notifications table
    for audit; posts to Slack if configured, prefixed to be visually
    distinct from a blocking approval request."""
    insert_notification(incident_id=incident_id, action=action, message=message)

    if not SLACK_WEBHOOK_URL:
        log_stage("slack_post", incident_id, "SKIP", reason="SLACK_WEBHOOK_URL not configured", type="notification")
        return True

    payload = {"text": f"ℹ️ *Auto-executed, no approval required* — `{incident_id}`\n{message}"}
    try:
        post_with_retry(lambda: httpx.post(SLACK_WEBHOOK_URL, json=payload, timeout=10.0))
        log_stage("slack_post", incident_id, "OK", type="notification")
        return True
    except httpx.HTTPError as e:
        log_stage("slack_post", incident_id, "FAIL", type="notification", error=str(e)[:80], note="retries exhausted")
        return False


def record_decision(
    incident_id: str, decision: str, decided_by: str, reason: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """decision must be 'approve' or 'deny' (mapped to approved/denied).
    `reason` (Item 6) is worth providing on a denial -- it's what a future
    approval request for the same pattern will show a reviewer.

    Item 1: also feeds the outcome back into per-pattern trust tracking --
    a human's decision here is exactly the "evidence accumulates" signal
    that widens (or, on a denial, permanently narrows) future autonomy for
    this pattern. Imported inline to avoid a circular import (agent.trust
    imports from agent.persistence, which this module also imports from)."""
    mapped = {"approve": "approved", "deny": "denied"}.get(decision, decision)
    result = update_approval_decision(incident_id, mapped, decided_by, reason=reason)

    from services.agent.persistence import get_pattern_key_for_incident
    from services.agent.trust import record_approval_outcome

    pattern_key = (result or {}).get("pattern_key") or get_pattern_key_for_incident(incident_id)
    if pattern_key:
        record_approval_outcome(pattern_key, incident_id, approved=(mapped == "approved"))

    return result


def execute_approved_action(incident_id: str, action: str, service: Optional[str] = None) -> dict[str, Any]:
    """Item 2: mint an approver token scoped to exactly this incident and
    this one action, then actually call the simulator to execute it. A
    token minted here literally cannot be reused for a different incident
    or after its 5-minute TTL -- see simulator/tokens.py ROLE_DEFAULT_TTL_SECONDS.

    Returns a dict the caller can turn into a CLI message or an API
    response: {"executed": bool, "result": {...} | None, "error": str | None}."""
    if action not in ("rollback_deploy",):
        return {"executed": False, "result": None, "error": f"'{action}' is not an executable simulator tool"}

    try:
        token = get_or_issue_token(incident_id, "approver")
    except TokenIssuanceError as e:
        return {"executed": False, "result": None, "error": f"could not obtain approver token: {e}"}

    payload: dict[str, Any] = {"incident_id": incident_id}
    if service:
        payload["service"] = service

    try:
        r = httpx.post(
            f"{SIMULATOR_BASE_URL}/tools/{action}",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        r.raise_for_status()
        result = r.json()
        insert_notification(
            incident_id=incident_id,
            action=action,
            message=f"`{action}` executed by human-approved approver token. Result: {result}",
        )
        return {"executed": True, "result": result, "error": None}
    except httpx.HTTPError as e:
        return {"executed": False, "result": None, "error": f"execution failed: {e}"}
