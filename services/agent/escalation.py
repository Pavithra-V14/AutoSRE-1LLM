from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from dotenv import load_dotenv

from services.agent.http_utils import post_with_retry
from services.agent.logging_utils import log_stage
from services.agent.persistence import get_stale_pending_approvals, get_system_metrics, mark_approval_escalated

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
ESCALATION_THRESHOLD_MINUTES = int(os.getenv("ESCALATION_THRESHOLD_MINUTES", "15"))
LATENCY_P99_ALERT_THRESHOLD_MS = int(os.getenv("LATENCY_P99_ALERT_THRESHOLD_MS", "30000"))
LATENCY_ALERT_COOLDOWN_MINUTES = int(os.getenv("LATENCY_ALERT_COOLDOWN_MINUTES", "60"))

_last_latency_alert_at: Optional[datetime] = None


def _minutes_pending(created_at: str) -> int:
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return 0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return int((now - created).total_seconds() // 60)


def _post_escalation(row: dict) -> bool:
    minutes = _minutes_pending(row.get("created_at", ""))
    text = (
        f"🚨 *ESCALATION — approval overdue* — `{row['incident_id']}`\n"
        f"Pending for *{minutes} minutes* (threshold: {ESCALATION_THRESHOLD_MINUTES}min). "
        f"Action: `{row['action']}`  Blast radius: `{row['blast_radius']}`\n"
        f"This incident's investigation is stalled waiting on a human decision. "
        f"Respond with:\n"
        f"`python scripts/handle_slack_response.py --incident {row['incident_id']} "
        f"--decision approve --by <your name>` or `--decision deny`"
    )

    if not SLACK_WEBHOOK_URL:
        log_stage("escalation", row["incident_id"], "SKIP", reason="SLACK_WEBHOOK_URL not configured")
        return False

    try:
        post_with_retry(lambda: httpx.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=10.0))
        return True
    except httpx.HTTPError as e:
        log_stage("escalation", row["incident_id"], "FAIL", error=str(e)[:100], note="retries exhausted")
        return False


def check_and_escalate(threshold_minutes: int = ESCALATION_THRESHOLD_MINUTES) -> int:
    """Call this periodically (see scripts/escalation_monitor.py). Returns
    the number of approvals escalated on this pass."""
    stale = get_stale_pending_approvals(threshold_minutes)
    escalated = 0

    for row in stale:
        sent = _post_escalation(row)
        mark_approval_escalated(row["id"])
        log_stage(
            "escalation", row["incident_id"], "WARN" if sent else "SKIP",
            minutes_pending=_minutes_pending(row.get("created_at", "")),
            action=row.get("action"),
        )
        escalated += 1

    return escalated


def check_p99_latency_alert(threshold_ms: int = LATENCY_P99_ALERT_THRESHOLD_MS) -> bool:
    """Call this periodically alongside check_and_escalate(). Returns
    True if an alert was actually sent this pass."""
    global _last_latency_alert_at

    metrics = get_system_metrics()
    if not metrics.get("configured"):
        return False

    p99 = metrics.get("latency_percentiles_ms", {}).get("p99", 0)
    sample_size = metrics.get("latency_percentiles_ms", {}).get("sample_size", 0)

    if sample_size == 0 or p99 < threshold_ms:
        return False

    now = datetime.now(timezone.utc)
    if _last_latency_alert_at is not None:
        minutes_since_last = (now - _last_latency_alert_at).total_seconds() / 60
        if minutes_since_last < LATENCY_ALERT_COOLDOWN_MINUTES:
            log_stage(
                "latency_alert", "-", "SKIP",
                reason=f"cooldown active, {int(LATENCY_ALERT_COOLDOWN_MINUTES - minutes_since_last)}min remaining",
                p99=p99,
            )
            return False

    text = (
        f"⚠️ *P99 LATENCY ALERT*\n"
        f"p99 LLM call latency is *{p99}ms*, above the {threshold_ms}ms threshold "
        f"(sample size: {sample_size} calls).\n"
        f"Check the /metrics dashboard's per-provider breakdown to see which provider "
        f"is driving this before assuming it's a system-wide issue."
    )

    if not SLACK_WEBHOOK_URL:
        log_stage("latency_alert", "-", "SKIP", reason="SLACK_WEBHOOK_URL not configured", p99=p99)
        return False

    try:
        post_with_retry(lambda: httpx.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=10.0))
        _last_latency_alert_at = now
        log_stage("latency_alert", "-", "WARN", p99=p99, threshold=threshold_ms)
        return True
    except httpx.HTTPError as e:
        log_stage("latency_alert", "-", "FAIL", error=str(e)[:100], note="retries exhausted")
        return False