from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Optional

from services.agent.persistence import find_active_incident, register_active_incident

DEDUP_WINDOW_SECONDS = 5 * 60  
_local_cache: dict[str, tuple[str, float]] = {}  


def compute_alert_fingerprint(alert_text: str) -> str:
    normalized = re.sub(r"\s+", " ", alert_text.strip().lower())
    normalized = re.sub(r"\d+(\.\d+)?", "#", normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass
class DedupResult:
    is_duplicate: bool
    existing_incident_id: Optional[str] = None
    fingerprint: str = ""


def check_and_register(alert_text: str, incident_id: str) -> DedupResult:
    """Call once per incoming alert, before starting an investigation.
    If a matching alert is already active within DEDUP_WINDOW_SECONDS,
    returns that incident_id instead of registering a new one -- the
    caller (agent/graph.py run_incident) should short-circuit rather than
    running a second full investigation."""
    fingerprint = compute_alert_fingerprint(alert_text)
    now = time.time()

    cached = _local_cache.get(fingerprint)
    if cached and cached[1] > now:
        return DedupResult(is_duplicate=True, existing_incident_id=cached[0], fingerprint=fingerprint)

    existing = find_active_incident(fingerprint, DEDUP_WINDOW_SECONDS)
    if existing:
        _local_cache[fingerprint] = (existing, now + DEDUP_WINDOW_SECONDS)
        return DedupResult(is_duplicate=True, existing_incident_id=existing, fingerprint=fingerprint)

    _local_cache[fingerprint] = (incident_id, now + DEDUP_WINDOW_SECONDS)
    register_active_incident(fingerprint, incident_id, DEDUP_WINDOW_SECONDS)
    return DedupResult(is_duplicate=False, fingerprint=fingerprint)


def clear_cache() -> None:
    """Used by tests/benchmark runs so fresh scenario runs don't get
    spuriously deduped against a previous test's fingerprint cache."""
    _local_cache.clear()
