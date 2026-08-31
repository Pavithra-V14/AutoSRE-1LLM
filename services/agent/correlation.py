from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from services.agent.persistence import get_recent_incident_services

CORRELATION_WINDOW_MINUTES = 15
SERVICE_DEPENDENCIES: dict[str, list[str]] = {
    "checkout-service": ["order-service", "inventory-service", "pricing-client"],
    "order-service": ["inventory-service"],
    "product-catalog-service": ["product-db"],
    "recommendation-service": [],
    "search-service": [],
    "inventory-service": [],
    "orders-db": [],
    "pricing-client": [],
}


def _reverse_dependencies() -> dict[str, list[str]]:
    """If A depends on B, then a problem in A is also worth surfacing when
    investigating B -- e.g. investigating inventory-service should also
    surface a recent checkout-service incident, since checkout depends on
    inventory. Built once from SERVICE_DEPENDENCIES rather than
    hand-maintained twice."""
    reverse: dict[str, list[str]] = {svc: [] for svc in SERVICE_DEPENDENCIES}
    for service, deps in SERVICE_DEPENDENCIES.items():
        for dep in deps:
            reverse.setdefault(dep, []).append(service)
    return reverse


_REVERSE_DEPENDENCIES = _reverse_dependencies()


def extract_primary_service(steps_taken: list[dict[str, Any]]) -> Optional[str]:
    """Scans the investigation's tool observations for the first
    successful get_metrics call and takes its top-level data key as the
    service under investigation -- matches the shape simulator scenario
    metrics dicts are keyed in (see simulator/scenarios.py)."""
    for step in steps_taken:
        if step["type"] != "observation":
            continue
        content = step["content"]
        if content.get("tool") != "get_metrics" or content.get("status") != "ok":
            continue
        data = content.get("data") or {}
        if data:
            return next(iter(data.keys()))
    return None


@dataclass
class RelatedIncident:
    incident_id: str
    service: str
    relationship: str  


def find_related_incidents(
    this_service: str, this_incident_id: str, within_minutes: int = CORRELATION_WINDOW_MINUTES
) -> list[RelatedIncident]:
    """The actual correlation check. Returns any recent/active incidents
    on a service this one depends on, or that depends on this one."""
    if not this_service:
        return []

    deps = set(SERVICE_DEPENDENCIES.get(this_service, []))
    dependents = set(_REVERSE_DEPENDENCIES.get(this_service, []))

    recent = get_recent_incident_services(within_minutes)
    related: list[RelatedIncident] = []

    for incident_id, services in recent.items():
        if incident_id == this_incident_id:
            continue
        overlap_deps = services & deps
        overlap_dependents = services & dependents
        for svc in overlap_deps:
            related.append(RelatedIncident(incident_id, svc, "depends_on"))
        for svc in overlap_dependents:
            related.append(RelatedIncident(incident_id, svc, "depended_on_by"))

    return related


def format_correlation_note(related: list[RelatedIncident]) -> str:
    if not related:
        return ""
    parts = [
        f"incident {r.incident_id} on `{r.service}` ({'a service this depends on' if r.relationship == 'depends_on' else 'a service that depends on this'})"
        for r in related[:3]
    ]
    return "Possible cascading correlation: " + "; ".join(parts) + "."
