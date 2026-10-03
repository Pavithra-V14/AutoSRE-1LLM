"""The agent's recommended remediation, and the rules that withhold it.

Until now the approval request inferred its "action" from whichever mutating
tool the planner happened to call, which an investigation-only agent never
does -- so the human reviewer was shown "investigation result (low
confidence)" instead of what the agent actually thought should be done. The
synthesis now states a recommendation explicitly:

    {"action": "none" | "restart_service" | "rollback_deploy",
     "service": "<service or null>", "rationale": "..."}

A recommendation is advice routed through the existing approval flow, never
an execution. Guardrails withhold it (action -> "none", reason recorded)
whenever acting on it would rest on shaky ground: see apply_action_guardrails.
"""

from __future__ import annotations

from typing import Any, Optional

from services.agent.schemas import ACTIONS
from services.agent.vocabulary import root_trigger

# A rollback only makes sense if the root cause is a change that can be
# rolled back.
ROLLBACKABLE_KINDS = ("deploy", "config", "flag")

NO_ACTION = {"action": "none", "service": None, "rationale": ""}


def normalize_action(graph: dict[str, Any]) -> dict[str, Any]:
    """Fills a missing recommendation with 'none' (models that ignore the new
    field must not fail validation) and tidies spellings. Never invents an
    action."""
    raw = graph.get("recommended_action")
    if not isinstance(raw, dict):
        graph["recommended_action"] = dict(NO_ACTION)
        return graph

    action = str(raw.get("action", "none") or "none").strip().lower().replace("-", "_").replace(" ", "_")
    service = raw.get("service")
    graph["recommended_action"] = {
        "action": action,
        "service": str(service).strip() if isinstance(service, str) and service.strip() else None,
        "rationale": str(raw.get("rationale", "") or ""),
    }
    return graph


def validate_action(graph: dict[str, Any]) -> Optional[str]:
    action = graph.get("recommended_action") or {}
    if action.get("action") not in ACTIONS:
        return f'recommended_action.action must be one of {ACTIONS}, got {action.get("action")!r}'
    if action["action"] != "none" and not action.get("service"):
        return f'recommended_action.service is required when action is {action["action"]!r}'
    return None


def _withhold(graph: dict[str, Any], reason: str) -> None:
    action = graph["recommended_action"]
    action["original_action"] = action["action"]
    action["withheld_reason"] = reason
    action["action"] = "none"


def apply_action_guardrails(graph: dict[str, Any], steps_taken: list[dict[str, Any]]) -> dict[str, Any]:
    """Run AFTER the confidence guardrails. Withholds a mutating
    recommendation when:

      - the confidence guard capped this run (a required signal was missing
        or tool output looked like prompt injection) -- fail closed: do not
        propose changing production on evidence we could not confirm;
      - rollback_deploy is proposed but the root cause is not a change that
        can be rolled back, or no deploy history was ever read.

    A withheld recommendation keeps `original_action` and the reason so the
    reviewer can see what the agent wanted and why it was held back."""
    graph.setdefault("recommended_action", dict(NO_ACTION))
    action = graph["recommended_action"]
    if action["action"] == "none":
        return graph

    if graph.get("confidence_capped_reasons"):
        _withhold(graph, "confidence was capped (" + ", ".join(graph["confidence_capped_reasons"]) + ")")
        return graph

    if action["action"] == "rollback_deploy":
        root = root_trigger(graph)
        root_kind = root.partition(":")[0] if root else None
        if root_kind not in ROLLBACKABLE_KINDS:
            _withhold(graph, f"root cause ({root or 'none established'}) is not a change that can be rolled back")
            return graph
        read_deploys = any(
            s["type"] == "observation" and s["content"].get("tool") == "get_deploy_history"
            and s["content"].get("status") == "ok"
            for s in steps_taken
        )
        if not read_deploys:
            _withhold(graph, "no deploy history was read, so the change being rolled back is unverified")
    return graph

