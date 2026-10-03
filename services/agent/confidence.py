"""Deterministic confidence guardrails.

The LLM proposes a confidence; these rules decide what the agent is allowed
to claim. Fail-closed behaviour (docs/architecture.md section 5) must not
depend on the model remembering to be humble: if a REQUIRED investigative
tool never returned data, the final confidence is capped and the gap is
disclosed in the rationale regardless of what the model wrote.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from services.agent.tools import MUTATING_TOOLS

# Below this the incident is routed to a human (needs_approval). Shared by
# synthesize_step and the benchmark's negative-scenario scoring so the two
# can never drift apart.
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.6"))

# Ceiling applied when a REQUIRED tool is unavailable. Must stay strictly
# below LOW_CONFIDENCE_THRESHOLD so a fail-closed run always reaches a human.
REQUIRED_DEGRADED_MAX_CONFIDENCE = min(
    float(os.getenv("REQUIRED_DEGRADED_MAX_CONFIDENCE", "0.5")),
    LOW_CONFIDENCE_THRESHOLD - 0.01,
)

UNAVAILABLE_STATUSES = frozenset({"unreachable", "empty", "denied"})


@dataclass(frozen=True)
class UnavailableSignal:
    tool: str
    role: str
    status: str


def unavailable_signals(steps_taken: list[dict[str, Any]]) -> list[UnavailableSignal]:
    """Read-only tools that never returned usable data during this
    investigation. A tool that failed once but later succeeded (or whose
    only failures were duplicates/invalid plans) is NOT unavailable -- the
    evidence is in hand."""
    succeeded: set[str] = set()
    failed: dict[str, UnavailableSignal] = {}

    for step in steps_taken:
        if step.get("type") != "observation":
            continue
        content = step.get("content", {})
        tool = content.get("tool")
        if not tool or tool in MUTATING_TOOLS:
            continue
        status = content.get("status")
        if status == "ok":
            succeeded.add(tool)
        elif status in UNAVAILABLE_STATUSES:
            # In-memory observations come from ToolCallResult.model_dump(), so
            # role is a ToolRole enum there but a plain string once persisted.
            raw_role = content.get("role", "preferred")
            role = str(getattr(raw_role, "value", raw_role))
            previous = failed.get(tool)
            # keep the strictest role seen for this tool
            if previous is None or (role == "required" and previous.role != "required"):
                failed[tool] = UnavailableSignal(tool=tool, role=role, status=status)

    return [signal for tool, signal in failed.items() if tool not in succeeded]


def injection_flagged_tools(steps_taken: list[dict[str, Any]]) -> list[str]:
    """Tools whose output contained injection-shaped text (flagged by
    services/agent/untrusted.py when the observation was recorded)."""
    seen: list[str] = []
    for step in steps_taken:
        if step.get("type") == "observation" and step.get("content", {}).get("injection_suspected"):
            tool = str(step["content"].get("tool"))
            if tool not in seen:
                seen.append(tool)
    return seen


def _cap(graph: dict[str, Any], reason: str) -> None:
    graph.setdefault("confidence_capped_reasons", [])
    if reason not in graph["confidence_capped_reasons"]:
        graph["confidence_capped_reasons"].append(reason)
    original = float(graph.get("confidence", 0.0) or 0.0)
    if original > REQUIRED_DEGRADED_MAX_CONFIDENCE:
        # keep the FIRST uncapped value if several guards fire
        graph.setdefault("confidence_capped_from", original)
        graph["confidence"] = REQUIRED_DEGRADED_MAX_CONFIDENCE


def apply_confidence_guardrails(graph: dict[str, Any], steps_taken: list[dict[str, Any]]) -> dict[str, Any]:
    """Returns `graph` (mutated) with confidence capped and gaps disclosed.

    - REQUIRED signal unavailable  -> fail closed: cap confidence, state
      explicitly that the signal could not be confirmed.
    - PREFERRED signal unavailable -> fail open: confidence untouched, gap
      still disclosed so reviewers know what the conclusion did not see.
    - Tool output looked like a prompt injection -> fail closed: the
      evidence may have been written to steer the conclusion."""
    signals = unavailable_signals(steps_taken)
    injected = injection_flagged_tools(steps_taken)
    if not signals and not injected:
        return graph

    required = [s for s in signals if s.role == "required"]
    preferred = [s for s in signals if s.role != "required"]
    notes: list[str] = []

    if required:
        _cap(graph, "required_tool_unavailable")
        notes.append(
            "Cannot confirm: REQUIRED signal(s) unavailable -- "
            + ", ".join(f"{s.tool} ({s.status})" for s in required)
            + "; conclusions depending on them are unverified."
        )
    if injected:
        _cap(graph, "prompt_injection_suspected")
        notes.append(
            "Possible prompt injection in tool output ("
            + ", ".join(injected)
            + "); that text was treated as data, but conclusions resting on it are unverified."
        )
    if preferred:
        notes.append(
            "Continuing without PREFERRED signal(s): "
            + ", ".join(f"{s.tool} ({s.status})" for s in preferred)
            + "."
        )

    existing = str(graph.get("confidence_rationale", "") or "").strip()
    graph["confidence_rationale"] = " | ".join(filter(None, [existing, *notes]))
    return graph
