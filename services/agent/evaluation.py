from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Tool-side degradation only. "invalid_plan" and "duplicate_call_skipped" are
# planner behaviour, not a degraded dependency; counting them made healthy
# runs look like they had recovered from a failure and skewed recovery_rate.
DEGRADATION_STATUSES = {"unreachable", "empty", "denied"}

MUTATING_TOOLS = {"restart_service", "rollback_deploy"}


def tool_selection_accuracy(steps_taken: list[dict[str, Any]], expected_tools: list[str]) -> dict[str, float]:
    """Compares tools actually called (read-only, investigative calls
    only -- remediation actions are excluded, this measures investigation
    thoroughness, not remediation choice) against the scenario's
    expected_tools set.

    Returns recall (did it call what it needed?), precision (were its
    calls actually useful, not wasted on irrelevant/duplicate tools?),
    and F1 (the single number for the aggregate report)."""
    expected = set(expected_tools)
    called = {
        s["content"].get("tool")
        for s in steps_taken
        if s["type"] == "plan" and s["content"].get("tool") not in MUTATING_TOOLS
    }
    called.discard(None)

    if not expected:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0}
    if not called:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    overlap = expected & called
    recall = len(overlap) / len(expected)
    precision = len(overlap) / len(called)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"recall": round(recall, 3), "precision": round(precision, 3), "f1": round(f1, 3)}


def had_degradation_event(steps_taken: list[dict[str, Any]]) -> bool:
    """Did this run hit at least one tool/provider degradation --
    something a 'clean' run wouldn't have encountered?"""
    for s in steps_taken:
        if s["type"] == "observation" and s["content"].get("status") in DEGRADATION_STATUSES:
            return True
    return False


def reached_valid_conclusion(result: dict[str, Any]) -> bool:
    """Did the run finish with an actual causal graph, as opposed to
    aborting (no LLM providers available, truncated/malformed output,
    max steps with nothing to show)? A deliberately low-confidence
    conclusion on a negative-case scenario (flaky_test_false_alarm etc.)
    still counts as valid -- 'no strong cause found' is a legitimate
    reached conclusion, not a failure to reach one."""
    hypothesis = result.get("hypothesis") or ""
    if hypothesis.startswith("ABORTED") or hypothesis.startswith("DEDUPLICATED"):
        return False
    causal_graph = result.get("causal_graph") or {}
    # synthesize_step marks graphs it had to fabricate on failure with
    # "aborted": True; an empty-but-honest low-confidence graph has no marker.
    return bool(causal_graph) and not causal_graph.get("aborted")


@dataclass
class RunEvaluation:
    scenario: str
    tool_selection_f1: float
    tool_selection_recall: float
    needed_approval: bool
    had_degradation: bool
    recovered: bool 
    reached_conclusion: bool
    cost_usd: float


def evaluate_run(scenario_key: str, expected_tools: list[str], result: dict[str, Any], cost_usd: float) -> RunEvaluation:
    steps_taken = result.get("steps_taken", [])
    accuracy = tool_selection_accuracy(steps_taken, expected_tools)
    degraded = had_degradation_event(steps_taken)
    concluded = reached_valid_conclusion(result)

    return RunEvaluation(
        scenario=scenario_key,
        tool_selection_f1=accuracy["f1"],
        tool_selection_recall=accuracy["recall"],
        needed_approval=bool(result.get("needs_approval")),
        had_degradation=degraded,
        recovered=(concluded if degraded else True),
        reached_conclusion=concluded,
        cost_usd=cost_usd,
    )


def aggregate_metrics(evaluations: list[RunEvaluation], success_flags: list[bool]) -> dict[str, Any]:
    """The five numbers for the end-of-benchmark report. success_flags is
    the existing ground-truth-match success list from run_benchmark.py,
    kept separate from RunEvaluation since that's a different (already
    existing) computation this module doesn't duplicate."""
    n = len(evaluations)
    if n == 0:
        return {"n": 0}

    degraded_runs = [e for e in evaluations if e.had_degradation]
    completed_runs = [e for e in evaluations if e.reached_conclusion]

    return {
        "n": n,
        "agent_success_rate": round(sum(success_flags) / n, 3) if success_flags else None,
        "tool_selection_accuracy_f1": round(sum(e.tool_selection_f1 for e in evaluations) / n, 3),
        "tool_selection_accuracy_recall": round(sum(e.tool_selection_recall for e in evaluations) / n, 3),
        "human_requests_total": sum(1 for e in evaluations if e.needed_approval),
        "human_request_rate": round(sum(1 for e in evaluations if e.needed_approval) / n, 3),
        "runs_with_degradation": len(degraded_runs),
        "recovery_rate": (
            round(sum(1 for e in degraded_runs if e.recovered) / len(degraded_runs), 3)
            if degraded_runs
            else None
        ),
        "cost_per_task_completion_usd": (
            round(sum(e.cost_usd for e in completed_runs) / len(completed_runs), 6)
            if completed_runs
            else None
        ),
    }
