from __future__ import annotations

from fakes import observation, plan
from services.agent.tools import ToolCallResult, ToolRole
from services.agent.confidence import (
    LOW_CONFIDENCE_THRESHOLD,
    REQUIRED_DEGRADED_MAX_CONFIDENCE,
    apply_confidence_guardrails,
    unavailable_signals,
)


def _graph(confidence=0.9, rationale="because"):
    return {"edges": [], "confidence": confidence, "confidence_rationale": rationale}


def test_cap_is_strictly_below_the_human_review_threshold():
    assert REQUIRED_DEGRADED_MAX_CONFIDENCE < LOW_CONFIDENCE_THRESHOLD


def test_required_tool_unreachable_fails_closed():
    steps = [observation("get_metrics"), observation("get_deploy_history", "unreachable", "required")]
    graph = apply_confidence_guardrails(_graph(0.92), steps)

    assert graph["confidence"] == REQUIRED_DEGRADED_MAX_CONFIDENCE
    assert graph["confidence_capped_from"] == 0.92
    assert "Cannot confirm" in graph["confidence_rationale"]
    assert "get_deploy_history (unreachable)" in graph["confidence_rationale"]
    assert graph["confidence_rationale"].startswith("because")  # model's own rationale is kept


def test_preferred_tool_unreachable_fails_open_but_discloses():
    steps = [observation("get_metrics"), observation("get_logs", "empty", "preferred")]
    graph = apply_confidence_guardrails(_graph(0.85), steps)

    assert graph["confidence"] == 0.85
    assert "confidence_capped_from" not in graph
    assert "PREFERRED" in graph["confidence_rationale"] and "get_logs" in graph["confidence_rationale"]


def test_already_low_confidence_is_not_raised():
    steps = [observation("get_metrics", "unreachable", "required")]
    assert apply_confidence_guardrails(_graph(0.2), steps)["confidence"] == 0.2


def test_tool_that_failed_then_succeeded_is_not_unavailable():
    steps = [observation("get_logs", "unreachable"), observation("get_logs", "ok")]
    assert unavailable_signals(steps) == []


def test_duplicate_and_invalid_observations_are_not_degradation():
    steps = [
        observation("get_logs", "ok"),
        observation("get_logs", "duplicate_call_skipped"),
        observation("nonexistent_tool", "invalid_plan"),
    ]
    assert unavailable_signals(steps) == []


def test_mutating_tool_denial_is_not_an_evidence_gap():
    assert unavailable_signals([observation("rollback_deploy", "denied")]) == []


def test_plan_steps_are_ignored():
    assert unavailable_signals([plan("get_logs")]) == []


def test_no_gaps_leaves_graph_untouched():
    graph = _graph(0.9)
    assert apply_confidence_guardrails(graph, [observation("get_metrics")]) == _graph(0.9)


def test_real_tool_results_use_enum_roles_and_still_fail_closed():
    """Regression: act_step stores ToolCallResult.model_dump(), where `role`
    is a ToolRole enum (str(ToolRole.REQUIRED) == 'ToolRole.REQUIRED'). A
    live run classed a REQUIRED gap as PREFERRED and never capped."""
    degraded = ToolCallResult(tool="get_deploy_history", role=ToolRole.REQUIRED, status="empty", data=None,
                              disclosure="REQUIRED tool 'get_deploy_history' is empty")
    steps = [{"type": "observation", "content": degraded.model_dump()}]

    signals = unavailable_signals(steps)
    assert [(s.tool, s.role) for s in signals] == [("get_deploy_history", "required")]

    graph = apply_confidence_guardrails(_graph(0.85), steps)
    assert graph["confidence"] == REQUIRED_DEGRADED_MAX_CONFIDENCE
    assert "Cannot confirm" in graph["confidence_rationale"]


def test_cap_records_why_it_fired_so_logs_do_not_blame_the_wrong_cause():
    unavailable = [observation("get_deploy_history", "unreachable", "required")]
    injected = [{"type": "observation", "content": {**observation("get_logs")["content"], "injection_suspected": True}}]

    assert apply_confidence_guardrails(_graph(0.9), unavailable)["confidence_capped_reasons"] == ["required_tool_unavailable"]
    assert apply_confidence_guardrails(_graph(0.9), injected)["confidence_capped_reasons"] == ["prompt_injection_suspected"]
    both = apply_confidence_guardrails(_graph(0.9), unavailable + injected)
    assert both["confidence_capped_reasons"] == ["required_tool_unavailable", "prompt_injection_suspected"]
    assert both["confidence_capped_from"] == 0.9  # the model's own value survives both guards


def test_injection_flag_alone_caps_confidence_and_discloses():
    steps = [{"type": "observation", "content": {**observation("get_logs")["content"], "injection_suspected": True}}]
    graph = apply_confidence_guardrails(_graph(0.95), steps)
    assert graph["confidence"] == REQUIRED_DEGRADED_MAX_CONFIDENCE
    assert "prompt injection" in graph["confidence_rationale"]
