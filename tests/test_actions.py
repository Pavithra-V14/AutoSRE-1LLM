from __future__ import annotations

import pytest

from fakes import observation, plan
from services.agent.actions import apply_action_guardrails, normalize_action, validate_action


def _graph(action="rollback_deploy", service="checkout-service", root="deploy:cfg-4471", **extra):
    return {
        "edges": [{"id": "e1", "from": root, "to": "mechanism:pool_exhaustion", "evidence_tool_call_id": "1"}],
        "confidence": 0.9,
        "recommended_action": {"action": action, "service": service, "rationale": "r"},
        **extra,
    }


STEPS = [plan("get_deploy_history"), observation("get_deploy_history")]


# --- normalize / validate -----------------------------------------------------------------------------

def test_missing_recommendation_becomes_none_instead_of_failing():
    graph = normalize_action({"edges": []})
    assert graph["recommended_action"]["action"] == "none"
    assert validate_action(graph) is None


@pytest.mark.parametrize("raw, expected", [("Rollback Deploy", "rollback_deploy"), ("restart-service", "restart_service"),
                                           ("NONE", "none")])
def test_spellings_are_normalized(raw, expected):
    graph = normalize_action({"recommended_action": {"action": raw, "service": "x", "rationale": ""}})
    assert graph["recommended_action"]["action"] == expected


def test_unknown_action_is_rejected_not_guessed():
    graph = normalize_action({"recommended_action": {"action": "delete_database", "service": "x"}})
    assert "must be one of" in validate_action(graph)


def test_a_mutating_action_needs_a_service():
    graph = normalize_action({"recommended_action": {"action": "restart_service", "service": "  "}})
    assert "service is required" in validate_action(graph)


def test_non_dict_recommendation_is_replaced():
    assert normalize_action({"recommended_action": "rollback"})["recommended_action"]["action"] == "none"


# --- guardrails ----------------------------------------------------------------------------------------------

def test_a_supported_rollback_is_kept():
    graph = apply_action_guardrails(_graph(), STEPS)
    assert graph["recommended_action"]["action"] == "rollback_deploy"
    assert "withheld_reason" not in graph["recommended_action"]


@pytest.mark.parametrize("reason", ["required_tool_unavailable", "prompt_injection_suspected"])
def test_nothing_is_recommended_when_confidence_was_capped(reason):
    """Fail closed: do not propose changing production on unconfirmed evidence."""
    graph = apply_action_guardrails(_graph(confidence_capped_reasons=[reason]), STEPS)
    action = graph["recommended_action"]
    assert action["action"] == "none" and action["original_action"] == "rollback_deploy"
    assert reason in action["withheld_reason"]


@pytest.mark.parametrize("root", ["traffic:flash_sale", "dependency:payment_gateway", "maintenance:reindex", "unknown:undetermined"])
def test_rollback_is_withheld_when_the_root_cause_is_not_a_change(root):
    graph = apply_action_guardrails(_graph(root=root), STEPS)
    assert graph["recommended_action"]["action"] == "none"
    assert "not a change that can be rolled back" in graph["recommended_action"]["withheld_reason"]


@pytest.mark.parametrize("root", ["deploy:rel-1", "config:cfg-1", "flag:dynamic_pricing"])
def test_rollback_is_allowed_for_every_rollbackable_kind(root):
    assert apply_action_guardrails(_graph(root=root), STEPS)["recommended_action"]["action"] == "rollback_deploy"


def test_rollback_without_reading_deploy_history_is_withheld():
    graph = apply_action_guardrails(_graph(), [plan("get_metrics"), observation("get_metrics")])
    assert "no deploy history was read" in graph["recommended_action"]["withheld_reason"]


def test_restart_is_not_subject_to_the_rollback_rules():
    graph = apply_action_guardrails(_graph(action="restart_service", root="traffic:spike"), [])
    assert graph["recommended_action"]["action"] == "restart_service"


def test_none_is_left_alone():
    graph = apply_action_guardrails(_graph(action="none", confidence_capped_reasons=["x"]), STEPS)
    assert graph["recommended_action"] == {"action": "none", "service": "checkout-service", "rationale": "r"}
