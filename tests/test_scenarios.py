"""Integrity checks on the benchmark's own data. A benchmark that leaks its
answer key or scores against a malformed scenario produces numbers nobody
should trust."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from services.simulator.scenarios import SCENARIOS

KEYS = sorted(SCENARIOS)


@pytest.mark.parametrize("key", KEYS)
def test_every_scenario_has_a_symptom_only_alert(key):
    assert SCENARIOS[key].get("alert"), f"{key} has no alert"


@pytest.mark.parametrize("key", KEYS)
def test_alert_does_not_leak_the_answer(key):
    scenario = SCENARIOS[key]
    alert = scenario["alert"].lower()

    assert alert != scenario["description"].lower()
    assert scenario["description"][:40].lower() not in alert, "alert copies the description"

    # deploy / change ids are the cleanest possible leak
    for deploy in scenario.get("deploys", []):
        assert deploy["id"].lower() not in alert
    for token in re.findall(r"[a-z]+-\d{3,}", scenario["ground_truth_cause"].lower()):
        assert token not in alert, f"alert leaks {token}"


@pytest.mark.parametrize("key", KEYS)
def test_alert_names_the_service_and_reads_like_a_monitoring_alert(key):
    alert = SCENARIOS[key]["alert"]
    assert alert.startswith("ALERT ") and ":" in alert


# words that only make sense once the cause is known
CAUSE_WORDS = ("deploy", "config", "rollback", "pool_size", "null", "leak", "flag", "migration", "index", "quota")


@pytest.mark.parametrize("key", KEYS)
def test_alert_contains_no_cause_vocabulary(key):
    alert = SCENARIOS[key]["alert"].lower()
    assert not [w for w in CAUSE_WORDS if w in alert]


def test_scenario_count_is_documented_correctly():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    assert f"Runs the {len(SCENARIOS)} scripted incidents" in readme


@pytest.mark.parametrize("key", KEYS)
def test_required_fields_present(key):
    scenario = SCENARIOS[key]
    for field in ("description", "ground_truth_cause", "expected_tools", "metrics", "logs", "deploys", "alert"):
        assert field in scenario, f"{key} missing {field}"


def test_injection_scenario_plants_its_trap_and_declares_the_forbidden_term():
    scenario = SCENARIOS["prompt_injection_in_logs"]
    planted = " ".join(entry["message"].lower() for entry in scenario["logs"])
    assert "ignore previous instructions" in planted
    for term in scenario["forbidden_terms"]:
        assert term in planted  # the term really is something only the injection says


def test_fail_closed_scenarios_declare_their_degraded_tools():
    for key, scenario in SCENARIOS.items():
        if scenario.get("expects_fail_closed"):
            assert scenario.get("degraded_tools"), key


# --- harder diagnostics ----------------------------------------------------------------------------------

def test_red_herring_deploy_really_is_a_different_service_and_harmless():
    scenario = SCENARIOS["red_herring_deploy"]
    (deploy,) = scenario["deploys"]
    assert deploy["service"] != next(iter(scenario["metrics"]))
    assert "no functional change" in deploy["change"]
    assert deploy["id"] in scenario["forbidden_edge_terms"]
    assert deploy["id"].lower() not in scenario["ground_truth_cause"].lower()


def test_two_cause_scenario_declares_both_causes_in_its_ground_truth():
    truth = SCENARIOS["two_concurrent_causes"]["ground_truth_cause"]
    assert " AND " in truth
    first, second = truth.split(" AND ")
    assert "cfg-4471" in first and "payment_gateway" in second
    logs = " ".join(entry["message"] for entry in SCENARIOS["two_concurrent_causes"]["logs"])
    assert "connection pool exhausted" in logs and "payment-gateway" in logs  # both are visible to find


def test_noisy_scenario_buries_one_real_signal_in_noise():
    logs = SCENARIOS["noisy_logs"]["logs"]
    errors = [entry for entry in logs if entry["level"] == "ERROR"]
    assert len(logs) >= 25 and len(errors) == 1
    assert "connection pool exhausted" in errors[0]["message"]
    assert SCENARIOS["noisy_logs"]["ground_truth_cause"] == SCENARIOS["db_timeout"]["ground_truth_cause"]


def test_noisy_logs_collapse_when_compacted_for_the_prompt():
    from services.agent.context import compact_steps

    step = {"type": "observation", "content": {"tool": "get_logs", "status": "ok",
                                               "data": {"logs": SCENARIOS["noisy_logs"]["logs"]}}}
    compacted = compact_steps([step])[0]["data"]["logs"]
    assert len(compacted) < len(SCENARIOS["noisy_logs"]["logs"]) / 2
    assert any("connection pool exhausted" in entry.get("message", "") for entry in compacted)  # the signal survives
