from __future__ import annotations

import pytest

from services.agent.trust import compute_pattern_key
from services.agent.vocabulary import (
    normalize_graph,
    normalize_node,
    pattern_key,
    root_trigger,
    validate_graph_vocabulary,
)


def _graph(*pairs):
    return {"edges": [{"id": f"e{i}", "from": a, "to": b, "evidence_tool_call_id": "1"}
                      for i, (a, b) in enumerate(pairs)]}


# --- normalize ------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("deploy:rel-892", "deploy:rel-892"),
    ("Deploy:REL-892", "deploy:rel-892"),
    ("config_change:cfg 4471", "config:cfg_4471"),
    ("Feature Flag:dynamic pricing v2", "flag:dynamic_pricing_v2"),
    ("deployment:rel-1", "deploy:rel-1"),
    ("Mechanism:Pool  Exhaustion", "mechanism:pool_exhaustion"),
    ("external:payment gateway", "dependency:payment_gateway"),
])
def test_near_miss_nodes_are_normalized(raw, expected):
    assert normalize_node(raw) == expected


@pytest.mark.parametrize("raw", ["ConfigChange", "latency spike", "", ":x", "deploy:"])
def test_nodes_without_a_usable_kind_are_left_for_the_validator(raw):
    assert ":" not in normalize_node(raw) or normalize_node(raw) == raw.strip()
    assert validate_graph_vocabulary(_graph((raw, "symptom:x"))) is not None


def test_a_node_is_never_guessed_into_a_kind():
    assert normalize_node("pool_exhaustion") == "pool_exhaustion"


# --- validate -----------------------------------------------------------------------------------

def test_valid_graph_passes():
    graph = _graph(("deploy:cfg-4471", "mechanism:pool_exhaustion"),
                   ("mechanism:pool_exhaustion", "symptom:latency_spike"))
    assert validate_graph_vocabulary(graph) is None


def test_unknown_kind_is_rejected_with_the_allowed_list():
    error = validate_graph_vocabulary(_graph(("gremlin:x", "symptom:y")))
    assert error and "deploy" in error and "e0" in error


def test_a_trigger_cannot_be_an_effect():
    error = validate_graph_vocabulary(_graph(("symptom:latency_spike", "deploy:rel-1")))
    assert error and "cause -> effect" in error


def test_empty_graph_is_valid_vocabulary():
    assert validate_graph_vocabulary({"edges": []}) is None


def test_normalize_graph_mutates_every_edge():
    graph = normalize_graph(_graph(("Config_Change:A B", "Symptom:High Latency")))
    assert (graph["edges"][0]["from"], graph["edges"][0]["to"]) == ("config:a_b", "symptom:high_latency")


# --- pattern key ----------------------------------------------------------------------------------

def test_root_trigger_is_the_node_that_is_only_ever_a_cause():
    graph = _graph(("mechanism:m", "symptom:s"), ("deploy:rel-1", "mechanism:m"))
    assert root_trigger(graph) == "deploy:rel-1"


def test_pattern_key_ignores_which_deploy_it_was():
    """The point of the vocabulary: two different deploys causing the same
    failure are ONE pattern, so trust can accumulate samples."""
    a = _graph(("deploy:cfg-4471", "mechanism:pool_exhaustion"), ("mechanism:pool_exhaustion", "symptom:latency"))
    b = _graph(("deploy:cfg-9999", "mechanism:pool_exhaustion"))
    assert pattern_key(a) == pattern_key(b) == "deploy/pool_exhaustion"


def test_pattern_key_separates_different_trigger_kinds():
    assert pattern_key(_graph(("deploy:x", "mechanism:m"))) != pattern_key(_graph(("config:x", "mechanism:m")))


def test_pattern_key_is_none_for_legacy_free_text_graphs():
    assert pattern_key(_graph(("ConfigChange", "PoolExhaustion"))) is None


def test_compute_pattern_key_uses_vocabulary_then_falls_back_to_legacy():
    assert compute_pattern_key(_graph(("deploy:rel-1", "mechanism:oom_kill"))) == "deploy/oom_kill"
    assert compute_pattern_key(_graph(("ConfigChange", "Pool Exhaustion"))) == "pool_exhaustion"
    assert compute_pattern_key({"edges": []}) == "unclassified"
    assert compute_pattern_key(None) == "unclassified"
