from __future__ import annotations

from services.agent.evidence import weak_citations


def _obs(tool, data, status="ok"):
    return {"type": "observation", "content": {"tool": tool, "status": status, "data": data}}


STEPS = [
    {"type": "plan", "content": {"tool": "get_deploy_history"}},
    _obs("get_deploy_history", {"deploys": [{"id": "cfg-4471", "change": "db.pool_size: 50 -> 10"}]}),   # 1
    {"type": "plan", "content": {"tool": "get_logs"}},
    _obs("get_logs", {"logs": [{"level": "ERROR", "message": "connection pool exhausted, waiting"}]}),   # 3
    {"type": "plan", "content": {"tool": "get_metrics"}},
    _obs("get_metrics", {"checkout-service": {"p99_latency_ms": [1, 2, 3]}}),                           # 5
]


def _graph(*edges):
    return {"edges": [{"id": f"e{i}", "from": a, "to": b, "evidence_tool_call_id": c} for i, (a, b, c) in enumerate(edges)]}


def test_supported_edges_are_not_flagged():
    graph = _graph(("deploy:cfg-4471", "mechanism:pool_exhaustion", "1"),   # id appears in the deploy list; 'pool' too
                   ("mechanism:pool_exhaustion", "symptom:latency_spike", "3"))
    assert weak_citations(graph, STEPS) == []


def test_edge_citing_unrelated_evidence_is_flagged():
    graph = _graph(("dependency:payment_gateway", "mechanism:upstream_timeout", "1"))
    assert weak_citations(graph, STEPS) == ["e0"]


def test_synonym_stems_match():
    """'exhaustion' vs 'exhausted' share a stem; models use both."""
    assert weak_citations(_graph(("config:x_y", "mechanism:pool_exhaustion", "3")), STEPS) == []


def test_generic_only_nodes_are_not_judged():
    """latency/error/spike appear in everything, so they cannot show support."""
    assert weak_citations(_graph(("unknown:undetermined", "symptom:latency_spike", "1")), STEPS) == []


def test_unresolvable_citations_are_left_to_the_other_check():
    graph = _graph(("deploy:x", "mechanism:y", "observation-1"), ("deploy:x", "mechanism:y", "99"))
    assert weak_citations(graph, STEPS) == []


def test_citing_a_plan_step_is_not_evaluated_here():
    assert weak_citations(_graph(("deploy:rel-892", "mechanism:null_pointer", "0")), STEPS) == []


def test_empty_graph():
    assert weak_citations({"edges": []}, STEPS) == []
