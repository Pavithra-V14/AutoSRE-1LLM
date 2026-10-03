from __future__ import annotations

import pytest

from fakes import FakeRouter, llm
from services.agent.judge import CausalGraphJudge, RUBRIC
from services.agent.structured import StructuredOutputError

GRAPH = {
    "edges": [{"id": "e1", "from": "config_change", "to": "pool_exhaustion", "evidence_tool_call_id": "1"},
              {"id": "e2", "from": "pool_exhaustion", "to": "high_latency", "evidence_tool_call_id": "3"}],
    "confidence": 0.9,
    "model_rationale": "Pool size was cut and latency followed.",
    "confidence_rationale": "Pool size was cut and latency followed. | Possible cascading correlation: incident X on other-service",
}


def _judge(*responses):
    router = FakeRouter(list(responses))
    return CausalGraphJudge(router=router), router


def test_correct_verdict_is_a_success():
    judge, router = _judge(llm('{"verdict": "correct", "score": 0.95, "rationale": "same cause"}'))
    verdict = judge.evaluate(ground_truth="deploy:cfg-4471 -> pool_size=10 -> latency_spike", graph=GRAPH)
    assert verdict.success and verdict.score == 0.95 and verdict.verdict == "correct"
    assert router.calls[0]["json_mode"] is True


@pytest.mark.parametrize("verdict", ["partial", "incorrect"])
def test_non_correct_verdicts_are_not_a_success(verdict):
    judge, _ = _judge(llm(f'{{"verdict": "{verdict}", "score": 0.4, "rationale": "r"}}'))
    assert not judge.evaluate(ground_truth="gt", graph=GRAPH).success


def test_prompt_contains_ground_truth_and_edges_but_not_system_appended_notes():
    judge, router = _judge(llm('{"verdict": "correct", "score": 1, "rationale": "r"}'))
    judge.evaluate(ground_truth="THE-GROUND-TRUTH", graph=GRAPH)
    prompt = router.calls[0]["prompt"]

    assert "THE-GROUND-TRUTH" in prompt
    assert "config_change -> pool_exhaustion" in prompt
    assert "Pool size was cut and latency followed." in prompt
    assert "Possible cascading correlation" not in prompt  # about OTHER incidents; must not be graded


def test_falls_back_to_confidence_rationale_for_graphs_without_a_snapshot():
    graph = {**GRAPH}
    graph.pop("model_rationale")
    judge, router = _judge(llm('{"verdict": "correct", "score": 1, "rationale": "r"}'))
    judge.evaluate(ground_truth="gt", graph=graph)
    assert "Pool size was cut" in router.calls[0]["prompt"]


def test_empty_graph_is_rendered_explicitly():
    judge, router = _judge(llm('{"verdict": "incorrect", "score": 0, "rationale": "nothing"}'))
    judge.evaluate(ground_truth="gt", graph={"edges": [], "confidence": 0.1})
    assert "(no causal edges)" in router.calls[0]["prompt"]


def test_invalid_verdict_is_reasked_then_accepted():
    judge, router = _judge(
        llm('{"verdict": "mostly", "score": 0.5, "rationale": "r"}'),
        llm('{"verdict": "partial", "score": 0.5, "rationale": "r"}'),
    )
    assert judge.evaluate(ground_truth="gt", graph=GRAPH).verdict == "partial"
    assert '"verdict" must be one of' in router.calls[1]["prompt"]


@pytest.mark.parametrize("bad", ['{"verdict": "correct", "score": 7, "rationale": "r"}',
                                 '{"verdict": "correct", "score": 0.9, "rationale": ""}',
                                 '{"verdict": "correct", "score": true, "rationale": "r"}'])
def test_out_of_contract_verdicts_are_rejected(bad):
    judge, _ = _judge(llm(bad), llm(bad))
    with pytest.raises(StructuredOutputError):
        judge.evaluate(ground_truth="gt", graph=GRAPH)


def test_preferred_provider_is_forwarded():
    router = FakeRouter([llm('{"verdict": "correct", "score": 1, "rationale": "r"}')])
    CausalGraphJudge(router=router, prefer="gemini").evaluate(ground_truth="gt", graph=GRAPH)
    assert router.calls[0]["prefer"] == "gemini"


def test_rubric_states_that_omitted_hops_and_terminal_symptoms_are_not_fatal():
    assert "MAY omit or merge intermediate hops" in RUBRIC


def test_judge_grades_at_the_configured_temperature_zero():
    judge, router = _judge(llm('{"verdict": "correct", "score": 1, "rationale": "r"}'))
    judge.evaluate(ground_truth="gt", graph=GRAPH)
    assert router.calls[0]["temperature"] == 0.0
