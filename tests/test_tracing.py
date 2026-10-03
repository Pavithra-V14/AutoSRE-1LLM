from __future__ import annotations

import pytest

from fakes import FakeRouter, llm, observation, plan
from services.agent import graph as g
from services.agent import tracing
from services.agent.llm_router import LLMRouter


@pytest.fixture
def exporter():
    exp = tracing.configure_tracing("memory")
    yield exp
    tracing.configure_tracing("off")


def _names(exp):
    return [s.name for s in exp.get_finished_spans()]


def _by_name(exp, name):
    return next(s for s in exp.get_finished_spans() if s.name == name)


# --- off by default / safe -------------------------------------------------------------------------------

def test_tracing_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("AUTOSRE_TRACING", raising=False)
    assert tracing.configure_tracing() is None and tracing.enabled() is False


def test_spans_are_harmless_no_ops_when_off():
    tracing.configure_tracing("off")
    with tracing.span("x", a=1) as sp:
        sp.set(b=2)  # must not raise
    assert tracing.enabled() is False


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="AUTOSRE_TRACING"):
        tracing.configure_tracing("datadog")


def test_enabling_tracing_does_not_install_a_global_provider(exporter):
    """The guardrails libraries run with OTEL_SDK_DISABLED; our spans must not
    switch their telemetry on."""
    from opentelemetry import trace

    assert "autosre" not in str(type(trace.get_tracer_provider()).__name__).lower()
    with tracing.span("probe"):
        pass
    assert _names(exporter) == ["probe"]


# --- span behaviour ------------------------------------------------------------------------------------------

def test_attributes_are_primitives_and_none_is_skipped(exporter):
    with tracing.span("s", n=3, f=1.5, ok=True, text="t", skipped=None, obj=["a"]):
        pass
    attrs = dict(_by_name(exporter, "s").attributes)
    assert attrs == {"n": 3, "f": 1.5, "ok": True, "text": "t", "obj": "['a']"}


def test_nested_spans_form_a_tree(exporter):
    with tracing.span("parent"):
        with tracing.span("child"):
            pass
    parent, child = _by_name(exporter, "parent"), _by_name(exporter, "child")
    assert child.parent.span_id == parent.context.span_id
    assert child.context.trace_id == parent.context.trace_id


def test_an_exception_is_recorded_and_reraised_unchanged(exporter):
    with pytest.raises(RuntimeError, match="boom"):
        with tracing.span("failing"):
            raise RuntimeError("boom")
    span = _by_name(exporter, "failing")
    assert span.status.status_code.name == "ERROR"
    assert any(event.name == "exception" for event in span.events)


# --- instrumentation -------------------------------------------------------------------------------------------

def _provider(monkeypatch, text="{}", provider="groq"):
    def _try_groq(self, prompt, system, json_mode=False, temperature=0.0, json_schema=None):
        return llm(text, provider=provider, input_tokens=111, output_tokens=22)

    monkeypatch.setattr(LLMRouter, "_try_groq", _try_groq)


def test_each_llm_call_is_a_span_with_usage_but_never_the_prompt(exporter, monkeypatch):
    _provider(monkeypatch)
    LLMRouter().complete("SECRET-PROMPT-TEXT with oncall@example.com", system="sys-text", json_mode=True)

    attrs = dict(_by_name(exporter, "llm.complete").attributes)
    assert attrs["provider"] == "groq" and attrs["input_tokens"] == 111 and attrs["output_tokens"] == 22
    assert attrs["json_mode"] is True and "latency_ms" in attrs
    flattened = " ".join(str(v) for v in attrs.values())
    assert "SECRET-PROMPT-TEXT" not in flattened and "oncall@example.com" not in flattened and "sys-text" not in flattened


def test_a_failed_llm_call_marks_the_span_errored(exporter):
    with pytest.raises(Exception):
        LLMRouter().complete("p")  # no provider keys in tests -> every provider unconfigured
    assert _by_name(exporter, "llm.complete").status.status_code.name == "ERROR"


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr(g, "insert_trace_step", lambda **kw: None)
    monkeypatch.setattr(g, "redact_pii", lambda text, entities=None: text)
    monkeypatch.setattr("services.agent.untrusted.redact_pii", lambda text, entities=None: text)
    monkeypatch.setattr(g, "post_notification", lambda **kw: None)
    monkeypatch.setattr(g, "post_approval_request", lambda **kw: None)
    monkeypatch.setattr(g, "find_related_incidents", lambda *a, **kw: [])


def _state(steps):
    return {"incident_id": "inc-9", "alert_text": "ALERT checkout-service: errors", "steps_taken": steps, "hypothesis": None,
            "confidence": 0.0, "causal_graph": None, "needs_approval": False, "done": False,
            "synthesis_provider": None, "synthesis_input_tokens": 0, "synthesis_output_tokens": 0}


def test_plan_step_is_a_span_tagged_with_the_incident(exporter, agent, monkeypatch):
    monkeypatch.setattr(g, "router", FakeRouter([llm('{"action": "conclude", "reasoning": "x"}')]))
    steps = [plan("get_logs"), observation("get_logs")]
    g.plan_step(_state(steps))

    attrs = dict(_by_name(exporter, "agent.plan").attributes)
    assert attrs["incident_id"] == "inc-9" and attrs["done"] is True and attrs["steps"] == 2


def test_act_step_span_records_tool_and_status_not_the_data(exporter, agent, monkeypatch):
    class Executor:
        def execute(self, request):
            from services.agent.tools import ToolCallResult
            return ToolCallResult(tool=request.tool, role=request.role, status="ok", data={"logs": [{"message": "pii@example.com"}]})

    monkeypatch.setattr(g, "executor", Executor())
    g.act_step(_state([plan("get_logs")]))

    attrs = dict(_by_name(exporter, "agent.act").attributes)
    assert attrs["tool"] == "get_logs" and attrs["status"] == "ok" and attrs["injection_suspected"] is False
    assert "pii@example.com" not in " ".join(str(v) for v in attrs.values())


def test_synthesis_span_carries_the_outcome(exporter, agent, monkeypatch):
    graph = ('{"edges": [{"id": "e1", "from": "deploy:rel-1", "to": "mechanism:oom", "evidence_tool_call_id": "1"}], '
             '"confidence": 0.9, "confidence_rationale": "r", "recommended_action": {"action": "none", "service": null, "rationale": ""}}')
    monkeypatch.setattr(g, "router", FakeRouter([llm(graph)]))
    g.synthesize_step(_state([plan("get_logs"), observation("get_logs")]))

    attrs = dict(_by_name(exporter, "agent.synthesize").attributes)
    assert attrs["confidence"] == 0.9 and attrs["edges"] == 1 and attrs["aborted"] is False


def test_tracing_does_not_change_what_the_agent_returns(agent, monkeypatch):
    def run():
        monkeypatch.setattr(g, "router", FakeRouter([llm('{"action": "conclude", "reasoning": "x"}')]))
        return g.plan_step(_state([plan("get_logs"), observation("get_logs")]))

    tracing.configure_tracing("off")
    off = run()
    tracing.configure_tracing("memory")
    try:
        on = run()
    finally:
        tracing.configure_tracing("off")
    assert off == on


def test_a_real_router_call_inside_a_node_becomes_its_child_span(exporter, monkeypatch):
    """Uses the real LLMRouter (provider stubbed) so the nesting is real."""
    _provider(monkeypatch)
    with tracing.span("agent.plan"):
        LLMRouter().complete("p")
    assert _by_name(exporter, "llm.complete").parent.span_id == _by_name(exporter, "agent.plan").context.span_id
