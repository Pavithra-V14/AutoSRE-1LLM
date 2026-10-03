"""Groq schema-constrained decoding and its fallback, against a fake client."""

from __future__ import annotations

import sys
import types

import pytest

from services.agent.llm_router import LLMRouter, ProviderError
from services.agent.schemas import JUDGE_SCHEMA, SYNTHESIS_SCHEMA


def _install_fake_groq(monkeypatch, behaviours):
    """`behaviours` is a list of exceptions/None consumed per create() call;
    records the response_format of every call."""
    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            behaviour = behaviours.pop(0) if behaviours else None
            if isinstance(behaviour, Exception):
                raise behaviour
            message = types.SimpleNamespace(content='{"ok": true}')
            choice = types.SimpleNamespace(message=message, finish_reason="stop")
            usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=7)
            return types.SimpleNamespace(choices=[choice], usage=usage)

    class _Groq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=_Groq))
    return calls


@pytest.fixture
def router(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    return LLMRouter()


def test_schema_is_sent_as_strict_json_schema(monkeypatch, router):
    calls = _install_fake_groq(monkeypatch, [])
    router._try_groq("p", "s", True, 0.0, json_schema=SYNTHESIS_SCHEMA)

    fmt = calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "causal_graph"
    assert fmt["json_schema"]["schema"] is SYNTHESIS_SCHEMA["schema"]


def test_without_a_schema_json_mode_is_json_object(monkeypatch, router):
    calls = _install_fake_groq(monkeypatch, [])
    router._try_groq("p", "s", True, 0.0)
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_no_json_mode_sends_no_response_format(monkeypatch, router):
    calls = _install_fake_groq(monkeypatch, [])
    router._try_groq("p", "s", False, 0.0)
    assert "response_format" not in calls[0]


def test_schema_rejected_with_400_falls_back_to_json_object(monkeypatch, router):
    calls = _install_fake_groq(monkeypatch, [Exception("Error code: 400 - response_format json_schema is not supported")])
    resp = router._try_groq("p", "s", True, 0.0, json_schema=JUDGE_SCHEMA)

    assert resp.text == '{"ok": true}'
    assert [c["response_format"]["type"] for c in calls] == ["json_schema", "json_object"]


@pytest.mark.parametrize("message", ["Error code: 429 - rate limit on tokens per day", "Connection reset by peer",
                                     "Error code: 500 - internal error"])
def test_rate_limits_and_outages_are_not_mistaken_for_schema_rejection(monkeypatch, router, message):
    calls = _install_fake_groq(monkeypatch, [Exception(message)])
    with pytest.raises(ProviderError):
        router._try_groq("p", "s", True, 0.0, json_schema=JUDGE_SCHEMA)
    assert len(calls) == 1  # no pointless retry


def test_a_400_unrelated_to_schema_is_not_retried(monkeypatch, router):
    calls = _install_fake_groq(monkeypatch, [Exception("Error code: 400 - model not found")])
    with pytest.raises(ProviderError):
        router._try_groq("p", "s", True, 0.0, json_schema=JUDGE_SCHEMA)
    assert len(calls) == 1


def test_structured_output_can_be_disabled(monkeypatch, router):
    import services.agent.llm_router as module

    monkeypatch.setattr(module, "STRUCTURED_OUTPUT_ENABLED", False)
    calls = _install_fake_groq(monkeypatch, [])
    router._try_groq("p", "s", True, 0.0, json_schema=JUDGE_SCHEMA)
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_complete_passes_the_schema_through_only_when_given(monkeypatch):
    seen: list[dict] = []

    def _try_groq(self, prompt, system, json_mode=False, temperature=0.0, **kwargs):
        seen.append(kwargs)
        from fakes import llm
        return llm("{}", provider="groq")

    monkeypatch.setattr(LLMRouter, "_try_groq", _try_groq)
    router = LLMRouter()
    router.complete("p")
    router.complete("p", json_schema=JUDGE_SCHEMA)
    assert seen == [{}, {"json_schema": JUDGE_SCHEMA}]


# --- the schemas themselves ----------------------------------------------------------------------------------

def _objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from _objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _objects(value)


@pytest.mark.parametrize("definition", [SYNTHESIS_SCHEMA, JUDGE_SCHEMA])
def test_schemas_satisfy_strict_mode_rules(definition):
    """Strict decoding needs every property required and no extras, at every
    object level -- otherwise the provider rejects the schema outright."""
    for obj in _objects(definition["schema"]):
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])


def test_synthesis_schema_matches_the_actions_the_code_accepts():
    from services.agent.actions import validate_action

    enum = SYNTHESIS_SCHEMA["schema"]["properties"]["recommended_action"]["properties"]["action"]["enum"]
    for action in enum:
        graph = {"recommended_action": {"action": action, "service": "svc", "rationale": ""}}
        assert validate_action(graph) is None
