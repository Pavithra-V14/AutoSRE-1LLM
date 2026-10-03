from __future__ import annotations

import pytest

from fakes import llm
from services.agent.llm_router import LLMRouter, ProviderError


def _order(router: LLMRouter, **kwargs) -> list[str]:
    return [fn.__name__.removeprefix("_try_") for fn in router._provider_chain(**kwargs)]


def test_default_order():
    assert _order(LLMRouter(), prefer=None) == ["groq", "gemini", "openrouter"]


def test_prefer_moves_a_provider_to_the_front_without_dropping_others():
    assert _order(LLMRouter(), prefer="gemini") == ["gemini", "groq", "openrouter"]


def test_demote_moves_providers_to_the_back_but_keeps_them():
    assert _order(LLMRouter(), prefer=None, demote=("groq",)) == ["gemini", "openrouter", "groq"]
    assert _order(LLMRouter(), prefer=None, demote=("groq", "gemini")) == ["openrouter", "groq", "gemini"]


def test_prefer_and_demote_compose():
    assert _order(LLMRouter(), prefer="openrouter", demote=("openrouter",)) == ["groq", "gemini", "openrouter"]


def test_unknown_preferred_provider_is_rejected():
    with pytest.raises(ValueError):
        LLMRouter()._provider_chain(prefer="nonexistent")


def test_demoted_provider_is_tried_last_but_still_tried(monkeypatch):
    calls: list[str] = []

    def _try_groq(self, prompt, system, json_mode=False, temperature=0.0):
        calls.append("groq")
        return llm("from groq", provider="groq")

    def _try_gemini(self, prompt, system, json_mode=False, temperature=0.0):
        calls.append("gemini")
        raise ProviderError("gemini: down")

    def _try_openrouter(self, prompt, system, json_mode=False, temperature=0.0):
        calls.append("openrouter")
        raise ProviderError("openrouter: down")

    monkeypatch.setattr(LLMRouter, "_try_groq", _try_groq)
    monkeypatch.setattr(LLMRouter, "_try_gemini", _try_gemini)
    monkeypatch.setattr(LLMRouter, "_try_openrouter", _try_openrouter)

    # groq demoted: gemini and openrouter fail first, then groq is still used.
    resp = LLMRouter().complete("p", demote=("groq",))
    assert resp.text == "from groq"
    assert calls == ["gemini", "openrouter", "groq"]


def _capture_temperature(monkeypatch):
    seen: list[float] = []

    def _try_groq(self, prompt, system, json_mode=False, temperature=0.0):
        seen.append(temperature)
        return llm("ok", provider="groq")

    monkeypatch.setattr(LLMRouter, "_try_groq", _try_groq)
    return seen


def test_temperature_defaults_to_the_configured_value(monkeypatch):
    import services.agent.llm_router as module

    seen = _capture_temperature(monkeypatch)
    monkeypatch.setattr(module, "DEFAULT_TEMPERATURE", 0.0)
    LLMRouter().complete("p")
    assert seen == [0.0]


def test_explicit_temperature_overrides_the_default_including_zero(monkeypatch):
    import services.agent.llm_router as module

    seen = _capture_temperature(monkeypatch)
    monkeypatch.setattr(module, "DEFAULT_TEMPERATURE", 0.7)
    router = LLMRouter()
    router.complete("p")                    # default
    router.complete("p", temperature=0.0)   # explicit zero must not fall back to the default
    router.complete("p", temperature=0.3)
    assert seen == [0.7, 0.0, 0.3]
