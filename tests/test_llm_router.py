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

    def _try_groq(self, prompt, system, json_mode=False, temperature=0.0, json_schema=None):
        calls.append("groq")
        return llm("from groq", provider="groq")

    def _try_gemini(self, prompt, system, json_mode=False, temperature=0.0, json_schema=None):
        calls.append("gemini")
        raise ProviderError("gemini: down")

    def _try_openrouter(self, prompt, system, json_mode=False, temperature=0.0, json_schema=None):
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

    def _try_groq(self, prompt, system, json_mode=False, temperature=0.0, json_schema=None):
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


# --- rate limits: honor the provider's own retry delay --------------------------------------------------------------------

import time as _time  # noqa: E402

from services.agent import llm_router as _lr  # noqa: E402
from services.agent.llm_router import configured_provider_order, parse_retry_after_seconds  # noqa: E402


@pytest.mark.parametrize("message, seconds", [
    ("Rate limit reached for model. Please try again in 2m3.6s. Need more tokens?", 123.6),
    ("tokens per day (TPD): Limit 200000, Used 199896. Please try again in 2m36.3s.", 156.3),
    ("Please try again in 41.5s", 41.5),
    ("Please try again in 1h2m10s.", 3730.0),
    ("Please try again in 850ms", 0.85),
    ("429 RESOURCE_EXHAUSTED. Please retry in 17.389s.", 17.389),
    ("{'retryDelay': '17s'}", 17.0),
])
def test_retry_delay_is_read_from_the_error_message(message, seconds):
    assert parse_retry_after_seconds(message) == pytest.approx(seconds)


@pytest.mark.parametrize("message", ["Connection reset by peer", "Error code: 429 - rate limited", "", "try again later", None])
def test_messages_without_a_delay_give_none(message):
    assert parse_retry_after_seconds(message) is None


def test_a_stated_delay_replaces_blind_exponential_backoff():
    _lr.reset_circuit_breaker()
    # base cooldown would be 3600 s (quota class); the provider says 5 minutes
    applied = _lr._record_failure("groq", 3600, retry_after=300.0)
    assert applied == 301
    assert 295 < _lr._is_in_cooldown("groq") <= 301


def test_without_a_stated_delay_backoff_is_unchanged():
    _lr.reset_circuit_breaker()
    assert _lr._record_failure("groq", 60) == 60
    assert _lr._record_failure("groq", 60) == 120  # doubles on the second consecutive failure


def test_a_stated_delay_is_still_capped():
    _lr.reset_circuit_breaker()
    assert _lr._record_failure("groq", 60, retry_after=10_000_000) == _lr.LLM_MAX_COOLDOWN_SECONDS


def test_a_rate_limited_call_sets_the_cooldown_from_the_message(monkeypatch):
    def _try_groq(self, prompt, system, json_mode=False, temperature=0.0, json_schema=None):
        raise ProviderError("groq: Error code: 429 - rate limit reached. Please try again in 90s.")

    monkeypatch.setattr(LLMRouter, "_try_groq", _try_groq)
    with pytest.raises(ProviderError):
        LLMRouter().complete("p")
    assert 85 < _lr._is_in_cooldown("groq") <= 91


def test_seconds_until_available(monkeypatch):
    router = LLMRouter()
    assert router.seconds_until_available() == 0.0

    for name in ("groq", "gemini", "openrouter"):
        _lr._provider_cooldowns[name] = _time.time() + {"groq": 40, "gemini": 3000, "openrouter": 90}[name]
    assert 38 < router.seconds_until_available() <= 40  # the soonest to reopen

    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,openrouter")  # groq is not in play
    assert 88 < router.seconds_until_available() <= 90


# --- provider order ---------------------------------------------------------------------------------------------------------

def test_provider_order_is_configurable(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini, groq")
    assert configured_provider_order() == ["gemini", "groq"]
    assert _order(LLMRouter(), prefer=None) == ["gemini", "groq"]


def test_a_provider_left_out_of_the_order_is_never_tried(monkeypatch):
    """Keeps OpenRouter's free routed models out of the run entirely."""
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "groq,gemini")
    assert "openrouter" not in _order(LLMRouter(), prefer=None)


def test_an_explicit_per_call_preference_overrides_the_configured_order(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "groq")
    assert _order(LLMRouter(), prefer="gemini") == ["gemini", "groq"]


def test_unknown_provider_in_the_order_is_rejected_not_ignored(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "groq,claude")
    with pytest.raises(ValueError, match="unknown provider"):
        configured_provider_order()


def test_empty_order_is_an_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", " , ")
    with pytest.raises(ProviderError, match="names no providers"):
        LLMRouter()._provider_chain(prefer=None)
