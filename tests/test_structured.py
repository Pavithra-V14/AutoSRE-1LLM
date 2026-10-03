from __future__ import annotations

import pytest

from fakes import FakeRouter, llm
from services.agent.llm_router import ProviderError
from services.agent.structured import StructuredOutputError, complete_json

GARBLED = '{"edges": [], "confidence": 0. nine, "confidence_rationale": "x"}'
GOOD = '{"edges": [], "confidence": 0.9, "confidence_rationale": "x"}'


def test_first_attempt_success_requests_json_mode():
    router = FakeRouter([llm(GOOD)])
    result = complete_json(router, "p", system="s")
    assert result.data["confidence"] == 0.9
    assert result.attempts == 1
    assert router.calls[0]["json_mode"] is True


def test_garbled_output_is_reasked_with_the_specific_error_and_recovers():
    router = FakeRouter([llm(GARBLED, input_tokens=100, output_tokens=40), llm(GOOD, input_tokens=120, output_tokens=30)])
    result = complete_json(router, "original prompt", max_attempts=2)

    assert result.data["confidence"] == 0.9
    assert result.attempts == 2
    # usage is summed across attempts so cost accounting stays honest
    assert (result.input_tokens, result.output_tokens) == (220, 70)
    repair = router.calls[1]["prompt"]
    assert "original prompt" in repair
    assert "invalid JSON" in repair
    assert "0. nine" in repair  # the model is shown exactly what it got wrong


def test_gives_up_after_max_attempts_with_full_diagnostics():
    router = FakeRouter([llm(GARBLED, finish_reason="length"), llm(GARBLED, finish_reason="length")])
    with pytest.raises(StructuredOutputError) as exc:
        complete_json(router, "p", max_attempts=2)

    err = exc.value
    assert err.attempts == 2
    assert err.last_text == GARBLED
    assert err.truncated is True
    assert err.error_pos is not None
    assert err.input_tokens == 20  # both attempts counted


def test_validator_failure_triggers_reask():
    def validator(data):
        return None if data.get("verdict") in ("a", "b") else 'verdict must be "a" or "b"'

    router = FakeRouter([llm('{"verdict": "zzz"}'), llm('{"verdict": "a"}')])
    result = complete_json(router, "p", validator=validator, max_attempts=2)
    assert result.data == {"verdict": "a"}
    assert 'verdict must be "a" or "b"' in router.calls[1]["prompt"]


def test_normalizer_runs_before_validator():
    router = FakeRouter([llm('{"n": "5"}')])
    result = complete_json(
        router, "p",
        normalizer=lambda d: {**d, "n": int(d["n"])},
        validator=lambda d: None if isinstance(d["n"], int) else "n must be an int",
        max_attempts=1,
    )
    assert result.data == {"n": 5}


def test_provider_error_propagates_unwrapped():
    router = FakeRouter([ProviderError("all providers failed")])
    with pytest.raises(ProviderError):
        complete_json(router, "p")


def test_single_attempt_mode_does_not_retry():
    router = FakeRouter([llm(GARBLED)])
    with pytest.raises(StructuredOutputError):
        complete_json(router, "p", max_attempts=1)
    assert len(router.calls) == 1


def test_prefer_is_forwarded_only_when_set():
    router = FakeRouter([llm(GOOD), llm(GOOD)])
    complete_json(router, "p")
    complete_json(router, "p", prefer="gemini")
    assert router.calls[0]["prefer"] is None
    assert router.calls[1]["prefer"] == "gemini"


def test_provider_that_returned_garbage_is_demoted_for_the_reask():
    """Real failure: a free routed model answered `User Safety: safe` instead
    of JSON, and the re-ask went straight back to it."""
    router = FakeRouter([llm("User Safety: safe", provider="openrouter"), llm(GOOD, provider="gemini")])
    result = complete_json(router, "p", max_attempts=2)

    assert result.provider == "gemini"
    assert router.calls[0]["demote"] == ()
    assert router.calls[1]["demote"] == ("openrouter",)


def test_temperature_is_forwarded_only_when_set():
    router = FakeRouter([llm(GOOD), llm(GOOD)])
    complete_json(router, "p")
    complete_json(router, "p", temperature=0.0)
    assert router.calls[0]["temperature"] is None
    assert router.calls[1]["temperature"] == 0.0
