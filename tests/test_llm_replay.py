from __future__ import annotations

import pytest

from fakes import llm
from services.agent import llm_replay
from services.agent.llm_replay import ReplayMiss, normalize, request_key
from services.agent.llm_router import LLMRouter, ProviderError


@pytest.fixture(autouse=True)
def fixtures(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_REPLAY_DIR", str(tmp_path))
    monkeypatch.delenv("LLM_REPLAY_MODE", raising=False)
    return tmp_path


def _live_provider(monkeypatch, text="live answer"):
    calls: list[str] = []

    def _try_groq(self, prompt, system, json_mode=False, temperature=0.0, json_schema=None):
        calls.append(prompt)
        return llm(text, provider="groq", input_tokens=123, output_tokens=45)

    monkeypatch.setattr(LLMRouter, "_try_groq", _try_groq)
    return calls


# --- masking ---------------------------------------------------------------------------------------------------

def test_timestamps_and_uuids_are_masked():
    text = "at 2026-10-03 16:36:13.872779 and 2026-10-03T16:00:00Z incident 18cb6199-8054-438f-9128-edc0c8800017"
    assert normalize(text) == "at <TS> and <TS> incident <ID>"


def test_the_same_request_on_a_different_day_has_the_same_key():
    a = request_key("Current time: 2026-10-03 16:00:00\nincident 11111111-1111-1111-1111-111111111111", "s", True, 0.0)
    b = request_key("Current time: 2027-01-09 03:15:42\nincident 22222222-2222-2222-2222-222222222222", "s", True, 0.0)
    assert a == b


@pytest.mark.parametrize("changed", [
    {"prompt": "different"}, {"system": "other"}, {"json_mode": False}, {"temperature": 0.7},
    {"json_schema": {"name": "judge_verdict"}},
])
def test_any_real_difference_changes_the_key(changed):
    base = dict(prompt="p", system="s", json_mode=True, temperature=0.0, json_schema=None)
    assert request_key(**base) != request_key(**{**base, **changed})


def test_provider_choice_is_not_part_of_the_key():
    """Replay is about what the model said, not who said it, so the same
    recording serves a run that routes differently."""
    import inspect
    assert "prefer" not in inspect.signature(request_key).parameters


# --- record then replay --------------------------------------------------------------------------------------------

def test_a_recording_replays_without_touching_any_provider(monkeypatch):
    calls = _live_provider(monkeypatch)

    monkeypatch.setenv("LLM_REPLAY_MODE", "record")
    recorded = LLMRouter().complete("what happened at 2026-10-03 16:00:00?", system="sys", json_mode=True)
    assert recorded.text == "live answer" and len(calls) == 1

    monkeypatch.setenv("LLM_REPLAY_MODE", "replay")
    calls.clear()
    replayed = LLMRouter().complete("what happened at 2027-05-05 09:09:09?", system="sys", json_mode=True)

    assert calls == []  # no provider was called
    assert (replayed.text, replayed.provider, replayed.input_tokens, replayed.output_tokens) == ("live answer", "groq", 123, 45)


def test_replay_without_a_recording_fails_loudly_and_says_how_to_fix_it(monkeypatch):
    monkeypatch.setenv("LLM_REPLAY_MODE", "replay")
    with pytest.raises(ReplayMiss) as exc:
        LLMRouter().complete("never recorded", system="s")
    assert "LLM_REPLAY_MODE=record" in str(exc.value) and "never recorded" in str(exc.value)
    assert isinstance(exc.value, ProviderError)  # callers already handle it as an unavailable LLM


def test_replay_never_falls_through_to_a_live_provider(monkeypatch):
    calls = _live_provider(monkeypatch)
    monkeypatch.setenv("LLM_REPLAY_MODE", "replay")
    with pytest.raises(ReplayMiss):
        LLMRouter().complete("never recorded")
    assert calls == []


def test_off_mode_neither_records_nor_replays(monkeypatch, fixtures):
    _live_provider(monkeypatch)
    LLMRouter().complete("p")
    assert list(fixtures.iterdir()) == []


def test_failed_provider_calls_are_not_recorded(monkeypatch, fixtures):
    def _try_groq(self, prompt, system, json_mode=False, temperature=0.0, json_schema=None):
        raise ProviderError("groq: 429")

    monkeypatch.setattr(LLMRouter, "_try_groq", _try_groq)
    monkeypatch.setenv("LLM_REPLAY_MODE", "record")
    with pytest.raises(ProviderError):
        LLMRouter().complete("p")
    assert list(fixtures.iterdir()) == []


def test_fixture_files_are_readable_json_with_a_prompt_preview(monkeypatch, fixtures):
    import json

    _live_provider(monkeypatch)
    monkeypatch.setenv("LLM_REPLAY_MODE", "record")
    LLMRouter().complete("explain the outage at 2026-10-03 16:00:00", system="s")

    (path,) = fixtures.iterdir()
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["prompt_preview"] == "explain the outage at <TS>"
    assert record["response"]["text"] == "live answer"


def test_invalid_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_REPLAY_MODE", "sometimes")
    with pytest.raises(ValueError):
        llm_replay.mode()


def test_temperature_participates_in_replay_matching(monkeypatch):
    _live_provider(monkeypatch)
    monkeypatch.setenv("LLM_REPLAY_MODE", "record")
    LLMRouter().complete("p", temperature=0.0)

    monkeypatch.setenv("LLM_REPLAY_MODE", "replay")
    LLMRouter().complete("p", temperature=0.0)  # hit
    with pytest.raises(ReplayMiss):
        LLMRouter().complete("p", temperature=0.9)
