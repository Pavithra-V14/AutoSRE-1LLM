from __future__ import annotations

import json

import pytest

from services.agent.json_utils import extract_json


def test_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_markdown_fence_with_language_tag():
    assert extract_json('Here you go:\n```json\n{"a": 1}\n```\nDone.') == {"a": 1}


def test_prose_around_object():
    assert extract_json('Sure! {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}


def test_trailing_comma_is_repaired():
    assert extract_json('{"edges": [{"id": "e1"},], "confidence": 0.9,}') == {
        "edges": [{"id": "e1"}],
        "confidence": 0.9,
    }


def test_truncated_output_is_closed():
    out = extract_json('{"edges": [{"id": "e1", "from": "a", "to": "b"}], "confidence_rationale": "cut o')
    assert out["edges"][0]["id"] == "e1"
    assert out["confidence_rationale"] == "cut o"


def test_truncated_after_key_colon_becomes_null():
    assert extract_json('{"edges": [], "confidence":') == {"edges": [], "confidence": None}


def test_garbled_value_is_not_guessed():
    """The real failure: the model wrote `0. nine`. Repair must not invent a
    number -- it must surface an error so the caller re-asks."""
    with pytest.raises(json.JSONDecodeError):
        extract_json('{"edges": [], "confidence": 0. nine, "confidence_rationale": "x"}')


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", '"just a string"', "42"])
def test_non_object_raises(text):
    with pytest.raises(json.JSONDecodeError):
        extract_json(text)
