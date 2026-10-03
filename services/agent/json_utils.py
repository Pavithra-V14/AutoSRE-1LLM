"""Tolerant JSON-object extraction for LLM output.

Strict prompting and provider JSON modes are the primary defenses against
malformed output; this module is the fallback layer, and
services/agent/structured.py adds a bounded re-ask on top of it. Only
*structural* damage is repaired here (code fences, surrounding prose,
trailing commas, truncation) -- never content. A garbled value such as
``"confidence": 0. nine`` is deliberately NOT guessed at; it is surfaced as
a JSONDecodeError so the caller can re-ask the model rather than invent a
number.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _candidates(text: str) -> Iterator[str]:
    """Yields progressively more aggressive slices of `text` to try, most
    likely first, without duplicates."""
    seen: set[str] = set()

    def emit(candidate: str) -> Iterator[str]:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate

    yield from emit(text)

    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                yield from emit(chunk)

    start = text.find("{")
    if start != -1:
        end = text.rfind("}")
        if end > start:
            yield from emit(text[start : end + 1])
        yield from emit(text[start:])  # no closing brace at all -> likely truncated


def _close_truncated(text: str) -> str:
    """Closes an unterminated string and any open brackets/braces, for
    output cut off by a max-token limit. Returns `text` unchanged if it is
    already balanced."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()

    if not stack and not in_string:
        return text

    repaired = text + ('"' if in_string else "")
    repaired = repaired.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1]
    elif repaired.endswith(":"):
        repaired += " null"
    return repaired + "".join(reversed(stack))


def _repair(candidate: str) -> str:
    return _close_truncated(_TRAILING_COMMA.sub(r"\1", candidate))


def extract_json(text: str) -> dict[str, Any]:
    """Returns the first JSON *object* recoverable from `text`.

    Raises json.JSONDecodeError (with `pos` relative to the first candidate
    tried, so error offsets in logs stay meaningful) when nothing usable can
    be recovered, or when the payload parses but is not a JSON object."""
    text = (text or "").strip()
    first_error: json.JSONDecodeError | None = None

    for candidate in _candidates(text):
        for variant in (candidate, _repair(candidate)):
            try:
                value = json.loads(variant)
            except json.JSONDecodeError as e:
                first_error = first_error or e
                continue
            if isinstance(value, dict):
                return value
            first_error = first_error or json.JSONDecodeError(
                f"expected a JSON object, got {type(value).__name__}", variant, 0
            )

    raise first_error or json.JSONDecodeError("no JSON object found in LLM response", text, 0)
