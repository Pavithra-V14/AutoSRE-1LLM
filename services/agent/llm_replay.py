"""Record and replay LLM calls, so the benchmark can run offline.

A live benchmark needs API quota (a full run exhausted Groq's daily limit and
Gemini's quota in one session) and is not repeatable. Replay makes a run free
and deterministic:

    LLM_REPLAY_MODE=record  python tools/run_benchmark.py --smoke   # once, live
    LLM_REPLAY_MODE=replay  python tools/run_benchmark.py --smoke   # CI, offline

Fixtures live in LLM_REPLAY_DIR (default tests/fixtures/llm_replay), one JSON
file per distinct request.

What identifies a request: the *normalized* prompt and system text, JSON mode,
temperature and the schema name. Timestamps and UUIDs are masked first --
scenario data stamps entries relative to "now" and every incident has a fresh
id, so without masking a recording could never match a later run. Provider
choice (prefer/demote) is deliberately NOT part of the key: replay is about
what the model said, not who said it.

Trade-off, stated plainly: any change to a prompt changes its key. A replay
run then fails loudly with ReplayMiss, naming the request, and the fixtures
must be re-recorded. That is the point -- replay verifies the harness and the
code around the model deterministically; it cannot verify a new prompt, which
needs a live run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from services.agent.llm_router import LLMResponse, ProviderError

_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)


class ReplayMiss(ProviderError):
    """Replay mode, but no recording matches this request."""


def mode() -> str:
    value = os.getenv("LLM_REPLAY_MODE", "off").strip().lower()
    if value not in ("off", "record", "replay"):
        raise ValueError(f"LLM_REPLAY_MODE must be off, record or replay, got {value!r}")
    return value


def fixture_dir() -> Path:
    return Path(os.getenv("LLM_REPLAY_DIR", "tests/fixtures/llm_replay"))


def normalize(text: str) -> str:
    """Masks the parts of a prompt that legitimately differ between runs."""
    return _UUID.sub("<ID>", _TIMESTAMP.sub("<TS>", text or ""))


def request_key(
    prompt: str, system: str, json_mode: bool, temperature: float, json_schema: Optional[dict[str, Any]] = None
) -> str:
    payload = json.dumps(
        {
            "prompt": normalize(prompt),
            "system": normalize(system),
            "json_mode": bool(json_mode),
            "temperature": round(float(temperature), 3),
            "schema": (json_schema or {}).get("name"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _path(key: str) -> Path:
    return fixture_dir() / f"{key}.json"


def lookup(key: str, prompt: str) -> LLMResponse:
    path = _path(key)
    if not path.exists():
        preview = normalize(prompt)[:160].replace("\n", " ")
        raise ReplayMiss(
            f"no recorded LLM response for request {key} (prompt starts: {preview!r}). "
            "A prompt or code change invalidated the fixtures -- re-record with "
            "LLM_REPLAY_MODE=record, or run live."
        )
    data = json.loads(path.read_text(encoding="utf-8"))["response"]
    return LLMResponse(**data)


def store(key: str, prompt: str, response: LLMResponse) -> None:
    directory = fixture_dir()
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "key": key,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_preview": normalize(prompt)[:200],
        "response": {
            "text": response.text,
            "provider": response.provider,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "finish_reason": response.finish_reason,
        },
    }
    _path(key).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
