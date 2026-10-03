"""Structured (JSON) LLM output with a bounded re-ask.

A single garbled token in an otherwise correct answer -- a real run
produced ``"confidence": 0. nine,`` -- used to discard the whole response
and abort the investigation. complete_json() instead:

  1. asks the provider for JSON mode,
  2. extracts/repairs the object (services/agent/json_utils.py),
  3. runs an optional caller validator (schema, enum membership, ...),
  4. on failure, re-asks ONCE (LLM_JSON_MAX_ATTEMPTS, default 2) showing the
     model exactly what was wrong with its previous answer.

Token usage is summed across attempts so cost accounting stays honest, and
the final failure carries everything needed to debug it from the trace.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from services.agent.json_utils import extract_json
from services.agent.llm_router import LLMResponse, LLMRouter
from services.agent.logging_utils import log_stage

JSON_MAX_ATTEMPTS = max(1, int(os.getenv("LLM_JSON_MAX_ATTEMPTS", "2")))
_PREVIOUS_RESPONSE_CHARS = 4000

# A validator returns None when the object is acceptable, otherwise a short
# human-readable reason that is shown to the model on the re-ask.
Validator = Callable[[dict[str, Any]], Optional[str]]
Normalizer = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class StructuredResult:
    data: dict[str, Any]
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    attempts: int
    finish_reason: str


class StructuredOutputError(Exception):
    """Raised when every attempt produced unusable output. ProviderError
    (all providers down) is NOT wrapped -- it propagates unchanged."""

    def __init__(
        self,
        reason: str,
        *,
        last_text: str,
        finish_reason: str,
        error_pos: Optional[int],
        attempts: int,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.last_text = last_text
        self.finish_reason = finish_reason
        self.error_pos = error_pos
        self.attempts = attempts
        self.provider = provider
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _describe_json_error(e: json.JSONDecodeError, text: str) -> str:
    start = max(0, e.pos - 60)
    context = text[start : e.pos + 60].replace("\n", " ")
    return f"invalid JSON ({e.msg} at line {e.lineno}, column {e.colno}) near: ...{context}..."


def _repair_prompt(prompt: str, previous_text: str, problem: str, truncated: bool) -> str:
    shown = previous_text[:_PREVIOUS_RESPONSE_CHARS]
    hint = (
        " Your previous response was cut off by the output limit -- be more concise."
        if truncated
        else ""
    )
    return (
        f"{prompt}\n\n"
        "---\n"
        f"Your previous response could not be used: {problem}.{hint}\n"
        f"Previous response:\n{shown}\n\n"
        "Respond again with ONLY the corrected, complete JSON object. Numbers "
        "must be plain digits (e.g. 0.85), never words."
    )


def complete_json(
    router: LLMRouter,
    prompt: str,
    *,
    system: str = "",
    validator: Optional[Validator] = None,
    normalizer: Optional[Normalizer] = None,
    max_attempts: Optional[int] = None,
    prefer: Optional[str] = None,
    temperature: Optional[float] = None,
    json_schema: Optional[dict[str, Any]] = None,
    incident_id: str = "-",
    stage: str = "llm_json",
) -> StructuredResult:
    attempts_allowed = max(1, max_attempts if max_attempts is not None else JSON_MAX_ATTEMPTS)
    current_prompt = prompt
    in_tokens = out_tokens = latency_ms = 0

    last_text = ""
    last_finish = "unknown"
    last_provider = ""
    last_problem = "no attempt made"
    last_pos: Optional[int] = None
    bad_providers: list[str] = []

    for attempt in range(1, attempts_allowed + 1):
        kwargs: dict[str, Any] = {"system": system, "json_mode": True}
        if prefer:
            kwargs["prefer"] = prefer
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_schema is not None:
            kwargs["json_schema"] = json_schema
        if bad_providers:
            # A provider that just produced unusable output is unlikely to
            # fix it on an identical re-ask -- try someone else first.
            kwargs["demote"] = tuple(bad_providers)
        resp: LLMResponse = router.complete(current_prompt, **kwargs)

        in_tokens += resp.input_tokens
        out_tokens += resp.output_tokens
        latency_ms += resp.latency_ms
        last_text, last_finish = resp.text or "", resp.finish_reason
        last_provider = resp.provider

        problem: Optional[str] = None
        data: Optional[dict[str, Any]] = None
        try:
            data = extract_json(last_text)
            last_pos = None
        except json.JSONDecodeError as e:
            problem = _describe_json_error(e, last_text)
            last_pos = e.pos

        if data is not None:
            if normalizer is not None:
                data = normalizer(data)
            if validator is not None:
                problem = validator(data)

        if problem is None and data is not None:
            if attempt > 1:
                log_stage(stage, incident_id, "OK", note=f"recovered on attempt {attempt}/{attempts_allowed}", provider=resp.provider)
            return StructuredResult(
                data=data,
                provider=resp.provider,
                model=resp.model,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_ms=latency_ms,
                attempts=attempt,
                finish_reason=resp.finish_reason,
            )

        last_problem = problem or "unknown problem"
        if resp.provider not in bad_providers:
            bad_providers.append(resp.provider)
        will_retry = attempt < attempts_allowed
        log_stage(
            stage, incident_id, "WARN",
            reason=last_problem[:120], attempt=f"{attempt}/{attempts_allowed}",
            provider=resp.provider, finish_reason=resp.finish_reason,
            action="re-asking model" if will_retry else "giving up",
        )
        if will_retry:
            current_prompt = _repair_prompt(prompt, last_text, last_problem, resp.finish_reason == "length")

    raise StructuredOutputError(
        last_problem,
        last_text=last_text,
        finish_reason=last_finish,
        error_pos=last_pos,
        attempts=attempts_allowed,
        provider=last_provider,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        latency_ms=latency_ms,
    )
