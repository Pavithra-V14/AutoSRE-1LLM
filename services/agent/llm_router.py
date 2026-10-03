from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import httpx
from dotenv import load_dotenv

from services.agent import tracing
from services.agent.logging_utils import log_stage

load_dotenv()


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    finish_reason: str = "unknown"


class ProviderError(Exception):
    pass

MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))
# Investigation and grading want repeatable answers, not creative ones: the
# same evidence should produce the same conclusion. Providers' own defaults
# (~1.0) were a measurable source of run-to-run variance.
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
# Schema-constrained decoding where supported (set false to force plain JSON mode).
STRUCTURED_OUTPUT_ENABLED = os.getenv("LLM_STRUCTURED_OUTPUT", "true").strip().lower() != "false"
LLM_RATE_LIMIT_COOLDOWN_SECONDS = int(os.getenv("LLM_RATE_LIMIT_COOLDOWN_SECONDS", "60"))
LLM_QUOTA_COOLDOWN_SECONDS = int(os.getenv("LLM_QUOTA_COOLDOWN_SECONDS", "3600"))
LLM_OVERSIZED_COOLDOWN_SECONDS = int(os.getenv("LLM_OVERSIZED_COOLDOWN_SECONDS", "86400"))
LLM_MAX_COOLDOWN_SECONDS = int(os.getenv("LLM_MAX_COOLDOWN_SECONDS", "86400"))
_provider_cooldowns: dict[str, float] = {}
_provider_consecutive_failures: dict[str, int] = {}


KNOWN_PROVIDERS = ("groq", "gemini", "openrouter")

# "try again in 2m3.6s" (Groq), "retry in 17.3s" (Gemini), "retryDelay": "17s"
_RETRY_PHRASE = re.compile(r"(?:try again|retry)\w*\s+in\s+(?P<t>(?:\d+(?:\.\d+)?\s*(?:ms|h|m|s)\s*)+)", re.IGNORECASE)
_RETRY_DELAY_FIELD = re.compile(r"retrydelay\W{1,6}(?P<t>\d+(?:\.\d+)?\s*s)", re.IGNORECASE)
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", re.IGNORECASE)
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def parse_retry_after_seconds(error_message: str) -> Optional[float]:
    """Seconds the provider says to wait, if its error message says so.

    Providers state the real wait in the 429 body. Ignoring it and applying a
    blind 60 s x 2^n cooldown meant either hammering a limit that would not
    lift for hours (a daily token cap) or sitting out a window that had
    already reopened."""
    text = error_message or ""
    match = _RETRY_PHRASE.search(text) or _RETRY_DELAY_FIELD.search(text)
    if not match:
        return None
    total = sum(float(value) * _UNIT_SECONDS[unit.lower()] for value, unit in _DURATION_PART.findall(match.group("t")))
    return total if total > 0 else None


def configured_provider_order() -> list[str]:
    """LLM_PROVIDER_ORDER, e.g. "groq,gemini" to leave out OpenRouter's free
    routed models (they have answered JSON requests with a safety-classifier
    verdict). Unknown names are rejected rather than silently ignored."""
    raw = os.getenv("LLM_PROVIDER_ORDER", ",".join(KNOWN_PROVIDERS))
    order = [name.strip().lower() for name in raw.split(",") if name.strip()]
    unknown = [name for name in order if name not in KNOWN_PROVIDERS]
    if unknown:
        raise ValueError(f"LLM_PROVIDER_ORDER has unknown provider(s) {unknown}; expected from {list(KNOWN_PROVIDERS)}")
    return order


def _classify_failure(error_message: str) -> tuple[str, int]:
    """Returns (failure_class, base_cooldown_seconds) -- the BASE value
    before exponential backoff is applied. Order matters -- check the
    more specific 'quota exhausted' pattern before the generic
    rate-limit one, since a quota message often also contains '429'."""
    lowered = error_message.lower()
    if "resource_exhausted" in lowered or ("quota" in lowered and "429" in lowered) or "exceeded your current quota" in lowered:
        return "quota_exhausted", LLM_QUOTA_COOLDOWN_SECONDS
    if "413" in lowered or "request too large" in lowered or "too large for model" in lowered:
        return "oversized_request", LLM_OVERSIZED_COOLDOWN_SECONDS
    if "429" in lowered or "rate limit" in lowered:
        return "rate_limited", LLM_RATE_LIMIT_COOLDOWN_SECONDS
    return "other", 0  # not a rate/quota/size issue -- no cooldown, may well be transient/one-off


def _record_failure(provider_name: str, base_cooldown: int, retry_after: Optional[float] = None) -> int:
    """Applies exponential backoff: 1st failure -> base_cooldown,
    2nd consecutive failure -> base_cooldown*2, 3rd -> base_cooldown*4,
    etc., capped at LLM_MAX_COOLDOWN_SECONDS. Sets the actual cooldown and
    returns the seconds applied, for logging."""
    consecutive = _provider_consecutive_failures.get(provider_name, 0) + 1
    _provider_consecutive_failures[provider_name] = consecutive
    if retry_after is not None:
        # The provider told us when the limit lifts: trust it (plus a second of
        # slack) instead of guessing with exponential backoff.
        actual_cooldown = min(int(retry_after) + 1, LLM_MAX_COOLDOWN_SECONDS)
    else:
        actual_cooldown = min(base_cooldown * (2 ** (consecutive - 1)), LLM_MAX_COOLDOWN_SECONDS)
    _provider_cooldowns[provider_name] = time.time() + actual_cooldown
    return actual_cooldown


def _record_success(provider_name: str) -> None:
    """A provider that succeeds is no longer 'consecutively failing' --
    reset its backoff so the NEXT failure (if any) starts back at the
    base cooldown, not wherever the exponential curve left off."""
    _provider_consecutive_failures[provider_name] = 0


def _is_in_cooldown(provider_name: str) -> Optional[float]:
    """Returns seconds remaining in cooldown, or None if not cooling down."""
    until = _provider_cooldowns.get(provider_name)
    if until is None:
        return None
    remaining = until - time.time()
    if remaining <= 0:
        del _provider_cooldowns[provider_name]
        return None
    return remaining


def reset_circuit_breaker() -> None:
    """Clears all cooldowns AND consecutive-failure counts -- used by
    tests, and available for a manual reset if you know a provider's
    quota just refreshed."""
    _provider_cooldowns.clear()
    _provider_consecutive_failures.clear()


class LLMRouter:
    """Tries providers in order until one succeeds. Logs the provider used."""

    def __init__(self) -> None:
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openrouter/free")

    def seconds_until_available(self) -> float:
        """0 if some configured provider can be called now; otherwise how long
        until the soonest one comes out of cooldown. Long-running tools use it
        to wait for a rate limit to lift instead of abandoning a run."""
        remaining: list[float] = []
        for name in configured_provider_order():
            left = _is_in_cooldown(name)
            if left is None:
                return 0.0
            remaining.append(left)
        return min(remaining) if remaining else 0.0

    def _provider_chain(
        self, prefer: Optional[str], demote: Iterable[str] = ()
    ) -> list[Callable[..., LLMResponse]]:
        order = configured_provider_order()
        if not order:
            raise ProviderError("LLM_PROVIDER_ORDER names no providers")
        chain = [getattr(self, f"_try_{name}") for name in order]
        if prefer:
            if prefer not in KNOWN_PROVIDERS:
                raise ValueError(f"unknown LLM provider '{prefer}' (expected groq, gemini or openrouter)")
            preferred = [fn for fn in chain if fn.__name__ == f"_try_{prefer}"]
            if not preferred:
                # asked for a provider the configured order excludes: honor the
                # explicit per-call request rather than ignoring it
                preferred = [getattr(self, f"_try_{prefer}")]
            chain = preferred + [fn for fn in chain if fn not in preferred]
        demoted = {f"_try_{name}" for name in demote}
        # Stable partition: demoted providers keep their relative order but go
        # last. They are still tried if everything else fails -- demotion
        # reorders, it never removes a provider.
        return [fn for fn in chain if fn.__name__ not in demoted] + [fn for fn in chain if fn.__name__ in demoted]

    def complete(
        self,
        prompt: str,
        system: str = "",
        json_mode: bool = False,
        prefer: Optional[str] = None,
        demote: Iterable[str] = (),
        temperature: Optional[float] = None,
        json_schema: Optional[dict] = None,
    ) -> LLMResponse:
        """One LLM call, traced. Span attributes are numbers and names only --
        never the prompt or the response (they may carry PII)."""
        with tracing.span(
            "llm.complete", json_mode=json_mode, prefer=prefer or "", schema=(json_schema or {}).get("name", ""),
        ) as sp:
            resp = self._complete(prompt, system, json_mode, prefer, demote, temperature, json_schema)
            sp.set(
                provider=resp.provider, model=resp.model, input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens, latency_ms=resp.latency_ms, finish_reason=resp.finish_reason,
            )
            return resp

    def _complete(
        self,
        prompt: str,
        system: str,
        json_mode: bool,
        prefer: Optional[str],
        demote: Iterable[str],
        temperature: Optional[float],
        json_schema: Optional[dict],
    ) -> LLMResponse:
        """json_mode asks the provider to constrain output to a single JSON
        object (Groq response_format, Gemini response_mime_type). OpenRouter
        is deliberately excluded: its free routed models reject
        response_format unpredictably, which would turn a formatting hint
        into a hard failure -- callers still parse defensively either way.
        `prefer` moves one provider to the front of the fallback order
        (used by the benchmark judge to avoid grading a model with itself).
        `demote` moves providers to the back of the order -- used when a
        provider just returned unusable output (e.g. a free routed model
        answering with a safety-classifier verdict instead of JSON).
        `temperature` overrides LLM_TEMPERATURE for this call.
        `json_schema` ({"name", "schema"}) asks for schema-constrained
        decoding where the provider supports it (Groq); other providers fall
        back to plain JSON mode, so callers must still validate."""
        resolved_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature

        # Record/replay (see services/agent/llm_replay.py). Imported here
        # because that module imports this one.
        from services.agent import llm_replay

        replay_mode = llm_replay.mode()
        replay_key = (
            llm_replay.request_key(prompt, system, json_mode, resolved_temperature, json_schema)
            if replay_mode != "off"
            else ""
        )
        if replay_mode == "replay":
            recorded = llm_replay.lookup(replay_key, prompt)
            log_stage("llm_provider", "-", "OK", provider=recorded.provider, model=recorded.model, source="replay")
            return recorded
        errors = []
        for fn in self._provider_chain(prefer, demote):
            provider_name = fn.__name__.replace("_try_", "")

            cooldown_remaining = _is_in_cooldown(provider_name)
            if cooldown_remaining is not None:
                log_stage(
                    "llm_provider", "-", "SKIP", provider=provider_name,
                    reason=f"in cooldown, {int(cooldown_remaining)}s remaining",
                )
                errors.append(f"{provider_name}: in cooldown ({int(cooldown_remaining)}s remaining)")
                continue

            try:
                extra = {"json_schema": json_schema} if json_schema else {}
                resp = fn(prompt, system, json_mode, resolved_temperature, **extra)
                _record_success(provider_name)
                log_stage(
                    "llm_provider", "-", "OK", provider=provider_name, model=resp.model,
                    finish_reason=resp.finish_reason,
                )
                if resp.finish_reason == "length":
                    log_stage(
                        "llm_provider", "-", "WARN", provider=provider_name,
                        reason="response truncated (finish_reason=length) -- "
                        "raise LLM_MAX_OUTPUT_TOKENS if this recurs",
                    )
                if replay_mode == "record":
                    llm_replay.store(replay_key, prompt, resp)
                return resp
            except ProviderError as e:
                failure_class, base_cooldown = _classify_failure(str(e))
                retry_after = parse_retry_after_seconds(str(e)) if base_cooldown > 0 else None
                actual_cooldown = _record_failure(provider_name, base_cooldown, retry_after) if base_cooldown > 0 else 0
                log_stage(
                    "llm_provider", "-", "FAIL", provider=provider_name, reason=str(e)[:100],
                    failure_class=failure_class,
                    cooldown=f"{actual_cooldown}s (consecutive={_provider_consecutive_failures.get(provider_name, 0)})" if actual_cooldown > 0 else "none",
                )
                errors.append(str(e))
                continue
        raise ProviderError(
            f"All providers failed or are unconfigured: {'; '.join(errors)}"
        )

    def _try_groq(
        self, prompt: str, system: str, json_mode: bool = False, temperature: float = 0.0,
        json_schema: Optional[dict] = None,
    ) -> LLMResponse:
        if not self.groq_key:
            raise ProviderError("groq: no API key configured")
        from groq import Groq

        start = time.time()
        client = Groq(api_key=self.groq_key)
        use_schema = bool(json_schema) and STRUCTURED_OUTPUT_ENABLED

        def _create(response_format: Optional[dict]):
            extra: dict = {"response_format": response_format} if response_format else {}
            return client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                temperature=temperature,
                **extra,
            )

        if use_schema:
            response_format: Optional[dict] = {
                "type": "json_schema",
                "json_schema": {"name": json_schema["name"], "schema": json_schema["schema"], "strict": True},
            }
        else:
            response_format = {"type": "json_object"} if json_mode else None

        try:
            try:
                resp = _create(response_format)
            except Exception as e:
                # A model/account that rejects schema decoding must degrade to
                # plain JSON mode, not fail the call. Only schema-shaped 400s
                # qualify; rate limits and outages still raise.
                text = str(e).lower()
                schema_rejected = use_schema and "400" in text and any(
                    k in text for k in ("response_format", "json_schema", "schema", "strict")
                )
                if not schema_rejected:
                    raise
                log_stage("llm_provider", "-", "WARN", provider="groq", reason="schema output rejected -- using json_object")
                resp = _create({"type": "json_object"})
        except Exception as e:
            raise ProviderError(f"groq: {e}")

        latency_ms = int((time.time() - start) * 1000)
        return LLMResponse(
            text=resp.choices[0].message.content,
            provider="groq",
            model=self.groq_model,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=latency_ms,
            finish_reason=resp.choices[0].finish_reason or "unknown",
        )

    def _try_gemini(
        self, prompt: str, system: str, json_mode: bool = False, temperature: float = 0.0,
        json_schema: Optional[dict] = None,
    ) -> LLMResponse:
        if not self.gemini_key:
            raise ProviderError("gemini: no API key configured")

        from google import genai
        from google.genai import types

        start = time.time()

        client = genai.Client(api_key=self.gemini_key)

        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system or None,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    temperature=temperature,
                    response_mime_type="application/json" if json_mode else None,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except Exception as e:
            raise ProviderError(f"gemini: {e}")

        latency_ms = int((time.time() - start) * 1000)
        usage = getattr(resp, "usage_metadata", None)
        raw_finish_reason = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else None
        finish_reason = "length" if raw_finish_reason and "MAX_TOKENS" in str(raw_finish_reason) else str(raw_finish_reason or "unknown")
        return LLMResponse(
            text=resp.text,
            provider="gemini",
            model="gemini-2.5-flash",
            input_tokens=getattr(usage, "prompt_token_count", 0),
            output_tokens=getattr(usage, "candidates_token_count", 0),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )

    def _try_openrouter(
        self, prompt: str, system: str, json_mode: bool = False, temperature: float = 0.0,
        json_schema: Optional[dict] = None,
    ) -> LLMResponse:
        if not self.openrouter_key:
            raise ProviderError("openrouter: no API key configured")

        start = time.time()
        model = self.openrouter_model
        try:
            r = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": temperature,
                },
                timeout=30.0,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise ProviderError(f"openrouter: {e}")
        if not data.get("choices"):
            raise ProviderError(f"openrouter: no choices in response: {data}")
        choice = data["choices"][0]

        latency_ms = int((time.time() - start) * 1000)
        usage = data.get("usage", {})
        return LLMResponse(
            text=choice["message"]["content"],
            provider="openrouter",
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason", "unknown"),
        )