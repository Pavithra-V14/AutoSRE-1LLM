"""Distributed tracing for investigations (OpenTelemetry).

Why: one benchmark scenario took 128 s while its neighbours took 30 s, and
nothing said whether the time went to a rate-limited LLM call, a slow tool or
the judge. Spans answer that: every incident is a trace, with a child span per
agent step and per LLM call.

    AUTOSRE_TRACING=off      (default) nothing is created; spans are no-ops
    AUTOSRE_TRACING=console  print spans to stdout (local debugging)
    AUTOSRE_TRACING=otlp     export over OTLP/HTTP; standard OTEL_* env vars
                             apply: OTEL_EXPORTER_OTLP_ENDPOINT,
                             OTEL_EXPORTER_OTLP_HEADERS, OTEL_SERVICE_NAME

Deliberately a separate switch from OTEL_SDK_DISABLED: that variable is set to
"true" for the guardrails libraries' own telemetry. This module builds its own
TracerProvider and never installs a global one, so turning tracing on does not
turn their telemetry on.

Privacy: spans carry structure and numbers only -- tool names, statuses,
token counts, latencies, confidence. Never prompts, tool output, alert text or
log lines, which may contain PII or secrets.
"""

from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

_tracer: Any = None
_provider: Any = None

_PRIMITIVES = (str, bool, int, float)


class _NoOpSpan:
    def set(self, **attributes: Any) -> None:  # noqa: D401 - mirrors _Span.set
        return None


class _Span:
    def __init__(self, span: Any) -> None:
        self._span = span

    def set(self, **attributes: Any) -> None:
        for key, value in attributes.items():
            if value is not None:
                self._span.set_attribute(key, value if isinstance(value, _PRIMITIVES) else str(value))


def configure_tracing(mode: Optional[str] = None) -> Any:
    """Enables tracing per AUTOSRE_TRACING (or `mode`). Safe to call more than
    once. Returns the in-memory exporter for mode "memory" (tests), else None."""
    global _tracer, _provider
    mode = (mode or os.getenv("AUTOSRE_TRACING", "off")).strip().lower()
    if mode in ("", "off", "false", "0"):
        _tracer = _provider = None
        return None
    if mode not in ("console", "otlp", "memory"):
        raise ValueError(f"AUTOSRE_TRACING must be off, console or otlp, got {mode!r}")

    # The SDK refuses to create real tracers while OTEL_SDK_DISABLED is true.
    os.environ["OTEL_SDK_DISABLED"] = "false"

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "autosre")}))
    exporter = None
    if mode == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif mode == "memory":
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    _provider = provider
    _tracer = provider.get_tracer("autosre")
    return exporter


def enabled() -> bool:
    return _tracer is not None


def shutdown() -> None:
    """Flushes batched spans; call before the process exits."""
    if _provider is not None:
        _provider.force_flush()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """A child of the current span (or a new trace). A no-op when tracing is
    off. An exception escaping the block is recorded on the span and
    re-raised unchanged -- tracing must never alter behaviour."""
    if _tracer is None:
        yield _NoOpSpan()
        return
    with _tracer.start_as_current_span(name) as raw:
        wrapped = _Span(raw)
        wrapped.set(**attributes)
        yield wrapped


def traced(name: str, describe: Optional[Callable[[Any], dict[str, Any]]] = None) -> Callable:
    """Decorator for graph nodes (state -> state). `describe` maps the node's
    RESULT to extra attributes; the incident id is always attached."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: Any) -> Any:
            with span(name, incident_id=state.get("incident_id")) as sp:
                result = fn(state)
                if describe is not None and enabled():
                    sp.set(**describe(result))
                return result

        return wrapper

    return decorator
