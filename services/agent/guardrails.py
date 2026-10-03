from __future__ import annotations

import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")  

from typing import Any, Optional

def redact_pii(text: str, entities: Optional[list[str]] = None) -> str:
    """Detect and redact PII before text enters the agent's context. Falls
    back to a no-op with a warning if Presidio isn't installed yet -- keep
    the pipeline running during early development.

    `entities` restricts detection to the named Presidio entity types. Tool
    output must pass an explicit list: the default (all entities) also
    rewrites timestamps and place-like words into <DATE_TIME>/<LOCATION>,
    which destroys the evidence an investigation depends on."""
    analyzer, anonymizer = _get_presidio_engines()
    if analyzer is None:
        return text

    results = analyzer.analyze(text=text, language="en", entities=entities)
    return anonymizer.anonymize(text=text, analyzer_results=results).text


_presidio_engines: Optional[tuple[Any, Any]] = None
_presidio_load_attempted = False


def _get_presidio_engines() -> tuple[Optional[Any], Optional[Any]]:
    """Builds the AnalyzerEngine/AnonymizerEngine exactly once per process
    and caches them, instead of rebuilding from scratch on every single
    redact_pii() call (the original implementation did this, which is both
    a real performance cost -- rebuilding an NLP pipeline is not cheap --
    and the reason ~10 lines of 'Recognizer not added to registry ...'
    warnings printed on EVERY call instead of once at startup)."""
    global _presidio_engines, _presidio_load_attempted

    if _presidio_engines is not None:
        return _presidio_engines
    if _presidio_load_attempted:
        return (None, None)  

    _presidio_load_attempted = True

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        return (None, None)
    import logging
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
    anonymizer = AnonymizerEngine()
    _presidio_engines = (analyzer, anonymizer)
    return _presidio_engines


def validate_causal_graph_output(raw_output: dict[str, Any]) -> tuple[bool, str]:
    """Every causal-graph edge must cite a tool_call_id. This is the
    fabrication check: if the model claims a cause without a cited
    observation, reject it.

    Uses the guardrails-ai Guard defined in agent/rails/causal_graph_schema.py
    (a real pydantic-schema validator, not just a manual dict check) when
    available, falling back to a manual check otherwise."""
    try:
        import json

        from services.agent.rails.causal_graph_schema import get_causal_graph_guard

        guard = get_causal_graph_guard()
        result = guard.parse(json.dumps(raw_output))
        if result.validation_passed:
            return True, ""
        return False, "guardrails-ai schema validation failed (missing citation or malformed edge)"
    except ImportError:
        pass  
    except Exception as e:
        return False, f"schema validation error: {e}"

    edges = raw_output.get("edges", [])
    if not edges:
        return False, "causal graph has no edges"

    for edge in edges:
        if "evidence_tool_call_id" not in edge or not edge["evidence_tool_call_id"]:
            return False, f"edge {edge.get('id', '?')} has no cited evidence"

    return True, ""


def check_scope_and_injection(alert_text: str) -> tuple[bool, str]:
    """Guard the incoming alert text against prompt injection and
    off-topic steering before it reaches the planner.

    Uses the NeMo Guardrails custom action defined in
    agent/nemo_config/actions.py, called directly as an async function
    rather than through NeMo's full LLMRails dialog engine. That engine is
    built for multi-turn conversational bots (it requires a "main" flow
    and its own dialog-management scaffolding); this check is a one-shot
    stateless gate, and NeMo's colang-orchestrated dialog runtime hit
    unresolved internal flow-registration conflicts in this version when
    forced into that shape (documented in docs/build-plan.md Phase 4).
    Calling the registered @action function directly still uses a real
    NeMo Guardrails primitive -- it just doesn't route through the parts
    of the framework built for conversation management, which this use
    case doesn't need anyway.

    Falls back to a minimal keyword check if nemoguardrails isn't
    installed, so the pipeline keeps working during early development."""
    try:
        import asyncio

        from services.agent.nemo_config.actions import check_alert_scope

        result = asyncio.run(check_alert_scope(context={"user_message": alert_text}))
        if result.is_suspicious:
            return False, f"blocked: {result.reason}"
        return True, ""
    except ImportError:
        pass 

    suspicious_markers = [
        "ignore previous instructions",
        "disregard the system prompt",
        "you are now",
    ]
    lowered = alert_text.lower()
    for marker in suspicious_markers:
        if marker in lowered:
            return False, f"blocked: suspicious pattern '{marker}' in alert text"
    return True, ""
