"""JSON Schemas for provider-enforced structured output.

With a schema, Groq constrains decoding so the response CANNOT contain a
garbled token like ``0. nine`` or a missing field -- the failure class that
used to abort runs is removed at the source instead of repaired afterwards.
Strict mode requires every property listed in ``required`` and
``additionalProperties: false`` at every object.

The schemas enforce *shape*. Vocabulary, citations and calibration are still
checked by validators in graph.py, because a schema cannot express them (and a
provider that ignores the schema, such as the OpenRouter fallback, must not
be trusted to have followed it).
"""

from __future__ import annotations

from typing import Any

ACTIONS = ["none", "restart_service", "rollback_deploy"]

SYNTHESIS_SCHEMA: dict[str, Any] = {
    "name": "causal_graph",
    "schema": {
        "type": "object",
        "properties": {
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "evidence_tool_call_id": {"type": "string"},
                    },
                    "required": ["id", "from", "to", "evidence_tool_call_id"],
                    "additionalProperties": False,
                },
            },
            "confidence": {"type": "number"},
            "confidence_rationale": {"type": "string"},
            "recommended_action": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ACTIONS},
                    "service": {"type": ["string", "null"]},
                    "rationale": {"type": "string"},
                },
                "required": ["action", "service", "rationale"],
                "additionalProperties": False,
            },
        },
        "required": ["edges", "confidence", "confidence_rationale", "recommended_action"],
        "additionalProperties": False,
    },
}

JUDGE_SCHEMA: dict[str, Any] = {
    "name": "judge_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
            "score": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["verdict", "score", "rationale"],
        "additionalProperties": False,
    },
}
