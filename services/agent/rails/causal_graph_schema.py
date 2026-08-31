from __future__ import annotations

import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true") 

from pydantic import BaseModel, Field


class CausalEdge(BaseModel):
    id: str
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    evidence_tool_call_id: str 


class CausalGraphSchema(BaseModel):
    edges: list[CausalEdge]
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_rationale: str = Field(
        default="",
        description=(
            "Item 3: why the confidence landed where it did -- e.g. which "
            "evidence supports it, or what's missing/uncertain. Optional "
            "(defaults to empty) so causal graphs produced before this field "
            "existed still validate."
        ),
    )


def get_causal_graph_guard():
    """Lazily constructed -- importing guardrails-ai at module load time
    would slow down every import of agent.guardrails even when this
    validator specifically isn't needed yet."""
    from guardrails import Guard

    return Guard.for_pydantic(output_class=CausalGraphSchema)
