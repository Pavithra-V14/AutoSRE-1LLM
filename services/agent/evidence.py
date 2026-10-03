"""Does the cited evidence actually support the edge?

The existing citation check (graph._resolve_citations) proves a cited step
exists and returned data. That stops invented step numbers but not a real
citation attached to an unrelated claim ("pool_exhaustion", citing the
deploy list). This is a cheap, deterministic second check: at least one
meaningful word from the edge's nodes must appear in the cited observation.

It deliberately errs towards NOT flagging -- word matching is crude and
models legitimately use synonyms -- so a flag is a prompt for a human to look
(`weak_citations` in the graph, shown with the approval request), never a
reason to discard an answer.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Too generic to show support: nearly every incident observation contains them.
_STOPWORDS = frozenset({
    "error", "errors", "rate", "high", "spike", "increase", "increased", "service", "latency", "failure",
    "failures", "status", "response", "request", "requests", "issue", "problem", "unknown", "undetermined",
    "change", "mechanism", "symptom",
})
_STEM_LEN = 5
_WORD = re.compile(r"[a-z0-9]+")


def _stems(text: str) -> set[str]:
    return {w[:_STEM_LEN] for w in _WORD.findall(text.lower()) if len(w) >= 4 and w not in _STOPWORDS}


def _slug(node: str) -> str:
    return node.partition(":")[2] or node


def weak_citations(graph: dict[str, Any], steps_taken: list[dict[str, Any]]) -> list[str]:
    """Ids of edges whose cited observation shares no meaningful word with
    the edge's nodes. Edges with an unresolvable citation, or nodes made only
    of generic words, are skipped (other checks cover the former; there is
    nothing to compare for the latter)."""
    weak: list[str] = []
    for edge in graph.get("edges", []):
        citation = str(edge.get("evidence_tool_call_id", ""))
        if not citation.isdigit() or int(citation) >= len(steps_taken):
            continue
        step = steps_taken[int(citation)]
        if step.get("type") != "observation":
            continue

        wanted = _stems(_slug(str(edge.get("from", "")))) | _stems(_slug(str(edge.get("to", ""))))
        if not wanted:
            continue

        content = step.get("content", {})
        evidence = _stems(json.dumps(content.get("data"), default=str) + " " + str(content.get("tool", "")))
        if not (wanted & evidence):
            weak.append(str(edge.get("id", "?")))
    return weak
