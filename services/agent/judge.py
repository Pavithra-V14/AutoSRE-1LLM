"""LLM judge for benchmark scoring.

The original scorer split the ground-truth string into words and required
50% of them to appear as substrings in the agent's edge names. That rewards
vocabulary, not correctness: ``config_change -> pool_exhaustion ->
high_latency`` scored 0.33 against ``deploy:cfg-4471 -> pool_size=10 ->
connection_exhaustion -> latency_spike`` and FAILED despite being the right
answer. The judge compares *meaning*: does the agent's causal chain identify
the same root cause and mechanism as the ground truth?

Design notes:
- The judge sees only the ground truth and the agent's graph -- not the
  scenario's raw telemetry -- so it cannot re-investigate and rationalize.
- Success is the judge's categorical verdict ("correct"), not a thresholded
  float, so the pass/fail boundary is explicit and auditable.
- Set JUDGE_PROVIDER (groq | gemini | openrouter) to grade with a different
  model family than the agent runs on, avoiding self-preference bias.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from services.agent.llm_router import LLMRouter
from services.agent.structured import complete_json

VERDICTS = ("correct", "partial", "incorrect")
# Grading must be repeatable: the same graph should get the same verdict.
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))

JUDGE_SYSTEM = (
    "You are a strict, impartial grader of incident root-cause analyses. You "
    "compare an agent's causal graph against the ground-truth cause and "
    "judge MEANING, not wording."
)

RUBRIC = (
    "Grading rubric. Two things decide correctness: (1) the ROOT CAUSE -- "
    "the originating trigger (a specific deploy/config change, an external "
    "outage, a traffic event, ...) and (2) the MECHANISM -- how that "
    "trigger produced the failure (e.g. pool exhaustion, thundering herd, "
    "rate limiting).\n"
    '- "correct": the agent\'s graph names the same root cause AND the same '
    "mechanism as the ground truth. Different node names, casing, synonyms "
    "or granularity are fine (e.g. 'config_change' and 'deploy cfg-4471' "
    "are the same if they describe the same change). The agent MAY omit or "
    "merge intermediate hops and MAY stop at the first user-visible symptom "
    "(e.g. 'error rate spike' instead of 'crash loop', or 'latency' instead "
    "of 'checkout errors') -- that alone is NOT a reason to downgrade. "
    "Extra harmless detail is fine.\n"
    '- "partial": the root cause is right but the mechanism is missing, '
    "vague or wrong; OR only symptoms are named with no root cause; OR the "
    "graph mixes the correct cause with a materially wrong one.\n"
    '- "incorrect": the graph blames a different root cause, is empty, or '
    "asserts something the ground truth contradicts.\n"
    "Rules: judge only the causal content. Do NOT reward high stated "
    "confidence, long rationales or confident tone, and do not penalize "
    "naming style. If the agent invents an unrelated cause, it is not "
    "correct even if some symptoms overlap."
)


@dataclass(frozen=True)
class JudgeVerdict:
    verdict: str
    score: float
    rationale: str
    provider: str
    input_tokens: int
    output_tokens: int

    @property
    def success(self) -> bool:
        return self.verdict == "correct"


def _validate_verdict(data: dict[str, Any]) -> Optional[str]:
    if data.get("verdict") not in VERDICTS:
        return f'"verdict" must be one of {list(VERDICTS)}'
    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
        return '"score" must be a number between 0 and 1'
    if not isinstance(data.get("rationale"), str) or not data["rationale"].strip():
        return '"rationale" must be a non-empty string'
    return None


def _model_rationale(graph: dict[str, Any]) -> str:
    """The model's own rationale. The pipeline appends notes to
    confidence_rationale (fail-closed disclosures, correlation with OTHER
    incidents); grading those would penalize the agent for system text, so
    prefer the pre-annotation snapshot taken in synthesize_step."""
    return str(graph.get("model_rationale") or graph.get("confidence_rationale") or "(none)")


def _render_graph(graph: dict[str, Any]) -> str:
    edges = graph.get("edges") or []
    if not edges:
        return "(no causal edges)"
    return "\n".join(f"- {e.get('from', '?')} -> {e.get('to', '?')}" for e in edges)


class CausalGraphJudge:
    def __init__(self, router: Optional[LLMRouter] = None, prefer: Optional[str] = None) -> None:
        self._router = router or LLMRouter()
        self._prefer = prefer or os.getenv("JUDGE_PROVIDER") or None

    def evaluate(self, *, ground_truth: str, graph: dict[str, Any]) -> JudgeVerdict:
        """Raises ProviderError / StructuredOutputError when no verdict can
        be obtained -- the caller decides the fallback, so a degraded judge
        is never silently counted as a pass or a fail."""
        prompt = (
            f"{RUBRIC}\n\n"
            f"Ground-truth cause chain:\n{ground_truth}\n\n"
            f"Agent's causal graph (edges):\n{_render_graph(graph)}\n\n"
            f"Agent's stated rationale:\n{_model_rationale(graph)}\n\n"
            'Respond as JSON: {"verdict": "correct|partial|incorrect", '
            '"score": 0.0-1.0, "rationale": "one or two sentences"}\n\n'
            "Respond with ONLY raw JSON. Numbers must be plain digits."
        )
        result = complete_json(
            self._router,
            prompt,
            system=JUDGE_SYSTEM,
            validator=_validate_verdict,
            prefer=self._prefer,
            temperature=JUDGE_TEMPERATURE,
            stage="judge_json",
        )
        data = result.data
        return JudgeVerdict(
            verdict=data["verdict"],
            score=round(float(data["score"]), 3),
            rationale=data["rationale"].strip(),
            provider=result.provider,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
