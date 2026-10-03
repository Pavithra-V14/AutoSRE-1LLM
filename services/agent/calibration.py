"""Is the judge any good? Measure it against labelled cases.

Every benchmark number rests on the judge's verdicts. Without a check against
independent labels, "66.7%" could mean the agent is mediocre or that the
judge is lenient or harsh. This module scores the judge on cases where the
right verdict is known:

  agreement  -- fraction of cases where the judge matched the label;
  kappa      -- Cohen's kappa, agreement beyond what chance would give
                (0.6+ is substantial, 0.8+ is near-perfect);
  confusion  -- which verdicts it swaps, e.g. grading "partial" as "correct"
                (a lenient judge) or the reverse (a harsh one).

The labels in benchmarks/judge_labels.json are a seed set written by the
developer. Replace and extend them with labels from people reviewing real
runs: the more independent the labelling, the more the numbers mean.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol

from services.agent.judge import VERDICTS


class _Judge(Protocol):
    def evaluate(self, *, ground_truth: str, graph: dict[str, Any]) -> Any: ...


def agreement(human: list[str], judged: list[str]) -> float:
    if not human:
        return 0.0
    return sum(1 for h, j in zip(human, judged) if h == j) / len(human)


def cohen_kappa(human: list[str], judged: list[str]) -> float:
    """(observed - expected) / (1 - expected). When chance agreement is
    already 1 (both raters always give the same single label) kappa is
    undefined; it is 1.0 if they agree and 0.0 otherwise."""
    n = len(human)
    if n == 0:
        return 0.0
    observed = agreement(human, judged)
    human_counts, judged_counts = Counter(human), Counter(judged)
    expected = sum((human_counts[label] / n) * (judged_counts[label] / n) for label in set(human) | set(judged))
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def confusion_matrix(human: list[str], judged: list[str]) -> dict[str, dict[str, int]]:
    matrix = {h: {j: 0 for j in VERDICTS} for h in VERDICTS}
    for h, j in zip(human, judged):
        matrix[h][j] += 1
    return matrix


@dataclass
class CaseResult:
    case_id: str
    label: str
    verdict: Optional[str]    # None if the judge could not produce one
    note: str = ""
    error: str = ""


@dataclass
class CalibrationReport:
    results: list[CaseResult]
    agreement: float
    kappa: float
    pass_agreement: float   # agreement on the only decision that matters: "correct" or not
    confusion: dict[str, dict[str, int]]
    unscored: int = 0
    disagreements: list[CaseResult] = field(default_factory=list)


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data["cases"]
    for case in cases:
        if case["label"] not in VERDICTS:
            raise ValueError(f"case {case.get('id')}: label must be one of {VERDICTS}, got {case['label']!r}")
    return cases


def _graph_from_case(case: dict[str, Any]) -> dict[str, Any]:
    edges = [
        {"id": f"e{i}", "from": e["from"], "to": e["to"], "evidence_tool_call_id": "1"}
        for i, e in enumerate(case.get("edges", []))
    ]
    rationale = case.get("rationale", "")
    return {"edges": edges, "confidence": 0.8, "confidence_rationale": rationale, "model_rationale": rationale}


def evaluate_judge(
    judge: _Judge,
    cases: Iterable[dict[str, Any]],
    *,
    retries: int = 0,
    wait_for: Optional[Callable[[], float]] = None,
    max_wait_seconds: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
) -> CalibrationReport:
    """Runs the judge over every case. A case the judge cannot grade (provider
    outage, unusable output) is counted as unscored and excluded from the
    statistics -- never silently treated as a match or a miss.

    On a free tier the limit is usually a rate window, not an outage: with
    `retries` and `wait_for` (seconds until a provider is available) a failed
    case waits for the window to lift -- up to `max_wait_seconds`, so a daily
    cap that will not lift today still ends the run promptly -- and is retried
    before being given up on."""
    results: list[CaseResult] = []
    for case in cases:
        attempt = 0
        while True:
            try:
                verdict = judge.evaluate(ground_truth=case["ground_truth"], graph=_graph_from_case(case)).verdict
                results.append(CaseResult(case["id"], case["label"], verdict, case.get("note", "")))
                break
            except Exception as e:
                wait = wait_for() if wait_for else 0.0
                if attempt < retries and 0 < wait <= max_wait_seconds:
                    attempt += 1
                    sleep(wait)
                    continue
                results.append(CaseResult(case["id"], case["label"], None, case.get("note", ""), error=str(e)[:120]))
                break

    scored = [r for r in results if r.verdict is not None]
    human = [r.label for r in scored]
    judged = [r.verdict for r in scored if r.verdict is not None]
    return CalibrationReport(
        results=results,
        agreement=agreement(human, judged),
        kappa=cohen_kappa(human, judged),
        pass_agreement=agreement([h == "correct" for h in human], [j == "correct" for j in judged]) if scored else 0.0,  # type: ignore[arg-type]
        confusion=confusion_matrix(human, judged),
        unscored=len(results) - len(scored),
        disagreements=[r for r in scored if r.verdict != r.label],
    )
