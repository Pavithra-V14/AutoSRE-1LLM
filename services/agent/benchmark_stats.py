"""Statistics for repeated benchmark runs.

An LLM agent is not deterministic: the same scenario scored 0.30, 0.60, 0.85
and 0.30 confidence across four runs. One run per scenario therefore cannot
tell a real regression from noise. Running each scenario N times gives:

  * a per-scenario pass rate and a stable/flaky classification, and
  * a confidence interval on the overall success rate, so "66.7% -> 61.1%"
    can be recognised as inside the noise (or not).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ScenarioStats:
    scenario: str
    runs: int
    passes: int
    mean_score: float

    @property
    def pass_rate(self) -> float:
        return self.passes / self.runs if self.runs else 0.0

    @property
    def status(self) -> str:
        """stable_pass: every run passed; stable_fail: none did; flaky: mixed.
        With a single run everything is 'unmeasured' -- one sample cannot
        show stability either way."""
        if self.runs < 2:
            return "unmeasured"
        if self.passes == self.runs:
            return "stable_pass"
        if self.passes == 0:
            return "stable_fail"
        return "flaky"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Preferred over the normal
    approximation because it stays inside [0, 1] and behaves at small n --
    a benchmark of ~19 scenarios is small n."""
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize_scenarios(results: Iterable[dict[str, Any]]) -> list[ScenarioStats]:
    """Groups run results by scenario, preserving first-seen order. Each
    result needs `scenario`, `success` and `match_ratio`."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["scenario"], []).append(result)

    stats = []
    for scenario, runs in grouped.items():
        stats.append(ScenarioStats(
            scenario=scenario,
            runs=len(runs),
            passes=sum(1 for r in runs if r["success"]),
            mean_score=round(sum(float(r.get("match_ratio", 0.0)) for r in runs) / len(runs), 3),
        ))
    return stats


def status_counts(stats: Iterable[ScenarioStats]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in stats:
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts
