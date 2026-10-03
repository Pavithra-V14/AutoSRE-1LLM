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


# --- baselines and regression gating -------------------------------------------------------------------

BASELINE_VERSION = 1
# Per-scenario pass rates move in steps of 1/runs. With 3 runs one flaky
# failure is a 0.33 drop, which is noise; two is 0.67, which is not.
DEFAULT_MAX_DROP = 0.34


def results_payload(stats: list[ScenarioStats], runs_per_scenario: int) -> dict[str, Any]:
    """The machine-readable result of a benchmark run; also the baseline file
    format (a baseline is just a saved result you chose to trust)."""
    passes = sum(s.passes for s in stats)
    runs = sum(s.runs for s in stats)
    low, high = wilson_interval(passes, runs)
    return {
        "version": BASELINE_VERSION,
        "runs_per_scenario": runs_per_scenario,
        "overall": {
            "passes": passes, "runs": runs,
            "rate": round(passes / runs, 4) if runs else 0.0,
            "ci_low": round(low, 4), "ci_high": round(high, 4),
        },
        "scenarios": {
            s.scenario: {
                "passes": s.passes, "runs": s.runs, "pass_rate": round(s.pass_rate, 4),
                "mean_score": s.mean_score, "status": s.status,
            }
            for s in stats
        },
    }


@dataclass(frozen=True)
class Comparison:
    regressions: list[tuple[str, float, float]]    # (scenario, baseline rate, current rate)
    improvements: list[tuple[str, float, float]]
    new_scenarios: list[str]
    missing_scenarios: list[str]                   # in the baseline, not run now (e.g. --smoke)
    baseline_rate: float
    current_rate: float
    current_ci: tuple[float, float]
    overall_regressed: bool

    @property
    def failed(self) -> bool:
        return bool(self.regressions) or self.overall_regressed


def compare_to_baseline(
    stats: list[ScenarioStats], baseline: dict[str, Any], max_drop: float = DEFAULT_MAX_DROP
) -> Comparison:
    """Per scenario: a regression is a pass rate more than `max_drop` below
    the baseline's. Overall (over shared scenarios only): a regression only
    when the current rate's confidence interval lies entirely below the
    baseline's, i.e. the drop is bigger than noise at this sample size.
    Scenarios absent from the current run (a smoke subset) or from the
    baseline (new ones) are reported, never counted as failures."""
    base_scenarios: dict[str, Any] = baseline.get("scenarios", {})
    epsilon = 1e-9

    regressions: list[tuple[str, float, float]] = []
    improvements: list[tuple[str, float, float]] = []
    new: list[str] = []
    for item in stats:
        base = base_scenarios.get(item.scenario)
        if base is None:
            new.append(item.scenario)
            continue
        base_rate = float(base["pass_rate"])
        if item.pass_rate < base_rate - max_drop - epsilon:
            regressions.append((item.scenario, base_rate, item.pass_rate))
        elif item.pass_rate > base_rate + max_drop + epsilon:
            improvements.append((item.scenario, base_rate, item.pass_rate))

    ran = {s.scenario for s in stats}
    missing = sorted(set(base_scenarios) - ran)

    # Compare like with like: only scenarios present on BOTH sides. A brand-new
    # scenario that fails must not read as a regression in everything else, and
    # a smoke subset is judged against the same scenarios in the baseline.
    shared = [s for s in stats if s.scenario in base_scenarios]
    base_passes = sum(int(base_scenarios[s.scenario]["passes"]) for s in shared)
    base_runs = sum(int(base_scenarios[s.scenario]["runs"]) for s in shared)
    baseline_rate = (base_passes / base_runs) if base_runs else 0.0

    passes, runs = sum(s.passes for s in shared), sum(s.runs for s in shared)
    ci = wilson_interval(passes, runs)
    # Both rates are estimates. Call the drop real only when the intervals are
    # disjoint: the best the current run could plausibly be is still worse than
    # the worst the baseline could plausibly be.
    base_low = wilson_interval(base_passes, base_runs)[0]
    return Comparison(
        regressions=regressions,
        improvements=improvements,
        new_scenarios=new,
        missing_scenarios=missing,
        baseline_rate=baseline_rate,
        current_rate=(passes / runs) if runs else 0.0,
        current_ci=ci,
        overall_regressed=runs > 0 and base_runs > 0 and ci[1] < base_low - epsilon,
    )
