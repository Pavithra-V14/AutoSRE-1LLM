from __future__ import annotations

import pytest

from services.agent.benchmark_stats import ScenarioStats, status_counts, summarize_scenarios, wilson_interval


def _runs(scenario, *outcomes):
    return [{"scenario": scenario, "success": ok, "match_ratio": 1.0 if ok else 0.0} for ok in outcomes]


def test_pass_rate_and_mean_score():
    (stats,) = summarize_scenarios(_runs("a", True, False, True, True))
    assert (stats.runs, stats.passes, stats.pass_rate, stats.mean_score) == (4, 3, 0.75, 0.75)


@pytest.mark.parametrize("outcomes, status", [
    ((True, True, True), "stable_pass"),
    ((False, False, False), "stable_fail"),
    ((True, False, True), "flaky"),
    ((True,), "unmeasured"),   # one sample cannot show stability either way
    ((False,), "unmeasured"),
])
def test_status_classification(outcomes, status):
    (stats,) = summarize_scenarios(_runs("a", *outcomes))
    assert stats.status == status


def test_groups_by_scenario_preserving_first_seen_order():
    results = _runs("b", True) + _runs("a", False) + _runs("b", False)
    stats = summarize_scenarios(results)
    assert [s.scenario for s in stats] == ["b", "a"]
    assert [s.runs for s in stats] == [2, 1]


def test_status_counts():
    stats = [ScenarioStats("a", 3, 3, 1.0), ScenarioStats("b", 3, 0, 0.0), ScenarioStats("c", 3, 1, 0.3),
             ScenarioStats("d", 3, 2, 0.6)]
    assert status_counts(stats) == {"stable_pass": 1, "stable_fail": 1, "flaky": 2}


def test_empty_input():
    assert summarize_scenarios([]) == []
    assert ScenarioStats("a", 0, 0, 0.0).pass_rate == 0.0


# --- Wilson interval ------------------------------------------------------------------------------

def test_wilson_matches_known_value():
    low, high = wilson_interval(12, 18)  # 66.7% of 18
    assert low == pytest.approx(0.4375, abs=0.01)
    assert high == pytest.approx(0.8372, abs=0.01)


@pytest.mark.parametrize("successes, n", [(0, 10), (10, 10), (0, 1), (1, 1), (3, 7)])
def test_wilson_stays_within_zero_and_one_and_brackets_the_estimate(successes, n):
    low, high = wilson_interval(successes, n)
    assert 0.0 <= low <= successes / n <= high <= 1.0


def test_interval_narrows_with_more_runs():
    narrow = wilson_interval(60, 90)
    wide = wilson_interval(12, 18)  # same 66.7% rate, 5x fewer samples
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_no_samples_is_a_zero_interval_not_a_crash():
    assert wilson_interval(0, 0) == (0.0, 0.0)
