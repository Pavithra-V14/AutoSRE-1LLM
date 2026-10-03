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


# --- baselines / regression gating ----------------------------------------------------------------------------

from services.agent.benchmark_stats import compare_to_baseline, results_payload  # noqa: E402


def _stats(**rates):
    """scenario=(passes, runs)"""
    return [ScenarioStats(name, runs, passes, passes / runs) for name, (passes, runs) in rates.items()]


def test_payload_round_trips_into_a_baseline():
    payload = results_payload(_stats(a=(3, 3), b=(1, 3)), runs_per_scenario=3)
    assert payload["overall"]["passes"] == 4 and payload["overall"]["runs"] == 6
    assert payload["scenarios"]["b"]["status"] == "flaky" and payload["scenarios"]["a"]["pass_rate"] == 1.0
    assert 0 < payload["overall"]["ci_low"] < payload["overall"]["rate"] < payload["overall"]["ci_high"] <= 1


def test_identical_results_have_no_regression():
    base = results_payload(_stats(a=(3, 3), b=(1, 3)), 3)
    comparison = compare_to_baseline(_stats(a=(3, 3), b=(1, 3)), base)
    assert not comparison.failed and comparison.regressions == []


def test_one_flaky_failure_in_three_runs_is_noise_not_a_regression():
    base = results_payload(_stats(a=(3, 3)), 3)
    assert not compare_to_baseline(_stats(a=(2, 3)), base).failed


def test_two_failures_in_three_runs_is_a_regression():
    base = results_payload(_stats(a=(3, 3)), 3)
    comparison = compare_to_baseline(_stats(a=(1, 3)), base)
    assert comparison.failed and comparison.regressions == [("a", 1.0, pytest.approx(1 / 3))]


def test_max_drop_is_configurable():
    base = results_payload(_stats(a=(3, 3)), 3)
    assert compare_to_baseline(_stats(a=(2, 3)), base, max_drop=0.2).failed


def test_improvements_are_reported_but_never_fail():
    base = results_payload(_stats(a=(0, 3)), 3)
    comparison = compare_to_baseline(_stats(a=(3, 3)), base)
    assert comparison.improvements and not comparison.failed


def test_new_and_missing_scenarios_are_informational():
    base = results_payload(_stats(a=(3, 3), old=(3, 3)), 3)
    comparison = compare_to_baseline(_stats(a=(3, 3), brand_new=(0, 3)), base)
    assert comparison.new_scenarios == ["brand_new"] and comparison.missing_scenarios == ["old"]
    assert not comparison.failed  # a smoke subset must not fail for scenarios it did not run


def test_overall_regression_needs_the_whole_interval_below_the_baseline():
    big_base = results_payload(_stats(**{f"s{i}": (3, 3) for i in range(10)}), 3)
    # 30 runs, 12 passing: CI roughly 25%-58%, entirely below a 100% baseline
    collapsed = compare_to_baseline(_stats(**{f"s{i}": (1, 3) if i < 6 else (2, 3) for i in range(10)}), big_base)
    assert collapsed.overall_regressed

    # a handful of runs cannot prove an overall drop: the intervals overlap
    tiny_base = results_payload(_stats(a=(3, 3)), 3)
    assert not compare_to_baseline(_stats(a=(1, 3)), tiny_base).overall_regressed


def test_overall_rate_is_compared_only_over_scenarios_both_runs_contain():
    base = results_payload(_stats(a=(3, 3), hard=(0, 3)), 3)  # baseline overall 50%
    comparison = compare_to_baseline(_stats(a=(3, 3)), base)  # smoke ran only the easy one
    assert comparison.baseline_rate == 1.0 and not comparison.failed


def test_empty_baseline_means_everything_is_new():
    comparison = compare_to_baseline(_stats(a=(3, 3)), {})
    assert comparison.new_scenarios == ["a"] and not comparison.failed


def test_a_failing_new_scenario_does_not_make_the_shared_ones_look_regressed():
    base = results_payload(_stats(**{f"s{i}": (3, 3) for i in range(10)}), 3)
    current = _stats(**{f"s{i}": (3, 3) for i in range(10)}, brand_new=(0, 3))
    comparison = compare_to_baseline(current, base)
    assert comparison.current_rate == 1.0 and not comparison.failed
