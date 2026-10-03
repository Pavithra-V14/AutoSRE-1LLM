"""run_benchmark.main() end to end, with the agent, judge and database faked."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import run_benchmark  # noqa: E402
from services.agent.judge import JudgeVerdict  # noqa: E402
from services.simulator.scenarios import SCENARIOS, SMOKE_SCENARIOS  # noqa: E402


class FakeJudge:
    """Judges every causal run correct unless the test says otherwise."""
    verdict = "correct"

    def evaluate(self, *, ground_truth, graph):
        return JudgeVerdict(self.verdict, 0.9 if self.verdict == "correct" else 0.4, "because", "groq", 100, 10)


@pytest.fixture
def bench(monkeypatch):
    saved: list[dict] = []
    ran: list[str] = []

    def fake_run_incident(alert_text, scenario=None, skip_dedup=False):
        ran.append(scenario)
        spec = SCENARIOS[scenario]
        low = spec.get("expects_low_confidence") or spec.get("expects_fail_closed")
        rationale = " ".join(spec.get("degraded_tools", [])) or "r"
        return {
            "incident_id": "inc", "alert_text": alert_text, "steps_taken": [], "hypothesis": None,
            "confidence": 0.3 if low else 0.9, "needs_approval": False, "done": True,
            "synthesis_provider": "groq", "synthesis_input_tokens": 100, "synthesis_output_tokens": 20,
            "causal_graph": {"edges": [{"id": "e1", "from": "deploy:x", "to": "mechanism:y", "evidence_tool_call_id": "1"}],
                             "confidence": 0.3 if low else 0.9, "confidence_rationale": rationale,
                             "model_rationale": rationale},
        }

    monkeypatch.setattr(run_benchmark, "run_incident", fake_run_incident)
    monkeypatch.setattr(run_benchmark, "CausalGraphJudge", FakeJudge)
    monkeypatch.setattr(run_benchmark, "insert_benchmark_run", lambda **kw: saved.append(kw))
    monkeypatch.setattr(run_benchmark, "record_benchmark_outcome", lambda *a, **k: None)
    FakeJudge.verdict = "correct"

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["run_benchmark.py", *argv])
        run_benchmark.main()

    run.saved, run.ran = saved, ran
    return run


def test_smoke_runs_exactly_the_smoke_subset(bench):
    bench("--smoke")
    assert bench.ran == SMOKE_SCENARIOS


def test_the_agent_is_given_the_symptom_alert_never_the_description(bench, monkeypatch):
    alerts = []
    original = run_benchmark.run_incident

    def spy(alert_text, **kw):
        alerts.append(alert_text)
        return original(alert_text, **kw)

    monkeypatch.setattr(run_benchmark, "run_incident", spy)
    bench("--scenario", "db_timeout")
    assert alerts == [SCENARIOS["db_timeout"]["alert"]]
    assert SCENARIOS["db_timeout"]["description"] not in alerts[0]


def test_runs_repeats_each_scenario_and_numbers_the_rows(bench):
    bench("--scenario", "db_timeout", "--runs", "3")
    assert bench.ran == ["db_timeout"] * 3
    assert [row["run_index"] for row in bench.saved] == [1, 2, 3]


def test_verdict_and_method_are_persisted(bench):
    bench("--scenario", "db_timeout")
    (row,) = bench.saved
    assert row["score_method"] == "judge" and row["score"] == 0.9 and "because" in row["score_rationale"]


def test_flag_scenarios_are_persisted_as_flags(bench):
    bench("--scenario", "flaky_test_false_alarm")
    assert bench.saved[0]["score_method"] == "flags" and bench.saved[0]["success"] is True


def test_scenario_and_smoke_are_mutually_exclusive(bench):
    with pytest.raises(SystemExit) as exc:
        bench("--scenario", "db_timeout", "--smoke")
    assert exc.value.code == 2


def test_runs_must_be_positive(bench):
    with pytest.raises(SystemExit) as exc:
        bench("--runs", "0")
    assert exc.value.code == 2


def test_output_file_is_written_with_per_scenario_stats(bench, tmp_path):
    out = tmp_path / "nested" / "results.json"
    bench("--scenario", "db_timeout", "--runs", "2", "--output", str(out))

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["runs_per_scenario"] == 2
    assert payload["overall"]["passes"] == 2 and payload["scenarios"]["db_timeout"]["status"] == "stable_pass"


def test_min_success_rate_gate_exits_1(bench):
    FakeJudge.verdict = "incorrect"
    with pytest.raises(SystemExit) as exc:
        bench("--scenario", "db_timeout", "--min-success-rate", "0.5")
    assert exc.value.code == 1


def test_missing_baseline_is_a_notice_not_a_failure(bench, tmp_path, capsys):
    bench("--scenario", "db_timeout", "--baseline", str(tmp_path / "absent.json"))
    assert "no baseline" in capsys.readouterr().out


def test_baseline_round_trip_then_a_regression_fails(bench, tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    bench("--scenario", "db_timeout", "--runs", "3", "--save-baseline", str(baseline))
    assert json.loads(baseline.read_text(encoding="utf-8"))["scenarios"]["db_timeout"]["pass_rate"] == 1.0

    bench("--scenario", "db_timeout", "--runs", "3", "--baseline", str(baseline))  # same behaviour: passes

    FakeJudge.verdict = "incorrect"  # the agent got worse
    with pytest.raises(SystemExit) as exc:
        bench("--scenario", "db_timeout", "--runs", "3", "--baseline", str(baseline))
    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "REGRESSION  db_timeout: pass rate 100% -> 0%" in out


def test_smoke_against_a_full_baseline_does_not_fail_for_scenarios_it_skipped(bench, tmp_path, capsys):
    baseline = tmp_path / "full.json"
    bench("--runs", "1", "--save-baseline", str(baseline))
    assert len(json.loads(baseline.read_text(encoding="utf-8"))["scenarios"]) == len(SCENARIOS)

    bench("--smoke", "--baseline", str(baseline))
    assert "not run this time" in capsys.readouterr().out


def test_few_runs_against_a_baseline_prints_the_noise_warning(bench, tmp_path, capsys):
    baseline = tmp_path / "b.json"
    bench("--scenario", "db_timeout", "--save-baseline", str(baseline))
    bench("--scenario", "db_timeout", "--baseline", str(baseline))
    assert "noisy" in capsys.readouterr().out


def test_smoke_list_covers_every_kind_of_behaviour():
    kinds = {
        "causal": any(not SCENARIOS[n].get("expects_low_confidence") and not SCENARIOS[n].get("expects_fail_closed")
                      and "forbidden_terms" not in SCENARIOS[n] for n in SMOKE_SCENARIOS),
        "low_confidence": any(SCENARIOS[n].get("expects_low_confidence") for n in SMOKE_SCENARIOS),
        "fail_closed": any(SCENARIOS[n].get("expects_fail_closed") for n in SMOKE_SCENARIOS),
        "injection": any("forbidden_terms" in SCENARIOS[n] for n in SMOKE_SCENARIOS),
        "decoy": any("forbidden_edge_terms" in SCENARIOS[n] for n in SMOKE_SCENARIOS),
        "degraded": any(SCENARIOS[n].get("degraded_tools") and not SCENARIOS[n].get("expects_fail_closed") for n in SMOKE_SCENARIOS),
    }
    assert all(kinds.values()), kinds
