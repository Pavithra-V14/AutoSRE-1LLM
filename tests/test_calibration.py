from __future__ import annotations

from pathlib import Path

import pytest

from services.agent.calibration import agreement, cohen_kappa, confusion_matrix, evaluate_judge, load_cases
from services.agent.judge import VERDICTS, JudgeVerdict

LABELS = Path(__file__).resolve().parent.parent / "benchmarks" / "judge_labels.json"


# --- the statistics ----------------------------------------------------------------------------------------

def test_agreement():
    assert agreement(["a", "b", "c"], ["a", "b", "x"]) == pytest.approx(2 / 3)
    assert agreement([], []) == 0.0


def test_kappa_is_one_for_perfect_agreement_and_zero_for_chance_level():
    labels = ["correct", "partial", "incorrect", "correct", "partial", "incorrect"]
    assert cohen_kappa(labels, labels) == 1.0
    # judge says "correct" regardless: agreement 2/6 but no better than chance
    assert cohen_kappa(labels, ["correct"] * 6) == pytest.approx(0.0)


def test_kappa_matches_a_hand_computed_value():
    human = ["correct"] * 4 + ["incorrect"] * 4
    judged = ["correct"] * 3 + ["incorrect"] + ["incorrect"] * 3 + ["correct"]
    # observed 6/8 = .75; expected .5*.5 + .5*.5 = .5; kappa = (.75-.5)/(1-.5) = .5
    assert cohen_kappa(human, judged) == pytest.approx(0.5)


def test_kappa_is_below_zero_when_the_judge_is_worse_than_chance():
    assert cohen_kappa(["correct", "incorrect"] * 3, ["incorrect", "correct"] * 3) < 0


@pytest.mark.parametrize("human, judged, expected", [
    (["correct"] * 3, ["correct"] * 3, 1.0),    # everyone always says the same, and agrees
    (["correct"] * 3, ["partial"] * 3, 0.0),    # constant but different
    ([], [], 0.0),
])
def test_kappa_degenerate_cases_do_not_divide_by_zero(human, judged, expected):
    assert cohen_kappa(human, judged) == expected


def test_confusion_matrix_counts_rows_as_human_and_columns_as_judge():
    matrix = confusion_matrix(["correct", "correct", "partial"], ["correct", "partial", "correct"])
    assert matrix["correct"] == {"correct": 1, "partial": 1, "incorrect": 0}
    assert matrix["partial"]["correct"] == 1


# --- evaluating a judge ----------------------------------------------------------------------------------------

class ScriptedJudge:
    def __init__(self, verdicts):
        self.verdicts = dict(verdicts)

    def evaluate(self, *, ground_truth, graph):
        verdict = self.verdicts[graph["edges"][0]["from"]] if graph["edges"] else self.verdicts["<empty>"]
        if isinstance(verdict, Exception):
            raise verdict
        return JudgeVerdict(verdict, 0.5, "r", "groq", 1, 1)


def _case(case_id, label, source="x:a"):
    return {"id": case_id, "label": label, "ground_truth": "gt", "edges": [{"from": source, "to": "symptom:s"}], "note": case_id}


def test_report_flags_leniency_and_harshness():
    cases = [_case("lenient", "partial", "x:1"), _case("harsh", "correct", "x:2"), _case("fine", "incorrect", "x:3")]
    report = evaluate_judge(ScriptedJudge({"x:1": "correct", "x:2": "partial", "x:3": "incorrect"}), cases)

    assert report.agreement == pytest.approx(1 / 3)
    assert [d.case_id for d in report.disagreements] == ["lenient", "harsh"]
    assert report.confusion["partial"]["correct"] == 1 and report.confusion["correct"]["partial"] == 1


def test_pass_agreement_only_cares_about_correct_versus_not():
    cases = [_case("a", "partial", "x:1"), _case("b", "incorrect", "x:2")]
    report = evaluate_judge(ScriptedJudge({"x:1": "incorrect", "x:2": "partial"}), cases)
    assert report.agreement == 0.0 and report.pass_agreement == 1.0  # both "not correct"


def test_ungradeable_cases_are_excluded_not_counted_as_matches_or_misses():
    cases = [_case("ok", "correct", "x:1"), _case("down", "correct", "x:2")]
    report = evaluate_judge(ScriptedJudge({"x:1": "correct", "x:2": RuntimeError("provider down")}), cases)

    assert report.unscored == 1 and report.agreement == 1.0
    assert [r.error for r in report.results if r.verdict is None] == ["provider down"]


def test_empty_edges_are_passed_to_the_judge_as_an_empty_graph():
    case = {"id": "e", "label": "incorrect", "ground_truth": "gt", "edges": []}
    assert evaluate_judge(ScriptedJudge({"<empty>": "incorrect"}), [case]).agreement == 1.0


# --- the seed label file ------------------------------------------------------------------------------------------

def test_seed_set_is_valid_and_covers_every_verdict():
    cases = load_cases(LABELS)
    assert len(cases) >= 12
    assert {c["label"] for c in cases} == set(VERDICTS)
    assert len({c["id"] for c in cases}) == len(cases)


def test_seed_set_is_honest_about_who_labelled_it():
    import json

    assert "NOT by independent reviewers" in json.loads(LABELS.read_text(encoding="utf-8"))["_about"]


def test_invalid_label_is_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"cases": [{"id": "x", "label": "maybe", "ground_truth": "g", "edges": []}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="label must be one of"):
        load_cases(bad)


def test_seed_cases_cover_the_scenario_shapes_the_benchmark_has():
    ids = {c["id"] for c in load_cases(LABELS)}
    assert {"blames-the-decoy", "echoes-injection", "two-causes-one-found", "two-causes-both-found",
            "pool-different-vocabulary"} <= ids
