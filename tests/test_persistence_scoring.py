from __future__ import annotations

from services.agent import persistence


class _Table:
    def __init__(self, client):
        self.client = client

    def insert(self, row):
        self.client.attempts.append(row)
        self.row = row
        return self

    def execute(self):
        if self.client.reject_scoring_columns and "score_method" in self.row:
            raise Exception('column "score_method" of relation "benchmark_runs" does not exist')
        self.client.saved.append(self.row)


class _Client:
    def __init__(self, reject_scoring_columns):
        self.reject_scoring_columns = reject_scoring_columns
        self.attempts: list[dict] = []
        self.saved: list[dict] = []

    def table(self, name):
        assert name == "benchmark_runs"
        return _Table(self)


def _insert(monkeypatch, client, **extra):
    monkeypatch.setattr(persistence, "_client_or_none", lambda: client)
    persistence.insert_benchmark_run(
        run_id="00000000-0000-0000-0000-000000000000", scenario="db_timeout", success=True,
        ground_truth_match=True, total_cost_usd=0.001, cpso=0.0, steps_taken=6, **extra,
    )


def test_verdict_columns_are_saved_when_the_migration_is_applied(monkeypatch):
    client = _Client(reject_scoring_columns=False)
    _insert(monkeypatch, client, score_method="judge", score=0.95, score_rationale="same cause", run_index=2)

    (row,) = client.saved
    assert (row["score_method"], row["score"], row["score_rationale"], row["run_index"]) == ("judge", 0.95, "same cause", 2)


def test_old_database_still_gets_the_pass_fail_record(monkeypatch, capsys):
    """Before 002_benchmark_scoring.sql is applied the insert must degrade,
    not lose the run."""
    client = _Client(reject_scoring_columns=True)
    monkeypatch.setattr(persistence, "_warned_missing_scoring_columns", False)
    _insert(monkeypatch, client, score_method="judge", score=0.95, score_rationale="r", run_index=1)

    (row,) = client.saved
    assert row["success"] is True and "score_method" not in row
    assert "002_benchmark_scoring.sql" in capsys.readouterr().out


def test_the_migration_hint_is_printed_once_not_per_row(monkeypatch, capsys):
    client = _Client(reject_scoring_columns=True)
    monkeypatch.setattr(persistence, "_warned_missing_scoring_columns", False)
    _insert(monkeypatch, client, score_method="judge", score=1.0, score_rationale="r", run_index=1)
    _insert(monkeypatch, client, score_method="judge", score=1.0, score_rationale="r", run_index=2)
    assert capsys.readouterr().out.count("002_benchmark_scoring.sql") == 1


def test_unrelated_database_errors_are_not_swallowed_as_missing_columns(monkeypatch, capsys):
    class _Boom(_Client):
        def table(self, name):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(persistence, "_client_or_none", lambda: _Boom(False))
    persistence.insert_benchmark_run("id", "s", True, True, 0.0, 0.0, 1, score_method="judge")  # must not raise
    assert "connection refused" in capsys.readouterr().out


def test_callers_that_pass_no_scoring_fields_are_unaffected(monkeypatch):
    client = _Client(reject_scoring_columns=False)
    _insert(monkeypatch, client)
    assert client.saved[0]["score_method"] is None


def test_long_rationales_are_truncated(monkeypatch):
    client = _Client(reject_scoring_columns=False)
    _insert(monkeypatch, client, score_method="judge", score=1.0, score_rationale="x" * 5000, run_index=1)
    assert len(client.saved[0]["score_rationale"]) == 1000
