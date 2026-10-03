from __future__ import annotations

import json
from datetime import datetime

import pytest

from fakes import observation, plan
from services.agent import context
from services.agent.context import (
    compact_steps,
    derive_signals,
    describe_series,
    dumps_compact,
    now_line,
    parse_ts,
    system_context,
    topology_block,
)


# --- describe_series ------------------------------------------------------------------------------------

def test_sustained_rise_reports_onset_ratio_and_state():
    line = describe_series("checkout.p99", [120, 130, 125, 480, 510, 495])
    assert "baseline ~125" in line and "peak 510" in line and "4.1x" in line
    assert "rises at point 4 of 6" in line and "3 of 6 points elevated" in line
    assert "still elevated" in line


def test_one_point_spike_is_called_a_transient_blip():
    """The flaky_test_false_alarm shape: the signal the agent needs to stay unsure."""
    line = describe_series("auth.p99", [95, 98, 340, 100, 97])
    assert "single-point blip" in line and "transient" in line


def test_zero_baseline_is_reported_as_from_zero():
    line = describe_series("checkout.pods_pending", [0, 0, 6, 6, 6])
    assert "from 0" in line and "rises at point 3 of 5" in line


def test_tiny_noise_at_exactly_twice_baseline_is_not_the_onset():
    line = describe_series("svc.error_rate", [0.001, 0.001, 0.002, 0.08, 0.09, 0.085])
    assert "rises at point 4 of 6" in line


def test_recovery_is_reported():
    assert "recovered by the last point" in describe_series("x", [100, 100, 500, 500, 100, 100])


def test_flat_series_has_no_rise():
    assert "no significant rise" in describe_series("x", [100, 101, 99, 100, 102])


@pytest.mark.parametrize("values", [[1, 2], [], ["a", "b", "c"], [1, None, 3, 4], [True, False, True]])
def test_unusable_series_returns_none(values):
    assert describe_series("x", values) is None


# --- derived signals from observations --------------------------------------------------------------------

def _obs(tool, data):
    return {"type": "observation", "content": {"tool": tool, "status": "ok", "data": data, "role": "required", "disclosure": None}}


DEPLOY_AT = "2026-10-03 16:00:00"


def _deploy_and_logs(*log_rows, deploy_ts=DEPLOY_AT):
    return [
        _obs("get_deploy_history", {"deploys": [{"id": "cfg-4471", "ts": deploy_ts, "service": "checkout", "change": "db.pool_size: 50 -> 10"}]}),
        _obs("get_logs", {"logs": [{"ts": ts, "level": lvl, "message": "m"} for ts, lvl in log_rows]}),
    ]


def test_deploy_followed_by_errors_reports_the_gap():
    lines = derive_signals(_deploy_and_logs(("2026-10-03 16:04:00", "ERROR"), ("2026-10-03 16:09:00", "ERROR")))
    (line,) = lines
    assert "cfg-4471" in line and "first ERROR log 4 min after it" in line
    assert "no error-level logs before it" in line and "9 min before the newest log line" in line


def test_errors_that_predate_the_deploy_are_called_out():
    (line,) = derive_signals(_deploy_and_logs(("2026-10-03 15:50:00", "ERROR"), ("2026-10-03 16:05:00", "ERROR")))
    assert "1 error-level log(s) came BEFORE it" in line


def test_deploy_after_all_errors_cannot_be_the_cause():
    (line,) = derive_signals(_deploy_and_logs(("2026-10-03 15:50:00", "FATAL")))
    assert "cannot have caused them" in line


def test_empty_deploy_history_is_stated_explicitly():
    steps = [_obs("get_deploy_history", {"deploys": []})]
    assert derive_signals(steps) == ["deploy history: no deploys or config changes in the window"]


def test_nothing_to_derive_before_any_observation():
    assert derive_signals([plan("get_logs")]) == []
    assert system_context("alert", [plan("get_logs")]).startswith("Current time:")


def test_failed_observations_are_ignored():
    bad = {"type": "observation", "content": {"tool": "get_metrics", "status": "unreachable", "data": None}}
    assert derive_signals([bad]) == []


def test_metric_and_deploy_signals_combine():
    steps = [_obs("get_metrics", {"checkout-service": {"p99_latency_ms": [120, 130, 125, 480, 510, 495]}})] + _deploy_and_logs(
        ("2026-10-03 16:04:00", "ERROR"))
    lines = derive_signals(steps)
    assert any("checkout-service.p99_latency_ms" in l for l in lines) and any("cfg-4471" in l for l in lines)


# --- time -----------------------------------------------------------------------------------------------------

def test_now_line_uses_the_tool_timestamp_format():
    stamp = now_line().split("Current time: ")[1].split(" (")[0]
    assert datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")


@pytest.mark.parametrize("value, ok", [("2026-10-03 16:36:13.872779", True), ("2026-10-03T16:36:13Z", True),
                                       ("nonsense", False), (None, False), (123, False)])
def test_parse_ts(value, ok):
    assert (parse_ts(value) is not None) is ok


# --- compaction -----------------------------------------------------------------------------------------------

def test_step_numbers_survive_compaction_because_citations_depend_on_them():
    steps = [plan("get_metrics"), observation("get_metrics"), plan("get_logs"), observation("get_logs")]
    assert [e["step"] for e in compact_steps(steps)] == [0, 1, 2, 3]
    assert [e["step"] for e in compact_steps(steps, start=2)] == [2, 3]


def test_nulls_and_empty_params_are_dropped():
    entry = compact_steps([plan("get_logs", {}), observation("get_logs", "unreachable")])
    assert "params" not in entry[0]
    assert "data" not in entry[1] and entry[1]["disclosure"]


def test_repeated_log_lines_collapse_into_one_with_a_count():
    logs = [{"ts": f"2026-10-03 16:0{i}:00", "level": "ERROR", "message": f"pool exhausted after {i * 10} ms"} for i in range(5)]
    logs.append({"ts": "2026-10-03 16:09:00", "level": "WARN", "message": "something else"})
    step = {"type": "observation", "content": {"tool": "get_logs", "status": "ok", "data": {"logs": logs}}}

    compacted = compact_steps([step])[0]["data"]["logs"]

    assert len(compacted) == 2
    assert compacted[0]["count"] == 5 and compacted[0]["last_ts"] == "2026-10-03 16:04:00"
    assert "count" not in compacted[1]


def test_distinct_log_lines_are_capped_with_a_note():
    logs = [{"ts": "t", "level": "INFO", "message": f"unique message {chr(97 + i % 26)}{chr(97 + i // 26)}"} for i in range(40)]
    step = {"type": "observation", "content": {"tool": "get_logs", "status": "ok", "data": {"logs": logs}}}
    compacted = compact_steps([step])[0]["data"]["logs"]
    assert len(compacted) == context.MAX_LOG_LINES + 1
    assert "omitted" in compacted[-1]["note"]


def test_long_strings_are_clipped():
    step = {"type": "observation", "content": {"tool": "get_logs", "status": "ok", "data": {"logs": [{"message": "x" * 900}]}}}
    message = compact_steps([step])[0]["data"]["logs"][0]["message"]
    assert len(message) < 400 and message.endswith("[truncated]")


def test_injection_flag_survives_compaction():
    step = {"type": "observation", "content": {"tool": "get_logs", "status": "ok", "data": {}, "injection_suspected": True}}
    assert compact_steps([step])[0]["injection_suspected"] is True


def test_compact_prompt_is_substantially_smaller_than_the_old_indented_dump():
    steps = [plan("get_deploy_history"), observation("get_deploy_history"), plan("get_metrics"),
             _obs("get_metrics", {"checkout-service": {"p99_latency_ms": [120, 130, 125, 480, 510, 495],
                                                       "error_rate": [0.001, 0.001, 0.002, 0.08, 0.09, 0.085]}}),
             plan("get_logs"), _obs("get_logs", {"logs": [{"ts": "2026-10-03 16:00:00", "level": "ERROR",
                                                           "message": "connection pool exhausted"}] * 6})]
    old = json.dumps(steps, indent=2)
    new = dumps_compact(compact_steps(steps))
    assert len(new) < 0.6 * len(old)


# --- topology -----------------------------------------------------------------------------------------------

def test_topology_shows_both_directions_for_services_in_the_alert():
    block = topology_block("ALERT order-service: error rate above 30%", [])
    assert "order-service: depends on [inventory-service, orders-db]" in block
    assert "relied on by [checkout-service]" in block
    assert "NOT evidence" in block


def test_external_dependencies_are_labelled():
    assert "payment-gateway (external)" in topology_block("ALERT checkout-service: errors", [])


def test_services_seen_in_metrics_are_included_even_if_not_in_the_alert():
    steps = [_obs("get_metrics", {"inventory-service": {"p99_latency_ms": [1, 2, 3]}})]
    assert "inventory-service" in topology_block("ALERT something is slow", steps)


def test_unknown_services_produce_no_block():
    assert topology_block("ALERT mystery-service: errors", []) == ""


def test_topology_path_can_be_overridden(tmp_path, monkeypatch):
    custom = tmp_path / "t.json"
    custom.write_text(json.dumps({"services": {"alpha": {"depends_on": ["beta"]}}, "external": []}), encoding="utf-8")
    monkeypatch.setenv("TOPOLOGY_PATH", str(custom))
    assert "alpha: depends on [beta]" in topology_block("ALERT alpha down", [])


def test_missing_topology_file_degrades_to_no_block(monkeypatch):
    monkeypatch.setenv("TOPOLOGY_PATH", "/nonexistent/topology.json")
    assert topology_block("ALERT checkout-service: errors", []) == ""
