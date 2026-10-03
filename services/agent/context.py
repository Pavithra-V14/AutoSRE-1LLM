"""Context engineering: what the model is shown, and in what shape.

Three jobs, all deterministic (no LLM calls):

  derive_signals  -- compute what code can compute reliably (when a metric
                     left its baseline, whether a spike was a one-point blip,
                     how long after a deploy the errors began) so the model
                     reasons about conclusions instead of re-deriving them
                     from raw number arrays;
  compact_steps   -- the investigation trace in a fraction of the tokens
                     (no indentation/nulls, repeated log lines collapsed);
  system_context  -- the current time and the service dependency map. This
                     is background, explicitly labelled as NOT evidence.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

# --- time ------------------------------------------------------------------------------------

_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def now_line() -> str:
    """Same clock and format as tool timestamps, so the model can build a
    valid `since` instead of guessing at ISO-8601 variants."""
    return f"Current time: {datetime.now().strftime(_TS_FORMAT)} (same clock and format as tool timestamps)"


def parse_ts(value: Any) -> Optional[datetime]:
    """Tolerant timestamp parse; returns a naive datetime or None."""
    if not isinstance(value, str):
        return None
    text = value.strip().replace("T", " ").replace("Z", "")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=None)


# --- derived signals -----------------------------------------------------------------------------

ELEVATION_RATIO = 2.0          # strictly greater than this x baseline counts as elevated
MIN_SUSTAINED_POINTS = 2       # fewer elevated points than this is a blip, not an incident
_ERROR_LEVELS = ("ERROR", "FATAL", "CRITICAL")


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def describe_series(name: str, values: list[float]) -> Optional[str]:
    """One line about a metric series, or None if it is not a usable series.

    Baseline = median of the first third (at least 2 points); elevated =
    strictly above ELEVATION_RATIO x baseline (any positive value when the
    baseline is 0). Series carry no timestamps, so positions are reported as
    "point i of n"."""
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    n = len(numbers)
    if n < 3 or n != len(values):
        return None

    baseline = statistics.median(numbers[: max(2, n // 3)])
    threshold = baseline * ELEVATION_RATIO
    elevated = [i for i, v in enumerate(numbers) if v > threshold and v > 0]
    peak = max(numbers)

    if not elevated:
        return f"{name}: no significant rise (baseline ~{_fmt(baseline)}, peak {_fmt(peak)})"

    ratio = f"{peak / baseline:.1f}x" if baseline > 0 else "from 0"
    if len(elevated) < MIN_SUSTAINED_POINTS:
        return (
            f"{name}: single-point blip (peak {_fmt(peak)} vs baseline ~{_fmt(baseline)}, {ratio}) "
            "that returned to baseline -- looks transient"
        )

    onset = elevated[0]
    recovered = numbers[-1] <= threshold
    tail = "recovered by the last point" if recovered else "still elevated at the last point"
    return (
        f"{name}: baseline ~{_fmt(baseline)} -> peak {_fmt(peak)} ({ratio}); rises at point {onset + 1} of {n}, "
        f"{len(elevated)} of {n} points elevated, {tail}"
    )


def _ok_observations(steps_taken: list[dict[str, Any]], tool: str) -> list[dict[str, Any]]:
    return [
        s["content"] for s in steps_taken
        if s["type"] == "observation" and s["content"].get("tool") == tool and s["content"].get("status") == "ok"
        and isinstance(s["content"].get("data"), dict)
    ]


def _metric_lines(steps_taken: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for obs in _ok_observations(steps_taken, "get_metrics"):
        for service, series in obs["data"].items():
            if not isinstance(series, dict):
                continue
            for metric, values in series.items():
                if isinstance(values, list):
                    line = describe_series(f"{service}.{metric}", values)
                    if line and line not in lines:
                        lines.append(line)
    return lines


def _deploy_log_lines(steps_taken: list[dict[str, Any]]) -> list[str]:
    deploy_obs = _ok_observations(steps_taken, "get_deploy_history")
    log_obs = _ok_observations(steps_taken, "get_logs")
    if not deploy_obs:
        return []

    deploys = [d for obs in deploy_obs for d in obs["data"].get("deploys", []) if isinstance(d, dict)]
    if not deploys:
        return ["deploy history: no deploys or config changes in the window"]

    entries = []
    for obs in log_obs:
        for entry in obs["data"].get("logs", []):
            ts = parse_ts(entry.get("ts")) if isinstance(entry, dict) else None
            if ts:
                entries.append((ts, str(entry.get("level", "")).upper(), entry))
    entries.sort(key=lambda e: e[0])
    errors = [e for e in entries if e[1] in _ERROR_LEVELS]

    lines: list[str] = []
    for deploy in deploys:
        deployed_at = parse_ts(deploy.get("ts"))
        label = f"deploy {deploy.get('id', '?')} ({str(deploy.get('change', ''))[:80]})"
        if deployed_at is None:
            continue
        before = sum(1 for e in errors if e[0] < deployed_at)
        after = next((e for e in errors if e[0] >= deployed_at), None)
        if after:
            gap = (after[0] - deployed_at).total_seconds() / 60
            note = f"first {after[1]} log {gap:.0f} min after it" + (
                f"; {before} error-level log(s) came BEFORE it" if before else "; no error-level logs before it"
            )
        elif errors:
            note = "all error-level logs predate it -- it cannot have caused them"
        else:
            note = "no error-level logs seen"
        newest = entries[-1][0] if entries else None
        age = f", {(newest - deployed_at).total_seconds() / 60:.0f} min before the newest log line" if newest else ""
        lines.append(f"{label} at {deployed_at.strftime(_TS_FORMAT)}: {note}{age}")
    return lines


def derive_signals(steps_taken: list[dict[str, Any]]) -> list[str]:
    """Facts computed from the observations gathered so far. Empty until
    there is something to compute from."""
    return _metric_lines(steps_taken) + _deploy_log_lines(steps_taken)


def derived_signals_block(steps_taken: list[dict[str, Any]]) -> str:
    lines = derive_signals(steps_taken)
    if not lines:
        return ""
    return "Derived signals (computed by code from the observations above -- reliable):\n" + "\n".join(f"- {l}" for l in lines)


# --- compact trace ---------------------------------------------------------------------------------

MAX_LOG_LINES = 25
MAX_STRING_CHARS = 300
_DIGITS = re.compile(r"\d+")


def _clip(text: Any) -> Any:
    if isinstance(text, str) and len(text) > MAX_STRING_CHARS:
        return text[:MAX_STRING_CHARS] + "...[truncated]"
    return text


def _compact_logs(logs: list[Any]) -> list[Any]:
    """Collapses repeats of the same message (ignoring numbers) into one line
    with a count and the first/last timestamp; keeps order of first sight."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    passthrough: list[Any] = []
    for entry in logs:
        if not isinstance(entry, dict):
            passthrough.append(_clip(entry))
            continue
        key = (str(entry.get("level", "")), _DIGITS.sub("#", str(entry.get("message", ""))))
        group = groups.get(key)
        if group is None:
            groups[key] = {"ts": entry.get("ts"), "level": entry.get("level"),
                           "message": _clip(entry.get("message")), "count": 1}
        else:
            group["count"] += 1
            group["last_ts"] = entry.get("ts")
    compacted = [{k: v for k, v in g.items() if not (k == "count" and v == 1)} for g in groups.values()]
    omitted = len(compacted) - MAX_LOG_LINES
    compacted = compacted[:MAX_LOG_LINES]
    if omitted > 0:
        compacted.append({"note": f"{omitted} further distinct log line(s) omitted"})
    return compacted + passthrough


def _compact_data(data: Any) -> Any:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            out[key] = _compact_logs(value) if key == "logs" and isinstance(value, list) else _compact_data(value)
        return out
    if isinstance(data, list):
        return [_compact_data(v) for v in data]
    return _clip(data)


def compact_steps(steps_taken: list[dict[str, Any]], start: int = 0) -> list[dict[str, Any]]:
    """Trace entries for prompts. `step` is the entry's index in
    steps_taken -- citations (evidence_tool_call_id) refer to it, so it must
    survive compaction unchanged. Nulls, repeated roles and verbose
    reasoning are dropped; nothing the model needs to cite is."""
    compact: list[dict[str, Any]] = []
    for index, step in enumerate(steps_taken):
        if index < start:
            continue
        content = step.get("content", {})
        if step["type"] == "plan":
            entry: dict[str, Any] = {"step": index, "type": "plan", "tool": content.get("tool"), "role": content.get("role")}
            if content.get("params"):
                entry["params"] = content["params"]
            if content.get("reasoning"):
                entry["why"] = _clip(str(content["reasoning"])[:160])
        elif step["type"] == "observation":
            entry = {"step": index, "type": "observation", "tool": content.get("tool"), "status": content.get("status")}
            if content.get("data") is not None:
                entry["data"] = _compact_data(content["data"])
            if content.get("disclosure"):
                entry["disclosure"] = _clip(content["disclosure"])
            if content.get("injection_suspected"):
                entry["injection_suspected"] = True
        else:
            entry = {"step": index, "type": step["type"], "content": _clip(str(content))}
        compact.append(entry)
    return compact


def dumps_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


# --- system context (topology) ---------------------------------------------------------------------

_DEFAULT_TOPOLOGY = Path(__file__).with_name("topology.json")


@lru_cache(maxsize=4)
def _load_topology(path: str) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"services": {}, "external": []}


def topology() -> dict[str, Any]:
    return _load_topology(os.getenv("TOPOLOGY_PATH", str(_DEFAULT_TOPOLOGY)))


def _services_mentioned(text: str, steps_taken: list[dict[str, Any]], known: set[str]) -> list[str]:
    haystack = text.lower()
    for obs in _ok_observations(steps_taken, "get_metrics"):
        haystack += " " + " ".join(obs["data"].keys()).lower()
    return sorted(name for name in known if name in haystack)


def topology_block(alert_text: str, steps_taken: list[dict[str, Any]]) -> str:
    """Dependency lines for the services this incident touches, in both
    directions, so a slow dependency can be recognised as a cause of its
    callers' errors. Empty when none of the services are in the catalog."""
    catalog = topology()
    services: dict[str, dict[str, Any]] = catalog.get("services", {})
    external = set(catalog.get("external", []))
    known = set(services) | {d for s in services.values() for d in s.get("depends_on", [])}

    lines: list[str] = []
    for name in _services_mentioned(alert_text, steps_taken, known):
        depends_on = [f"{d} (external)" if d in external else d for d in services.get(name, {}).get("depends_on", [])]
        depended_on_by = sorted(s for s, spec in services.items() if name in spec.get("depends_on", []))
        if depends_on or depended_on_by:
            lines.append(
                f"- {name}: depends on [{', '.join(depends_on) or 'nothing tracked'}]; "
                f"relied on by [{', '.join(depended_on_by) or 'nothing tracked'}]"
            )
    if not lines:
        return ""
    return "Service dependencies (background from the service catalog -- NOT evidence about this incident):\n" + "\n".join(lines)


def system_context(alert_text: str, steps_taken: list[dict[str, Any]]) -> str:
    parts = [now_line(), topology_block(alert_text, steps_taken), derived_signals_block(steps_taken)]
    return "\n\n".join(p for p in parts if p)
