from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.simulator.scenarios import SCENARIOS
from services.simulator.tokens import issue_token, verify_token


def _log(event: str, incident_id: str, status: str, **details: object) -> None:
    ts = time.strftime("%H:%M:%S")
    detail_str = " ".join(f"{k}={v}" for k, v in details.items())
    print(f"[{ts}] SIMULATOR  {event:<16} incident={incident_id[:8]:<8} {status:<8} {detail_str}", flush=True)


def _parse_ts(value: str) -> Optional[datetime]:
    """Tolerant timestamp parser -- the LLM sends its own guessed format
    (typically ISO 8601, e.g. '2026-08-29T05:00:00Z'), while scenario log
    entries use space-separated timestamps ('2026-08-29 05:12:57.918767').
    Tries a couple of normalizations before giving up; returns None (not
    an exception) on anything unparseable, so a malformed 'since' value
    degrades to 'ignore the filter' rather than crashing the request.

    Always returns a naive datetime (strips tzinfo if present) -- Python
    3.11+'s fromisoformat parses a trailing 'Z' as UTC and returns a
    timezone-AWARE datetime, while scenario timestamps (generated via
    datetime.utcnow()) are naive. Comparing the two raises TypeError --
    the exact same class of bug already hit once in agent/escalation.py's
    _minutes_pending(), fixed here the same way: normalize to naive."""
    value = value.strip()
    candidates = [value, value.replace("T", " ").replace("Z", "")]
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is not None:
                parsed = parsed.replace(tzinfo=None)
            return parsed
        except ValueError:
            continue
    return None


app = FastAPI(title="Genesis Kit Incident Simulator")
DASHBOARD_ORIGIN = "http://localhost:3000"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[DASHBOARD_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# incident_id -> active scenario key, set via /chaos/inject
_active_incidents: dict[str, str] = {}


class InjectRequest(BaseModel):
    incident_id: str
    scenario: str


class ToolRequest(BaseModel):
    incident_id: str
    service: Optional[str] = None
    since: Optional[str] = None  


class TokenRequest(BaseModel):
    incident_id: str
    role: str


@app.post("/chaos/inject")
def inject(req: InjectRequest) -> dict[str, Any]:
    if req.scenario not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario '{req.scenario}'")
    _active_incidents[req.incident_id] = req.scenario
    _log("inject", req.incident_id, "OK", scenario=req.scenario)
    return {
        "incident_id": req.incident_id,
        "scenario": req.scenario,
        "description": SCENARIOS[req.scenario]["description"],
    }


@app.post("/auth/token")
def auth_token(req: TokenRequest) -> dict[str, Any]:
    """Item 2: issues a role-scoped, expiring token for an incident. This
    is the credential the simulator will actually check on every /tools/*
    call -- see simulator/tokens.py for why this (not the agent's own
    Python policy check) is the real enforcement boundary."""
    try:
        token = issue_token(req.incident_id, req.role)
    except ValueError as e:
        _log("issue_token", req.incident_id, "FAIL", role=req.role, error=str(e))
        raise HTTPException(400, str(e))
    _log("issue_token", req.incident_id, "OK", role=req.role)
    return {"token": token, "incident_id": req.incident_id, "role": req.role}


def _authorize(incident_id: str, tool: str, authorization: Optional[str]) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        _log("authorize", incident_id, "DENY", tool=tool, reason="missing/malformed header")
        raise HTTPException(401, "missing or malformed Authorization header")

    token = authorization.removeprefix("Bearer ")
    result = verify_token(token, incident_id, tool)
    if not result.valid:
        _log("authorize", incident_id, "DENY", tool=tool, role=result.role, reason=result.error)
        raise HTTPException(403, result.error)
    _log("authorize", incident_id, "ALLOW", tool=tool, role=result.role)


def _scenario_for(incident_id: str) -> dict[str, Any]:
    key = _active_incidents.get(incident_id)
    if not key:
        return {}
    return SCENARIOS[key]


def _is_degraded(scenario: dict[str, Any], tool_name: str) -> bool:
    return tool_name in scenario.get("degraded_tools", [])


def _primary_service(scenario: dict[str, Any]) -> Optional[str]:
    """The service this scenario's data actually belongs to -- derived
    from the metrics dict's key, or deploy history's service field as a
    fallback for scenarios with no metrics (e.g. deploy_rollback_missing_metrics).
    Used by _service_matches() below to fix a real bug: every tool
    endpoint used to ignore req.service entirely and return this
    scenario's data regardless of which service was actually asked for --
    a real run showed the agent asking for pricing-api's logs and
    silently getting checkout-service's logs back instead, wasting a full
    LLM call + tool round-trip re-querying data it already had."""
    metrics = scenario.get("metrics") or {}
    if metrics:
        return next(iter(metrics.keys()))
    deploys = scenario.get("deploys") or []
    if deploys:
        return deploys[0].get("service")
    return None


def _service_matches(scenario: dict[str, Any], requested_service: Optional[str]) -> bool:
    """No service specified -> return everything (backward compatible
    with every existing call site that doesn't pass one). A service IS
    specified -> only return real data if it matches this scenario's
    actual service. Asking for a different service's data is a
    legitimate 'no data here' case (empty result), not a degradation --
    matches the same real-vs-degraded distinction already made for
    degraded_tools."""
    if not requested_service:
        return True
    primary = _primary_service(scenario)
    return primary is None or requested_service == primary


@app.post("/tools/get_metrics")
def get_metrics(req: ToolRequest, authorization: Optional[str] = Header(None)) -> dict[str, Any]:
    _authorize(req.incident_id, "get_metrics", authorization)
    scenario = _scenario_for(req.incident_id)
    if _is_degraded(scenario, "get_metrics"):
        _log("tool_call", req.incident_id, "WARN", tool="get_metrics", note="degraded -- returning empty")
        return {}  # tool "went quiet" -- 200 OK, empty body
    if not _service_matches(scenario, req.service):
        _log("tool_call", req.incident_id, "OK", tool="get_metrics", note=f"no data for service={req.service}")
        return {}
    if req.since:
        _log("tool_call", req.incident_id, "OK", tool="get_metrics", note="since ignored -- metric series have no per-point timestamps")
    else:
        _log("tool_call", req.incident_id, "OK", tool="get_metrics")
    return scenario.get("metrics", {})


@app.post("/tools/get_logs")
def get_logs(req: ToolRequest, authorization: Optional[str] = Header(None)) -> dict[str, Any]:
    _authorize(req.incident_id, "get_logs", authorization)
    scenario = _scenario_for(req.incident_id)
    if _is_degraded(scenario, "get_logs"):
        _log("tool_call", req.incident_id, "WARN", tool="get_logs", note="degraded -- returning empty")
        return {}
    if not _service_matches(scenario, req.service):
        _log("tool_call", req.incident_id, "OK", tool="get_logs", note=f"no data for service={req.service}")
        return {"logs": []}

    logs = scenario.get("logs", [])
    if req.since:
        since_dt = _parse_ts(req.since)
        if since_dt is None:
            _log("tool_call", req.incident_id, "WARN", tool="get_logs", note=f"unparseable since={req.since!r}, ignoring filter")
        else:
            before_count = len(logs)
            logs = [entry for entry in logs if (parsed := _parse_ts(entry["ts"])) is not None and parsed >= since_dt]
            _log("tool_call", req.incident_id, "OK", tool="get_logs", since=req.since, filtered=f"{len(logs)}/{before_count}")
            return {"logs": logs}

    _log("tool_call", req.incident_id, "OK", tool="get_logs")
    return {"logs": logs}


@app.post("/tools/get_deploy_history")
def get_deploy_history(req: ToolRequest, authorization: Optional[str] = Header(None)) -> dict[str, Any]:
    _authorize(req.incident_id, "get_deploy_history", authorization)
    scenario = _scenario_for(req.incident_id)
    if _is_degraded(scenario, "get_deploy_history"):
        _log("tool_call", req.incident_id, "WARN", tool="get_deploy_history", note="degraded -- returning empty")
        return {}
    if not _service_matches(scenario, req.service):
        _log("tool_call", req.incident_id, "OK", tool="get_deploy_history", note=f"no data for service={req.service}")
        return {"deploys": []}

    deploys = scenario.get("deploys", [])
    if req.since:
        since_dt = _parse_ts(req.since)
        if since_dt is None:
            _log("tool_call", req.incident_id, "WARN", tool="get_deploy_history", note=f"unparseable since={req.since!r}, ignoring filter")
        else:
            before_count = len(deploys)
            deploys = [d for d in deploys if (parsed := _parse_ts(d["ts"])) is not None and parsed >= since_dt]
            _log("tool_call", req.incident_id, "OK", tool="get_deploy_history", since=req.since, filtered=f"{len(deploys)}/{before_count}")
            return {"deploys": deploys}

    _log("tool_call", req.incident_id, "OK", tool="get_deploy_history")
    return {"deploys": deploys}


@app.post("/tools/restart_service")
def restart_service(req: ToolRequest, authorization: Optional[str] = Header(None)) -> dict[str, Any]:
    _authorize(req.incident_id, "restart_service", authorization)
    _log("tool_call", req.incident_id, "OK", tool="restart_service", service=req.service)
    return {"status": "restarted", "service": req.service}


@app.post("/tools/rollback_deploy")
def rollback_deploy(req: ToolRequest, authorization: Optional[str] = Header(None)) -> dict[str, Any]:
    _log("tool_call", req.incident_id, "OK", tool="rollback_deploy", service=req.service)
    return {"status": "rolled_back", "service": req.service}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}