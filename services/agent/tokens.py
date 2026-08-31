from __future__ import annotations

import os
from typing import Optional

import httpx

from services.agent.logging_utils import log_stage

SIMULATOR_BASE_URL = os.getenv("SIMULATOR_BASE_URL", "http://localhost:8000")
_token_cache: dict[tuple[str, str], str] = {}


class TokenIssuanceError(Exception):
    pass


def get_or_issue_token(incident_id: str, role: str) -> str:
    key = (incident_id, role)
    if key in _token_cache:
        log_stage("get_token", incident_id, "OK", role=role, source="cache")
        return _token_cache[key]

    try:
        r = httpx.post(
            f"{SIMULATOR_BASE_URL}/auth/token",
            json={"incident_id": incident_id, "role": role},
            timeout=10.0,
        )
        r.raise_for_status()
        token = r.json()["token"]
    except httpx.HTTPError as e:
        raise TokenIssuanceError(f"could not obtain '{role}' token for incident {incident_id}: {e}")

    _token_cache[key] = token
    log_stage("get_token", incident_id, "OK", role=role, source="issued")
    return token


def clear_cache() -> None:
    """Used by tests/benchmark runs that reuse incident_ids across scenarios."""
    _token_cache.clear()
