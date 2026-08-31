from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

TOKEN_SECRET = os.getenv("SIMULATOR_TOKEN_SECRET", "dev-only-insecure-secret-change-me")
TOOL_ROLE_REQUIREMENTS: dict[str, set[str]] = {
    "get_metrics": {"investigator", "approver"},
    "get_logs": {"investigator", "approver"},
    "get_deploy_history": {"investigator", "approver"},
    "restart_service": {"investigator", "approver"}, 
    "rollback_deploy": {"approver"}, 
}

ROLE_DEFAULT_TTL_SECONDS = {
    "investigator": 30 * 60,
    "approver": 5 * 60, 
}


def _sign(payload_b64: str) -> str:
    return hmac.new(TOKEN_SECRET.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()


def issue_token(incident_id: str, role: str, ttl_seconds: Optional[int] = None) -> str:
    if role not in ROLE_DEFAULT_TTL_SECONDS:
        raise ValueError(f"unknown role '{role}'")

    ttl = ttl_seconds if ttl_seconds is not None else ROLE_DEFAULT_TTL_SECONDS[role]
    payload = {
        "incident_id": incident_id,
        "role": role,
        "exp": int(time.time()) + ttl,
    }
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    signature = _sign(payload_b64)
    return f"{payload_b64}.{signature}"


@dataclass
class TokenValidationResult:
    valid: bool
    role: Optional[str] = None
    incident_id: Optional[str] = None
    error: Optional[str] = None


def verify_token(token: str, incident_id: str, tool: str) -> TokenValidationResult:
    """Checks: well-formed, signature valid, not expired, incident_id
    matches, and role is in the tool's required set. Any single failure is
    reported specifically -- the caller (the simulator endpoint) turns this
    into a 403 with a clear reason, which is what ends up in the agent's
    own trace as a 'denied' disclosure."""
    try:
        payload_b64, signature = token.split(".", 1)
    except ValueError:
        return TokenValidationResult(valid=False, error="malformed token")

    expected_signature = _sign(payload_b64)
    if not hmac.compare_digest(signature, expected_signature):
        return TokenValidationResult(valid=False, error="invalid token signature")

    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode()))
    except Exception:
        return TokenValidationResult(valid=False, error="malformed token payload")

    if payload.get("exp", 0) < time.time():
        return TokenValidationResult(valid=False, error="token expired")

    token_incident_id = payload.get("incident_id")
    if token_incident_id != incident_id:
        return TokenValidationResult(
            valid=False,
            error=f"token scoped to incident '{token_incident_id}', not '{incident_id}'",
        )

    role = payload.get("role")
    allowed_roles = TOOL_ROLE_REQUIREMENTS.get(tool, set())
    if role not in allowed_roles:
        return TokenValidationResult(
            valid=False,
            role=role,
            incident_id=token_incident_id,
            error=f"role '{role}' is not authorized to call '{tool}' (requires one of {sorted(allowed_roles)})",
        )

    return TokenValidationResult(valid=True, role=role, incident_id=token_incident_id)
