"""
Tool layer: schema-validated calls into the incident simulator, with the
load-bearing / decorative role declared by the planner at plan time (not
inside an error handler under latency pressure -- see docs/architecture.md
section 5).

REQUIRED tools that go unreachable -> fail closed (agent must say so).
PREFERRED tools that go unreachable -> fail open (agent continues, discloses
the gap in its trace and final report).

Note: an earlier version of this file had _check_policy() raise
PolicyDenied on a disallowed call, but nothing ever caught that exception
-- a policy violation would crash the whole graph run instead of producing
the 'denied' ToolCallResult status the rest of the codebase (act_step in
agent/graph.py) already expected. Fixed as part of Item 2's token-based
enforcement rework: policy denials and simulator-side token denials both
now return status='denied' through the normal result path.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field

from services.agent.logging_utils import log_stage
from services.agent.tokens import TokenIssuanceError, get_or_issue_token

SIMULATOR_BASE_URL = os.getenv("SIMULATOR_BASE_URL", "http://localhost:8100")


class ToolRole(str, Enum):
    REQUIRED = "required"    # load-bearing: unreachable -> fail closed
    PREFERRED = "preferred"  # decorative: unreachable -> fail open + disclose


class ToolCallRequest(BaseModel):
    """What the planner emits before a tool is actually invoked."""

    tool: Literal["get_metrics", "get_logs", "get_deploy_history",
                   "restart_service", "rollback_deploy"]
    role: ToolRole
    params: dict[str, Any] = Field(default_factory=dict)
    incident_id: str


class ToolCallResult(BaseModel):
    tool: str
    role: ToolRole
    status: Literal["ok", "empty", "unreachable", "denied"]
    data: Optional[dict[str, Any]] = None
    disclosure: Optional[str] = None  # set when a PREFERRED tool degrades


# Actions tagged by blast radius -- feeds the human-approval trigger.
BLAST_RADIUS = {
    "get_metrics": "none",
    "get_logs": "none",
    "get_deploy_history": "none",
    "restart_service": "low",       # auto-executable
    "rollback_deploy": "medium",    # needs one-click Slack approval
}

# Item 4: which tools actually change production state, as opposed to just
# reading it. Read-only calls (get_metrics etc.) shouldn't generate a
# notification on every call -- that would be noise, not visibility.
MUTATING_TOOLS = {"restart_service", "rollback_deploy"}


# The only request params each simulator endpoint actually reads (see
# ToolRequest in services/simulator/main.py). Anything else a planner adds
# -- e.g. {"tail": 500} on get_logs -- is silently ignored server-side, so
# two calls differing only in such params return identical data.
TOOL_SUPPORTED_PARAMS: dict[str, frozenset[str]] = {
    "get_metrics": frozenset({"service", "since"}),
    "get_logs": frozenset({"service", "since"}),
    "get_deploy_history": frozenset({"service", "since"}),
    "restart_service": frozenset({"service"}),
    "rollback_deploy": frozenset({"service"}),
}


def canonical_params(tool: str, params: dict[str, Any]) -> dict[str, Any]:
    """The subset of `params` the tool actually honors, with empty values
    dropped. This -- not the planner's raw params -- defines call identity."""
    supported = TOOL_SUPPORTED_PARAMS.get(tool, frozenset())
    return {k: params[k] for k in sorted(params) if k in supported and params[k] not in (None, "")}


def call_signature(tool: str, params: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    canonical = canonical_params(tool, params if isinstance(params, dict) else {})
    return tool, tuple((k, str(v)) for k, v in canonical.items())


class BudgetExceeded(Exception):
    pass


class PolicyDenied(Exception):
    pass


class ToolExecutor:
    """Pre-flight validation (schema/policy/budget/authz), then dispatch.

    Item 2: authz is now enforced twice, deliberately. _check_policy() below
    is a fast, local, defense-in-depth check -- but the real boundary is the
    role-scoped token attached to every simulator call (see agent/tokens.py
    and simulator/tokens.py). If _check_policy somehow missed a case, or a
    future refactor bypassed it, the simulator would still reject the call
    server-side with a 403. That's the point: the credential, not the
    application logic, makes the disallowed action impossible.
    """

    def __init__(self, max_calls_per_incident: Optional[int] = None, agent_role: str = "investigator"):
        self.max_calls_per_incident = (
            max_calls_per_incident
            if max_calls_per_incident is not None
            else int(os.getenv("MAX_TOOL_CALLS_PER_INCIDENT", "8"))
        )
        self.agent_role = agent_role
        self._call_counts: dict[str, int] = {}

    def _check_budget(self, incident_id: str) -> None:
        count = self._call_counts.get(incident_id, 0)
        if count >= self.max_calls_per_incident:
            raise BudgetExceeded(
                f"incident {incident_id} exceeded {self.max_calls_per_incident} tool calls"
            )
        self._call_counts[incident_id] = count + 1

    def _check_policy(self, request: ToolCallRequest, agent_role: str) -> Optional[str]:
        """Returns a denial reason string if policy disallows the call,
        None if it's allowed. Deliberately returns rather than raises --
        a policy denial is an expected, handleable outcome (the graph
        routes it into the approval flow via status='denied'), not an
        exceptional program state that should crash the investigation."""
        if agent_role == "investigator" and request.tool == "rollback_deploy":
            return "investigator role cannot execute rollback_deploy"
        return None

    def execute(
        self, request: ToolCallRequest, agent_role: Optional[str] = None
    ) -> ToolCallResult:
        role = agent_role or self.agent_role
        self._check_budget(request.incident_id)

        policy_denial = self._check_policy(request, role)
        if policy_denial:
            log_stage("local_policy_check", request.incident_id, "DENIED", tool=request.tool, role=role, reason=policy_denial)
            return ToolCallResult(
                tool=request.tool,
                role=request.role,
                status="denied",
                data=None,
                disclosure=f"policy denied: {policy_denial}",
            )

        try:
            token = get_or_issue_token(request.incident_id, role)
        except TokenIssuanceError as e:
            log_stage("get_token", request.incident_id, "FAIL", role=role, error=str(e)[:80])
            # Can't get a token at all (simulator unreachable at the auth
            # endpoint specifically) -- treat the same as the tool itself
            # being unreachable, since without a token no call can succeed.
            return self._degrade(request, "unreachable")

        try:
            r = httpx.post(
                f"{SIMULATOR_BASE_URL}/tools/{request.tool}",
                # Only honored params are sent, and incident_id is applied
                # last so a model-supplied param can never override it.
                json={**canonical_params(request.tool, request.params), "incident_id": request.incident_id},
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )

            if r.status_code == 403:
                # This is the real enforcement boundary firing -- the
                # simulator itself rejected the token's role for this tool.
                # See simulator/tokens.py TOOL_ROLE_REQUIREMENTS.
                reason = r.json().get("detail", "authorization denied by simulator")
                log_stage("simulator_token_check", request.incident_id, "DENIED", tool=request.tool, reason=reason[:80])
                return ToolCallResult(
                    tool=request.tool,
                    role=request.role,
                    status="denied",
                    data=None,
                    disclosure=f"simulator denied (role-scoped token): {reason}",
                )

            r.raise_for_status()
            data = r.json()

            if not data:
                return self._degrade(request, "empty")

            return ToolCallResult(
                tool=request.tool, role=request.role, status="ok", data=data
            )

        except httpx.HTTPError:
            return self._degrade(request, "unreachable")

    def _degrade(self, request: ToolCallRequest, status: Literal["empty", "unreachable"]) -> ToolCallResult:
        if request.role == ToolRole.REQUIRED:
            # Fail closed: do not guess. The graph node calling this must
            # surface this as an explicit "cannot confirm/deny" statement.
            return ToolCallResult(
                tool=request.tool,
                role=request.role,
                status=status,
                data=None,
                disclosure=(
                    f"REQUIRED tool '{request.tool}' is {status}; "
                    "investigation cannot confirm or deny this signal."
                ),
            )
        # Fail open: continue, but disclose the gap.
        return ToolCallResult(
            tool=request.tool,
            role=request.role,
            status=status,
            data=None,
            disclosure=(
                f"PREFERRED tool '{request.tool}' is {status}; "
                "continuing without this signal."
            ),
        )
