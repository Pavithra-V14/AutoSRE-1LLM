from __future__ import annotations

import argparse
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Optional, TypedDict

import httpx
from langgraph.graph import StateGraph, END
from pydantic import ValidationError

from services.agent.guardrails import (
    check_scope_and_injection,
    redact_pii,
    validate_causal_graph_output,
)
from services.agent.approval import post_approval_request, post_notification
from services.agent.correlation import extract_primary_service, find_related_incidents, format_correlation_note
from services.agent.dedup import DEDUP_WINDOW_SECONDS, check_and_register
from services.agent.llm_router import LLMRouter, ProviderError
from services.agent.logging_utils import log_stage
from services.agent.persistence import (
    compute_equivalent_cost_usd,
    flush_pending_writes,
    get_denial_notes_for_pattern,
    insert_trace_step,
)
from services.agent.tokens import get_or_issue_token
from services.agent.tools import ToolCallRequest, ToolExecutor, ToolRole, BLAST_RADIUS, MUTATING_TOOLS
from typing import get_args
TOOL_NAMES = get_args(ToolCallRequest.model_fields["tool"].annotation)
from services.agent.trust import compute_pattern_key, evaluate_trust, record_auto_approval

MAX_STEPS = 8
MAX_COST_PER_INCIDENT_USD = float(os.getenv("MAX_COST_PER_INCIDENT_USD", "1.00"))
MAX_LATENCY_MS_PER_INCIDENT = int(os.getenv("MAX_LATENCY_MS_PER_INCIDENT", "60000"))
SIMULATOR_BASE_URL = os.getenv("SIMULATOR_BASE_URL", "http://localhost:8000")


class ScenarioInjectionError(Exception):
    pass


def inject_scenario(incident_id: str, scenario: str) -> None:
    """Tell the simulator which scenario this incident_id should serve.
    Must run before any tool call, or the simulator has no data to return
    and every REQUIRED tool call will look 'unreachable/empty' by default."""
    try:
        r = httpx.post(
            f"{SIMULATOR_BASE_URL}/chaos/inject",
            json={"incident_id": incident_id, "scenario": scenario},
            timeout=10.0,
        )
        r.raise_for_status()
        log_stage("inject_scenario", incident_id, "OK", scenario=scenario)
    except httpx.HTTPError as e:
        log_stage("inject_scenario", incident_id, "FAIL", scenario=scenario, error=str(e)[:80])
        raise ScenarioInjectionError(
            f"could not inject scenario '{scenario}' at {SIMULATOR_BASE_URL}: {e}"
        )


class IncidentState(TypedDict):
    incident_id: str
    alert_text: str
    steps_taken: list[dict[str, Any]]
    hypothesis: Optional[str]
    confidence: float
    causal_graph: Optional[dict[str, Any]]
    needs_approval: bool
    done: bool
    synthesis_provider: Optional[str]
    synthesis_input_tokens: int
    synthesis_output_tokens: int


router = LLMRouter()
executor = ToolExecutor()

AGENT_FAST_PATH = os.getenv("AGENT_FAST_PATH", "false").strip().lower() == "true"
AGENT_FAST_PATH_MAX_STEPS = int(os.getenv("AGENT_FAST_PATH_MAX_STEPS", "10"))
EFFECTIVE_MAX_STEPS = AGENT_FAST_PATH_MAX_STEPS if AGENT_FAST_PATH else MAX_STEPS
FAST_PATH_TOOLS: list[tuple[str, str]] = [
    ("get_deploy_history", "required"),
    ("get_metrics", "required"),
    ("get_logs", "required"),
]


def _prefetch_baseline_evidence(incident_id: str) -> list[dict[str, Any]]:
    """Fires FAST_PATH_TOOLS concurrently (via a small thread pool, since
    ToolExecutor.execute() is a blocking httpx call) and returns them as
    plan+observation step pairs in the exact shape plan_step/act_step
    normally produce -- so everything downstream (duplicate-call
    detection, correlation, evaluation scoring, the dashboard trace
    viewer) treats a prefetched step identically to an LLM-decided one,
    with only the plan's 'reasoning' field disclosing it was auto-fetched.

    Note on MAX_STEPS: pre-seeding 3 pairs (6 entries) against the
    default MAX_STEPS=8 leaves room for exactly one more real
    LLM-decided step before the budget check in plan_step kicks in --
    intentional (the whole point is brevity), but worth knowing if you
    also want deeper follow-up investigation on top of the fast path;
    raise MAX_STEPS if so."""
    def _call(tool: str, role: str) -> tuple[dict[str, Any], Any]:
        request = ToolCallRequest(tool=tool, role=ToolRole(role), params={}, incident_id=incident_id)
        result = executor.execute(request)
        plan = {
            "tool": tool,
            "role": role,
            "params": {},
            "reasoning": "auto-prefetched baseline evidence (AGENT_FAST_PATH)",
        }
        return plan, result

    with ThreadPoolExecutor(max_workers=len(FAST_PATH_TOOLS), thread_name_prefix="fast-path-prefetch") as pool:
        futures = [pool.submit(_call, tool, role) for tool, role in FAST_PATH_TOOLS]
        pairs = [f.result() for f in futures]  # submission order, not completion order

    prefetched: list[dict[str, Any]] = []
    for i, (plan, result) in enumerate(pairs):
        observation = result.model_dump()
        prefetched.append({"type": "plan", "content": plan})
        prefetched.append({"type": "observation", "content": observation})

        insert_trace_step(
            incident_id=incident_id, step_index=i * 2, step_type="plan", content=plan,
            tool=plan["tool"], tool_role=plan["role"],
        )
        insert_trace_step(
            incident_id=incident_id, step_index=i * 2 + 1, step_type="observation", content=observation,
            tool=result.tool, tool_role=result.role.value, status=result.status,
        )

    log_stage(
        "fast_path_prefetch", incident_id, "OK",
        tools=",".join(t for t, _ in FAST_PATH_TOOLS), entries=len(prefetched),
    )
    return prefetched


STRICT_JSON_INSTRUCTION = (
    "Respond with ONLY raw JSON -- no markdown code fences (no ```), no "
    "explanation before or after, no commentary. Your entire response must "
    "be parseable directly as JSON."
)

TOOL_REFERENCE = (
    "Tool reference -- what each one actually returns:\n"
    "- get_metrics: per-service p99_latency_ms and error_rate time series. "
    "Does NOT return request_rate, throughput, or replica counts.\n"
    "- get_logs: recent log lines (timestamp, level, message).\n"
    "- get_deploy_history: recent deploys/config changes for the service, "
    "each with an id, timestamp, and a one-line description of the change. "
    "Deploys and config changes are a very common root cause of latency "
    "and error-rate spikes -- check this early, not just metrics/logs.\n"
    "- restart_service: restarts a service (a remediation action, not an "
    "investigation tool).\n"
    "- rollback_deploy: rolls back the most recent deploy (a remediation "
    "action requiring approval, not an investigation tool)."
)


def _extract_json(text: str) -> dict[str, Any]:
    """LLMs -- Gemini especially -- routinely wrap JSON in markdown code
    fences or add a sentence of commentary despite being told not to.
    Strict prompting (STRICT_JSON_INSTRUCTION above) is the primary
    defense; this is the fallback layer, matching the project's pattern
    elsewhere of not trusting a single guardrail. Tries, in order: raw
    parse, stripped code-fence parse, then first-{-to-last-} substring."""
    text = text.strip()
    last_error: Optional[json.JSONDecodeError] = None

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        last_error = e

    if "```" in text:
        fenced = text.split("```")
        for chunk in fenced:
            chunk = chunk.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{") and chunk.endswith("}"):
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError as e:
                    last_error = e
                    continue

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            last_error = e

    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("could not extract valid JSON from LLM response", text, 0)

PLAN_HISTORY_KEEP_RECENT = int(os.getenv("PLAN_HISTORY_KEEP_RECENT", "4"))


def _summarize_step(step: dict[str, Any]) -> str:
    content = step.get("content", {})
    if step["type"] == "plan":
        return f"planned: {content.get('tool')}({content.get('role')})"
    if step["type"] == "observation":
        tool = content.get("tool")
        status = content.get("status")
        disclosure = content.get("disclosure")
        if disclosure:
            return f"observed: {tool} -> {status} ({str(disclosure)[:80]})"
        return f"observed: {tool} -> {status}"
    return str(step)[:100]


def _build_plan_history_context(steps_taken: list[dict[str, Any]]) -> str:
    if len(steps_taken) <= PLAN_HISTORY_KEEP_RECENT:
        return json.dumps(steps_taken, indent=2)

    older = steps_taken[: -PLAN_HISTORY_KEEP_RECENT]
    recent = steps_taken[-PLAN_HISTORY_KEEP_RECENT:]
    summary_lines = "\n".join(f"  {i + 1}. {_summarize_step(s)}" for i, s in enumerate(older))
    return (
        f"Summary of earlier steps (full detail omitted to keep this prompt "
        f"from growing unbounded):\n{summary_lines}\n\n"
        f"Full detail of the {len(recent)} most recent steps:\n{json.dumps(recent, indent=2)}"
    )


def guard_input(state: IncidentState) -> IncidentState:
    ok, reason = check_scope_and_injection(state["alert_text"])
    if not ok:
        state["done"] = True
        state["hypothesis"] = f"REJECTED: {reason}"
        log_stage("guard_input", state["incident_id"], "BLOCKED", reason=reason[:100])
    else:
        log_stage("guard_input", state["incident_id"], "PASS")
    return state


def _cumulative_cost_so_far(steps_taken: list[dict[str, Any]]) -> float:
    """Item 7: sums equivalent cost across every plan step taken so far.
    Uses the same EQUIVALENT_PAID_RATES basis as the benchmark's CPSO
    calculation (agent/persistence.py), since the free-tier providers
    genuinely cost $0 and a real dollar ceiling needs a real dollar basis
    to check against."""
    total = 0.0
    for step in steps_taken:
        if step["type"] == "plan":
            total += compute_equivalent_cost_usd(
                step.get("provider", ""), step.get("input_tokens", 0), step.get("output_tokens", 0)
            )
    return total


def plan_step(state: IncidentState) -> IncidentState:
    """Ask the LLM: given what we know so far, what's the next tool call
    and its role (required vs preferred)?"""
    if state["done"] or len(state["steps_taken"]) >= EFFECTIVE_MAX_STEPS:
        state["done"] = True
        if not state.get("hypothesis"):
            log_stage("plan_step", state["incident_id"], "ABORT", reason="max steps reached", steps=len(state["steps_taken"]), ceiling=EFFECTIVE_MAX_STEPS)
        return state

    cumulative_cost = _cumulative_cost_so_far(state["steps_taken"])
    if cumulative_cost >= MAX_COST_PER_INCIDENT_USD:
        state["done"] = True
        state["hypothesis"] = (
            f"ABORTED: cost budget exceeded (${cumulative_cost:.4f} >= "
            f"${MAX_COST_PER_INCIDENT_USD:.2f} ceiling) after "
            f"{len(state['steps_taken'])} step(s)"
        )
        log_stage("plan_step", state["incident_id"], "ABORT", reason="cost budget exceeded", cost=f"${cumulative_cost:.4f}")
        return state

    cumulative_latency_ms = sum(
        s.get("latency_ms") or 0 for s in state["steps_taken"] if s["type"] == "plan"
    )
    if cumulative_latency_ms >= MAX_LATENCY_MS_PER_INCIDENT:
        state["done"] = True
        state["hypothesis"] = (
            f"ABORTED: latency budget exceeded ({cumulative_latency_ms}ms >= "
            f"{MAX_LATENCY_MS_PER_INCIDENT}ms ceiling) after "
            f"{len(state['steps_taken'])} step(s)"
        )
        log_stage("plan_step", state["incident_id"], "ABORT", reason="latency budget exceeded", latency_ms=cumulative_latency_ms)
        return state

    history = _build_plan_history_context(state["steps_taken"])
    valid_tools = ", ".join(f'"{t}"' for t in TOOL_NAMES)
    prompt = (
        f"Incident alert: {redact_pii(state['alert_text'])}\n\n"
        f"{TOOL_REFERENCE}\n\n"
        f"Steps taken so far:\n{history}\n\n"
        "What is the next tool to call to investigate this incident? "
        f'The "tool" field MUST be exactly one of: {valid_tools} -- no other '
        "tool names exist, do not invent one. Do NOT call a tool with the "
        "same params as a call already shown in 'Steps taken so far' above "
        "-- that data is already visible to you, re-use it instead of "
        "re-fetching it.\n"
        'Respond as JSON: {"tool": "...", "role": "required|preferred", '
        '"params": {...}, "reasoning": "..."}\n\n'
        f"{STRICT_JSON_INSTRUCTION}"
    )
    try:
        resp = router.complete(
            prompt,
            system=(
                "You are an SRE investigation agent. Plan one tool call at "
                "a time. Investigate broadly before concluding -- for a "
                "latency or error-rate incident, check deploy history "
                "alongside metrics and logs rather than only re-querying "
                "metrics."
            ),
        )
    except ProviderError as e:
        state["done"] = True
        state["hypothesis"] = f"ABORTED: all LLM providers failed ({e})"
        log_stage("plan_step", state["incident_id"], "ABORT", reason="all LLM providers failed")
        return state

    try:
        plan = _extract_json(resp.text)
    except json.JSONDecodeError:
        state["done"] = True
        state["hypothesis"] = "ABORTED: planner returned non-JSON output"
        log_stage(
            "plan_step", state["incident_id"], "ABORT",
            reason="non-JSON planner output", provider=resp.provider,
            finish_reason=resp.finish_reason,
            raw=resp.text[:120].replace("\n", " "),
        )
        return state

    state["steps_taken"].append({
        "type": "plan",
        "content": plan,
        "provider": resp.provider,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "latency_ms": resp.latency_ms,
    })

    insert_trace_step(
        incident_id=state["incident_id"],
        step_index=len(state["steps_taken"]) - 1,
        step_type="plan",
        content=plan,
        tool=plan.get("tool"),
        tool_role=plan.get("role"),
        provider=resp.provider,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        latency_ms=resp.latency_ms,
    )
    log_stage(
        "plan_step", state["incident_id"], "OK",
        tool=plan.get("tool"), role=plan.get("role"), provider=resp.provider,
        tokens=f"{resp.input_tokens}in/{resp.output_tokens}out",
    )
    return state


def act_step(state: IncidentState) -> IncidentState:
    if state["done"] or not state["steps_taken"]:
        return state

    last_plan = state["steps_taken"][-1]["content"]
    duplicate_index = next(
        (
            i for i, s in enumerate(state["steps_taken"][:-1])
            if s["type"] == "plan"
            and s["content"].get("tool") == last_plan.get("tool")
            and s["content"].get("params") == last_plan.get("params")
        ),
        None,
    )
    if duplicate_index is not None:
        dup_result = {
            "tool": last_plan.get("tool"),
            "role": last_plan.get("role"),
            "status": "duplicate_call_skipped",
            "data": None,
            "disclosure": (
                f"identical call already made at step {duplicate_index} -- "
                "not re-fetching, see that step's observation for the data"
            ),
        }
        state["steps_taken"].append({"type": "observation", "content": dup_result})
        insert_trace_step(
            incident_id=state["incident_id"],
            step_index=len(state["steps_taken"]) - 1,
            step_type="observation",
            content=dup_result,
            tool=dup_result["tool"],
            status="duplicate_call_skipped",
        )
        log_stage(
            "act_step", state["incident_id"], "WARN",
            reason="duplicate call skipped", tool=dup_result["tool"], original_step=duplicate_index,
        )
        return state

    try:
        request = ToolCallRequest(
            tool=last_plan["tool"],
            role=ToolRole(last_plan["role"]),
            params=last_plan.get("params", {}),
            incident_id=state["incident_id"],
        )
    except (ValidationError, ValueError, KeyError) as e:
        invalid_result = {
            "tool": last_plan.get("tool", "<missing>"),
            "role": last_plan.get("role", "<missing>"),
            "status": "invalid_plan",
            "data": None,
            "disclosure": f"planner produced an invalid tool call, discarded: {e}",
        }
        state["steps_taken"].append({"type": "observation", "content": invalid_result})
        insert_trace_step(
            incident_id=state["incident_id"],
            step_index=len(state["steps_taken"]) - 1,
            step_type="observation",
            content=invalid_result,
            tool=invalid_result["tool"],
            status="invalid_plan",
        )
        log_stage(
            "act_step", state["incident_id"], "WARN",
            reason="invalid tool call from planner", tool=invalid_result["tool"],
            error=str(e)[:100].replace("\n", " "),
        )
        return state

    result = executor.execute(request)
    state["steps_taken"].append({"type": "observation", "content": result.model_dump()})

    insert_trace_step(
        incident_id=state["incident_id"],
        step_index=len(state["steps_taken"]) - 1,
        step_type="observation",
        content=result.model_dump(),
        tool=result.tool,
        tool_role=result.role.value,
        status=result.status,
    )

    status_label = {"ok": "OK", "empty": "WARN", "unreachable": "WARN", "denied": "DENIED"}.get(result.status, "WARN")
    log_stage(
        "act_step", state["incident_id"], status_label,
        tool=result.tool, role=result.role.value,
        disclosure=(result.disclosure[:80] if result.disclosure else "-"),
    )

    if result.status == "denied":
        state["needs_approval"] = True

    if (
        result.status == "ok"
        and result.tool in MUTATING_TOOLS
        and BLAST_RADIUS.get(result.tool, "none") == "low"
    ):
        post_notification(
            incident_id=state["incident_id"],
            action=result.tool,
            message=f"`{result.tool}` executed automatically (low blast radius, no approval required).",
        )
        log_stage("notify", state["incident_id"], "INFO", action=result.tool, reason="low blast radius auto-exec")

    return state


def check_done(state: IncidentState) -> str:
    if state["done"] or len(state["steps_taken"]) >= EFFECTIVE_MAX_STEPS * 2:
        return "synthesize"
    return "plan"


def synthesize_step(state: IncidentState) -> IncidentState:
    """Produce the final causal graph, citing tool_call evidence for every
    edge. Rejects and retries once if the citation check fails."""
    history = json.dumps(state["steps_taken"], indent=2)
    prompt = (
        f"Given this investigation trace:\n{history}\n\n"
        "Produce a causal graph as JSON: "
        '{"edges": [{"id": "...", "from": "...", "to": "...", '
        '"evidence_tool_call_id": "..."}], "confidence": 0.0-1.0, '
        '"confidence_rationale": "one or two sentences on what evidence '
        'supports this confidence level, or what is missing/uncertain"}\n\n'
        f"{STRICT_JSON_INSTRUCTION}"
    )
    try:
        resp = router.complete(prompt, system="Cite evidence for every causal claim.")
    except ProviderError as e:
        state["causal_graph"] = {"edges": [], "confidence": 0.0}
        state["done"] = True
        insert_trace_step(
            incident_id=state["incident_id"],
            step_index=len(state["steps_taken"]),
            step_type="synthesis",
            content=state["causal_graph"],
            status="aborted",
        )
        log_stage("synthesize_step", state["incident_id"], "ABORT", reason="ProviderError")
        return state

    try:
        graph = _extract_json(resp.text)
        synth_provider = resp.provider
        synth_tokens = (resp.input_tokens, resp.output_tokens, resp.latency_ms)
    except json.JSONDecodeError as e:
        state["causal_graph"] = {"edges": [], "confidence": 0.0}
        state["done"] = True

        error_pos = getattr(e, "pos", None)
        if error_pos is not None:
            window_start = max(0, error_pos - 150)
            window_end = min(len(resp.text), error_pos + 150)
            context_window = resp.text[window_start:window_end].replace("\n", " ")
        else:
            context_window = resp.text[-150:].replace("\n", " ")

        insert_trace_step(
            incident_id=state["incident_id"],
            step_index=len(state["steps_taken"]),
            step_type="synthesis",
            content={
                **state["causal_graph"],
                "raw_response_on_failure": resp.text,
                "parse_error_char_offset": error_pos,
            },
            status="aborted",
        )
        
        truncation_note = (
            " -- response was TRUNCATED (finish_reason=length); raise "
            "LLM_MAX_OUTPUT_TOKENS in .env"
            if resp.finish_reason == "length"
            else ""
        )
        log_stage(
            "synthesize_step", state["incident_id"], "ABORT",
            reason=f"JSONDecodeError{truncation_note}",
            finish_reason=resp.finish_reason,
            error_at_char=error_pos,
            context_around_error=context_window,
            note="full raw response saved to traces row for this step" if not truncation_note else None,
        )
        return state

    valid, error = validate_causal_graph_output(graph)
    if not valid:
        graph = {"edges": [], "confidence": 0.0, "validation_error": error}
        log_stage("synthesize_step", state["incident_id"], "WARN", reason="schema validation failed", detail=error[:80])

    state["causal_graph"] = graph
    state["confidence"] = graph.get("confidence", 0.0)
    confidence_rationale = graph.get("confidence_rationale", "")
    log_stage(
        "synthesize_step", state["incident_id"], "OK",
        edges=len(graph.get("edges", [])), confidence=f"{state['confidence']:.0%}", provider=synth_provider,
    )

    primary_service = extract_primary_service(state["steps_taken"])
    related_incidents = (
        find_related_incidents(primary_service, state["incident_id"])
        if primary_service
        else []
    )
    correlation_note = format_correlation_note(related_incidents)
    if correlation_note:
        confidence_rationale = (
            f"{confidence_rationale} | {correlation_note}" if confidence_rationale else correlation_note
        )
        graph["confidence_rationale"] = confidence_rationale  # so the trace/dashboard shows it too
        log_stage("correlation", state["incident_id"], "WARN", service=primary_service, related=len(related_incidents))
    elif primary_service:
        log_stage("correlation", state["incident_id"], "OK", service=primary_service, related=0)

    blast_radii = [
        BLAST_RADIUS.get(s["content"].get("tool", ""), "none")
        for s in state["steps_taken"]
        if s["type"] == "plan"
    ]
    high_blast = any(b in ("medium", "high") for b in blast_radii)
    low_confidence = state["confidence"] < 0.6
    state["needs_approval"] = high_blast or low_confidence
    state["done"] = True

    insert_trace_step(
        incident_id=state["incident_id"],
        step_index=len(state["steps_taken"]),
        step_type="synthesis",
        content=graph,
        provider=synth_provider,
        input_tokens=synth_tokens[0],
        output_tokens=synth_tokens[1],
        latency_ms=synth_tokens[2],
    )

    state["synthesis_provider"] = synth_provider
    state["synthesis_input_tokens"] = synth_tokens[0]
    state["synthesis_output_tokens"] = synth_tokens[1]

    if state["needs_approval"]:
        risky_action = next(
            (
                s["content"].get("tool")
                for s in state["steps_taken"]
                if s["type"] == "plan"
                and BLAST_RADIUS.get(s["content"].get("tool", ""), "none") in ("medium", "high")
            ),
            "investigation result (low confidence)",
        )
        risky_blast_radius = BLAST_RADIUS.get(risky_action, "low")
        pattern_key = compute_pattern_key(graph)
        log_stage("pattern_key", state["incident_id"], "INFO", key=pattern_key, action=risky_action, blast_radius=risky_blast_radius)

        trust_decision = evaluate_trust(pattern_key, risky_blast_radius, state["confidence"])
        log_stage(
            "trust_eval", state["incident_id"],
            "OK" if trust_decision.auto_approve_eligible else "INFO",
            eligible=trust_decision.auto_approve_eligible,
            samples=trust_decision.sample_size,
            score=f"{trust_decision.trust_score:.0%}",
        )

        if trust_decision.auto_approve_eligible:
            record_auto_approval(pattern_key, state["incident_id"])
            state["needs_approval"] = False
            post_notification(
                incident_id=state["incident_id"],
                action=risky_action,
                message=(
                    f"`{risky_action}` auto-executed based on pattern trust "
                    f"(`{pattern_key}`, {trust_decision.trust_score:.0%} over "
                    f"{trust_decision.sample_size} prior observations). "
                    f"{trust_decision.reason}"
                ),
            )
            log_stage("auto_approve", state["incident_id"], "OK", action=risky_action, pattern=pattern_key)
        else:
            prior_denials = get_denial_notes_for_pattern(pattern_key)
            rationale = confidence_rationale or (
                f"(no rationale provided by the model; trust note: {trust_decision.reason})"
            )
            if prior_denials:
                notes = "; ".join(
                    f"denied by {d.get('decided_by', '?')}"
                    + (f" because: {d['denial_reason']}" if d.get("denial_reason") else " (no reason given)")
                    for d in prior_denials
                )
                rationale += f" | Prior human feedback on this pattern: {notes}"
                log_stage("feedback_loop", state["incident_id"], "WARN", pattern=pattern_key, prior_denials=len(prior_denials))

            post_approval_request(
                incident_id=state["incident_id"],
                action=risky_action,
                blast_radius=risky_blast_radius,
                confidence=state["confidence"],
                evidence=graph,
                alternatives_considered=[
                    s["content"] for s in state["steps_taken"] if s["type"] == "plan"
                ],
                confidence_rationale=rationale,
                pattern_key=pattern_key,
                service=primary_service,
            )
            log_stage("approval_request", state["incident_id"], "BLOCKED", action=risky_action, blast_radius=risky_blast_radius)

    return state


def build_graph() -> StateGraph:
    graph = StateGraph(IncidentState)
    graph.add_node("guard_input", guard_input)
    graph.add_node("plan", plan_step)
    graph.add_node("act", act_step)
    graph.add_node("synthesize", synthesize_step)

    graph.set_entry_point("guard_input")
    graph.add_edge("guard_input", "plan")
    graph.add_edge("plan", "act")
    graph.add_conditional_edges("act", check_done, {"plan": "plan", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_incident(
    alert_text: str, scenario: Optional[str] = None, skip_dedup: bool = False
) -> IncidentState:
    incident_id = str(uuid.uuid4())
    log_stage("run_incident", incident_id, "INFO", scenario=scenario or "-", alert=alert_text[:60])
    if not skip_dedup:
        dedup_result = check_and_register(alert_text, incident_id)
        if dedup_result.is_duplicate:
            log_stage("dedup_check", incident_id, "BLOCKED", matches=dedup_result.existing_incident_id)
            return {
                "incident_id": dedup_result.existing_incident_id,
                "alert_text": alert_text,
                "steps_taken": [],
                "hypothesis": (
                    f"DEDUPLICATED: matches an active investigation "
                    f"({dedup_result.existing_incident_id}) within the last "
                    f"{DEDUP_WINDOW_SECONDS // 60} minutes -- not starting a "
                    "duplicate investigation"
                ),
                "confidence": 0.0,
                "causal_graph": None,
                "needs_approval": False,
                "done": True,
                "synthesis_provider": None,
                "synthesis_input_tokens": 0,
                "synthesis_output_tokens": 0,
            }
        log_stage("dedup_check", incident_id, "OK", fingerprint=dedup_result.fingerprint)

    if scenario:
        inject_scenario(incident_id, scenario)

    get_or_issue_token(incident_id, "investigator")
    log_stage("issue_token", incident_id, "OK", role="investigator")

    initial_steps_taken: list[dict[str, Any]] = []
    if AGENT_FAST_PATH:
        initial_steps_taken = _prefetch_baseline_evidence(incident_id)

    app = build_graph()
    initial: IncidentState = {
        "incident_id": incident_id,
        "alert_text": alert_text,
        "steps_taken": initial_steps_taken,
        "hypothesis": None,
        "confidence": 0.0,
        "causal_graph": None,
        "needs_approval": False,
        "done": False,
        "synthesis_provider": None,
        "synthesis_input_tokens": 0,
        "synthesis_output_tokens": 0,
    }
    result = app.invoke(initial)

    flush_pending_writes()

    log_stage(
        "run_incident", incident_id, "DONE",
        needs_approval=result.get("needs_approval"),
        confidence=f"{result.get('confidence', 0.0):.0%}",
        steps=len(result.get("steps_taken", [])),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="db_timeout")
    parser.add_argument(
        "--alert",
        default="Latency spike detected on checkout-service, p99 up 4x in last 10 minutes.",
    )
    args = parser.parse_args()

    try:
        result = run_incident(args.alert, scenario=args.scenario)
    except ScenarioInjectionError as e:
        print(f"ERROR: {e}")
        print("Is the simulator running? Try: uvicorn simulator.main:app --port 8000")
        raise SystemExit(1)

    print(json.dumps(result, indent=2, default=str))