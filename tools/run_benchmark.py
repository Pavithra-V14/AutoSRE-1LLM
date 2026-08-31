from __future__ import annotations

import argparse
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.agent.evaluation import RunEvaluation, aggregate_metrics, evaluate_run
from services.agent.graph import run_incident, ScenarioInjectionError
from services.agent.persistence import compute_equivalent_cost_usd, insert_benchmark_run
from services.agent.trust import compute_pattern_key, record_benchmark_outcome
from services.simulator.scenarios import SCENARIOS

SUCCESS_THRESHOLD = 0.5  


def _extract_nodes(ground_truth: str) -> list[str]:
    """'deploy:cfg-4471 -> pool_size=10 -> connection_exhaustion -> latency_spike'
    -> ['deploy', 'cfg', '4471', 'pool', 'size', '10', 'connection',
        'exhaustion', 'latency', 'spike']"""
    raw_nodes = [n.strip() for n in ground_truth.split("->")]
    words: list[str] = []
    for node in raw_nodes:
        words.extend(w.lower() for w in re.split(r"[^a-zA-Z0-9]+", node) if len(w) > 2)
    return words


def _graph_text(causal_graph: dict[str, Any]) -> str:
    parts = []
    for edge in causal_graph.get("edges", []):
        parts.append(str(edge.get("from", "")))
        parts.append(str(edge.get("to", "")))
        parts.append(str(edge.get("id", "")))
    return " ".join(parts).lower()


def score_against_ground_truth(causal_graph: dict[str, Any], ground_truth: str) -> tuple[bool, float]:
    nodes = _extract_nodes(ground_truth)
    if not nodes:
        return False, 0.0
    graph_text = _graph_text(causal_graph)
    matched = sum(1 for n in nodes if n in graph_text)
    match_ratio = matched / len(nodes)
    return match_ratio >= SUCCESS_THRESHOLD, round(match_ratio, 3)


def run_one_scenario(scenario_key: str, scenario: dict[str, Any]) -> dict[str, Any]:
    alert_text = f"Incident detected: {scenario['description']}"
    start = time.time()
    expected_tools = scenario.get("expected_tools", [])

    try:
        result = run_incident(alert_text, scenario=scenario_key, skip_dedup=True)
    except ScenarioInjectionError as e:
        empty_result = {"hypothesis": f"ABORTED: {e}", "causal_graph": {"edges": []}, "steps_taken": [], "needs_approval": False}
        return {
            "scenario": scenario_key,
            "success": False,
            "match_ratio": 0.0,
            "cost_usd": 0.0,
            "steps_taken": 0,
            "wall_clock_s": round(time.time() - start, 2),
            "error": f"simulator unreachable: {e}",
            "evaluation": evaluate_run(scenario_key, expected_tools, empty_result, 0.0),
        }
    except Exception as e:  # LLM providers all failing, etc.
        empty_result = {"hypothesis": f"ABORTED: {e}", "causal_graph": {"edges": []}, "steps_taken": [], "needs_approval": False}
        return {
            "scenario": scenario_key,
            "success": False,
            "match_ratio": 0.0,
            "cost_usd": 0.0,
            "steps_taken": 0,
            "wall_clock_s": round(time.time() - start, 2),
            "error": f"agent run failed: {e}",
            "evaluation": evaluate_run(scenario_key, expected_tools, empty_result, 0.0),
        }

    causal_graph = result.get("causal_graph") or {"edges": []}
    success, match_ratio = score_against_ground_truth(causal_graph, scenario["ground_truth_cause"])

    pattern_key = compute_pattern_key(causal_graph)
    record_benchmark_outcome(pattern_key, result.get("incident_id", ""), success)

    total_cost = 0.0
    for step in result.get("steps_taken", []):
        if step["type"] == "plan":
            total_cost += compute_equivalent_cost_usd(
                step.get("provider", ""), step.get("input_tokens", 0), step.get("output_tokens", 0)
            )
    total_cost += compute_equivalent_cost_usd(
        result.get("synthesis_provider", ""),
        result.get("synthesis_input_tokens", 0),
        result.get("synthesis_output_tokens", 0),
    )
    total_cost = round(total_cost, 6)

    return {
        "scenario": scenario_key,
        "success": success,
        "match_ratio": match_ratio,
        "cost_usd": total_cost,
        "steps_taken": len(result.get("steps_taken", [])),
        "wall_clock_s": round(time.time() - start, 2),
        "confidence": result.get("confidence", 0.0),
        "needs_approval": result.get("needs_approval", False),
        "pattern_key": pattern_key,
        "error": None,
        "evaluation": evaluate_run(scenario_key, expected_tools, result, total_cost),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None, help="run a single scenario by key")
    parser.add_argument("--min-success-rate", type=float, default=None,
                         help="exit 1 if success rate falls below this (for CI)")
    args = parser.parse_args()

    scenarios = (
        {args.scenario: SCENARIOS[args.scenario]}
        if args.scenario
        else SCENARIOS
    )

    run_id = str(uuid.uuid4())
    results = []
    print(f"Running benchmark ({len(scenarios)} scenario(s))...\n")

    for key, scenario in scenarios.items():
        r = run_one_scenario(key, scenario)
        results.append(r)

        status = "PASS" if r["success"] else "FAIL"
        error_note = f"  [{r['error']}]" if r["error"] else ""
        print(
            f"  [{status}] {key:30s} match={r['match_ratio']:.2f} "
            f"steps={r['steps_taken']:2d} cost=${r['cost_usd']:.6f} "
            f"({r['wall_clock_s']}s){error_note}"
        )

        insert_benchmark_run(
            run_id=run_id,
            scenario=key,
            success=r["success"],
            ground_truth_match=r["success"],
            total_cost_usd=r["cost_usd"],
            cpso=0.0, 
            steps_taken=r["steps_taken"],
            tool_selection_f1=r["evaluation"].tool_selection_f1,
            needed_approval=r["evaluation"].needed_approval,
            had_degradation=r["evaluation"].had_degradation,
            recovered=r["evaluation"].recovered,
        )

    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / n if n else 0.0
    mean_cost = sum(r["cost_usd"] for r in results) / n if n else 0.0
    cpso = (mean_cost / success_rate) if success_rate > 0 else float("inf")

    evaluations: list[RunEvaluation] = [r["evaluation"] for r in results]
    success_flags = [r["success"] for r in results]
    metrics = aggregate_metrics(evaluations, success_flags)

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"1. Agent success rate:          {n_success}/{n} ({success_rate:.1%})")
    print(
        f"2. Tool selection accuracy:     F1={metrics['tool_selection_accuracy_f1']:.1%}  "
        f"(recall={metrics['tool_selection_accuracy_recall']:.1%} -- did runs call the tools "
        f"they actually needed?)"
    )
    print(
        f"3. Human requests:              {metrics['human_requests_total']}/{n} runs "
        f"({metrics['human_request_rate']:.1%} needed approval)"
    )
    recovery_str = (
        f"{metrics['recovery_rate']:.1%} over {metrics['runs_with_degradation']} degraded run(s)"
        if metrics["recovery_rate"] is not None
        else "undefined (no run hit a degradation this pass)"
    )
    print(f"4. Recovery rate:               {recovery_str}")
    completion_cost_str = (
        f"${metrics['cost_per_task_completion_usd']:.6f}"
        if metrics["cost_per_task_completion_usd"] is not None
        else "undefined (no run reached a conclusion)"
    )
    print(f"5. Cost per task completion:    {completion_cost_str}")
    print("=" * 60)
    print(f"\nMean cost/incident (equivalent paid rate): ${mean_cost:.6f}")
    print(f"CPSO (Cost Per Successful Outcome):        ${cpso:.6f}" if cpso != float("inf")
          else "CPSO: undefined (0 successes)")
    print(
        "\nNote: cost figures use EQUIVALENT_PAID_RATES in agent/persistence.py "
        "-- actual spend on the free-tier stack is $0, this states what the "
        "same token volume would cost on comparable paid models, so CPSO and "
        "cost-per-completion are meaningful benchmark numbers rather than "
        "trivially zero. Tool selection accuracy is scored against each "
        "scenario's expected_tools list (services/simulator/scenarios.py) -- "
        "read-only investigative tools only, remediation actions excluded."
    )

    if args.min_success_rate is not None and success_rate < args.min_success_rate:
        print(
            f"\nFAIL: success rate {success_rate:.1%} is below the "
            f"required {args.min_success_rate:.1%} threshold."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
