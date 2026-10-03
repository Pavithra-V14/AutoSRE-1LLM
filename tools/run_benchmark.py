from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.agent.benchmark_stats import (
    DEFAULT_MAX_DROP,
    compare_to_baseline,
    results_payload,
    status_counts,
    summarize_scenarios,
    wilson_interval,
)
from services.agent.evaluation import RunEvaluation, aggregate_metrics, evaluate_run
from services.agent.graph import run_incident, ScenarioInjectionError
from services.agent.judge import CausalGraphJudge
from services.agent.persistence import compute_equivalent_cost_usd, insert_benchmark_run
from services.agent.scoring import score_against_ground_truth, score_run  # noqa: F401 (score_against_ground_truth re-exported for callers of the old scorer)
from services.agent.trust import compute_pattern_key, record_benchmark_outcome
from services.simulator.scenarios import SCENARIOS, SMOKE_SCENARIOS
import os
from dotenv import load_dotenv
load_dotenv(override=True)
def run_one_scenario(
    scenario_key: str, scenario: dict[str, Any], judge: Optional[CausalGraphJudge]
) -> dict[str, Any]:
    # Symptom-only alert. scenario["description"] states the cause and must
    # never be shown to the agent (see ALERTS in scenarios.py).
    alert_text = scenario["alert"]
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
            "score_method": "aborted",
            "score_rationale": f"simulator unreachable: {e}",
            "judge_cost_usd": 0.0,
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
            "score_method": "aborted",
            "score_rationale": f"agent run failed: {e}",
            "judge_cost_usd": 0.0,
            "cost_usd": 0.0,
            "steps_taken": 0,
            "wall_clock_s": round(time.time() - start, 2),
            "error": f"agent run failed: {e}",
            "evaluation": evaluate_run(scenario_key, expected_tools, empty_result, 0.0),
        }

    causal_graph = result.get("causal_graph") or {"edges": []}
    score = score_run(scenario, result, judge=judge)

    pattern_key = compute_pattern_key(causal_graph)
    record_benchmark_outcome(pattern_key, result.get("incident_id", ""), score.success)

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
        "success": score.success,
        "match_ratio": score.score,
        "score_method": score.method,
        "score_rationale": score.rationale,
        "judge_cost_usd": round(score.judge_cost_usd, 6),
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
    parser.add_argument("--runs", type=int, default=int(os.getenv("BENCHMARK_RUNS", "1")),
                         help="run each scenario N times and report per-scenario pass rate, a 95%% "
                              "confidence interval and which scenarios are flaky (default 1; the "
                              "--min-success-rate gate then applies to the pass rate over ALL runs)")
    parser.add_argument("--smoke", action="store_true",
                         help="run only the small SMOKE_SCENARIOS subset (fast; for PR checks)")
    parser.add_argument("--output", default=None, help="write machine-readable results (JSON) to this path")
    parser.add_argument("--save-baseline", default=None,
                         help="write this run's results as a baseline file (commit it once you trust the numbers)")
    parser.add_argument("--baseline", default=None,
                         help="compare against a saved baseline and exit 1 on a regression; a missing file "
                              "is a notice, not an error (first run)")
    parser.add_argument("--max-drop", type=float, default=DEFAULT_MAX_DROP,
                         help="largest per-scenario pass-rate drop vs the baseline that still counts as noise "
                              f"(default {DEFAULT_MAX_DROP}: one flaky failure in 3 runs)")
    parser.add_argument("--no-judge", action="store_true",
                         help="grade with the legacy word-overlap scorer instead of the LLM judge "
                              "(offline / no extra LLM calls; negative scenarios are still scored on their flags)")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    judge = None if args.no_judge else CausalGraphJudge()

    if args.scenario and args.smoke:
        parser.error("--scenario and --smoke are mutually exclusive")
    if args.scenario:
        scenarios = {args.scenario: SCENARIOS[args.scenario]}
    elif args.smoke:
        scenarios = {name: SCENARIOS[name] for name in SMOKE_SCENARIOS}
    else:
        scenarios = SCENARIOS

    run_id = str(uuid.uuid4())
    results = []
    repeat = f" x {args.runs} runs" if args.runs > 1 else ""
    print(f"Running benchmark ({len(scenarios)} scenario(s){repeat})...\n")

    schedule = [(key, scenario, i) for key, scenario in scenarios.items() for i in range(1, args.runs + 1)]
    for key, scenario, run_number in schedule:
        r = run_one_scenario(key, scenario, judge)
        results.append(r)

        status = "PASS" if r["success"] else "FAIL"
        error_note = f"  [{r['error']}]" if r["error"] else ""
        label = f"{key} #{run_number}/{args.runs}" if args.runs > 1 else key
        print(
            f"  [{status}] {label:34s} score={r['match_ratio']:.2f} via={r['score_method']:<16s}"
            f"steps={r['steps_taken']:2d} cost=${r['cost_usd']:.6f} "
            f"({r['wall_clock_s']}s){error_note}"
        )
        print(f"         -> {r['score_rationale'][:200]}")

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
            score_method=r["score_method"],
            score=r["match_ratio"],
            score_rationale=r["score_rationale"],
            run_index=run_number,
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
    low, high = wilson_interval(n_success, n)
    print(f"1. Agent success rate:          {n_success}/{n} ({success_rate:.1%}, 95% CI {low:.0%}-{high:.0%})")
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

    stats = summarize_scenarios(results)
    if args.runs > 1:
        print("\nPer-scenario stability (" + f"{args.runs} runs each):")
        for item in sorted(stats, key=lambda x: (x.pass_rate, x.scenario)):
            print(f"  {item.passes}/{item.runs}  {item.status:<12s} mean_score={item.mean_score:.2f}  {item.scenario}")
        counts = status_counts(stats)
        print("Stability: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if counts.get("flaky"):
            print(
                f"NOTE: {counts['flaky']} scenario(s) are flaky -- single-run results for them are noise. "
                "Compare pass rates, not individual PASS/FAIL lines."
            )

    by_method: dict[str, int] = {}
    for r in results:
        by_method[r["score_method"]] = by_method.get(r["score_method"], 0) + 1
    print("\nScored via: " + ", ".join(f"{m}={c}" for m, c in sorted(by_method.items())))
    if by_method.get("lexical_fallback"):
        print(
            f"WARNING: the LLM judge was unavailable for {by_method['lexical_fallback']} scenario(s); "
            "those used the word-overlap fallback and are less reliable -- re-run when the judge is reachable."
        )
    judge_cost = sum(r["judge_cost_usd"] for r in results)
    if judge_cost:
        print(f"Judge cost (equivalent paid rate, NOT included in CPSO below): ${judge_cost:.6f}")

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

    payload = results_payload(stats, args.runs)
    for path, label in ((args.output, "results"), (args.save_baseline, "baseline")):
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nWrote {label} to {target}")

    failed = False

    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            print(
                f"\nNOTICE: no baseline at {baseline_path} -- nothing to compare against. "
                "Re-run with --save-baseline once you trust these numbers."
            )
        else:
            if args.runs < 3:
                print(
                    f"\nNOTE: comparing {args.runs} run(s) per scenario to a baseline is noisy; "
                    "use --runs 3 or more for a trustworthy gate."
                )
            comparison = compare_to_baseline(stats, json.loads(baseline_path.read_text(encoding="utf-8")), args.max_drop)
            print(
                f"\nBaseline comparison ({baseline_path}): current {comparison.current_rate:.1%} "
                f"(95% CI {comparison.current_ci[0]:.0%}-{comparison.current_ci[1]:.0%}) vs baseline "
                f"{comparison.baseline_rate:.1%} over the shared scenarios"
            )
            for name, was, now in comparison.regressions:
                print(f"  REGRESSION  {name}: pass rate {was:.0%} -> {now:.0%}")
            for name, was, now in comparison.improvements:
                print(f"  improved    {name}: pass rate {was:.0%} -> {now:.0%}")
            if comparison.new_scenarios:
                print(f"  new (no baseline yet): {', '.join(comparison.new_scenarios)}")
            if comparison.missing_scenarios:
                print(f"  not run this time: {len(comparison.missing_scenarios)} scenario(s) from the baseline")
            if comparison.overall_regressed:
                print("  REGRESSION  overall: the drop is larger than run-to-run noise at this sample size")
            if comparison.failed:
                print("\nFAIL: regression against the baseline.")
                failed = True

    if args.min_success_rate is not None and success_rate < args.min_success_rate:
        print(
            f"\nFAIL: success rate {success_rate:.1%} is below the "
            f"required {args.min_success_rate:.1%} threshold."
        )
        failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
