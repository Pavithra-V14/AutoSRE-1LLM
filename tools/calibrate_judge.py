"""Measure the benchmark judge against labelled cases.

    uv run python tools/calibrate_judge.py
    uv run python tools/calibrate_judge.py --min-agreement 0.8 --min-kappa 0.6   # gate
    JUDGE_PROVIDER=gemini uv run python tools/calibrate_judge.py                 # try another judge

See services/agent/calibration.py for what the numbers mean. The default
label file is a developer-written seed set (benchmarks/judge_labels.json);
point --labels at a set labelled by independent reviewers for numbers that
should carry weight.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.agent.calibration import evaluate_judge, load_cases  # noqa: E402
from services.agent.judge import VERDICTS, CausalGraphJudge  # noqa: E402

DEFAULT_LABELS = Path(__file__).resolve().parent.parent / "benchmarks" / "judge_labels.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--min-agreement", type=float, default=None, help="exit 1 below this agreement")
    parser.add_argument("--min-kappa", type=float, default=None, help="exit 1 below this Cohen's kappa")
    parser.add_argument("--max-wait", type=float, default=120.0,
                        help="seconds to wait for a rate limit to lift before giving up on a case (0 = never wait)")
    parser.add_argument("--retries", type=int, default=3, help="retries per case after waiting out a rate limit")
    args = parser.parse_args()

    cases = load_cases(Path(args.labels))
    print(f"Judging {len(cases)} labelled case(s) from {args.labels}...\n")
    judge = CausalGraphJudge()
    report = evaluate_judge(
        judge, cases,
        retries=args.retries if args.max_wait > 0 else 0,
        wait_for=judge.seconds_until_available,
        max_wait_seconds=args.max_wait,
        sleep=lambda seconds: (print(f"  ...rate limited; waiting {seconds:.0f}s", flush=True), time.sleep(seconds))[1],
    )

    for r in report.results:
        mark = "  ?  " if r.verdict is None else ("  ok " if r.verdict == r.label else " MISS")
        print(f"{mark} {r.case_id:34s} human={r.label:9s} judge={r.verdict or '(none)':9s} {r.error}")

    print(f"\nAgreement:            {report.agreement:.1%}")
    print(f"Cohen's kappa:        {report.kappa:.2f}   (0.6+ substantial, 0.8+ near-perfect)")
    print(f"Pass/fail agreement:  {report.pass_agreement:.1%}   (does the judge agree on 'correct' vs not?)")
    if report.unscored:
        print(f"WARNING: {report.unscored} case(s) could not be graded and are excluded from the numbers.")

    print("\nConfusion (rows = human label, columns = judge verdict):")
    print(f"{'':12s}" + "".join(f"{v:>11s}" for v in VERDICTS))
    for human in VERDICTS:
        print(f"{human:12s}" + "".join(f"{report.confusion[human][v]:>11d}" for v in VERDICTS))

    lenient = sum(1 for r in report.disagreements if r.label != "correct" and r.verdict == "correct")
    harsh = sum(1 for r in report.disagreements if r.label == "correct" and r.verdict != "correct")
    if report.disagreements:
        print(f"\nDisagreements: {len(report.disagreements)} (judge too lenient on {lenient}, too harsh on {harsh})")
        for r in report.disagreements:
            print(f"  - {r.case_id}: human={r.label}, judge={r.verdict}  ({r.note})")

    failed = False
    if args.min_agreement is not None and report.agreement < args.min_agreement:
        print(f"\nFAIL: agreement {report.agreement:.1%} < {args.min_agreement:.1%}")
        failed = True
    if args.min_kappa is not None and report.kappa < args.min_kappa:
        print(f"\nFAIL: kappa {report.kappa:.2f} < {args.min_kappa:.2f}")
        failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
