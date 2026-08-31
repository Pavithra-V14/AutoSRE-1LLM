from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.agent.escalation import (
    ESCALATION_THRESHOLD_MINUTES,
    LATENCY_P99_ALERT_THRESHOLD_MS,
    check_and_escalate,
    check_p99_latency_alert,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run a single check and exit")
    parser.add_argument("--interval", type=int, default=60, help="seconds between checks (default 60)")
    parser.add_argument(
        "--threshold",
        type=int,
        default=ESCALATION_THRESHOLD_MINUTES,
        help=f"minutes an approval can sit pending before escalating (default {ESCALATION_THRESHOLD_MINUTES})",
    )
    parser.add_argument(
        "--latency-threshold-ms",
        type=int,
        default=LATENCY_P99_ALERT_THRESHOLD_MS,
        help=f"p99 latency (ms) that triggers a latency alert (default {LATENCY_P99_ALERT_THRESHOLD_MS})",
    )
    args = parser.parse_args()

    print(
        f"Escalation monitor: approval threshold={args.threshold}min, "
        f"p99 latency threshold={args.latency_threshold_ms}ms, poll interval={args.interval}s"
    )

    def _run_checks() -> None:
        n = check_and_escalate(args.threshold)
        if n:
            print(f"Escalated {n} stale approval(s).")
        if check_p99_latency_alert(args.latency_threshold_ms):
            print("Sent a p99 latency alert.")

    if args.once:
        _run_checks()
        print("Checked once.")
        return

    try:
        while True:
            _run_checks()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nEscalation monitor stopped.")


if __name__ == "__main__":
    main()