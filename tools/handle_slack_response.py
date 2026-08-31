from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.agent.approval import execute_approved_action, record_decision
from services.agent.persistence import get_pending_approvals


def list_pending() -> None:
    pending = get_pending_approvals()
    if not pending:
        print("No pending approvals (or Supabase isn't configured -- see .env.example).")
        return

    print(f"{len(pending)} pending approval(s):\n")
    for row in pending:
        print(
            f"  incident={row['incident_id']}  action={row['action']}  "
            f"service={row.get('service', 'n/a')}  "
            f"blast_radius={row['blast_radius']}  confidence={row['confidence']:.0%}  "
            f"pattern={row.get('pattern_key', 'n/a')}  created={row['created_at']}"
        )
        if row.get("confidence_rationale"):
            print(f"    rationale: {row['confidence_rationale']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="show all pending approvals")
    parser.add_argument("--incident", help="incident_id to decide on")
    parser.add_argument("--decision", choices=["approve", "deny"])
    parser.add_argument("--by", help="your name, for the audit trail")
    parser.add_argument(
        "--reason",
        help="Item 6: why you denied this (only meaningful with --decision deny).",
    )
    args = parser.parse_args()

    if args.list:
        list_pending()
        return

    if not (args.incident and args.decision and args.by):
        parser.error("--incident, --decision, and --by are required unless using --list")

    pending = get_pending_approvals(args.incident)
    action = pending[0]["action"] if pending else None
    service = pending[0].get("service") if pending else None

    if args.decision == "deny" and not args.reason:
        print("Tip: pass --reason so future approvals for this pattern show why this was denied.")

    result = record_decision(args.incident, args.decision, args.by, reason=args.reason)
    if not result:
        print(
            f"Could not record decision for incident {args.incident} "
            "(no pending row found, or Supabase not configured)."
        )
        return

    print(f"Recorded: incident {args.incident} -> {result['status']} by {args.by}")

    if args.decision == "approve" and action:
        exec_result = execute_approved_action(args.incident, action, service)
        if exec_result["executed"]:
            print(f"Executed: {action} -> {exec_result['result']}")
        else:
            print(f"({exec_result['error']})")


if __name__ == "__main__":
    main()
