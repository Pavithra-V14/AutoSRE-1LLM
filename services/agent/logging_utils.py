from __future__ import annotations

import os
import time
ENABLED = os.getenv("LOG_STAGES", "1") != "0"

STATUS_COLOR = {
    "OK": "\033[92m",       # green
    "PASS": "\033[92m",
    "BLOCKED": "\033[91m",  # red
    "DENIED": "\033[91m",
    "ABORT": "\033[91m",
    "FAIL": "\033[91m",
    "WARN": "\033[93m",     # yellow
    "SKIP": "\033[93m",
    "INFO": "\033[94m",     # blue
}
RESET = "\033[0m"
_NO_COLOR = os.getenv("NO_COLOR") is not None or not os.getenv("TERM")


def log_stage(stage: str, incident_id: str, status: str, **details: object) -> None:
    if not ENABLED:
        return

    ts = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
    short_id = (incident_id or "")[:8] or "--------"
    detail_str = " ".join(f"{k}={v}" for k, v in details.items())

    color = "" if _NO_COLOR else STATUS_COLOR.get(status, "")
    reset = "" if _NO_COLOR else RESET

    print(
        f"[{ts}] {stage:<18} incident={short_id} {color}{status:<8}{reset} {detail_str}",
        flush=True,
    )
