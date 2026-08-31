from __future__ import annotations

from nemoguardrails.actions import action

SUSPICIOUS_MARKERS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard the system prompt",
    "disregard previous instructions",
    "you are now",
    "new instructions:",
    "system prompt:",
    "act as if",
]


class ScopeCheckResult:
    def __init__(self, is_suspicious: bool, reason: str = ""):
        self.is_suspicious = is_suspicious
        self.reason = reason


@action(name="CheckAlertScopeAction")
async def check_alert_scope(context: dict | None = None, **kwargs) -> ScopeCheckResult:
    user_message = (context or {}).get("user_message", "") if context else ""
    lowered = user_message.lower()

    for marker in SUSPICIOUS_MARKERS:
        if marker in lowered:
            return ScopeCheckResult(is_suspicious=True, reason=f"matched pattern: '{marker}'")

    return ScopeCheckResult(is_suspicious=False)
