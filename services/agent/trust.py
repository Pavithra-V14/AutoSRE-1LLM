from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from services.agent.persistence import get_pattern_trust_stats, record_trust_event

MIN_SAMPLE_SIZE = 5
TRUST_THRESHOLD = 0.9 
MIN_CONFIDENCE_FOR_AUTO_APPROVAL = 0.75 

def compute_pattern_key(causal_graph: Optional[dict[str, Any]]) -> str:
    """Slugs the causal graph's first edge's 'to' node (the proximate
    cause/symptom) into a stable-ish key, e.g. 'connection_pool_exhaustion'.
    Falls back to 'unclassified' if no graph or edges are available (e.g.
    a rejected input or an aborted synthesis)."""
    if not causal_graph:
        return "unclassified"

    edges = causal_graph.get("edges", [])
    if not edges:
        return "unclassified"

    proximate_cause = str(edges[0].get("to", "")).strip().lower()
    if not proximate_cause:
        return "unclassified"

    slug = re.sub(r"[^a-z0-9]+", "_", proximate_cause).strip("_")
    return slug[:64] or "unclassified"  


@dataclass
class TrustDecision:
    pattern_key: str
    sample_size: int
    trust_score: float
    auto_approve_eligible: bool
    reason: str


def evaluate_trust(pattern_key: str, blast_radius: str, confidence: float) -> TrustDecision:
    """The actual graduated-autonomy decision. Only ever returns
    auto_approve_eligible=True for blast_radius == 'medium' -- high-risk
    actions always require a human, no matter how trusted the pattern is."""
    if blast_radius != "medium":
        return TrustDecision(
            pattern_key=pattern_key,
            sample_size=0,
            trust_score=0.0,
            auto_approve_eligible=False,
            reason=f"trust-based auto-approval only applies to 'medium' blast radius (got '{blast_radius}')",
        )

    stats = get_pattern_trust_stats(pattern_key)
    total = stats.get("total", 0)
    positive = stats.get("positive", 0)
    negative = stats.get("negative", 0)
    trust_score = (positive / total) if total > 0 else 0.0

    if total < MIN_SAMPLE_SIZE:
        return TrustDecision(
            pattern_key=pattern_key,
            sample_size=total,
            trust_score=trust_score,
            auto_approve_eligible=False,
            reason=f"only {total} prior observation(s) for this pattern, need {MIN_SAMPLE_SIZE}+",
        )

    if negative > 0:
        return TrustDecision(
            pattern_key=pattern_key,
            sample_size=total,
            trust_score=trust_score,
            auto_approve_eligible=False,
            reason=f"pattern has {negative} denial/failure event(s) on record -- disqualified from auto-approval",
        )

    if trust_score < TRUST_THRESHOLD:
        return TrustDecision(
            pattern_key=pattern_key,
            sample_size=total,
            trust_score=trust_score,
            auto_approve_eligible=False,
            reason=f"trust score {trust_score:.0%} below the {TRUST_THRESHOLD:.0%} threshold",
        )

    if confidence < MIN_CONFIDENCE_FOR_AUTO_APPROVAL:
        return TrustDecision(
            pattern_key=pattern_key,
            sample_size=total,
            trust_score=trust_score,
            auto_approve_eligible=False,
            reason=(
                f"pattern is trusted ({trust_score:.0%} over {total} samples) but this "
                f"incident's own confidence ({confidence:.0%}) is below the "
                f"{MIN_CONFIDENCE_FOR_AUTO_APPROVAL:.0%} floor -- trust widens autonomy, "
                "it doesn't replace per-incident judgment"
            ),
        )

    return TrustDecision(
        pattern_key=pattern_key,
        sample_size=total,
        trust_score=trust_score,
        auto_approve_eligible=True,
        reason=(
            f"pattern '{pattern_key}' has a clean {trust_score:.0%} trust score over "
            f"{total} prior observations with zero denials, and this incident's "
            f"confidence ({confidence:.0%}) clears the floor -- auto-executing with "
            "a notification instead of a blocking approval"
        ),
    )


def record_approval_outcome(pattern_key: str, incident_id: str, approved: bool) -> None:
    record_trust_event(
        pattern_key, "approved" if approved else "denied", incident_id
    )


def record_benchmark_outcome(pattern_key: str, incident_id: str, success: bool) -> None:
    record_trust_event(
        pattern_key, "benchmark_success" if success else "benchmark_fail", incident_id
    )


def record_auto_approval(pattern_key: str, incident_id: str) -> None:
    record_trust_event(pattern_key, "auto_approved", incident_id)
