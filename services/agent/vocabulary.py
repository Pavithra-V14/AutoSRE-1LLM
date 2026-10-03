"""Controlled vocabulary for causal-graph nodes.

Free-text node names ("config_change", "db_pool_size_reduction",
"ConfigChange") made two things unreliable:

  * pattern trust (services/agent/trust.py) keys on the graph, so the same
    failure fragmented into many one-off patterns and never accumulated the
    samples needed for graduated autonomy;
  * grading and de-duplication had to guess at synonyms.

Every node is now ``<kind>:<slug>``:

  trigger kinds  deploy | config | flag | traffic | dependency | capacity |
                 maintenance | transient | unknown      (where it started)
  mechanism      how the trigger produced the failure   (pool_exhaustion)
  symptom        what was observed                      (latency_spike)

Edges point from cause to effect, so an edge's ``to`` is always a mechanism
or a symptom -- a trigger is never an effect.
"""

from __future__ import annotations

import re
from typing import Any, Optional

TRIGGER_KINDS: tuple[str, ...] = (
    "deploy", "config", "flag", "traffic", "dependency", "capacity",
    "maintenance", "transient", "unknown",
)
EFFECT_KINDS: tuple[str, ...] = ("mechanism", "symptom")
NODE_KINDS: tuple[str, ...] = TRIGGER_KINDS + EFFECT_KINDS

# Spellings models reach for despite the instructions -> canonical kind.
_KIND_ALIASES = {
    "deployment": "deploy", "release": "deploy", "rollout": "deploy",
    "config_change": "config", "configuration": "config", "change": "config",
    "feature_flag": "flag", "feature": "flag",
    "traffic_spike": "traffic", "load": "traffic",
    "external": "dependency", "dependency_outage": "dependency", "upstream": "dependency",
    "resource": "capacity", "quota": "capacity",
    "scheduled_maintenance": "maintenance", "planned": "maintenance",
    "blip": "transient", "noise": "transient",
    "effect": "symptom", "impact": "symptom", "cause": "mechanism",
}

_SLUG_RE = re.compile(r"[^a-z0-9._\-]+")
_NODE_RE = re.compile(r"^(?P<kind>[a-z_]+):(?P<slug>[a-z0-9][a-z0-9._\-]*)$")

VOCABULARY_GUIDANCE = (
    'Node naming -- every edge "from" and "to" MUST be "<kind>:<slug>" with a '
    "lowercase snake_case slug, where <kind> is exactly one of:\n"
    "  trigger kinds (where the problem started): "
    + ", ".join(TRIGGER_KINDS)
    + "\n  mechanism (how the trigger caused the failure), symptom (what was observed)\n"
    'Edges run cause -> effect, so "to" must be a mechanism or a symptom, '
    "never a trigger. For a deploy or config change put its id in the slug "
    '(e.g. "deploy:rel-892"). If no cause can be established use '
    '"unknown:undetermined".\n'
    "Examples:\n"
    '  deploy:cfg-4471 -> mechanism:connection_pool_exhaustion -> symptom:latency_spike\n'
    '  dependency:payment_gateway -> mechanism:upstream_timeout -> symptom:checkout_errors\n'
    '  transient:metric_blip -> symptom:latency_spike   (self-resolved, low confidence)'
)


def normalize_node(raw: Any) -> str:
    """Best-effort canonicalization: case, separators, kind aliases. Anything
    that still does not fit ``<kind>:<slug>`` is returned stripped but
    otherwise unchanged so the validator can report it -- a node is never
    guessed into a kind."""
    text = str(raw or "").strip()
    kind, sep, slug = text.partition(":")
    if not sep:
        return text
    kind = re.sub(r"[\s\-]+", "_", kind.strip().lower())
    kind = _KIND_ALIASES.get(kind, kind)
    slug = _SLUG_RE.sub("_", slug.strip().lower()).strip("_.-")
    return f"{kind}:{slug}" if kind and slug else text


def normalize_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Normalizes every edge's nodes in place. Tolerates malformed edges --
    schema validation reports those."""
    for edge in graph.get("edges") or []:
        if isinstance(edge, dict):
            for key in ("from", "to"):
                if key in edge:
                    edge[key] = normalize_node(edge[key])
    return graph


def validate_graph_vocabulary(graph: dict[str, Any]) -> Optional[str]:
    """None if every node conforms, else a reason the model can act on."""
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        edge_id = edge.get("id", "?")
        for key in ("from", "to"):
            node = str(edge.get(key, ""))
            match = _NODE_RE.match(node)
            if not match or match.group("kind") not in NODE_KINDS:
                return (
                    f'edge {edge_id}: "{key}" value {node!r} is not "<kind>:<slug>" with kind one of '
                    f"{list(NODE_KINDS)}"
                )
        to_kind = _NODE_RE.match(str(edge.get("to", ""))).group("kind")  # type: ignore[union-attr]
        if to_kind in TRIGGER_KINDS:
            return (
                f'edge {edge_id}: "to" is a {to_kind!r} trigger -- edges run cause -> effect, '
                'so "to" must be a mechanism or a symptom'
            )
    return None


def _split(node: str) -> tuple[str, str]:
    match = _NODE_RE.match(node)
    return (match.group("kind"), match.group("slug")) if match else ("", "")


def root_trigger(graph: Optional[dict[str, Any]]) -> Optional[str]:
    """First node that is only ever a cause (never an effect) and is a
    trigger kind."""
    edges = [e for e in (graph or {}).get("edges") or [] if isinstance(e, dict)]
    effects = {str(e.get("to", "")) for e in edges}
    for edge in edges:
        source = str(edge.get("from", ""))
        if source not in effects and _split(source)[0] in TRIGGER_KINDS:
            return source
    return None


def pattern_key(graph: Optional[dict[str, Any]]) -> Optional[str]:
    """Stable key for pattern trust: ``<trigger kind>/<first effect slug>``,
    e.g. ``deploy/connection_pool_exhaustion``. Specific ids are dropped on
    purpose -- a *different* deploy causing the *same* failure is the same
    pattern. Returns None for graphs not in vocabulary form so the caller
    can fall back to the legacy key."""
    root = root_trigger(graph)
    if root is None:
        return None
    for edge in (graph or {}).get("edges") or []:
        if isinstance(edge, dict) and str(edge.get("from", "")) == root:
            kind, slug = _split(str(edge.get("to", "")))
            if kind in EFFECT_KINDS and slug:
                return f"{_split(root)[0]}/{slug}"[:64]
    return None
