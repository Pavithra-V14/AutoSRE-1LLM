"""The alert must reach the planner intact; only real PII is redacted."""

from __future__ import annotations

import pytest

from services.agent import guardrails
from services.agent.untrusted import PII_ENTITIES

ALERTS = [
    "ALERT checkout-service: p99 latency 4x baseline and error rate above 5% for the last 10 minutes.",
    "ALERT inventory-service: error rate above 30% for the last 5 minutes.",
    "ALERT orders-db: write latency 30x baseline and write failures reported.",
    "ALERT search-service: p99 latency 4x baseline for the last 30 minutes.",
]


def test_the_entity_list_names_only_personal_identifiers():
    assert "DATE_TIME" not in PII_ENTITIES and "US_DRIVER_LICENSE" not in PII_ENTITIES and "LOCATION" not in PII_ENTITIES
    assert {"EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"} <= set(PII_ENTITIES)


def test_entities_are_forwarded_to_the_analyzer(monkeypatch):
    seen = {}

    class Analyzer:
        def analyze(self, text, language, entities):
            seen["entities"] = entities
            return []

    class Anonymizer:
        def anonymize(self, text, analyzer_results):
            return type("R", (), {"text": text})()

    monkeypatch.setattr(guardrails, "_get_presidio_engines", lambda: (Analyzer(), Anonymizer()))
    assert guardrails.redact_pii("hello", entities=PII_ENTITIES) == "hello"
    assert seen["entities"] == PII_ENTITIES


def test_a_missing_nlp_model_degrades_instead_of_crashing(monkeypatch, capsys):
    pytest.importorskip("presidio_analyzer")
    import presidio_analyzer.nlp_engine as nlp

    def boom(self):
        raise OSError("[E050] Can't find model 'en_core_web_sm'")

    monkeypatch.setattr(nlp.NlpEngineProvider, "create_engine", boom)
    monkeypatch.setattr(guardrails, "_presidio_engines", None)
    monkeypatch.setattr(guardrails, "_presidio_load_attempted", False)

    assert guardrails.redact_pii("call me at 555-0100", entities=PII_ENTITIES) == "call me at 555-0100"
    assert "Presidio unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("alert", ALERTS)
def test_real_alerts_survive_redaction_unchanged(alert):
    """Run against the real engine when it is installed: the old default
    turned 'p99' into <US_DRIVER_LICENSE> and 'the last 10 minutes' into
    <DATE_TIME>."""
    pytest.importorskip("presidio_analyzer")
    guardrails._presidio_engines = None
    guardrails._presidio_load_attempted = False
    if guardrails._get_presidio_engines()[0] is None:
        pytest.skip("Presidio / spaCy model not available in this environment")
    assert guardrails.redact_pii(alert, entities=PII_ENTITIES) == alert


def test_actual_pii_is_still_redacted():
    pytest.importorskip("presidio_analyzer")
    guardrails._presidio_engines = None
    guardrails._presidio_load_attempted = False
    if guardrails._get_presidio_engines()[0] is None:
        pytest.skip("Presidio / spaCy model not available in this environment")
    out = guardrails.redact_pii("page oncall@example.com about checkout-service", entities=PII_ENTITIES)
    assert "oncall@example.com" not in out and "checkout-service" in out
