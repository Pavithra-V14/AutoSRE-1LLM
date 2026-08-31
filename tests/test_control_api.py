from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fresh_app(control_api_key: str | None):
    """CONTROL_API_KEY is read as a module-level constant at import time
    (see services/api/main.py) so auth behavior is fixed per-import --
    reload the module with the env var set/unset to test both states
    within one test file rather than needing separate files."""
    if control_api_key is None:
        os.environ.pop("CONTROL_API_KEY", None)
    else:
        os.environ["CONTROL_API_KEY"] = control_api_key

    import services.api.main as main_module
    importlib.reload(main_module)
    return main_module.app

class TestUnauthenticatedMode:
    @pytest.fixture(autouse=True)
    def client(self):
        app = _fresh_app(control_api_key=None)
        return TestClient(app)

    def test_health_reports_auth_not_configured(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "auth_configured": False}

    def test_pending_approvals_returns_empty_list_not_error(self, client):
        r = client.get("/approvals/pending")
        assert r.status_code == 200
        assert r.json() == []

    def test_metrics_summary_degrades_gracefully_with_no_supabase(self, client):
        r = client.get("/metrics/summary")
        assert r.status_code == 200
        assert r.json() == {"configured": False}

    def test_decide_with_invalid_decision_value_is_rejected(self, client):
        r = client.post(
            "/approvals/some-incident/decide",
            json={"decision": "maybe", "decided_by": "test"},
        )
        assert r.status_code == 400

    def test_decide_on_nonexistent_incident_is_404_not_500(self, client):
        r = client.post(
            "/approvals/does-not-exist/decide",
            json={"decision": "approve", "decided_by": "test"},
        )
        assert r.status_code == 404

    def test_decide_without_api_key_succeeds_when_auth_not_configured(self, client):
        r = client.post(
            "/approvals/does-not-exist/decide",
            json={"decision": "approve", "decided_by": "test"},
        )
        assert r.status_code == 404, "should reach business logic, not be blocked by auth"

    def test_escalation_check_endpoint_runs_without_error(self, client):
        r = client.post("/escalations/check")
        assert r.status_code == 200
        assert "escalated" in r.json()

    def test_wipe_data_without_confirmation_phrase_is_rejected(self, client):
        r = client.post("/admin/wipe-data", json={"confirmation": "delete everything please"})
        assert r.status_code == 400

    def test_wipe_data_with_correct_confirmation_degrades_gracefully_with_no_supabase(self, client):
        r = client.post("/admin/wipe-data", json={"confirmation": "DELETE ALL DATA"})
        assert r.status_code == 200
        assert r.json()["configured"] is False


# -- Auth-enforced behavior --------------------------------------------------

class TestAuthenticatedMode:
    SECRET = "test-secret-for-pytest"

    @pytest.fixture(autouse=True)
    def client(self):
        app = _fresh_app(control_api_key=self.SECRET)
        yield TestClient(app)
        os.environ.pop("CONTROL_API_KEY", None)  # don't leak into other test classes

    def test_health_reports_auth_configured(self, client):
        r = client.get("/health")
        assert r.json()["auth_configured"] is True

    def test_decide_without_key_is_rejected(self, client):
        r = client.post(
            "/approvals/some-incident/decide",
            json={"decision": "approve", "decided_by": "test"},
        )
        assert r.status_code == 401

    def test_decide_with_wrong_key_is_rejected(self, client):
        r = client.post(
            "/approvals/some-incident/decide",
            json={"decision": "approve", "decided_by": "test"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401

    def test_decide_with_correct_key_reaches_business_logic(self, client):
        # Correct key -> should get PAST auth to the 404 (no such
        # incident), not blocked at 401.
        r = client.post(
            "/approvals/some-incident/decide",
            json={"decision": "approve", "decided_by": "test"},
            headers={"X-API-Key": self.SECRET},
        )
        assert r.status_code == 404

    def test_reads_do_not_require_the_key(self, client):
        # GET endpoints stay open even with auth configured -- only
        # mutating endpoints are gated. See services/api/main.py docstring
        # for why.
        r = client.get("/approvals/pending")
        assert r.status_code == 200
        r2 = client.get("/metrics/summary")
        assert r2.status_code == 200

    def test_escalation_check_requires_key(self, client):
        r = client.post("/escalations/check")
        assert r.status_code == 401
        r2 = client.post("/escalations/check", headers={"X-API-Key": self.SECRET})
        assert r2.status_code == 200

    def test_wipe_data_requires_key_even_with_correct_confirmation(self, client):
        # Two independent safeguards on a destructive action: the API key
        # protects against an unauthorized caller, the confirmation phrase
        # protects against an authorized one clicking the wrong button.
        # Neither should substitute for the other.
        r = client.post("/admin/wipe-data", json={"confirmation": "DELETE ALL DATA"})
        assert r.status_code == 401

    def test_wipe_data_requires_confirmation_even_with_correct_key(self, client):
        r = client.post(
            "/admin/wipe-data",
            json={"confirmation": "wrong phrase"},
            headers={"X-API-Key": self.SECRET},
        )
        assert r.status_code == 400

    def test_wipe_data_with_both_correct_proceeds(self, client):
        r = client.post(
            "/admin/wipe-data",
            json={"confirmation": "DELETE ALL DATA"},
            headers={"X-API-Key": self.SECRET},
        )
        assert r.status_code == 200
        assert r.json()["configured"] is False  # no Supabase in this test env


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
