from __future__ import annotations

import base64
import json

import pytest

from services.agent import persistence


def _jwt(role):
    def b64(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{b64({'alg': 'HS256', 'typ': 'JWT'})}.{b64({'iss': 'supabase', 'role': role})}.signature"


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(persistence, "_warned_anon_key", False)
    monkeypatch.delenv("SUPABASE_REQUIRE_SERVICE_ROLE", raising=False)


@pytest.mark.parametrize("key, role", [
    (_jwt("anon"), "anon"),
    (_jwt("service_role"), "service_role"),
    ("not-a-jwt", None),
    ("", None),
    (None, None),
    ("a.b.c", None),
    ("eyJhbGciOiJIUzI1NiJ9." + base64.urlsafe_b64encode(b"[1,2]").decode() + ".x", None),  # payload is not an object
])
def test_key_role_reads_the_label_without_verifying(key, role):
    assert persistence.key_role(key) == role


def test_service_role_key_is_silent(capsys):
    persistence.check_backend_key(_jwt("service_role"))
    assert capsys.readouterr().out == ""


def test_anon_key_warns_once_and_never_prints_the_key(capsys):
    key = _jwt("anon")
    persistence.check_backend_key(key)
    persistence.check_backend_key(key)
    out = capsys.readouterr().out
    assert out.count("not 'service_role'") == 1
    assert "'anon'" in out and key not in out and "signature" not in out


def test_unknown_key_format_also_warns(capsys):
    persistence.check_backend_key("sb_secret_abc123")
    assert "'unknown'" in capsys.readouterr().out


def test_require_flag_turns_the_warning_into_an_error(monkeypatch):
    monkeypatch.setenv("SUPABASE_REQUIRE_SERVICE_ROLE", "true")
    with pytest.raises(RuntimeError, match="not a service_role key"):
        persistence.check_backend_key(_jwt("anon"))
    persistence.check_backend_key(_jwt("service_role"))  # allowed


def test_the_check_runs_when_the_client_is_built(monkeypatch):
    called = []
    monkeypatch.setattr(persistence, "SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setattr(persistence, "SUPABASE_KEY", _jwt("anon"))
    monkeypatch.setattr(persistence, "check_backend_key", lambda key: called.append(key))
    monkeypatch.setattr("supabase.create_client", lambda *a, **k: object())
    persistence._get_client()
    assert called == [_jwt("anon")]
