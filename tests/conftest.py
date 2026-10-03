from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _no_live_llm_and_clean_router_state(monkeypatch):
    """Unit tests must never reach a real LLM provider.

    Importing the agent loads .env, so real API keys are in the process
    environment; a test that stubbed only one provider would silently call
    the others (spending quota and making results depend on the network).
    Strip the keys and replay mode for every test -- tests that need a key set
    a fake one explicitly -- and reset the module-level provider cooldowns so
    one test's simulated outage cannot leak into the next."""
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "LLM_REPLAY_MODE", "JUDGE_PROVIDER"):
        monkeypatch.delenv(name, raising=False)

    from services.agent.llm_router import reset_circuit_breaker

    reset_circuit_breaker()
    yield
    reset_circuit_breaker()
