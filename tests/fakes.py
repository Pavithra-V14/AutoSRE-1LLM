"""Test doubles shared by the offline unit tests."""

from __future__ import annotations

from typing import Optional, Union

from services.agent.llm_router import LLMResponse


def llm(text: str, *, finish_reason: str = "stop", input_tokens: int = 10, output_tokens: int = 5,
        provider: str = "fake") -> LLMResponse:
    return LLMResponse(
        text=text, provider=provider, model="fake-model", input_tokens=input_tokens,
        output_tokens=output_tokens, latency_ms=7, finish_reason=finish_reason,
    )


class FakeRouter:
    """Returns scripted responses in order; an Exception entry is raised.
    Records every call so tests can assert on prompts and flags."""

    def __init__(self, responses: list[Union[LLMResponse, Exception]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, prompt: str, system: str = "", json_mode: bool = False,
                 prefer: Optional[str] = None, demote: tuple = (),
                 temperature: Optional[float] = None) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system, "json_mode": json_mode,
                           "prefer": prefer, "demote": tuple(demote), "temperature": temperature})
        if not self._responses:
            raise AssertionError("FakeRouter exhausted: the code made more LLM calls than the test scripted")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def observation(tool: str, status: str = "ok", role: str = "required", data: Optional[dict] = None) -> dict:
    return {
        "type": "observation",
        "content": {
            "tool": tool, "role": role, "status": status,
            "data": data if data is not None else ({"x": 1} if status == "ok" else None),
            "disclosure": None if status == "ok" else f"{tool} is {status}",
        },
    }


def plan(tool: str, params: Optional[dict] = None, role: str = "required") -> dict:
    return {"type": "plan", "content": {"tool": tool, "role": role, "params": params or {}, "reasoning": "t"}}
