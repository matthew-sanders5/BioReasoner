from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    """
    Minimal interface for any LLM backend.
    """

    def query(self, prompt: str, model_name: str) -> str:  # pragma: no cover - interface
        ...


@dataclass
class StubLLMClient:
    """
    Deterministic stub for tests or offline use.

    You can override `fixed_response` in tests to simulate different models.
    """

    fixed_response: str

    def query(self, prompt: str, model_name: str) -> str:
        return self.fixed_response


def query_model(prompt: str, model_name: str) -> str:
    """
    Convenience function to plug in a real LLM backend in *private* code.

    In the public repo, leave this unimplemented to avoid hard dependencies.
    """
    raise NotImplementedError(
        "query_model is intentionally left unimplemented. "
        "Use an LLMClient implementation (e.g., StubLLMClient) or "
        "provide your own backend in private code."
    )
