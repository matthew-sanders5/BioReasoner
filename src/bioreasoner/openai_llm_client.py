from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm_client import LLMClient


@dataclass
class OpenAILLMClient(LLMClient):
    """
    Public-release OpenAI client.

    - Optional dependency: `openai` (install with: pip install -e ".[openai]")
    - Requires env var: OPENAI_API_KEY
    - No hardcoded endpoints, no secrets committed
    """

    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 800

    def query(self, prompt: str, model_name: str) -> str:
        import os

        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} is not set. Export it or set it in your local .env."
            )

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                'OpenAI support requires the optional dependency: pip install -e ".[openai]"'
            ) from e

        client = OpenAI(api_key=api_key)

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a careful biological reasoning assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return resp.choices[0].message.content or ""
