from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .util.env import getenv_required


@dataclass
class OpenAILLMClient:
    """
    Public-release OpenAI client.

    - Requires OPENAI_API_KEY in the environment.
    - Uses the official `openai` Python SDK (optional dependency).
    - No private endpoints. No secrets committed.
    """

    api_key_env: str = "OPENAI_API_KEY"

    def complete(self, prompt: str, model: str, **kwargs: Any) -> str:
        api_key = getenv_required(self.api_key_env)

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI support requires the optional dependency: pip install -e '.[openai]'"
            ) from e

        client = OpenAI(api_key=api_key)

        # Default settings: conservative + reproducible-ish.
        temperature = float(kwargs.get("temperature", 0.0))
        max_tokens = int(kwargs.get("max_tokens", 800))

        # Chat Completions style via Responses API if available is better long-term,
        # but Chat Completions remains widely supported. Keep it simple here.
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful biological reasoning assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return resp.choices[0].message.content or ""
