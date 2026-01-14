from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .llm_client import LLMClient  # Protocol in your package


client = OpenAI()  # uses OPENAI_API_KEY from env


def _map_model_name(requested: str | None) -> str:
    """
    Map CLI model names to actual OpenAI model IDs.

    You can add aliases here (e.g., 'gpt-5.1-mini' -> 'gpt-4.1-mini').
    """
    if not requested:
        return "gpt-4.1-mini"

    # Alias your internal "gpt-5.1-mini" to a real model.
    if requested == "gpt-5.1-mini":
        return "gpt-4.1-mini"

    # Otherwise just pass through and let the API error if it is wrong.
    return requested


@dataclass
class OpenAILLMClient(LLMClient):
    default_model: str = "gpt-4.1-mini"

    def query(self, prompt: str, model_name: str) -> str:
        model = _map_model_name(model_name or self.default_model)
        response = client.responses.create(
            model=model,
            input=prompt,
        )
        return response.output_text
