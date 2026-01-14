from __future__ import annotations
import requests
from .llm_client import LLMClient

class OllamaLLMClient(LLMClient):
    def __init__(self, base_url="http://localhost:11434/api/generate"):
        self.base_url = base_url

    def query(self, prompt: str, model_name: str) -> str:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }

        resp = requests.post(self.base_url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns { "response": "..." }
        text = data.get("response", "")

        # BioReasoner expects raw JSON string like {"facts": [...]}
        return text
