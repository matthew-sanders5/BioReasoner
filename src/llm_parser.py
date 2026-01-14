from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Set, Tuple

# Map non-canonical fact IDs that the LLM might output
# to the engine's canonical fact IDs.
SYNONYM_MAP: dict[str, str] = {
    # Activity/state synonyms
    "PI3K__ACTIVITY__UP": "PI3K__STATE__ACTIVE",
    "AKT__ACTIVITY__UP": "AKT__STATE__ACTIVE",
    "GSK3__ACTIVITY__DOWN": "GSK3__STATE__INACTIVE",
    # You can extend this as you see more patterns, e.g.:
    # "APOPTOSIS__ACTIVITY__LOW": "APOPTOSIS__TENDENCY__LOW",
}

JSON_FACTS_KEY = "facts"


@dataclass
class ParsedLLMOutput:
    raw_text: str
    facts: Set[str]
    parsing_errors: List[str]


def _extract_json_fragment(text: str) -> str | None:
    """
    Extract the first top-level JSON object using brace matching.

    This guards against models that ignore instructions and wrap the JSON.
    """
    first = text.find("{")
    if first == -1:
        return None

    depth = 0
    for i, ch in enumerate(text[first:], start=first):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[first : i + 1]
    return None


def parse_llm_output(raw_text: str) -> ParsedLLMOutput:
    """
    Parse raw LLM output into a set of BioReasoner fact IDs.

    Expected canonical format:
      {"facts": ["BETA_CAT__LEVEL__UP", "DESTRUCTION_COMPLEX__ACTIVITY__LOW"]}
    """
    errors: List[str] = []
    facts: Set[str] = set()

    candidate = raw_text.strip()

    # Try whole string as JSON first
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        fragment = _extract_json_fragment(candidate)
        if fragment is None:
            errors.append("No JSON object found in LLM output.")
            return ParsedLLMOutput(raw_text=raw_text, facts=facts, parsing_errors=errors)
        try:
            data = json.loads(fragment)
        except json.JSONDecodeError as e:
            errors.append(f"Failed to parse extracted JSON fragment: {e}")
            return ParsedLLMOutput(raw_text=raw_text, facts=facts, parsing_errors=errors)

    if not isinstance(data, dict):
        errors.append(f"Top-level JSON is not an object (got {type(data)!r}).")
        return ParsedLLMOutput(raw_text=raw_text, facts=facts, parsing_errors=errors)

    if JSON_FACTS_KEY not in data:
        errors.append(f"JSON missing required key '{JSON_FACTS_KEY}'.")
        return ParsedLLMOutput(raw_text=raw_text, facts=facts, parsing_errors=errors)

    value = data[JSON_FACTS_KEY]
    if not isinstance(value, list):
        errors.append(f"'{JSON_FACTS_KEY}' value is not a list.")
        return ParsedLLMOutput(raw_text=raw_text, facts=facts, parsing_errors=errors)

    for item in value:
        if isinstance(item, str):
            facts.add(item.strip())
        else:
            errors.append(f"Ignoring non-string fact entry: {item!r}")

    # Normalize synonyms to canonical fact IDs
    normalized: Set[str] = set()
    for f in facts:
        normalized.add(SYNONYM_MAP.get(f, f))
    facts = normalized

    return ParsedLLMOutput(raw_text=raw_text, facts=facts, parsing_errors=errors)
