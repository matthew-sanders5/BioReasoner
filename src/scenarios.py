from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .facts import FactSet


@dataclass
class Scenario:
    """
    A single reasoning scenario loaded from YAML.

    Minimal schema:

    name: str
    description: optional str
    metadata: optional dict (cell_line, etc.)
    initial_facts: list[str]
    queries: optional list[str]
    """
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    initial_facts: FactSet = field(default_factory=FactSet)
    queries: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        name = data.get("name") or "unnamed_scenario"
        description = data.get("description")
        metadata = data.get("metadata") or {}

        initial_facts_raw: Iterable[str] = data.get("initial_facts") or []
        initial_facts = FactSet(initial_facts_raw)

        queries_raw: Iterable[str] = data.get("queries") or []
        queries = list(queries_raw)

        return cls(
            name=name,
            description=description,
            metadata=metadata,
            initial_facts=initial_facts,
            queries=queries,
        )


def load_scenario(path: str | Path) -> Scenario:
    """
    Load a Scenario from a YAML file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Scenario file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top of YAML file: {p}")

    return Scenario.from_dict(data)
