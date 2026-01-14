from __future__ import annotations

from typing import Iterable, Iterator, List, Set


class FactSet:
    """
    Minimal wrapper around a set of fact IDs (strings).

    Facts are just opaque strings like "WNT_PRESENT", "LRP6_P_S", "BETA_CAT_UP".
    """

    def __init__(self, initial: Iterable[str] | None = None) -> None:
        self._facts: Set[str] = set(initial) if initial is not None else set()

    def add(self, fact: str) -> None:
        self._facts.add(fact)

    def update(self, facts: Iterable[str]) -> None:
        self._facts.update(facts)

    def remove(self, fact: str) -> None:
        self._facts.discard(fact)

    def has(self, fact: str) -> bool:
        return fact in self._facts

    def to_list(self) -> List[str]:
        return sorted(self._facts)

    def copy(self) -> "FactSet":
        return FactSet(self._facts)

    def __contains__(self, fact: str) -> bool:  # type: ignore[override]
        return fact in self._facts

    def __iter__(self) -> Iterator[str]:
        # Sort for deterministic output
        return iter(sorted(self._facts))

    def __len__(self) -> int:
        return len(self._facts)

    def __repr__(self) -> str:
        return f"FactSet({sorted(self._facts)!r})"
