from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Set, Tuple

from .facts import FactSet
from .rules import Rule, sort_rules


@dataclass
class ReasoningStep:
    """
    One application of a rule in an inference run.
    """
    rule_name: str
    new_facts: FrozenSet[str]
    iteration_index: int
    rule_description: str | None = None
    rule_citation: str | None = None


@dataclass
class ReasoningResult:
    """
    Output of the reasoning engine.
    """
    final_facts: FactSet
    steps: List[ReasoningStep]
    contradictions: List[Tuple[str, str]]

    def to_dict(self) -> dict:
        return {
            "final_facts": sorted(list(self.final_facts)),
            "steps": [
                {
                    "iteration_index": step.iteration_index,
                    "rule": step.rule_name,
                    "new_facts": sorted(list(step.new_facts)),
                    "rule_description": step.rule_description,
                    "rule_citation": step.rule_citation,
                }
                for step in self.steps
            ],
            "contradictions": [
                {"fact_a": a, "fact_b": b}
                for (a, b) in self.contradictions
            ],
        }


class ReasoningEngine:
    """
    Simple forward-chaining engine.

    - Rules are monotonic (only add facts).
    - Contradictions are defined as mutually incompatible pairs of facts.
    """

    def __init__(
        self,
        rules: Iterable[Rule],
        contradiction_pairs: Iterable[Tuple[str, str]] | None = None,
    ) -> None:
        self.rules: List[Rule] = sort_rules(rules)
        # store contradictions as frozenset({a, b}) to be order-agnostic
        self._contradictions: Set[frozenset[str]] = set()
        if contradiction_pairs is not None:
            for a, b in contradiction_pairs:
                self.add_contradiction_pair(a, b)

    def add_contradiction_pair(self, a: str, b: str) -> None:
        self._contradictions.add(frozenset((a, b)))

    def run(self, initial_facts: FactSet, max_iterations: int = 1000) -> ReasoningResult:
        facts = initial_facts.copy()
        steps: List[ReasoningStep] = []

        iteration = 0
        changed = True

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for rule in self.rules:
                if not rule.is_applicable(facts):
                    continue

                new_facts = rule.new_facts_if_applied(facts)
                if not new_facts:
                    continue

                facts.update(new_facts)
                steps.append(
                    ReasoningStep(
                        rule_name=rule.name,
                        new_facts=frozenset(new_facts),
                        iteration_index=iteration,
                        rule_description=rule.description,
                        rule_citation=rule.citation,
                    )
                )
                changed = True

        contradictions = self._find_contradictions(facts)
        return ReasoningResult(final_facts=facts, steps=steps, contradictions=contradictions)

    def _find_contradictions(self, facts: FactSet) -> List[Tuple[str, str]]:
        present = set(facts.to_list())
        hits: List[Tuple[str, str]] = []

        for pair in self._contradictions:
            # pair is a frozenset of size 2
            if pair.issubset(present):
                a, b = sorted(pair)  # deterministic ordering
                hits.append((a, b))

        return hits
