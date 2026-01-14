from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Set, Tuple

from .engine import ReasoningResult
from .scenarios import Scenario
from .llm_prompts import build_llm_prompt
from .llm_client import LLMClient
from .llm_parser import ParsedLLMOutput, parse_llm_output
from . import vocab as V

# Facts we actually score TP/FP/FN on.
# You can tweak this as you refine the benchmark.
DEFAULT_EVAL_FACT_IDS: Set[str] = {
    # β-catenin levels
    V.BETA_CAT_LEVEL_UP,
    V.BETA_CAT_LEVEL_BASELINE,
    V.BETA_CAT_LEVEL_DOWN,
    # Destruction complex
    V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH,
    V.DESTRUCTION_COMPLEX_ACTIVITY_LOW,
    # AKT
    V.AKT_STATE_ACTIVE,
    V.AKT_STATE_INACTIVE,
    # GSK3
    V.GSK3_STATE_ACTIVE,
    V.GSK3_STATE_INACTIVE,
    # PI3K
    V.PI3K_STATE_ACTIVE,
    # Apoptosis
    V.APOPTOSIS_TENDENCY_HIGH,
    V.APOPTOSIS_TENDENCY_LOW,
}

@dataclass
class LLMEvaluationResult:
    scenario_name: str
    model_name: str

    engine_facts: List[str]
    llm_facts: List[str]

    true_positives: List[str]
    false_negatives: List[str]  # engine facts missed by LLM
    false_positives: List[str]  # LLM hallucinations vs engine

    llm_internal_contradictions: List[Tuple[str, str]]
    llm_vs_engine_contradictions: List[Tuple[str, str]]

    parsing_errors: List[str]
    raw_llm_output: str

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_contradictions(
    facts: Set[str],
    contradiction_pairs: Iterable[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """
    Given a set of facts and a library of mutually exclusive pairs,
    return all pairs present in 'facts'.
    """
    hits: List[Tuple[str, str]] = []
    for a, b in contradiction_pairs:
        if a in facts and b in facts:
            # deterministic ordering
            hits.append(tuple(sorted((a, b))))
    # remove duplicates in case contradictions list contains both (a,b) and (b,a)
    hits = sorted(set(hits))
    return hits


def evaluate_llm_on_scenario(
    *,
    scenario: Scenario,
    engine_result: ReasoningResult,
    llm_client: LLMClient,
    model_name: str,
    contradiction_pairs: Iterable[Tuple[str, str]],
) -> LLMEvaluationResult:
    """
    End-to-end evaluation pipeline for a single Scenario.

    1. Build prompt from Scenario.
    2. Query LLM.
    3. Parse LLM output → fact IDs.
    4. Compute TP / FP / FN and contradictions.
    """
    prompt = build_llm_prompt(scenario)
    raw_output = llm_client.query(prompt, model_name=model_name)

    parsed: ParsedLLMOutput = parse_llm_output(raw_output)

    engine_facts_set: Set[str] = set(engine_result.final_facts.to_list())
    llm_facts_set: Set[str] = set(parsed.facts)

    # Keep full sets for inspection
    full_engine_facts = engine_facts_set
    full_llm_facts = llm_facts_set

    # Restrict scoring to evaluation target facts
    eval_targets = DEFAULT_EVAL_FACT_IDS

    engine_eval = full_engine_facts & eval_targets
    llm_eval = full_llm_facts & eval_targets

    tp = sorted(engine_eval & llm_eval)
    fn = sorted(engine_eval - llm_eval)        # engine truth not predicted by LLM
    fp = sorted(llm_eval - engine_eval)        # LLM hallucinations within eval set

    # Internal contradictions among LLM facts
    llm_internal_contras = compute_contradictions(llm_facts_set, contradiction_pairs)

    # Contradictions where LLM asserts a fact that contradicts an engine fact
    llm_vs_engine_contras: List[Tuple[str, str]] = []
    for a, b in contradiction_pairs:
        if (a in engine_facts_set and b in llm_facts_set) or (
            b in engine_facts_set and a in llm_facts_set
        ):
            llm_vs_engine_contras.append(tuple(sorted((a, b))))
    llm_vs_engine_contras = sorted(set(llm_vs_engine_contras))

    return LLMEvaluationResult(
        scenario_name=scenario.name,
        model_name=model_name,
        engine_facts=sorted(engine_facts_set),
        llm_facts=sorted(llm_facts_set),
        true_positives=tp,
        false_negatives=fn,
        false_positives=fp,
        llm_internal_contradictions=llm_internal_contras,
        llm_vs_engine_contradictions=llm_vs_engine_contras,
        parsing_errors=parsed.parsing_errors,
        raw_llm_output=parsed.raw_text,
    )
