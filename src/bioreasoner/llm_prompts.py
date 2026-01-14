from __future__ import annotations

from typing import Iterable

from .scenarios import Scenario
from .facts import FactSet
from . import vocab as V

# Fact IDs that the LLM is allowed to output.
# Start with the ones you care about evaluating.
VALID_FACT_IDS: list[str] = [
    # β-catenin levels
    V.BETA_CAT_LEVEL_UP,
    V.BETA_CAT_LEVEL_BASELINE,
    V.BETA_CAT_LEVEL_DOWN,
    # Destruction complex
    V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH,
    V.DESTRUCTION_COMPLEX_ACTIVITY_LOW,
    # AKT state
    V.AKT_STATE_ACTIVE,
    V.AKT_STATE_INACTIVE,
    # GSK3 state
    V.GSK3_STATE_ACTIVE,
    V.GSK3_STATE_INACTIVE,
    # PI3K state
    V.PI3K_STATE_ACTIVE,
    # Apoptosis
    V.APOPTOSIS_TENDENCY_HIGH,
    V.APOPTOSIS_TENDENCY_LOW,
    # Presence facts (optional to include in scoring later)
    V.RTK_RECEPTOR_PRESENT,
    V.GROWTH_FACTOR_LIGAND_PRESENT,
    V.LRP6_PROTEIN_PRESENT,
    V.FRIZZLED_PROTEIN_PRESENT,
]

def _facts_to_list(facts: FactSet) -> list[str]:
    """
    Convert a FactSet to a sorted list of fact IDs.
    """
    return facts.to_list()


def format_initial_facts(facts: Iterable[str]) -> str:
    """
    Turn engine fact IDs into a human-readable bullet list.

    For now we just echo the raw IDs; later you can add a pretty-print map.
    """
    lines: list[str] = []
    for fact in facts:
        lines.append(f"- {fact}")
    return "\n".join(lines)


def build_llm_prompt(
    scenario: Scenario,
    *,
    include_instructions: bool = True,
) -> str:
    """
    Build a deterministic LLM prompt for a Scenario.

    Protocol: we tell the model to output ONLY a JSON object:

      {"facts": ["BETA_CAT__LEVEL__UP", "DESTRUCTION_COMPLEX__ACTIVITY__LOW"]}

    where entries are *BioReasoner fact IDs*.
    """
    header = f"Scenario: {scenario.name}\n"
    desc = ""
    if scenario.description:
        desc = f"Description: {scenario.description.strip()}\n\n"

    facts_block = "Initial biological facts:\n"
    facts_block += format_initial_facts(_facts_to_list(scenario.initial_facts)) + "\n\n"

    if not include_instructions:
        return header + desc + facts_block

    valid_facts_block = "Valid BioReasoner fact IDs you may output (choose any subset):\n"
    for fid in sorted(VALID_FACT_IDS):
        valid_facts_block += f"  - {fid}\n"

    instructions = (
        "You are an expert molecular and cellular biologist.\n"
        "You are given symbolic biological facts about signaling pathways:\n"
        "  - Wnt / LRP6 / β-catenin\n"
        "  - PI3K / AKT / GSK3 / apoptosis\n\n"
        "TASK:\n"
        "1. Reason forward based ONLY on these initial facts and standard biology.\n"
        "2. Decide qualitative states (UP, DOWN, BASELINE, ACTIVE, INACTIVE, etc.) "
        "   for relevant downstream components.\n"
        "3. You MUST restrict your predictions to the following BioReasoner fact IDs:\n"
        f"{valid_facts_block}\n"
        "4. Output ONLY a JSON object with a single key 'facts', whose value is a list "
        "   of BioReasoner fact IDs taken from the list above.\n"
        "   Example:\n"
        '   {\"facts\": [\"BETA_CAT__LEVEL__UP\", \"DESTRUCTION_COMPLEX__ACTIVITY__LOW\"]}\n\n'
        "5. Do not invent new fact IDs. Use only IDs exactly as written in the list.\n"
        "6. Do not explain your reasoning.\n"
        "7. Do not output any text outside of the JSON.\n"
    )

    return header + desc + facts_block + instructions
