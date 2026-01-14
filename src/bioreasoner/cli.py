from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import List, Tuple

from .engine import ReasoningEngine
from .rules import build_default_wnt_rules
from .scenarios import load_scenario
from . import vocab as V

# NEW imports for Phase 2
from .llm_client import StubLLMClient
from .llm_eval import evaluate_llm_on_scenario
# end NEW imports


DEFAULT_CONTRADICTIONS: List[Tuple[str, str]] = [
    # β-catenin mutually exclusive levels
    (V.BETA_CAT_LEVEL_UP, V.BETA_CAT_LEVEL_BASELINE),
    (V.BETA_CAT_LEVEL_UP, V.BETA_CAT_LEVEL_DOWN),
    (V.BETA_CAT_LEVEL_BASELINE, V.BETA_CAT_LEVEL_DOWN),
    # Destruction complex mutually exclusive activities
    (V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH, V.DESTRUCTION_COMPLEX_ACTIVITY_LOW),
    # AKT mutually exclusive activity states
    (V.AKT_STATE_ACTIVE, V.AKT_STATE_INACTIVE),
    # GSK3 mutually exclusive states
    (V.GSK3_STATE_ACTIVE, V.GSK3_STATE_INACTIVE),
    # Apoptotic tendency
    (V.APOPTOSIS_TENDENCY_HIGH, V.APOPTOSIS_TENDENCY_LOW),
]


def _build_engine() -> ReasoningEngine:
    """
    Helper: construct a ReasoningEngine with default rules and contradictions.
    """
    rules = build_default_wnt_rules()
    engine = ReasoningEngine(rules=rules, contradiction_pairs=DEFAULT_CONTRADICTIONS)
    return engine


def cmd_run(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario)
    scenario = load_scenario(scenario_path)

    engine = _build_engine()

    # Run reasoning first
    result = engine.run(scenario.initial_facts)

    # If JSON output is requested, dump structured payload and return.
    if getattr(args, "format", "text") == "json":
        payload = {
            "scenario": {
                "name": scenario.name,
                "description": scenario.description,
                "metadata": scenario.metadata,
                "initial_facts": sorted(list(scenario.initial_facts)),
                "queries": list(scenario.queries),
            },
            "result": result.to_dict(),
        }
        print(json.dumps(payload, indent=2))
        return 0

    # Default: existing human-readable text output.
    print(f"Scenario: {scenario.name}")
    if scenario.description:
        print(f"Description: {scenario.description}")
    if scenario.metadata:
        print("Metadata:")
        for k, v in scenario.metadata.items():
            print(f"  {k}: {v}")

    print("\nInitial facts:")
    for f in sorted(scenario.initial_facts):
        print(f"  - {f}")

    print("\nReasoning steps:")
    if result.steps:
        for step in result.steps:
            new_facts = ", ".join(sorted(step.new_facts))
            print(
                f"  [iter {step.iteration_index}] {step.rule_name} -> {new_facts}"
            )
            if step.rule_description:
                print(f"       {step.rule_description}")
    else:
        print("  (no rules fired)")

    print("\nFinal facts:")
    for f in sorted(result.final_facts):
        print(f"  - {f}")

    if result.contradictions:
        print("\nContradictions detected:")
        for a, b in result.contradictions:
            print(f"  * {a} <-> {b}")
    else:
        print("\nNo contradictions detected.")

    # Handle queries if provided
    if scenario.queries:
        print("\nQueries:")
        for q in scenario.queries:
            truth = "TRUE" if q in result.final_facts else "FALSE/UNKNOWN"
            print(f"  ? {q} -> {truth}")

    return 0


# NEW: eval-llm command
def cmd_eval_llm(args: argparse.Namespace) -> int:
    """
    Evaluate an LLM's predictions against the BioReasoner engine for a scenario.

    Example:
      bioreasoner eval-llm examples/akt_wnt_crosstalk_preserves_beta_cat.yaml --model gpt-4.1 --format json
    """
    scenario_path = Path(args.scenario)
    scenario = load_scenario(scenario_path)

    engine = _build_engine()
    engine_result = engine.run(scenario.initial_facts)

    try:
        if args.model.startswith("llama") or args.model.startswith("mistral"):
            from .ollama_llm_client import OllamaLLMClient
            print("Using OllamaLLMClient", file=sys.stderr)
            llm_client = OllamaLLMClient()
        else:
            from .local_llm_client import OpenAILLMClient  # type: ignore
            print("Using OpenAILLMClient", file=sys.stderr)
            llm_client = OpenAILLMClient()
    except ImportError:
        from .llm_client import StubLLMClient
        print("Using fallback StubLLMClient", file=sys.stderr)
        llm_client = StubLLMClient(fixed_response='{"facts": []}')

    eval_result = evaluate_llm_on_scenario(
        scenario=scenario,
        engine_result=engine_result,
        llm_client=llm_client,
        model_name=args.model,
        contradiction_pairs=DEFAULT_CONTRADICTIONS,
    )

    if args.format == "json":
        print(json.dumps(eval_result.to_json_dict(), indent=2))
        return 0

    # If you ever want a human-readable mode, implement it here.
    print("Only JSON format is currently supported for eval-llm.")
    return 1
# end NEW


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bioreasoner",
        description="BioReasoner: symbolic biological causal inference engine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser(
        "run",
        help="Run a reasoning scenario defined in a YAML file.",
    )
    run_parser.add_argument(
        "scenario",
        type=str,
        help="Path to the scenario YAML file.",
    )
    run_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    run_parser.set_defaults(func=cmd_run)

    # NEW: eval-llm
    eval_parser = subparsers.add_parser(
        "eval-llm",
        help="Evaluate an LLM against the engine for a scenario.",
    )
    eval_parser.add_argument(
        "scenario",
        type=str,
        help="Path to the scenario YAML file.",
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model name identifier (OpenAI model id or alias).",
    )
    eval_parser.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help="Output format (currently only 'json' is supported).",
    )
    eval_parser.set_defaults(func=cmd_eval_llm)
    # end NEW

    return parser


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1

    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
