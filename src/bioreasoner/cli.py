from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .engine import ReasoningEngine
from .rules import build_default_wnt_rules
from .scenarios import load_scenario

# Phase 2
from .llm_eval import evaluate_llm_on_scenario


# -------------------------
# Defaults / helpers
# -------------------------

from . import vocab as V

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
    rules = build_default_wnt_rules()
    return ReasoningEngine(rules=rules, contradiction_pairs=DEFAULT_CONTRADICTIONS)


def _safe_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "scenario"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _getenv(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _build_llm_client(provider: str) -> Any:
    """
    Provider policy (public release safe):
    - openai: optional, requires OPENAI_API_KEY and openai client implementation present
    - ollama: optional, requires BIOREASONER_OLLAMA_BASE_URL and ollama client implementation present
    """
    provider = provider.strip().lower()

    if provider == "ollama":
        # Optional dependency; only import when used.
        try:
            from .ollama_llm_client import OllamaLLMClient  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Ollama provider requested but ollama client is not available. "
                "Ensure src/bioreasoner/ollama_llm_client.py exists and dependencies are installed."
            ) from e
        return OllamaLLMClient()

    if provider == "openai":
        # OpenAI support is optional. We do NOT ship private/local clients.
        # If you later add a public openai client (e.g., openai_llm_client.py), wire it here.
        try:
            from .openai_llm_client import OpenAILLMClient  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI provider requested but OpenAI client is not available in this public release. "
                "If you want OpenAI support, add src/bioreasoner/openai_llm_client.py (public-safe) "
                "and set OPENAI_API_KEY."
            ) from e

        # Enforce env var presence (no secrets committed).
        if not _getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Export it or set it in your local .env.")
        return OpenAILLMClient()

    raise RuntimeError(f"Unknown provider '{provider}'. Use BIOREASONER_MODEL_PROVIDER=openai|ollama.")


# -------------------------
# CLI commands
# -------------------------

def cmd_engine(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario)
    out_path = Path(args.out)

    scenario = load_scenario(scenario_path)
    engine = _build_engine()
    result = engine.run(scenario.initial_facts)

    payload = {
        "scenario": {
            "path": str(scenario_path),
            "name": scenario.name,
            "description": scenario.description,
            "metadata": scenario.metadata,
            "initial_facts": sorted(list(scenario.initial_facts)),
            "queries": list(scenario.queries),
        },
        "engine_result": result.to_dict(),
    }

    if str(out_path) == "-":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _write_json(out_path, payload)

    return 0


def cmd_eval_llm(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario)
    out_path = Path(args.out)

    # Provider/model selection: CLI flags override env; env overrides defaults.
    provider = (args.provider or _getenv("BIOREASONER_MODEL_PROVIDER") or "").strip()
    model_name = (args.model or _getenv("BIOREASONER_MODEL_NAME") or "").strip()

    if not provider:
        raise RuntimeError(
            "No provider selected. Set BIOREASONER_MODEL_PROVIDER=openai|ollama "
            "or pass --provider openai|ollama."
        )
    if not model_name:
        raise RuntimeError(
            "No model name set. Set BIOREASONER_MODEL_NAME (e.g., gpt-4.1-mini or llama3.1) "
            "or pass --model."
        )

    scenario = load_scenario(scenario_path)
    engine = _build_engine()
    engine_result = engine.run(scenario.initial_facts)

    llm_client = _build_llm_client(provider)

    eval_result = evaluate_llm_on_scenario(
        scenario=scenario,
        engine_result=engine_result,
        llm_client=llm_client,
        model_name=model_name,
        contradiction_pairs=DEFAULT_CONTRADICTIONS,
    )

    payload = eval_result.to_json_dict()
    payload["_meta"] = {
        "scenario_path": str(scenario_path),
        "provider": provider,
        "model_name": model_name,
        "replicate": int(args.replicate),
    }

    if str(out_path) == "-":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _write_json(out_path, payload)

    return 0


def _iter_yaml_files(suite_dir: Path) -> Iterable[Path]:
    # recursive suite
    for p in sorted(suite_dir.rglob("*.yaml")):
        if p.name.startswith("."):
            continue
        yield p


def cmd_batch_eval(args: argparse.Namespace) -> int:
    suite_dir = Path(args.suite)
    outdir = Path(args.outdir)

    provider = (args.provider or _getenv("BIOREASONER_MODEL_PROVIDER") or "").strip()
    model_name = (args.model or _getenv("BIOREASONER_MODEL_NAME") or "").strip()
    replicate = int(args.replicate)

    if not provider:
        raise RuntimeError(
            "No provider selected. Set BIOREASONER_MODEL_PROVIDER=openai|ollama or pass --provider."
        )
    if not model_name:
        raise RuntimeError(
            "No model name set. Set BIOREASONER_MODEL_NAME or pass --model."
        )

    outdir.mkdir(parents=True, exist_ok=True)

    engine = _build_engine()
    llm_client = _build_llm_client(provider)

    manifest: List[Dict[str, Any]] = []
    n = 0

    for scenario_path in _iter_yaml_files(suite_dir):
        scenario = load_scenario(scenario_path)
        engine_result = engine.run(scenario.initial_facts)

        eval_result = evaluate_llm_on_scenario(
            scenario=scenario,
            engine_result=engine_result,
            llm_client=llm_client,
            model_name=model_name,
            contradiction_pairs=DEFAULT_CONTRADICTIONS,
        )

        slug = _safe_slug(scenario.name) + "__" + _safe_slug(scenario_path.stem)
        out_path = outdir / f"{slug}.json"

        payload = eval_result.to_json_dict()
        payload["_meta"] = {
            "scenario_path": str(scenario_path),
            "scenario_name": scenario.name,
            "provider": provider,
            "model_name": model_name,
            "replicate": replicate,
        }

        _write_json(out_path, payload)
        manifest.append(
            {
                "scenario_path": str(scenario_path),
                "scenario_name": scenario.name,
                "output": str(out_path),
            }
        )
        n += 1

    _write_json(outdir / "manifest.json", {"count": n, "items": manifest})
    return 0


def _extract_counts(obj: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Best-effort extraction of TP/FP/FN counts from eval JSON.
    Your eval_result likely contains these already; we handle a few common shapes.
    """
    # direct
    for keyset in [
        ("tp", "fp", "fn"),
        ("TP", "FP", "FN"),
        ("true_positives", "false_positives", "false_negatives"),
    ]:
        if all(k in obj for k in keyset):
            return int(obj[keyset[0]]), int(obj[keyset[1]]), int(obj[keyset[2]])

    # nested metrics
    if "metrics" in obj and isinstance(obj["metrics"], dict):
        m = obj["metrics"]
        for keyset in [
            ("tp", "fp", "fn"),
            ("TP", "FP", "FN"),
            ("true_positives", "false_positives", "false_negatives"),
        ]:
            if all(k in m for k in keyset):
                return int(m[keyset[0]]), int(m[keyset[1]]), int(m[keyset[2]])

    # fallback: none
    return 0, 0, 0


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    return float(0.0 if denom == 0 else (2 * tp) / denom)


def cmd_analyze(args: argparse.Namespace) -> int:
    inp = Path(args.input)
    out_path = Path(args.out)

    json_files = [p for p in inp.glob("*.json") if p.name != "manifest.json"]
    if not json_files:
        raise RuntimeError(f"No eval JSON files found in: {inp}")

    tp = fp = fn = 0
    per_file: List[Dict[str, Any]] = []

    for p in sorted(json_files):
        obj = _read_json(p)
        tpi, fpi, fni = _extract_counts(obj)
        tp += tpi
        fp += fpi
        fn += fni

        per_file.append(
            {
                "file": p.name,
                "tp": tpi,
                "fp": fpi,
                "fn": fni,
                "f1": _f1(tpi, fpi, fni),
                "meta": obj.get("_meta", {}),
            }
        )

    summary = {
        "n_files": len(json_files),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_f1": _f1(tp, fp, fn),
        # macro F1: average per-file F1 (best-effort)
        "macro_f1": float(sum(x["f1"] for x in per_file) / max(1, len(per_file))),
        "per_file": per_file,
    }

    if str(out_path) == "-":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _write_json(out_path, summary)
    return 0


def cmd_aggregate_replicates(args: argparse.Namespace) -> int:
    """
    Aggregate summary metrics across replicate folders by discovering summary files.

    Discovery rules (public-safe):
    - Find files named: summary.json OR *summary*.json under root recursively.
    - Average numeric scalar metrics across discovered summaries.
    """
    root = Path(args.root)
    out_path = Path(args.out)

    candidates = []
    for patt in ["summary.json", "*summary*.json"]:
        candidates.extend(root.rglob(patt))

    # Filter obvious non-summary blobs by requiring at least micro_f1 or macro_f1
    summaries: List[Dict[str, Any]] = []
    for p in sorted(set(candidates)):
        try:
            obj = _read_json(p)
        except Exception:
            continue
        if isinstance(obj, dict) and ("micro_f1" in obj or "macro_f1" in obj):
            obj["_path"] = str(p)
            summaries.append(obj)

    if not summaries:
        raise RuntimeError(f"No summary-like JSON files found under: {root}")

    # Aggregate scalar numeric keys present in all summaries
    keys = set.intersection(*[set(s.keys()) for s in summaries])
    scalar_keys = []
    for k in sorted(keys):
        if k.startswith("_"):
            continue
        if isinstance(summaries[0].get(k), (int, float)):
            scalar_keys.append(k)

    agg: Dict[str, Any] = {"n_summaries": len(summaries), "sources": [s["_path"] for s in summaries]}
    for k in scalar_keys:
        vals = [float(s[k]) for s in summaries]
        agg[k] = sum(vals) / len(vals)

    if str(out_path) == "-":
        print(json.dumps(agg, indent=2, sort_keys=True))
    else:
        _write_json(out_path, agg)
    return 0


# -------------------------
# Parser
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bioreasoner",
        description="BioReasoner: deterministic symbolic biological causal inference engine (with optional LLM eval harness).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # engine
    pe = sub.add_parser("engine", help="Run the deterministic engine on a scenario YAML.")
    pe.add_argument("--scenario", required=True, help="Path to scenario YAML.")
    pe.add_argument("--out", required=True, help="Path to write JSON output (use '-' for stdout).")
    pe.set_defaults(func=cmd_engine)

    # eval-llm
    pl = sub.add_parser("eval-llm", help="Evaluate an LLM on a scenario YAML.")
    pl.add_argument("--scenario", required=True, help="Path to scenario YAML.")
    pl.add_argument("--out", required=True, help="Path to write JSON output (use '-' for stdout).")
    pl.add_argument("--provider", default="", help="openai|ollama (overrides BIOREASONER_MODEL_PROVIDER).")
    pl.add_argument("--model", default="", help="Model name (overrides BIOREASONER_MODEL_NAME).")
    pl.add_argument("--replicate", type=int, default=1, help="Replicate index (for repeated runs).")
    pl.set_defaults(func=cmd_eval_llm)

    # batch-eval
    pb = sub.add_parser("batch-eval", help="Run LLM evaluation over a suite folder of YAML scenarios.")
    pb.add_argument("--suite", required=True, help="Suite folder containing YAML scenarios (recursive).")
    pb.add_argument("--outdir", required=True, help="Output directory for per-scenario JSON files.")
    pb.add_argument("--provider", default="", help="openai|ollama (overrides BIOREASONER_MODEL_PROVIDER).")
    pb.add_argument("--model", default="", help="Model name (overrides BIOREASONER_MODEL_NAME).")
    pb.add_argument("--replicate", type=int, default=1, help="Replicate index (for repeated runs).")
    pb.set_defaults(func=cmd_batch_eval)

    # analyze
    pa = sub.add_parser("analyze", help="Analyze a folder of eval JSON outputs and compute summary metrics.")
    pa.add_argument("--input", required=True, help="Folder containing eval JSON files.")
    pa.add_argument("--out", required=True, help="Path to write summary JSON (use '-' for stdout).")
    pa.set_defaults(func=cmd_analyze)

    # aggregate-replicates
    pr = sub.add_parser("aggregate-replicates", help="Aggregate summary metrics across replicate folders.")
    pr.add_argument("--root", required=True, help="Root folder containing replicate runs (recursive).")
    pr.add_argument("--out", required=True, help="Path to write aggregate JSON (use '-' for stdout).")
    pr.set_defaults(func=cmd_aggregate_replicates)

    return p


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        return 1
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
