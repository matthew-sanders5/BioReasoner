from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import statistics as stats


@dataclass
class EvalRecord:
    scenario_name: str
    model_name: str
    tp: int
    fp: int
    fn: int


def load_eval_file(path: Path) -> EvalRecord:
    data = json.loads(path.read_text())
    return EvalRecord(
        scenario_name=data["scenario_name"],
        model_name=data["model_name"],
        tp=len(data["true_positives"]),
        fp=len(data["false_positives"]),
        fn=len(data["false_negatives"]),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate BioReasoner eval-llm JSON outputs."
    )
    parser.add_argument(
        "glob",
        type=str,
        help="Glob pattern for eval JSON files, e.g. 'out/*.json'.",
    )
    args = parser.parse_args()

    paths = sorted(Path(".").glob(args.glob))
    if not paths:
        print(f"No files matched pattern: {args.glob}")
        return

    records: List[EvalRecord] = [load_eval_file(p) for p in paths]

    # Per-scenario metrics
    print("Per-scenario metrics:\n")
    for r in records:
        denom = r.tp + r.fn
        recall = r.tp / denom if denom > 0 else 0.0
        denom_p = r.tp + r.fp
        precision = r.tp / denom_p if denom_p > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        print(
            f"{r.scenario_name:40s} "
            f"TP={r.tp:2d} FP={r.fp:2d} FN={r.fn:2d} "
            f"P={precision:.2f} R={recall:.2f} F1={f1:.2f}"
        )

    # Global metrics
    total_tp = sum(r.tp for r in records)
    total_fp = sum(r.fp for r in records)
    total_fn = sum(r.fn for r in records)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )

    macro_f1 = stats.mean(
        [
            (
                lambda r: (
                    lambda p, r_: (
                        2 * p * r_ / (p + r_) if p + r_ > 0 else 0.0
                    )
                )(
                    r.tp / (r.tp + r.fp) if (r.tp + r.fp) > 0 else 0.0,
                    r.tp / (r.tp + r.fn) if (r.tp + r.fn) > 0 else 0.0,
                )
            )(r)
            for r in records
        ]
    )

    print("\nGlobal metrics:")
    print(f"  Micro-averaged: P={micro_precision:.2f} R={micro_recall:.2f} F1={micro_f1:.2f}")
    print(f"  Macro-averaged F1: {macro_f1:.2f}")


if __name__ == "__main__":
    main()
