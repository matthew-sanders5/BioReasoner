from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m bioreasoner.batch_eval",
        description="Run eval-llm on all scenario YAMLs in a directory.",
    )
    parser.add_argument(
        "examples_dir",
        nargs="?",
        default="examples",
        help="Directory containing scenario .yaml files (default: examples).",
    )
    parser.add_argument(
        "--glob",
        default="*.yaml",
        help="Glob pattern for scenario files within the directory (default: *.yaml).",
    )
    parser.add_argument(
        "--out-dir",
        default="out",
        help="Directory to write JSON eval outputs (default: out).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model name to pass to bioreasoner eval-llm (default: gpt-4.1-mini).",
    )
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not examples_dir.is_dir():
        print(f"Examples directory does not exist: {examples_dir}")
        return

    scenario_paths = sorted(examples_dir.glob(args.glob))
    if not scenario_paths:
        print(f"No files matched pattern {args.glob!r} in {examples_dir}")
        return

    print(
        f"Found {len(scenario_paths)} scenario(s) in {examples_dir} "
        f"matching {args.glob!r}."
    )

    for path in scenario_paths:
        base = path.stem
        out_file = out_dir / f"{base}.json"
        print(f"Running {path} -> {out_file}")

        with out_file.open("w") as f:
            subprocess.run(
                [
                    "bioreasoner",
                    "eval-llm",
                    str(path),
                    "--model",
                    args.model,
                    "--format",
                    "json",
                ],
                check=True,
                stdout=f,
            )

    print("All scenarios complete.")


if __name__ == "__main__":
    main()
