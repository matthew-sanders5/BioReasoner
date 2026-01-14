#!/usr/bin/env bash
set -euo pipefail

# Offline reproducibility: no OpenAI, no Ollama.
export BIOREASONER_MODEL_PROVIDER="stub"
export BIOREASONER_MODEL_NAME="stub"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${ROOT}/outputs/repro_offline/replicate=1"

mkdir -p "${OUTDIR}"

# Run batch eval over scenarios
bioreasoner batch-eval --suite "${ROOT}/scenarios" --outdir "${OUTDIR}" --replicate 1

# Analyze batch outputs into a summary
bioreasoner analyze --input "${OUTDIR}" --out "${ROOT}/outputs/repro_offline/summary.json"

# Aggregate (trivial here, but demonstrates the interface)
bioreasoner aggregate-replicates --root "${ROOT}/outputs/repro_offline" --out "${ROOT}/outputs/repro_offline/aggregate.json"

echo "Wrote:"
echo "  ${ROOT}/outputs/repro_offline/summary.json"
echo "  ${ROOT}/outputs/repro_offline/aggregate.json"
