# BioReasoner

BioReasoner is a deterministic, symbolic biological causal inference engine with an optional LLM evaluation harness.

It is designed to (1) compute ground-truth causal consequences from curated signaling scenarios, and (2) evaluate how well language models reproduce those consequences from natural-language prompts.

## What problem this solves

Biological pathway reasoning is sensitive to causal direction, context, and exceptions. Purely statistical or generative systems can be convincing yet wrong. BioReasoner provides:

- Deterministic ground truth via forward-chaining rules over symbolic facts
- Contradiction detection and provenance tracking
- Scenario-driven evaluation of LLM outputs against engine ground truth
- Batch evaluation and analysis to quantify hallucinations and omissions

---

## Architecture

### Phase 1: Symbolic engine (deterministic)
Given a scenario YAML:
- parses entities and facts
- applies forward-chaining rules
- emits inferred facts with provenance
- detects contradictions

### Phase 2: LLM evaluation harness (optional)
Given the same scenario YAML:
- generates a natural-language prompt from scenario state
- queries a model (OpenAI optional, Ollama optional)
- parses free-text outputs into symbolic facts
- scores against engine ground truth (TP / FP / FN, micro-F1, macro-F1)

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

---

## Quickstart

### Run the engine on a scenario
```bash
bioreasoner engine --scenario scenarios/wnt/lrp6_basic.yaml --out outputs/engine.json
```

### Evaluate an LLM (OpenAI optional)
```bash
export OPENAI_API_KEY="YOUR_KEY"
export BIOREASONER_MODEL_PROVIDER="openai"
export BIOREASONER_MODEL_NAME="gpt-4.1-mini"

bioreasoner eval-llm --scenario scenarios/wnt/lrp6_basic.yaml --out outputs/eval_openai.json
```

### Evaluate an LLM (Ollama optional)
```bash
export BIOREASONER_MODEL_PROVIDER="ollama"
export BIOREASONER_OLLAMA_BASE_URL="http://localhost:11434"
export BIOREASONER_MODEL_NAME="llama3.1"

bioreasoner eval-llm --scenario scenarios/wnt/lrp6_basic.yaml --out outputs/eval_ollama.json
```

### Batch evaluation
```bash
bioreasoner batch-eval --suite examples/tiny_suite --outdir outputs/batch_run --replicate 1
```

### Analyze results
```bash
bioreasoner analyze --input outputs/batch_run --out outputs/summary.json
```

---

## Reproducibility

- The symbolic engine is deterministic.
- LLM evaluation is nondeterministic; run replicates.

Recommended structure:
```text
outputs/
  suite/
    provider=model/
      replicate=1/
      replicate=2/
```

Aggregate replicates:
```bash
bioreasoner aggregate-replicates --root outputs/suite --out outputs/aggregate.json
```

---

## Examples

- `examples/` contains small scenarios for sanity checks.
- `scenarios/` contains curated biological signaling scenarios.

---

## Citation

See `CITATION.cff`.

---

## License

Apache License 2.0.
