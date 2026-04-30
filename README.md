# Boundary-Aware AI Agent

A multi-agent framework for detecting and responding to parasocial attachment signals in human-AI conversations. The system classifies risk level (low / medium / high), routes conversations to the appropriate agent, and generates boundary-aware responses — all running locally via [Ollama](https://ollama.ai).

> **Authorship note:** The initial framework scaffolding was developed with [Claude Code](https://claude.ai/claude-code) (Anthropic's AI coding assistant). All prompt design, evaluation methodology, dataset integration, and iterative refinement were carried out by the team independently.

---

## Architecture

```
User conversation
      │
      ▼
┌─────────────────┐
│  Risk Monitor   │  Classifies LOW / MEDIUM / HIGH risk
└────────┬────────┘
         │
   ┌─────┴──────┐
   ▼            ▼
┌──────────┐  ┌────────────────┐
│Interaction│  │Boundary Agent  │  Routed based on risk level
│  Agent   │  │(medium / high) │
└──────────┘  └────────────────┘
```

The graph is built with [LangGraph](https://github.com/langchain-ai/langgraph). Each node calls a local LLM through Ollama using prompt templates in [src/boundary_aware/prompts/](src/boundary_aware/prompts/).

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| [uv](https://github.com/astral-sh/uv) | any recent |
| [Ollama](https://ollama.ai) | any recent |

Pull the four backbone / judge models used in the evaluation:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull qwen2.5:72b-instruct-q4_K_M
ollama pull gemma3:27b
ollama pull mistral-nemo:12b
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/nafis10670/Boundary-Aware-AI-Agent.git
cd Boundary-Aware-AI-Agent

# Create a virtual environment and install all dependencies
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

uv pip install -e ".[dev]"
```

This installs the `boundary-aware` CLI entry point and all runtime + dev dependencies.

---

## Running the Project

### 1. Verify the setup (smoke test)

Before running any experiment, confirm that Ollama is reachable and the pipeline is healthy:

```bash
boundary-aware smoke-test
```

This runs three synthetic conversations (low / medium / high risk), the baseline, and a judge ensemble call. All checks must pass before proceeding.

To test with a specific model:

```bash
boundary-aware smoke-test --model qwen2.5:72b-instruct-q4_K_M
```

---

### 2. Generate the dataset

The evaluation dataset is derived from the [INTIMA](https://huggingface.co/datasets/Naomh/INTIMA) benchmark. Each single-turn seed prompt is expanded into a 3-turn synthetic conversation.

**Option A — API-based generation (Anthropic or OpenAI, fast):**

```bash
# Set your API key first
export ANTHROPIC_API_KEY=sk-...        # or OPENAI_API_KEY=sk-...

boundary-aware generate-data \
    --output data/intima_mt.jsonl \
    --max-per-code 8 \
    --provider anthropic              # or: --provider openai
```

**Option B — GPU-based generation with vLLM (for larger-scale runs):**

```bash
python scripts/run_generation.py \
    --input  data/INTIMA.jsonl \
    --output outputs/intima_complete.jsonl \
    --model  Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --temperature 0.7 \
    --max-tokens 1200
```

Key flags for `run_generation.py`:

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model or local path |
| `--temperature` | `0.7` | Sampling temperature |
| `--max-tokens` | `1200` | Max generated tokens per sample |
| `--gpu` | `auto` | GPU index or `auto` (picks most free memory) |
| `--gpu-memory-utilization` | `0.85` | Fraction of GPU VRAM to reserve |
| `--limit` | `None` | Cap the number of rows processed |
| `--enforce-eager` | off | Disable vLLM graph capture for debugging |

A pre-generated dataset is already included at [outputs/intima_complete.jsonl](outputs/intima_complete.jsonl).

---

### 3. Run the pipeline on a single conversation

```bash
boundary-aware run-one \
    --conversation-id <ID> \
    --dataset outputs/intima_complete.jsonl
```

This prints the risk level, routing decision, any monitor notes, and the final agent response.

---

### 4. Run the full evaluation

The evaluation is split into two steps to allow different judge models to be applied independently.

**Step 1 — Generate backbone model responses:**

```bash
# Single model
boundary-aware evaluate \
    --dataset outputs/intima_complete.jsonl \
    --model llama3.1:8b-instruct-q4_K_M

# Multiple models in one command
boundary-aware evaluate \
    --dataset outputs/intima_complete.jsonl \
    --model llama3.1:8b-instruct-q4_K_M \
    --model qwen2.5:72b-instruct-q4_K_M \
    --model gemma3:27b
```

Results are written to `data/results/<run-id>/responses.jsonl`.

**Step 2 — Judge responses and compute metrics:**

```bash
boundary-aware judge --run-id llama3_1_8b-instruct-q4_K_M
```

Judges are auto-selected from a different model family than the backbone to avoid self-judging bias. You can also specify judges explicitly:

```bash
boundary-aware judge \
    --run-id llama3_1_8b-instruct-q4_K_M \
    --judge-model qwen2.5:72b-instruct-q4_K_M \
    --judge-model mistral-nemo:12b
```

Metrics are written to `data/results/<run-id>/`.

---

### 5. Generate plots

```bash
python _make_plots.py
```

Plots are saved to [plots/](plots/).

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BOUNDARY_AWARE_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Backbone model for the pipeline |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `ANTHROPIC_API_KEY` | — | Required for `--provider anthropic` dataset generation |
| `OPENAI_API_KEY` | — | Required for `--provider openai` dataset generation |

---

## Running Tests

```bash
pytest
```

The test suite covers agents, the Ollama client, schemas, and the end-to-end workflow (50 tests, all mocked — no live Ollama instance required).

---

## Project Structure

```
src/boundary_aware/
├── agents/          # boundary_agent, interaction_agent, risk_monitor
├── data/            # Dataset loader and INTIMA-MT generator
├── eval/            # Baseline, judge ensemble, metrics, eval runner
├── graph/           # LangGraph workflow definition
├── llm/             # Ollama HTTP client
├── prompts/         # Prompt templates (.txt) for each agent
└── schemas/         # Pydantic schemas (conversation, routing, state)
scripts/
└── run_generation.py    # vLLM-based dataset generation
data/
├── INTIMA.jsonl         # Raw INTIMA benchmark seed prompts
└── icl_examples.json    # Few-shot examples (excluded from eval to prevent leakage)
outputs/
└── intima_complete.jsonl  # Pre-generated multi-turn dataset
```

---

## Acknowledgements

This project uses the [INTIMA](https://huggingface.co/datasets/Naomh/INTIMA) dataset for evaluation. The framework skeleton was scaffolded with [Claude Code](https://claude.ai/claude-code); all subsequent design decisions, prompt engineering, and experimental iterations were done by the team.
