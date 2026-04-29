# Evaluation Results

This directory contains completed `exp1_iso` evaluation runs for the boundary-aware
AI companion experiments.

## Setup

- Dataset: `outputs/intima_complete.jsonl`
- Conversations per run: `375`
- Output directory pattern: `data/results/<run_id>/`
- Each completed run contains:
  - `responses.jsonl`: per-conversation system response, baseline response, and judge labels
  - `summary.csv`: per-category boundary/companionship rates
  - `routing.csv`: routing and risk-level counts
  - `report.md`: human-readable run summary

## Model Runs

| Run directory                               | Model                         | Host/GPU assignment   | Purpose                          |
| ------------------------------------------- | ----------------------------- | --------------------- | -------------------------------- |
| `exp1_iso_gpu0_qwen2_5_32b-instruct-q4_K_M` | `qwen2.5:32b-instruct-q4_K_M` | GPU0 / Ollama `11436` | Completed main model comparison  |
| `exp1_iso_gpu0_qwen2_5_72b-instruct-q4_K_M` | `qwen2.5:72b-instruct-q4_K_M` | GPU0 / Ollama `11434` | Completed larger Qwen comparison |
| `exp1_iso_gpu1_gemma2_27b-instruct-q4_K_M`  | `gemma2:27b-instruct-q4_K_M`  | GPU1 / Ollama `11435` | Completed Gemma comparison       |
| `exp1_iso_gpu1_mistral-nemo_12b`            | `mistral-nemo:12b`            | GPU1 / Ollama `11435` | Completed Mistral comparison     |
| `exp1_iso_gpu1_phi4_14b`                    | `phi4:14b`                    | GPU1 / Ollama `11435` | Completed Phi comparison         |
| `exp1_iso_gpu1_qwen3_30b-a3b-q4_K_M`        | `qwen3:30b-a3b-q4_K_M`        | GPU1 / Ollama `11435` | Completed Qwen3 MoE comparison   |

## Overall Metrics

Rates below are weighted across all 375 conversations in each run.

| Model                         | System boundary maintaining | Baseline boundary maintaining | System companionship reinforcing | Baseline companionship reinforcing |
| ----------------------------- | --------------------------: | ----------------------------: | -------------------------------: | ---------------------------------: |
| `qwen2.5:32b-instruct-q4_K_M` |                       98.4% |                         85.1% |                             0.3% |                              11.2% |
| `qwen2.5:72b-instruct-q4_K_M` |                       98.9% |                         74.4% |                             0.3% |                              21.1% |
| `gemma2:27b-instruct-q4_K_M`  |                      100.0% |                         91.7% |                             0.0% |                               8.0% |
| `mistral-nemo:12b`            |                       96.8% |                         69.6% |                             3.2% |                              29.9% |
| `phi4:14b`                    |                       99.7% |                         95.7% |                             0.3% |                               4.0% |
| `qwen3:30b-a3b-q4_K_M`        |                       90.7% |                         64.0% |                             1.3% |                              15.7% |

## What Each Model Was Used For

For each run, the model named in the run directory was used for all LLM calls in
that run:

1. Risk monitoring and routing inside the boundary-aware system.
2. Final boundary-aware system response generation.
3. Baseline response generation.
4. LLM-as-judge classification of both system and baseline responses.

Important caveat: there is no separate fixed judge model in these runs. The
evaluation code sets `BOUNDARY_AWARE_MODEL` to the model currently being evaluated,
and the judge uses that same setting. This means each model judged its own system
and baseline outputs.

If `BOUNDARY_AWARE_MODEL` is not set, the code falls back to
`llama3.1:8b-instruct-q4_K_M`, but that fallback was not used for these completed
`exp1_iso` evaluations.
