# LLM Defect Detection in Software Issues

Research project evaluating open-source LLMs (**Llama 3.1 8B**, **Qwen 2.5 14B**, **Mistral 7B**) for detecting **ambiguity** and **incompleteness** defects in real-world software issues from the [TAWOS dataset](https://github.com/SOLAR-group/TAWOS).

All models run **locally via Ollama** — no API keys, no rate limits, full reproducibility.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Quick Start](#quick-start)
4. [Running the Pipeline](#running-the-pipeline)
5. [Step-by-Step Commands](#step-by-step-commands)
6. [Models](#models)
7. [Prompting Strategies](#prompting-strategies)
8. [Reproducibility](#reproducibility)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Results Structure](#results-structure)
11. [Notebooks Guide](#notebooks-guide)
12. [Using Your Real TAWOS Dataset](#using-your-real-tawos-dataset)
13. [Research Questions](#research-questions)
14. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
llm_defect_detection/
│
├── dataset/
│   ├── project_01.csv            ← Ground truth for project 1 (TAWOS issues)
│   ├── project_02.csv            ← … through project_05.csv
│   ├── project_03.csv
│   ├── project_04.csv
│   ├── project_05.csv
│   └── dataset_meta.json         ← seed, n, label counts — auto-generated (synthetic only)
│
├── evaluation/                   ← Auto-generated — per project + input mode
│   ├── __init__.py
│   ├── metrics.py                ← Accuracy, Precision, Recall, F1, Confusion Matrix
│   ├── evaluate_all.py           ← Aggregate all results + leaderboard
│   ├── project_01/
│   │   ├── title_only/
│   │   │   ├── summary_ambiguity.csv
│   │   │   ├── summary_incompleteness.csv
│   │   │   └── full_metrics.json
│   │   └── title_desc/
│   │       └── …
│   ├── project_02/ … project_05/
│   └── combined/                 ← Aggregate across all projects
│       ├── summary_ambiguity.csv
│       ├── summary_incompleteness.csv
│       └── full_metrics.json
│
├── figures/                      ← Auto-generated — mirrors evaluation layout
│   ├── project_01/
│   │   ├── title_only/
│   │   │   ├── f1_heatmap_ambiguity.png
│   │   │   ├── metrics_bar_ambiguity.png
│   │   │   ├── confusion_matrices.png
│   │   │   └── ensemble_vs_individual.png
│   │   └── title_desc/
│   │       └── …
│   └── combined/
│       ├── … (same figure set)
│       └── input_mode_comparison_ambiguity.png
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_prompt_engineering.ipynb
│   ├── 03_llm_inference.ipynb
│   ├── 04_ensemble.ipynb
│   ├── 05_evaluation_metrics.ipynb
│   └── 06_figures_analysis.ipynb
│
├── outputs/                      ← Paper tables, extra exports
│
├── prompts/
│   ├── __init__.py
│   └── prompt_templates.py       ← All prompt functions (4 strategies × 2 tasks × 2 input modes)
│
├── results/                      ← Auto-generated — one subfolder per model
│   ├── llama3.1/
│   │   ├── run_config.json
│   │   ├── requirements_snapshot.txt
│   │   └── project_01/
│   │       ├── title_only/
│   │       │   ├── llama3.1_ambiguity_zero_shot.csv
│   │       │   └── …  (8 CSVs: 2 tasks × 4 strategies)
│   │       └── title_desc/
│   │           └── …  (8 CSVs)
│   │   └── project_02/ … project_05/
│   ├── qwen2.5/   (same structure)
│   ├── mistral/   (same structure)
│   └── ensemble/
│       └── project_01/
│           ├── title_only/
│           │   └── ensemble_ambiguity_zero_shot.csv
│           └── title_desc/
│               └── …
│
├── scripts/
│   ├── generate_dataset.py       ← Create synthetic ground truth CSV
│   ├── llm_clients.py            ← MODEL_REGISTRY + Ollama/cloud clients
│   ├── run_inference.py          ← Run inference for ONE model
│   ├── ensemble.py               ← Majority-voting ensemble builder
│   ├── generate_figures.py       ← All publication figures
│   ├── reproducibility.py        ← Checksums, run config, env snapshot
│   └── run_pipeline.py           ← ★ Master pipeline (per-model, all steps)
│
└── requirements.txt
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

Download and install from **https://ollama.com** (available for macOS, Linux, Windows).

### 3. Start the Ollama server

```bash
ollama serve
```

Keep this running in a separate terminal throughout your experiments.

### 4. Pull the three research models

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull mistral:7b-instruct-v0.3
```

> Each model downloads once and is cached locally. Sizes: ~5GB, ~9GB, ~4.5GB respectively.

### 5. Verify setup

```bash
ollama list          # should show all three models
ollama run llama3.1:8b-instruct-q4_K_M "Say hello"   # quick test
```

---

## Quick Start

```bash
# See all available models and their aliases
python scripts/run_pipeline.py --list_models

# Test the full pipeline without Ollama (mock predictions)
python scripts/run_pipeline.py --model llama3.1 --mock

# Run the real pipeline for one model
python scripts/run_pipeline.py --model llama3.1
```

---

## Running the Pipeline

The pipeline is **per-model by design**. Run one model at a time, in any order, on any schedule. Evaluation and figures automatically include all models that have results at that point.

Each model is evaluated across **5 projects**, **2 input modes** (`title_only` and `title_desc`), **2 tasks** (ambiguity, incompleteness), and **4 prompting strategies** — 80 inference combinations per model (5 × 2 × 2 × 4).

### Typical workflow

```bash
# Day 1 — run llama3.1 (all projects, both input modes)
python scripts/run_pipeline.py --model llama3.1

# Day 2 — run qwen2.5
python scripts/run_pipeline.py --model qwen2.5

# Day 3 — run mistral (ensemble of all 3 is built automatically)
python scripts/run_pipeline.py --model mistral
```

After each run, evaluation and figures are regenerated to include all completed models.

### Resume an interrupted run

```bash
# Skips any (project, input_mode, task, strategy) combos already completed
python scripts/run_pipeline.py --model qwen2.5 --resume
```

### Run a specific project, input mode, task, or strategy

```bash
# Only project_02
python scripts/run_pipeline.py --model llama3.1 --project project_02

# Only title_only input mode (issue title passed to LLM, no description)
python scripts/run_pipeline.py --model llama3.1 --input_mode title_only

# Only title + description input mode
python scripts/run_pipeline.py --model llama3.1 --input_mode title_desc

# Only ambiguity task
python scripts/run_pipeline.py --model llama3.1 --task ambiguity

# Only zero_shot strategy
python scripts/run_pipeline.py --model mistral --strategy zero_shot

# Combine filters — e.g. project_01, title_only, ambiguity, cot
python scripts/run_pipeline.py --model qwen2.5 --project project_01 --input_mode title_only --task ambiguity --strategy cot
```

### Skip steps you don't need

```bash
# Inference only (skip ensemble, evaluate, figures)
python scripts/run_pipeline.py --model llama3.1 --skip_ensemble --skip_evaluate --skip_figures

# Re-run just evaluation + figures after all models finish
python scripts/run_pipeline.py --model mistral --skip_inference --skip_ensemble

# Re-run figures only (e.g. after style changes)
python scripts/run_pipeline.py --model mistral --skip_inference --skip_ensemble --skip_evaluate
```

---

## Step-by-Step Commands

Each script can also be run independently.

### Generate dataset (synthetic — for testing only)

```bash
python scripts/generate_dataset.py              # default: n=200, seed=42
python scripts/generate_dataset.py --n 200 --seed 42 --output dataset/ground_truth.csv
```

### Run inference (one model)

```bash
# All projects, both input modes
python scripts/run_inference.py --model llama3.1

# One project, title_only
python scripts/run_inference.py --model llama3.1 --project project_01 --input_mode title_only

# Resume after interruption
python scripts/run_inference.py --model llama3.1 --resume

python scripts/run_inference.py --list_models
```

### Build ensemble

```bash
# Auto-discovers all models with results
python scripts/ensemble.py

# Specify models, project, or input mode
python scripts/ensemble.py --models llama3.1 qwen2.5 mistral
python scripts/ensemble.py --project project_01 --input_mode title_only
```

### Evaluate results

```bash
# All available models, all projects, both input modes
python evaluation/evaluate_all.py

# Single model
python evaluation/evaluate_all.py --model llama3.1

# One project + input mode
python evaluation/evaluate_all.py --project project_02 --input_mode title_desc

# Specific set of models
python evaluation/evaluate_all.py --models llama3.1 qwen2.5 --project project_01
```

### Generate figures

```bash
# All projects and input modes
python scripts/generate_figures.py

# One slice
python scripts/generate_figures.py --project project_01 --input_mode title_only

python scripts/generate_figures.py --models llama3.1 qwen2.5 mistral
```

---

## Models

| Alias | Model String | Description | Backend |
|-------|-------------|-------------|---------|
| `llama3.1` | `llama3.1:8b-instruct-q4_K_M` | Llama 3.1 8B Instruct | Ollama (local) |
| `qwen2.5` | `qwen2.5:14b-instruct-q4_K_M` | Qwen 2.5 14B Instruct | Ollama (local) |
| `mistral` | `mistral:7b-instruct-v0.3` | Mistral 7B Instruct v0.3 | Ollama (local) |
| `gpt` | `gpt-4o-mini` | GPT-4o-mini | OpenAI API (optional) |
| `gemini` | `gemini-2.5-flash` | Gemini 2.5 Flash | Google AI API (optional) |

### Adding a new model

Edit `MODEL_REGISTRY` in `scripts/llm_clients.py` — add one entry:

```python
"your_alias": {
    "model_string": "modelname:tag",   # exact Ollama model name
    "backend":      "ollama",
    "rpm":          60,
    "description":  "Your Model Name",
},
```

No other file needs to change. The alias becomes available immediately as a `--model` value.

---

## Prompting Strategies

All prompts are defined in `prompts/prompt_templates.py`.

| Strategy | Description |
|----------|-------------|
| `zero_shot` | Task instruction only — no examples |
| `few_shot` | 3 static labelled examples + classify |
| `cot` | Chain-of-thought step-by-step reasoning |
| `few_shot_cot` | 3 CoT examples with reasoning + classify |

### Input modes

| Mode | What is sent to the LLM | Ground-truth column used |
|------|-------------------------|--------------------------|
| `title_only` | Issue title only | `Final_amb_T` / `Final_inc_T` |
| `title_desc` | Title + description | `Final_amb_TD` / `Final_inc_TD` |

Each strategy is applied independently to both tasks (**ambiguity** and **incompleteness**) and both input modes, giving **16 prompt variants per model per project**. With 5 projects × 3 models × 16 combos = **240 result files** total, plus ensemble files per (project, input_mode, task, strategy).

---

## Reproducibility

The following measures ensure experiments are fully reproducible:

| Mechanism | Where |
|-----------|-------|
| Fixed random seed (`seed=42`) for dataset generation | `generate_dataset.py` |
| `temperature=0, seed=42` for all Ollama calls | `llm_clients.py` |
| MD5 checksum of dataset saved and verified before each run | `reproducibility.py` |
| SHA-256 prefix of each prompt stored per prediction row | result CSVs |
| Exact model string (e.g. `llama3.1:8b-instruct-q4_K_M`) recorded per prediction | result CSVs |
| Full raw LLM response stored (never truncated) | result CSVs |
| `run_config.json` per model: timestamp, dataset hash, model versions, all settings | `results/<model>/` |
| `requirements_snapshot.txt` (`pip freeze`) per model run | `results/<model>/` |
| `dataset_meta.json`: seed, n, label counts | `dataset/` |

To exactly reproduce a past run, check `results/<model>/run_config.json` for all settings used.

---

## Evaluation Metrics

Computed for every (model, task, strategy) combination and for the ensemble:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall fraction of correct predictions |
| **Precision** | Of all predicted defects, how many are true defects |
| **Recall** | Of all true defects, how many were detected |
| **F1-score** | Harmonic mean of Precision and Recall (primary metric) |
| **Confusion Matrix** | TP, FP, TN, FN counts |

> **F1-score is the primary ranking metric** because the dataset may be class-imbalanced. Accuracy alone is misleading when one class dominates.

---

## Results Structure

After running all three models across all five projects, your `results/` folder looks like:

```
results/
├── llama3.1/
│   ├── run_config.json                         ← reproducibility record
│   ├── requirements_snapshot.txt               ← pip freeze at run time
│   └── project_01/
│       ├── title_only/
│       │   ├── llama3.1_ambiguity_zero_shot.csv
│       │   ├── llama3.1_ambiguity_few_shot.csv
│       │   ├── llama3.1_ambiguity_cot.csv
│       │   ├── llama3.1_ambiguity_few_shot_cot.csv
│       │   ├── llama3.1_incompleteness_zero_shot.csv
│       │   └── …  (8 CSVs per input mode)
│       └── title_desc/
│           └── …  (8 CSVs)
│   └── project_02/ … project_05/
├── qwen2.5/   (same structure)
├── mistral/   (same structure)
└── ensemble/
    └── project_01/
        ├── title_only/
        │   └── ensemble_ambiguity_zero_shot.csv
        └── title_desc/
            └── …
```

Each result CSV has these columns:

| Column | Description |
|--------|-------------|
| `issue_id` | Issue identifier |
| `title` | Issue title |
| `ground_truth` | Manually annotated label (0 or 1) — column chosen by task + input_mode |
| `prediction` | Model prediction (0, 1, or -1 for parse failure) |
| `raw_response` | Full untruncated LLM response |
| `prompt_hash` | SHA-256 prefix of the prompt sent |
| `model_version` | Exact model string used |
| `temperature` | Always 0 |
| `timestamp_utc` | When this prediction was made |
| `project` | Project stem, e.g. `project_01` |
| `input_mode` | `title_only` or `title_desc` |
| `task` | `ambiguity` or `incompleteness` |
| `strategy` | Prompting strategy used |

---

## Notebooks Guide

Run notebooks from the `notebooks/` folder. Each is self-contained.

| Notebook | Purpose |
|----------|---------|
| `01_dataset_exploration` | Load dataset, visualise label distributions, cross-tabulations |
| `02_prompt_engineering` | Preview all 8 prompt variants for any issue, estimate token counts |
| `03_llm_inference` | Interactive inference runner — set API keys here if using cloud models |
| `04_ensemble` | Build ensemble, inspect vote distributions and agreement rates |
| `05_evaluation_metrics` | Load summaries, pivot tables, best configuration per task |
| `06_figures_analysis` | Display all figures inline, answer all 4 research questions |

---

## Using Your Real TAWOS Dataset

Place one CSV per project in `dataset/`, named `project_01.csv` through `project_05.csv`. Required columns:

```
Issue_ID, Title, Description, Final_amb_T, Final_amb_TD, Final_inc_T, Final_inc_TD
```

| Column | Description |
|--------|-------------|
| `Issue_ID` | Unique issue identifier |
| `Title` | Issue title |
| `Description` | Issue description |
| `Final_amb_T` | Ambiguity label when evaluating title only (0 or 1) |
| `Final_amb_TD` | Ambiguity label when evaluating title + description (0 or 1) |
| `Final_inc_T` | Incompleteness label when evaluating title only (0 or 1) |
| `Final_inc_TD` | Incompleteness label when evaluating title + description (0 or 1) |

The pipeline automatically discovers all `*.csv` files in `dataset/`. Use `--project project_02` to run a single project. Labels are **independent** — an issue can be both ambiguous and incomplete, or neither.

Everything downstream reads from these files automatically. No other changes needed.

---

## Research Questions

| RQ | Question | Answered by |
|----|----------|-------------|
| RQ1 | Which LLM performs best for defect detection? | Leaderboard in `evaluate_all.py`, Notebook 06 |
| RQ2 | Which prompting strategy is most effective? | F1 heatmap figures, `summary_*.csv` pivot |
| RQ3 | Does ensemble improve over individual models? | `ensemble_vs_individual.png`, Notebook 06 |
| RQ4 | Which defect type is harder to detect? | Cross-task F1 comparison, Notebook 06 |

---

## Troubleshooting

**`OllamaNotRunning` error**
```bash
ollama serve   # start the server, keep it running
```

**`ModelNotPulled` error**
```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull mistral:7b-instruct-v0.3
```

**Check what models are pulled**
```bash
ollama list
```

**Test without Ollama (mock mode)**
```bash
python scripts/run_pipeline.py --model llama3.1 --mock
```

**Resume after interruption**
```bash
python scripts/run_pipeline.py --model llama3.1 --resume
```

**Resume a specific slice only** (e.g. project_03, title_only)
```bash
python scripts/run_pipeline.py --model llama3.1 --project project_03 --input_mode title_only --resume
```

**Dataset integrity error** — a ground-truth file changed since inference was run. Either restore the original file or re-run inference from scratch (remove the affected result CSVs or the model's results folder first).

**`-1` predictions in result CSV** — the model response could not be parsed as Yes/No. Check the `raw_response` column in that row. This usually means the model gave a long explanation without a clear Yes/No — the `cot` or `few_shot_cot` prompts instruct it to end with `Final Answer: Yes/No` to prevent this.
