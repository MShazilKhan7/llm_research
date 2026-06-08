"""
scripts/run_inference.py
Run inference for ONE model across all (or selected) tasks and strategies.
Designed to be run independently per model — no need to run all at once.

Key design
----------
- --model is REQUIRED. One run = one model. Run separately for each model.
- Results are saved to results/<model_alias>/  so each model has its own folder.
- Supports --resume: skips combinations already completed in a previous run.
- Saves run_config.json + requirements_snapshot.txt per model run.

Usage
-----
  # Run all tasks + strategies for llama3.1
  python scripts/run_inference.py --model llama3.1

  # Run only ambiguity task, zero_shot strategy
  python scripts/run_inference.py --model qwen2.5 --task ambiguity --strategy zero_shot

  # Resume an interrupted run (skips already-done combos)
  python scripts/run_inference.py --model mistral --resume

  # Mock run (no Ollama needed)
  python scripts/run_inference.py --model llama3.1 --mock

  # List all available models
  python scripts/run_inference.py --list_models
"""

from __future__ import annotations
import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts.prompt_templates import PROMPT_REGISTRY, PROMPT_STRATEGIES, TASKS
from scripts.llm_clients import (
    get_client, SUPPORTED_MODELS, OLLAMA_MODELS, MODEL_REGISTRY, get_model_description
)
from scripts.reproducibility import (
    save_run_config, save_env_snapshot, assert_dataset_unchanged,
    make_result_row, EXTENDED_FIELDNAMES, dataset_checksum,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TASK_TO_COL = {"ambiguity": "ambiguous", "incompleteness": "incomplete"}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    issues = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            issues.append(row)
    logger.info("Loaded %d issues from '%s'", len(issues), path)
    return issues


def result_path(output_dir: str, model_alias: str, task: str, strategy: str) -> str:
    """All results for a model go into results/<model_alias>/"""
    folder = os.path.join(output_dir, model_alias)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{model_alias}_{task}_{strategy}.csv")


def is_complete(path: str, expected_rows: int) -> bool:
    """Return True if the result CSV exists and has the expected number of data rows."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = sum(1 for _ in csv.DictReader(f))
        return rows >= expected_rows
    except Exception:
        return False


# ─────────────────────────────────────────────
# SINGLE COMBINATION RUNNER
# ─────────────────────────────────────────────

def run_combination(
    issues: list[dict],
    model_alias: str,
    task: str,
    strategy: str,
    mock: bool = False,
    output_dir: str = "results",
    expected_checksum: str | None = None,
    dataset_path: str = "dataset/ground_truth.csv",
    resume: bool = False,
) -> str:
    """
    Run inference for one (model, task, strategy) combination.
    Returns path to output CSV.
    """
    out_path = result_path(output_dir, model_alias, task, strategy)

    # Resume: skip if already complete
    if resume and is_complete(out_path, len(issues)):
        logger.info("  [SKIP — already done] %s", out_path)
        return out_path

    if expected_checksum:
        assert_dataset_unchanged(dataset_path, expected_checksum)

    client       = get_client(model_alias, mock=mock)
    prompt_fn    = PROMPT_REGISTRY[task][strategy]
    label_col    = TASK_TO_COL[task]
    model_string = MODEL_REGISTRY[model_alias]["model_string"]

    logger.info(
        "Running: model=%-12s  task=%-16s  strategy=%-14s",
        f"{model_alias} ({model_string})", task, strategy,
    )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXTENDED_FIELDNAMES)
        writer.writeheader()

        for i, issue in enumerate(issues):
            prompt = prompt_fn(issue["title"], issue["description"])
            label, raw = client.predict(prompt)

            writer.writerow(make_result_row(
                issue=issue,
                label_col=label_col,
                prediction=label,
                raw_response=raw,
                prompt_text=prompt,
                model_version=model_string,
            ))

            if (i + 1) % 20 == 0:
                logger.info("    %d / %d done", i + 1, len(issues))

    logger.info("  Saved → %s", out_path)
    return out_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run LLM inference for a single model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_inference.py --model llama3.1
  python scripts/run_inference.py --model qwen2.5 --task ambiguity
  python scripts/run_inference.py --model mistral --strategy zero_shot --resume
  python scripts/run_inference.py --list_models
        """,
    )
    p.add_argument("--model",       choices=SUPPORTED_MODELS, default=None,
                   help="Model alias to run (required unless --list_models)")
    p.add_argument("--task",        choices=TASKS,             default=None,
                   help="Specific task (default: all tasks)")
    p.add_argument("--strategy",    choices=PROMPT_STRATEGIES, default=None,
                   help="Specific strategy (default: all strategies)")
    p.add_argument("--dataset",     default="dataset/ground_truth.csv")
    p.add_argument("--output_dir",  default="results",
                   help="Root results directory (model subfolder created automatically)")
    p.add_argument("--mock",        action="store_true",
                   help="Use mock client — no Ollama needed")
    p.add_argument("--resume",      action="store_true",
                   help="Skip combinations that already have complete result CSVs")
    p.add_argument("--list_models", action="store_true",
                   help="Print available models and exit")
    return p.parse_args()


def main():
    args = parse_args()

    # ── List models ──────────────────────────────────────────────
    if args.list_models:
        print("\nAvailable models:")
        print(f"  {'Alias':<12}  {'Backend':<8}  {'Model String':<40}  Description")
        print("  " + "─" * 80)
        for alias, cfg in MODEL_REGISTRY.items():
            print(f"  {alias:<12}  {cfg['backend']:<8}  {cfg['model_string']:<40}  {cfg['description']}")
        print()
        sys.exit(0)

    if not args.model:
        print("ERROR: --model is required. Use --list_models to see options.")
        sys.exit(1)

    tasks      = [args.task]     if args.task     else TASKS
    strategies = [args.strategy] if args.strategy else PROMPT_STRATEGIES

    issues   = load_dataset(args.dataset)
    checksum = dataset_checksum(args.dataset)
    logger.info("Dataset checksum (md5): %s", checksum)

    # Save run config for this model
    model_output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_output_dir, exist_ok=True)

    save_run_config(
        output_dir=model_output_dir,
        dataset_path=args.dataset,
        models=[args.model],
        tasks=tasks,
        strategies=strategies,
        model_versions={args.model: MODEL_REGISTRY[args.model]["model_string"]},
        extra={"mock_mode": args.mock, "resume": args.resume},
    )
    save_env_snapshot(model_output_dir)

    total = len(tasks) * len(strategies)
    done  = 0

    logger.info("")
    logger.info("Model      : %s  (%s)", args.model, get_model_description(args.model))
    logger.info("Backend    : %s", MODEL_REGISTRY[args.model]["backend"])
    logger.info("Tasks      : %s", tasks)
    logger.info("Strategies : %s", strategies)
    logger.info("Combos     : %d", total)
    logger.info("")

    for task in tasks:
        for strategy in strategies:
            done += 1
            logger.info("── Combo %d/%d: %s | %s ──", done, total, task, strategy)
            run_combination(
                issues=issues,
                model_alias=args.model,
                task=task,
                strategy=strategy,
                mock=args.mock,
                output_dir=args.output_dir,
                expected_checksum=checksum,
                dataset_path=args.dataset,
                resume=args.resume,
            )

    logger.info("")
    logger.info("✓ Done — results in '%s/%s/'", args.output_dir, args.model)


if __name__ == "__main__":
    main()
