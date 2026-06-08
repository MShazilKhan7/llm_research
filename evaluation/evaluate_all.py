"""
evaluation/evaluate_all.py
Compute metrics for any subset of models that have completed inference.

Model-aware: reads from results/<model_alias>/ subfolders.
Ensemble CSVs are read from results/ensemble/.

Can be run after EACH model to see incremental results — no need to wait
for all models to finish.

Outputs (always merged/appended, never overwritten per-model):
  evaluation/summary_ambiguity.csv
  evaluation/summary_incompleteness.csv
  evaluation/full_metrics.json

Usage
-----
  # Evaluate all available results (models + ensemble)
  python evaluation/evaluate_all.py

  # Evaluate a specific model only
  python evaluation/evaluate_all.py --model llama3.1

  # Evaluate specific models + ensemble
  python evaluation/evaluate_all.py --models llama3.1 qwen2.5 ensemble
"""

from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import compute_metrics, format_metrics
from prompts.prompt_templates import TASKS, PROMPT_STRATEGIES
from scripts.llm_clients import SUPPORTED_MODELS, MODEL_REGISTRY, get_model_description

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────

def find_available_models(results_dir: str) -> list[str]:
    """Return model aliases that have a results subfolder."""
    found = []
    if not os.path.isdir(results_dir):
        return found
    for name in sorted(os.listdir(results_dir)):
        if name in SUPPORTED_MODELS and os.path.isdir(os.path.join(results_dir, name)):
            found.append(name)
    return found


def find_result_files(
    results_dir: str,
    models: list[str],
    task: str,
) -> list[tuple[str, str, str]]:
    """
    Returns [(label, csv_path, pred_col), ...] for all available
    model + ensemble files for the given task.
    """
    entries = []

    # Individual model files
    for model in models:
        for strategy in PROMPT_STRATEGIES:
            path = os.path.join(results_dir, model, f"{model}_{task}_{strategy}.csv")
            if os.path.exists(path):
                entries.append((f"{model} / {strategy}", path, "prediction"))

    # Ensemble files (live in results/ensemble/)
    ensemble_dir = os.path.join(results_dir, "ensemble")
    for strategy in PROMPT_STRATEGIES:
        path = os.path.join(ensemble_dir, f"ensemble_{task}_{strategy}.csv")
        if os.path.exists(path):
            entries.append((f"ensemble / {strategy}", path, "ensemble_prediction"))

    return entries


# ─────────────────────────────────────────────
# LOAD PREDICTIONS
# ─────────────────────────────────────────────

def load_predictions(path: str, pred_col: str = "prediction") -> tuple[list[int], list[int]]:
    y_true, y_pred = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y_true.append(int(row["ground_truth"]))
            y_pred.append(int(row[pred_col]))
    return y_true, y_pred


# ─────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────

def evaluate_all(
    results_dir: str = "results",
    output_dir: str = "evaluation",
    models: list[str] | None = None,
) -> dict:
    """
    Evaluate all available (or specified) model results.
    Returns the full metrics dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Auto-discover if not specified
    if models is None:
        models = find_available_models(results_dir)
        if not models:
            logger.error("No model result folders found in '%s'", results_dir)
            return {}
        logger.info("Models with results: %s", models)
    else:
        # Filter to only models in SUPPORTED_MODELS (exclude "ensemble" — handled separately)
        models = [m for m in models if m in SUPPORTED_MODELS]

    all_results: dict[str, list[dict]] = {task: [] for task in TASKS}

    for task in TASKS:
        logger.info("\n══════════════════════════════════════")
        logger.info("  Task: %s", task.upper())
        logger.info("══════════════════════════════════════")

        entries = find_result_files(results_dir, models, task)
        if not entries:
            logger.warning("  No result files found for task '%s'", task)
            continue

        for label, path, pred_col in entries:
            try:
                y_true, y_pred = load_predictions(path, pred_col)
                m = compute_metrics(y_true, y_pred)
                parts = label.split(" / ")
                row = {
                    "model":       parts[0],
                    "strategy":    parts[1] if len(parts) > 1 else "N/A",
                    "model_desc":  get_model_description(parts[0]) if parts[0] in MODEL_REGISTRY else parts[0],
                    **m,
                }
                all_results[task].append(row)
                print(format_metrics(m, title=label))
            except Exception as exc:
                logger.error("  Error processing %s: %s", path, exc)

        # Save per-task CSV summary
        summary_path = os.path.join(output_dir, f"summary_{task}.csv")
        _write_summary_csv(all_results[task], summary_path)
        logger.info("  Summary → %s", summary_path)

    # Save full JSON
    json_path = os.path.join(output_dir, "full_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nFull metrics → %s", json_path)

    _print_leaderboard(all_results)
    return all_results


def _write_summary_csv(rows: list[dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_leaderboard(all_results: dict):
    print("\n" + "═" * 75)
    print("  LEADERBOARD  (ranked by F1-score)")
    print("═" * 75)
    for task, rows in all_results.items():
        print(f"\n  ── {task.upper()} ──")
        valid = [r for r in rows if r.get("f1") is not None]
        for rank, r in enumerate(sorted(valid, key=lambda x: x["f1"], reverse=True), 1):
            desc = r.get("model_desc", r["model"])
            print(
                f"  {rank:2}. {r['model']:<12} | {r['strategy']:<14} "
                f"| F1={r['f1']:.4f}  Acc={r['accuracy']:.4f}  "
                f"P={r['precision']:.4f}  R={r['recall']:.4f}"
                f"  ({desc})"
            )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate LLM + ensemble results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/evaluate_all.py
  python evaluation/evaluate_all.py --model llama3.1
  python evaluation/evaluate_all.py --models llama3.1 qwen2.5 mistral
        """,
    )
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output_dir",  default="evaluation")
    p.add_argument("--model",       default=None,
                   help="Evaluate a single model only")
    p.add_argument("--models", nargs="+", default=None,
                   help="Evaluate specific models (space-separated)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --model takes priority over --models; if neither, auto-discover
    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models
    else:
        models = None

    evaluate_all(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        models=models,
    )
