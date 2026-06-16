"""
evaluation/evaluate_all.py
Compute metrics for any subset of models / projects / input modes.

Output layout
-------------
  evaluation/<project>/<input_mode>/
      summary_ambiguity.csv
      summary_incompleteness.csv
      full_metrics.json

  evaluation/combined/          ← aggregate across all projects
      summary_ambiguity.csv
      summary_incompleteness.csv
      full_metrics.json

Usage
-----
  # Evaluate everything available
  python evaluation/evaluate_all.py

  # One model only
  python evaluation/evaluate_all.py --model llama3.1

  # One project, one input mode
  python evaluation/evaluate_all.py --project project_01 --input_mode title_only

  # Specific models + project
  python evaluation/evaluate_all.py --models llama3.1 qwen2.5 --project project_02
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
from prompts.prompt_templates import TASKS, PROMPT_STRATEGIES, INPUT_MODES
from scripts.llm_clients import SUPPORTED_MODELS, MODEL_REGISTRY, get_model_description

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────

def find_available_models(results_dir: str) -> list[str]:
    found = []
    if not os.path.isdir(results_dir):
        return found
    for name in sorted(os.listdir(results_dir)):
        if name in SUPPORTED_MODELS and os.path.isdir(os.path.join(results_dir, name)):
            found.append(name)
    return found


def discover_projects(results_dir: str, models: list[str]) -> list[str]:
    projects = set()
    for model in models:
        model_dir = os.path.join(results_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for name in os.listdir(model_dir):
            if os.path.isdir(os.path.join(model_dir, name)) and name.startswith("project"):
                projects.add(name)
    return sorted(projects)


def find_result_files(
    results_dir: str,
    models: list[str],
    project: str,
    input_mode: str,
    task: str,
) -> list[tuple[str, str, str]]:
    """
    Returns [(label, csv_path, pred_col), ...] for individual models + ensemble.
    """
    entries = []

    for model in models:
        for strategy in PROMPT_STRATEGIES:
            path = os.path.join(
                results_dir, model, project, input_mode,
                f"{model}_{task}_{strategy}.csv"
            )
            if os.path.exists(path):
                entries.append((f"{model} / {strategy}", path, "prediction"))

    # Ensemble files
    for strategy in PROMPT_STRATEGIES:
        path = os.path.join(
            results_dir, "ensemble", project, input_mode,
            f"ensemble_{task}_{strategy}.csv"
        )
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
            gt = int(row["ground_truth"])
            pr = int(row[pred_col])
            if pr == -1:        # parse failure — skip
                continue
            y_true.append(gt)
            y_pred.append(pr)
    return y_true, y_pred


# ─────────────────────────────────────────────
# SINGLE (project, input_mode) EVALUATION
# ─────────────────────────────────────────────

def evaluate_slice(
    results_dir: str,
    output_dir: str,
    models: list[str],
    project: str,
    input_mode: str,
) -> dict:
    """Evaluate one (project, input_mode) slice. Returns full_metrics dict."""
    slice_out = os.path.join(output_dir, project, input_mode)
    os.makedirs(slice_out, exist_ok=True)

    all_results: dict[str, list[dict]] = {task: [] for task in TASKS}

    for task in TASKS:
        logger.info(
            "  [%s / %s / %s]", project, input_mode, task.upper()
        )
        entries = find_result_files(results_dir, models, project, input_mode, task)
        if not entries:
            logger.warning("    No result files found.")
            continue

        for label, path, pred_col in entries:
            try:
                y_true, y_pred = load_predictions(path, pred_col)
                if not y_true:
                    logger.warning("    Empty predictions for %s", label)
                    continue
                m = compute_metrics(y_true, y_pred)
                parts = label.split(" / ")
                row = {
                    "model":      parts[0],
                    "strategy":   parts[1] if len(parts) > 1 else "N/A",
                    "model_desc": get_model_description(parts[0]) if parts[0] in MODEL_REGISTRY else parts[0],
                    "project":    project,
                    "input_mode": input_mode,
                    **m,
                }
                all_results[task].append(row)
                print(format_metrics(m, title=f"{project} / {input_mode} / {label}"))
            except Exception as exc:
                logger.error("    Error processing %s: %s", path, exc)

        summary_path = os.path.join(slice_out, f"summary_{task}.csv")
        _write_summary_csv(all_results[task], summary_path)
        logger.info("    Summary → %s", summary_path)

    json_path = os.path.join(slice_out, "full_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ─────────────────────────────────────────────
# MAIN EVALUATE ALL
# ─────────────────────────────────────────────

def evaluate_all(
    results_dir: str = "results",
    output_dir: str = "evaluation",
    models: list[str] | None = None,
    projects: list[str] | None = None,
    input_modes: list[str] | None = None,
) -> dict:
    """
    Evaluate all (project, input_mode) slices, then write a combined summary.
    Returns a nested dict: {project: {input_mode: full_metrics}}.
    """
    os.makedirs(output_dir, exist_ok=True)

    if models is None:
        models = find_available_models(results_dir)
        if not models:
            logger.error("No model result folders found in '%s'", results_dir)
            return {}
        logger.info("Models with results: %s", models)
    else:
        models = [m for m in models if m in SUPPORTED_MODELS]

    if projects is None:
        projects = discover_projects(results_dir, models)
        logger.info("Projects discovered: %s", projects)

    if input_modes is None:
        input_modes = INPUT_MODES

    combined: dict = {}          # {project: {input_mode: full_metrics}}
    all_rows: dict = {task: [] for task in TASKS}   # for combined summary

    for project in projects:
        combined[project] = {}
        for input_mode in input_modes:
            logger.info("")
            logger.info("══ %s / %s ══", project.upper(), input_mode.upper())
            metrics = evaluate_slice(
                results_dir=results_dir,
                output_dir=output_dir,
                models=models,
                project=project,
                input_mode=input_mode,
            )
            combined[project][input_mode] = metrics
            for task in TASKS:
                all_rows[task].extend(metrics.get(task, []))

    # Write combined summary across all projects
    combined_dir = os.path.join(output_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    for task in TASKS:
        _write_summary_csv(all_rows[task], os.path.join(combined_dir, f"summary_{task}.csv"))
    with open(os.path.join(combined_dir, "full_metrics.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    logger.info("\nCombined summary → %s/", combined_dir)

    _print_leaderboard(all_rows)
    return combined


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def _write_summary_csv(rows: list[dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_leaderboard(all_rows: dict):
    print("\n" + "═" * 85)
    print("  LEADERBOARD  (ranked by F1-score, all projects combined)")
    print("═" * 85)
    for task, rows in all_rows.items():
        print(f"\n  ── {task.upper()} ──")
        valid = [r for r in rows if r.get("f1") is not None]
        for rank, r in enumerate(
            sorted(valid, key=lambda x: x["f1"], reverse=True), 1
        ):
            desc = r.get("model_desc", r["model"])
            print(
                f"  {rank:2}. {r['model']:<12} | {r.get('input_mode',''):<12} "
                f"| {r['strategy']:<14} | project={r.get('project',''):<12} "
                f"| F1={r['f1']:.4f}  Acc={r['accuracy']:.4f}  "
                f"P={r['precision']:.4f}  R={r['recall']:.4f}"
                f"  ({desc})"
            )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate LLM + ensemble results across projects and input modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/evaluate_all.py
  python evaluation/evaluate_all.py --model llama3.1
  python evaluation/evaluate_all.py --project project_01 --input_mode title_only
  python evaluation/evaluate_all.py --models llama3.1 qwen2.5 --project project_02
        """,
    )
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--output_dir",   default="evaluation")
    p.add_argument("--model",        default=None, help="Evaluate a single model only")
    p.add_argument("--models",       nargs="+", default=None)
    p.add_argument("--project",      default=None, help="Restrict to one project stem")
    p.add_argument("--input_mode",   choices=INPUT_MODES, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models     = [args.model]    if args.model    else (args.models or None)
    projects   = [args.project]  if args.project  else None
    input_modes = [args.input_mode] if args.input_mode else None

    evaluate_all(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        models=models,
        projects=projects,
        input_modes=input_modes,
    )