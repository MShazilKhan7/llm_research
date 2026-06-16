"""
scripts/ensemble.py
Majority-voting ensemble over individual LLM predictions.

Folder layout (mirrors run_inference.py)
-----------------------------------------
  results/<model>/<project>/<input_mode>/<model>_<task>_<strategy>.csv
  results/ensemble/<project>/<input_mode>/ensemble_<task>_<strategy>.csv

Usage
-----
  # Ensemble all models / projects / input modes that have results
  python scripts/ensemble.py

  # Restrict to specific dimensions
  python scripts/ensemble.py --models llama3.1 qwen2.5
  python scripts/ensemble.py --project project_01
  python scripts/ensemble.py --input_mode title_only
  python scripts/ensemble.py --models llama3.1 qwen2.5 --project project_02 --input_mode title_desc
"""

from __future__ import annotations
import argparse
import csv
import glob
import logging
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.llm_clients import SUPPORTED_MODELS
from prompts.prompt_templates import TASKS, PROMPT_STRATEGIES, INPUT_MODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def find_available_models(results_dir: str) -> list[str]:
    """Return model aliases that have a results subfolder (excludes 'ensemble')."""
    available = []
    if not os.path.isdir(results_dir):
        return available
    for name in sorted(os.listdir(results_dir)):
        if name in SUPPORTED_MODELS and os.path.isdir(os.path.join(results_dir, name)):
            available.append(name)
    return available


def discover_projects(results_dir: str, models: list[str]) -> list[str]:
    """Discover project names from existing result subfolders."""
    projects = set()
    for model in models:
        model_dir = os.path.join(results_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for name in os.listdir(model_dir):
            if os.path.isdir(os.path.join(model_dir, name)) and name.startswith("project"):
                projects.add(name)
    return sorted(projects)


def result_csv_path(
    results_dir: str,
    model: str,
    project: str,
    input_mode: str,
    task: str,
    strategy: str,
) -> str:
    return os.path.join(
        results_dir, model, project, input_mode,
        f"{model}_{task}_{strategy}.csv"
    )


def load_result_csv(path: str) -> dict[str, dict]:
    """Return {issue_id: row_dict}."""
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["issue_id"]] = row
    return data


def majority_vote(votes: list[int]) -> int:
    valid = [v for v in votes if v in (0, 1)]
    if not valid:
        return -1
    return Counter(valid).most_common(1)[0][0]


# ─────────────────────────────────────────────
# CORE ENSEMBLE BUILDER
# ─────────────────────────────────────────────

def build_ensemble(
    results_dir: str = "results",
    output_dir: str | None = None,
    models: list[str] | None = None,
    projects: list[str] | None = None,
    input_modes: list[str] | None = None,
) -> list[str]:
    """
    For every (project, input_mode, task, strategy) combination,
    read per-model CSVs, apply majority voting, and write ensemble CSV.

    Output: results/ensemble/<project>/<input_mode>/ensemble_<task>_<strategy>.csv
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "ensemble")

    if models is None:
        models = find_available_models(results_dir)
        logger.info("Auto-discovered models: %s", models)

    if len(models) < 2:
        logger.warning(
            "Ensemble needs ≥2 models. Found: %s. Run more models first.", models
        )
        return []

    if projects is None:
        projects = discover_projects(results_dir, models)
        logger.info("Auto-discovered projects: %s", projects)

    if input_modes is None:
        input_modes = INPUT_MODES

    written = []

    for project in projects:
        for input_mode in input_modes:
            for task in TASKS:
                for strategy in PROMPT_STRATEGIES:
                    # Load whichever models have this combo
                    model_data: dict[str, dict[str, dict]] = {}
                    for model in models:
                        path = result_csv_path(
                            results_dir, model, project, input_mode, task, strategy
                        )
                        if os.path.exists(path):
                            model_data[model] = load_result_csv(path)
                        else:
                            logger.debug("Missing: %s", path)

                    if len(model_data) < 2:
                        logger.warning(
                            "Only %d model(s) for %s/%s/%s/%s — skipping",
                            len(model_data), project, input_mode, task, strategy,
                        )
                        continue

                    common_ids = set.intersection(
                        *[set(d.keys()) for d in model_data.values()]
                    )
                    if not common_ids:
                        logger.warning(
                            "No common issues for %s/%s/%s/%s",
                            project, input_mode, task, strategy,
                        )
                        continue

                    ens_dir = os.path.join(output_dir, project, input_mode)
                    os.makedirs(ens_dir, exist_ok=True)
                    out_path = os.path.join(ens_dir, f"ensemble_{task}_{strategy}.csv")

                    fieldnames = [
                        "issue_id", "title", "ground_truth",
                        *[f"pred_{m}" for m in model_data],
                        "n_models_voted",
                        "n_yes_votes",
                        "ensemble_prediction",
                        "project",
                        "input_mode",
                        "task",
                        "strategy",
                    ]

                    first_model = next(iter(model_data))
                    rows_written = 0

                    with open(out_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()

                        for issue_id in model_data[first_model]:
                            if issue_id not in common_ids:
                                continue
                            ref   = model_data[first_model][issue_id]
                            votes = []
                            preds = {}
                            for m, d in model_data.items():
                                p = int(d[issue_id]["prediction"])
                                preds[f"pred_{m}"] = p
                                votes.append(p)

                            writer.writerow({
                                "issue_id":            issue_id,
                                "title":               ref["title"],
                                "ground_truth":        ref["ground_truth"],
                                **preds,
                                "n_models_voted":      len([v for v in votes if v != -1]),
                                "n_yes_votes":         sum(v for v in votes if v == 1),
                                "ensemble_prediction": majority_vote(votes),
                                "project":             project,
                                "input_mode":          input_mode,
                                "task":                task,
                                "strategy":            strategy,
                            })
                            rows_written += 1

                    logger.info(
                        "Ensemble → %s  (%d issues, models: %s)",
                        out_path, rows_written, list(model_data.keys()),
                    )
                    written.append(out_path)

    return written


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build majority-voting ensemble")
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--output_dir",   default=None,
                   help="Ensemble output root (default: results/ensemble/)")
    p.add_argument("--models",       nargs="+", default=None,
                   help="Model aliases to include (default: all with results)")
    p.add_argument("--project",      default=None,
                   help="Restrict to one project stem (default: all)")
    p.add_argument("--input_mode",   choices=INPUT_MODES, default=None,
                   help="'title_only' or 'title_desc' (default: both)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    projects    = [args.project]    if args.project    else None
    input_modes = [args.input_mode] if args.input_mode else None
    paths = build_ensemble(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        models=args.models,
        projects=projects,
        input_modes=input_modes,
    )
    print(f"\n✓ {len(paths)} ensemble file(s) written.")