"""
scripts/ensemble.py
Majority-voting ensemble over individual LLM predictions.

Now model-aware: reads from results/<model_alias>/ subfolders.
Can ensemble any subset of models that have completed inference.

Usage
-----
  # Ensemble all models that have results
  python scripts/ensemble.py

  # Ensemble specific models only
  python scripts/ensemble.py --models llama3.1 qwen2.5 mistral

  # Custom results dir
  python scripts/ensemble.py --results_dir results --output_dir results/ensemble
"""

from __future__ import annotations
import argparse
import csv
import logging
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.llm_clients import SUPPORTED_MODELS, OLLAMA_MODELS
from prompts.prompt_templates import TASKS, PROMPT_STRATEGIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def find_available_models(results_dir: str) -> list[str]:
    """Return model aliases that have a results subfolder."""
    available = []
    for name in os.listdir(results_dir):
        if os.path.isdir(os.path.join(results_dir, name)) and name in SUPPORTED_MODELS:
            available.append(name)
    return sorted(available)


def result_csv_path(results_dir: str, model: str, task: str, strategy: str) -> str:
    return os.path.join(results_dir, model, f"{model}_{task}_{strategy}.csv")


def load_result_csv(path: str) -> dict[str, dict]:
    """Return {issue_id: row_dict}."""
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["issue_id"]] = row
    return data


def majority_vote(votes: list[int]) -> int:
    """
    Majority vote over a list of 0/1 predictions.
    -1 (parse failures) are excluded.
    Returns 0 on tie or if all votes are invalid.
    """
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
) -> list[str]:
    """
    For every (task, strategy) pair, read per-model CSVs,
    apply majority voting, and write ensemble CSV.

    Ensemble CSVs go to results/ensemble/<task>_<strategy>.csv
    Returns list of written file paths.
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "ensemble")
    os.makedirs(output_dir, exist_ok=True)

    # Auto-discover models if not specified
    if models is None:
        models = find_available_models(results_dir)
        logger.info("Auto-discovered models with results: %s", models)

    if len(models) < 2:
        logger.warning(
            "Ensemble needs at least 2 models. Found: %s. "
            "Run inference for more models first.", models
        )
        return []

    written = []

    for task in TASKS:
        for strategy in PROMPT_STRATEGIES:
            # Load whichever models have this combo
            model_data: dict[str, dict[str, dict]] = {}
            for model in models:
                path = result_csv_path(results_dir, model, task, strategy)
                if os.path.exists(path):
                    model_data[model] = load_result_csv(path)
                else:
                    logger.debug("Missing: %s — skipping for this combo", path)

            if len(model_data) < 2:
                logger.warning(
                    "Only %d model(s) available for %s/%s — skipping ensemble",
                    len(model_data), task, strategy,
                )
                continue

            # Issue IDs present in ALL loaded models
            common_ids = set.intersection(*[set(d.keys()) for d in model_data.values()])
            if not common_ids:
                logger.warning("No common issues for %s/%s", task, strategy)
                continue

            out_path = os.path.join(output_dir, f"ensemble_{task}_{strategy}.csv")
            fieldnames = [
                "issue_id", "title", "ground_truth",
                *[f"pred_{m}" for m in model_data],
                "n_models_voted",
                "n_yes_votes",
                "ensemble_prediction",
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
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output_dir",  default=None,
                   help="Where to write ensemble CSVs (default: results/ensemble/)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model aliases to include (default: all with results)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = build_ensemble(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        models=args.models,
    )
    print(f"\n✓ {len(paths)} ensemble file(s) written.")
