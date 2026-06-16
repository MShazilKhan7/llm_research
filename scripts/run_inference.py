"""
scripts/run_inference.py
Run inference for ONE model across all (or selected) projects, tasks,
input modes, and strategies.

New dimensions vs. original
----------------------------
  --project    : restrict to one project (default: all *.csv in dataset/)
  --input_mode : "title_only" | "title_desc" | both (default: both)

Result folder layout
--------------------
  results/<model>/<project>/<input_mode>/<model>_<task>_<strategy>.csv

Usage
-----
  # Full run for llama3.1 across all projects and input modes
  python scripts/run_inference.py --model llama3.1

  # Only one project, title_only mode
  python scripts/run_inference.py --model llama3.1 --project project_01 --input_mode title_only

  # Resume an interrupted run
  python scripts/run_inference.py --model llama3.1 --resume

  # Mock run (no Ollama needed)
  python scripts/run_inference.py --model llama3.1 --mock

  # List available models
  python scripts/run_inference.py --list_models
"""

from __future__ import annotations
import argparse
import csv
import glob
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts.prompt_templates import PROMPT_REGISTRY, PROMPT_STRATEGIES, TASKS, INPUT_MODES
from scripts.llm_clients import (
    get_client, SUPPORTED_MODELS, MODEL_REGISTRY, get_model_description,
)
from scripts.reproducibility import (
    save_run_config, save_env_snapshot, assert_dataset_unchanged,
    make_result_row, EXTENDED_FIELDNAMES, dataset_checksum, GROUND_TRUTH_COL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = str(ROOT / "dataset")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def discover_projects(dataset_dir: str) -> list[str]:
    """Return sorted list of project stems (e.g. ['project_01','project_02'])."""
    files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    return sorted(Path(f).stem for f in files)


def load_dataset(path: str) -> list[dict]:
    issues = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            issues.append(row)
    logger.info("Loaded %d issues from '%s'", len(issues), path)
    return issues


def result_path(
    output_dir: str,
    model_alias: str,
    project: str,
    input_mode: str,
    task: str,
    strategy: str,
) -> str:
    """
    results/<model>/<project>/<input_mode>/<model>_<task>_<strategy>.csv
    """
    folder = os.path.join(output_dir, model_alias, project, input_mode)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{model_alias}_{task}_{strategy}.csv")


def is_complete(path: str, expected_rows: int) -> bool:
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
    project: str,
    input_mode: str,
    task: str,
    strategy: str,
    mock: bool = False,
    output_dir: str = "results",
    expected_checksum: str | None = None,
    dataset_path: str = "",
    resume: bool = False,
) -> str:
    out_path = result_path(output_dir, model_alias, project, input_mode, task, strategy)

    if resume and is_complete(out_path, len(issues)):
        logger.info("  [SKIP — already done] %s", out_path)
        return out_path

    if expected_checksum and dataset_path:
        assert_dataset_unchanged(dataset_path, expected_checksum)

    client       = get_client(model_alias, mock=mock)
    prompt_fn    = PROMPT_REGISTRY[task][strategy][input_mode]
    label_col    = GROUND_TRUTH_COL[(task, input_mode)]
    model_string = MODEL_REGISTRY[model_alias]["model_string"]

    logger.info(
        "Running: model=%-12s  project=%-12s  mode=%-12s  task=%-16s  strategy=%s",
        model_alias, project, input_mode, task, strategy,
    )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXTENDED_FIELDNAMES)
        writer.writeheader()

        for i, issue in enumerate(issues):
            prompt = prompt_fn(issue["Title"], issue.get("Description", ""))
            label, raw = client.predict(prompt)

            writer.writerow(make_result_row(
                issue=issue,
                label_col=label_col,
                prediction=label,
                raw_response=raw,
                prompt_text=prompt,
                model_version=model_string,
                project=project,
                input_mode=input_mode,
                task=task,
                strategy=strategy,
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
        description="Run LLM inference for a single model across projects and input modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_inference.py --model llama3.1
  python scripts/run_inference.py --model llama3.1 --project project_01
  python scripts/run_inference.py --model llama3.1 --input_mode title_only
  python scripts/run_inference.py --model llama3.1 --project project_02 --input_mode title_desc --task ambiguity
  python scripts/run_inference.py --model llama3.1 --resume
  python scripts/run_inference.py --model llama3.1 --mock
  python scripts/run_inference.py --list_models
        """,
    )
    p.add_argument("--model",       choices=SUPPORTED_MODELS, default=None,
                   help="Model alias (required unless --list_models)")
    p.add_argument("--project",     default=None,
                   help="Restrict to one project stem, e.g. 'project_01' (default: all)")
    p.add_argument("--input_mode",  choices=INPUT_MODES, default=None,
                   help="'title_only' or 'title_desc' (default: both)")
    p.add_argument("--task",        choices=TASKS,          default=None,
                   help="Restrict to one task (default: all)")
    p.add_argument("--strategy",    choices=PROMPT_STRATEGIES, default=None,
                   help="Restrict to one strategy (default: all)")
    p.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR,
                   help="Directory containing project CSV files")
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--mock",        action="store_true")
    p.add_argument("--resume",      action="store_true",
                   help="Skip combinations that already have complete result CSVs")
    p.add_argument("--list_models", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

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

    # Resolve dimension lists
    all_projects   = discover_projects(args.dataset_dir)
    if not all_projects:
        logger.error("No CSV files found in dataset dir '%s'", args.dataset_dir)
        sys.exit(1)

    projects   = [args.project]    if args.project    else all_projects
    modes      = [args.input_mode] if args.input_mode else INPUT_MODES
    tasks      = [args.task]       if args.task       else TASKS
    strategies = [args.strategy]   if args.strategy   else PROMPT_STRATEGIES

    logger.info("Model      : %s  (%s)", args.model, get_model_description(args.model))
    logger.info("Projects   : %s", projects)
    logger.info("Input modes: %s", modes)
    logger.info("Tasks      : %s", tasks)
    logger.info("Strategies : %s", strategies)
    total = len(projects) * len(modes) * len(tasks) * len(strategies)
    logger.info("Combos     : %d", total)

    model_out = os.path.join(args.output_dir, args.model)
    os.makedirs(model_out, exist_ok=True)

    dataset_paths = [
        os.path.join(args.dataset_dir, f"{p}.csv") for p in projects
    ]
    save_run_config(
        output_dir=model_out,
        dataset_paths=dataset_paths,
        models=[args.model],
        tasks=tasks,
        strategies=strategies,
        input_modes=modes,
        projects=projects,
        model_versions={args.model: MODEL_REGISTRY[args.model]["model_string"]},
        extra={"mock_mode": args.mock, "resume": args.resume},
    )
    save_env_snapshot(model_out)

    done = 0
    for project in projects:
        dataset_path = os.path.join(args.dataset_dir, f"{project}.csv")
        if not os.path.exists(dataset_path):
            logger.warning("Dataset not found: %s — skipping", dataset_path)
            continue
        issues   = load_dataset(dataset_path)
        checksum = dataset_checksum(dataset_path)

        for input_mode in modes:
            for task in tasks:
                for strategy in strategies:
                    done += 1
                    logger.info("── Combo %d/%d: %s | %s | %s | %s ──",
                                done, total, project, input_mode, task, strategy)
                    run_combination(
                        issues=issues,
                        model_alias=args.model,
                        project=project,
                        input_mode=input_mode,
                        task=task,
                        strategy=strategy,
                        mock=args.mock,
                        output_dir=args.output_dir,
                        expected_checksum=checksum,
                        dataset_path=dataset_path,
                        resume=args.resume,
                    )

    logger.info("✓ Done — results in '%s/%s/'", args.output_dir, args.model)


if __name__ == "__main__":
    main()