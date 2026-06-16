"""
scripts/run_pipeline.py
Per-model pipeline — run the FULL workflow for ONE model at a time.

Each step is independently skippable. You are never forced to run all
models together. Evaluation and figures automatically include all models that
have completed results at that point.

Supports multiple project datasets (project_01.csv … project_05.csv),
title_only vs title_desc input modes, resume, and granular filtering.

Usage
-----
  # Full pipeline for llama3.1 (all projects, both input modes)
  python scripts/run_pipeline.py --model llama3.1

  # Resume an interrupted run (skips completed inference combos)
  python scripts/run_pipeline.py --model qwen2.5 --resume

  # One project, title_only only
  python scripts/run_pipeline.py --model llama3.1 --project project_01 --input_mode title_only

  # Run inference only (skip ensemble/evaluate/figures for now)
  python scripts/run_pipeline.py --model mistral --skip_ensemble --skip_evaluate --skip_figures

  # Re-run just evaluation + figures after all models are done
  python scripts/run_pipeline.py --model llama3.1 --skip_inference --skip_ensemble

  # Mock run (no Ollama needed — for testing)
  python scripts/run_pipeline.py --model llama3.1 --mock

  # See available models
  python scripts/run_pipeline.py --list_models

Typical multi-model workflow
-----------------------------
  python scripts/run_pipeline.py --model llama3.1
  python scripts/run_pipeline.py --model qwen2.5
  python scripts/run_pipeline.py --model mistral
  # After all three: evaluation and figures include all models automatically.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.llm_clients import SUPPORTED_MODELS, OLLAMA_MODELS, MODEL_REGISTRY, get_model_description
from prompts.prompt_templates import TASKS, PROMPT_STRATEGIES, INPUT_MODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = str(ROOT / "dataset")


def banner(text: str):
    logger.info("")
    logger.info("━" * 60)
    logger.info("  %s", text)
    logger.info("━" * 60)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run per-model defect detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",           choices=SUPPORTED_MODELS, default=None,
                   help="Model alias to run (required unless --list_models)")
    p.add_argument("--project",         default=None,
                   help="Restrict to one project stem, e.g. 'project_01' (default: all CSVs in dataset/)")
    p.add_argument("--input_mode",      choices=INPUT_MODES, default=None,
                   help="'title_only' or 'title_desc' (default: both)")
    p.add_argument("--task",            choices=TASKS,             default=None,
                   help="Restrict to one task (default: all)")
    p.add_argument("--strategy",        choices=PROMPT_STRATEGIES, default=None,
                   help="Restrict to one strategy (default: all)")

    # Step toggles
    p.add_argument("--skip_inference",  action="store_true")
    p.add_argument("--skip_ensemble",   action="store_true")
    p.add_argument("--skip_evaluate",   action="store_true")
    p.add_argument("--skip_figures",    action="store_true")

    # Inference options
    p.add_argument("--mock",    action="store_true",
                   help="Use MockClient — no Ollama needed")
    p.add_argument("--resume",  action="store_true",
                   help="Skip already-completed inference combos")

    # Paths
    p.add_argument("--dataset_dir",  default=DEFAULT_DATASET_DIR,
                   help="Directory containing project CSV files (project_01.csv, …)")
    p.add_argument("--results_dir",  default=str(ROOT / "results"))
    p.add_argument("--eval_dir",     default=str(ROOT / "evaluation"))
    p.add_argument("--figures_dir",  default=str(ROOT / "figures"))

    p.add_argument("--list_models",  action="store_true",
                   help="Print available models and exit")
    return p.parse_args()


def main():
    args = parse_args()

    # ── List models ─────────────────────────────────────────────
    if args.list_models:
        print("\nAvailable models:")
        print(f"  {'Alias':<12}  {'Backend':<8}  {'Model String':<42}  Description")
        print("  " + "─" * 82)
        for alias, cfg in MODEL_REGISTRY.items():
            tag = " ← recommended" if alias in OLLAMA_MODELS else ""
            print(f"  {alias:<12}  {cfg['backend']:<8}  {cfg['model_string']:<42}  "
                  f"{cfg['description']}{tag}")
        print("\nOllama setup:")
        print("  1. Install  : https://ollama.com")
        print("  2. Start    : ollama serve")
        for alias in OLLAMA_MODELS:
            ms = MODEL_REGISTRY[alias]["model_string"]
            print(f"  3. Pull     : ollama pull {ms}")
        print()
        sys.exit(0)

    if not args.model:
        print("ERROR: --model is required. Use --list_models to see options.\n")
        argparse.ArgumentParser().print_help()
        sys.exit(1)

    model = args.model
    desc  = get_model_description(model)
    backend = MODEL_REGISTRY[model]["backend"]

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  Model   : %-42s║", f"{model}  ({desc})")
    logger.info("║  Backend : %-42s║", backend)
    logger.info("║  Mock    : %-42s║", str(args.mock))
    logger.info("╚══════════════════════════════════════════════════════╝")

    # ── Ollama startup hint ──────────────────────────────────────
    if backend == "ollama" and not args.mock:
        ms = MODEL_REGISTRY[model]["model_string"]
        logger.info("")
        logger.info("Pre-flight check for Ollama model '%s':", ms)
        logger.info("  Make sure Ollama is running : ollama serve")
        logger.info("  Make sure model is pulled   : ollama pull %s", ms)

    # Resolve experiment dimensions (shared across steps)
    from scripts.run_inference import discover_projects, load_dataset, run_combination
    from scripts.reproducibility import dataset_checksum, save_run_config, save_env_snapshot

    all_projects = discover_projects(args.dataset_dir)
    if not all_projects:
        logger.error("No CSV files found in dataset dir '%s'", args.dataset_dir)
        sys.exit(1)

    projects   = [args.project]    if args.project    else all_projects
    modes      = [args.input_mode] if args.input_mode else INPUT_MODES
    tasks      = [args.task]       if args.task       else TASKS
    strategies = [args.strategy]   if args.strategy   else PROMPT_STRATEGIES

    # ── STEP 1: Dataset ─────────────────────────────────────────
    banner("STEP 1: Dataset")
    logger.info("Dataset dir : %s", args.dataset_dir)
    logger.info("Projects    : %s", projects)
    logger.info("Input modes : %s", modes)
    for project in projects:
        path = os.path.join(args.dataset_dir, f"{project}.csv")
        if not os.path.exists(path):
            logger.error("Dataset not found: %s", path)
            sys.exit(1)
        logger.info("  ✓ %s", path)

    # ── STEP 2: Inference ────────────────────────────────────────
    if not args.skip_inference:
        banner(f"STEP 2: Inference  [{model}]")
        model_out = os.path.join(args.results_dir, model)
        os.makedirs(model_out, exist_ok=True)

        dataset_paths = [os.path.join(args.dataset_dir, f"{p}.csv") for p in projects]
        save_run_config(
            output_dir=model_out,
            dataset_paths=dataset_paths,
            models=[model],
            tasks=tasks,
            strategies=strategies,
            input_modes=modes,
            projects=projects,
            model_versions={model: MODEL_REGISTRY[model]["model_string"]},
            extra={"mock_mode": args.mock, "resume": args.resume},
        )
        save_env_snapshot(model_out)

        total = len(projects) * len(modes) * len(tasks) * len(strategies)
        done  = 0
        for project in projects:
            dataset_path = os.path.join(args.dataset_dir, f"{project}.csv")
            issues   = load_dataset(dataset_path)
            checksum = dataset_checksum(dataset_path)

            for input_mode in modes:
                for task in tasks:
                    for strategy in strategies:
                        done += 1
                        logger.info(
                            "── Combo %d/%d: %s | %s | %s | %s ──",
                            done, total, project, input_mode, task, strategy,
                        )
                        run_combination(
                            issues=issues,
                            model_alias=model,
                            project=project,
                            input_mode=input_mode,
                            task=task,
                            strategy=strategy,
                            mock=args.mock,
                            output_dir=args.results_dir,
                            expected_checksum=checksum,
                            dataset_path=dataset_path,
                            resume=args.resume,
                        )
    else:
        logger.info("Skipping inference (--skip_inference)")

    # ── STEP 3: Ensemble ─────────────────────────────────────────
    if not args.skip_ensemble:
        banner("STEP 3: Ensemble  [majority voting over all available models]")
        from scripts.ensemble import build_ensemble, find_available_models
        available = find_available_models(args.results_dir)
        if len(available) >= 2:
            build_ensemble(
                results_dir=args.results_dir,
                output_dir=os.path.join(args.results_dir, "ensemble"),
                models=available,
                projects=projects,
                input_modes=modes,
            )
        else:
            logger.info(
                "Only %d model(s) have results (%s). "
                "Ensemble needs ≥2. Run more models first, or skip with --skip_ensemble.",
                len(available), available,
            )
    else:
        logger.info("Skipping ensemble (--skip_ensemble)")

    # ── STEP 4: Evaluate ─────────────────────────────────────────
    if not args.skip_evaluate:
        banner("STEP 4: Evaluate  [all available models]")
        from evaluation.evaluate_all import evaluate_all, find_available_models as fam
        available = fam(args.results_dir)
        evaluate_all(
            results_dir=args.results_dir,
            output_dir=args.eval_dir,
            models=available if available else None,
            projects=projects,
            input_modes=modes,
        )
    else:
        logger.info("Skipping evaluation (--skip_evaluate)")

    # ── STEP 5: Figures ──────────────────────────────────────────
    if not args.skip_figures:
        combined_json = os.path.join(args.eval_dir, "combined", "full_metrics.json")
        if os.path.exists(combined_json) or any(
            os.path.exists(os.path.join(args.eval_dir, p, m, "full_metrics.json"))
            for p in projects for m in modes
        ):
            banner("STEP 5: Figures  [all evaluated models]")
            from scripts.generate_figures import generate_all_figures
            generate_all_figures(
                eval_dir=args.eval_dir,
                output_dir=args.figures_dir,
                projects=projects,
                input_modes=modes,
            )
        else:
            logger.info("No metrics JSON yet — skipping figures (run evaluate first)")
    else:
        logger.info("Skipping figures (--skip_figures)")

    logger.info("")
    logger.info("✓ Pipeline complete for model: %s", model)
    logger.info("  Results    → %s/%s/<project>/<input_mode>/", args.results_dir, model)
    logger.info("  Ensemble   → %s/ensemble/<project>/<input_mode>/", args.results_dir)
    logger.info("  Evaluation → %s/<project>/<input_mode>/", args.eval_dir)
    logger.info("  Figures    → %s/<project>/<input_mode>/", args.figures_dir)


if __name__ == "__main__":
    main()
