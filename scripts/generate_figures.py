"""
scripts/generate_figures.py
Generate publication-quality figures from evaluation metrics.

Figure output layout
--------------------
  figures/<project>/<input_mode>/
      f1_heatmap_<task>.png
      metrics_bar_<task>.png
      confusion_matrices.png
      ensemble_vs_individual.png

  figures/combined/             ← aggregate across all projects
      (same set of figures)

Usage
-----
  python scripts/generate_figures.py
  python scripts/generate_figures.py --project project_01 --input_mode title_only
  python scripts/generate_figures.py --models llama3.1 qwen2.5
  python scripts/generate_figures.py --metrics_json evaluation/combined/full_metrics.json --output_dir figures/combined
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from scripts.llm_clients import MODEL_REGISTRY, get_model_description
from prompts.prompt_templates import INPUT_MODES, TASKS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

STRATEGIES = ["zero_shot", "few_shot", "cot", "few_shot_cot"]
_PALETTE   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _model_color(models: list[str]) -> dict[str, str]:
    colors = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}
    colors["ensemble"] = "#2d2d2d"
    return colors


def _model_label(alias: str) -> str:
    if alias == "ensemble":
        return "Ensemble"
    desc = get_model_description(alias)
    return f"{alias}\n({desc})"


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def _get_val(data: list[dict], model: str, strategy: str, metric: str = "f1"):
    for row in data:
        if row["model"] == model and row["strategy"] == strategy:
            return row.get(metric)
    return None


def _best_per_model(data: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for row in data:
        m = row["model"]
        if row.get("f1") is None:
            continue
        if m not in best or row["f1"] > best[m]["f1"]:
            best[m] = row
    return best


def _models_in_data(data: list[dict]) -> list[str]:
    seen, result = set(), []
    for row in data:
        m = row["model"]
        if m not in seen:
            seen.add(m)
            result.append(m)
    if "ensemble" in result:
        result.remove("ensemble")
        result.append("ensemble")
    return result


# ─────────────────────────────────────────────
# FIGURE 1 — F1 HEATMAP
# ─────────────────────────────────────────────

def plot_f1_heatmap(task_data: list[dict], task: str, output_dir: str,
                    models: list[str] | None = None, subtitle: str = ""):
    all_models = models or _models_in_data(task_data)
    matrix = np.zeros((len(all_models), len(STRATEGIES)))

    for i, model in enumerate(all_models):
        for j, strat in enumerate(STRATEGIES):
            v = _get_val(task_data, model, strat, "f1")
            matrix[i, j] = v if v is not None else 0.0

    fig, ax = plt.subplots(figsize=(10, max(4, len(all_models) * 1.2)))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="F1-score")

    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels([s.replace("_", "\n") for s in STRATEGIES], fontsize=11)
    ax.set_yticks(range(len(all_models)))
    ax.set_yticklabels([_model_label(m) for m in all_models], fontsize=10)

    for i in range(len(all_models)):
        for j in range(len(STRATEGIES)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if v > 0.65 else "black")

    title = f"F1-score Heatmap — {task.capitalize()} Detection"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Prompting Strategy", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f"f1_heatmap_{task}.png"))


# ─────────────────────────────────────────────
# FIGURE 2 — GROUPED METRICS BAR
# ─────────────────────────────────────────────

def plot_metrics_bar(task_data: list[dict], task: str, output_dir: str,
                     models: list[str] | None = None, subtitle: str = ""):
    all_models = models or _models_in_data(task_data)
    best = _best_per_model(task_data)
    present = [m for m in all_models if m in best]
    if not present:
        return

    metrics  = ["accuracy", "precision", "recall", "f1"]
    m_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    x        = np.arange(len(present))
    width    = 0.18
    offsets  = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(max(10, len(present) * 3), 5))

    for metric, color, offset in zip(metrics, m_colors, offsets):
        vals = [best[m].get(metric) or 0 for m in present]
        bars = ax.bar(x + offset * width, vals, width, label=metric.capitalize(),
                      color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    labels = [f"{_model_label(m)}\n[{best[m]['strategy']}]" for m in present]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    title = f"Best Performance per Model — {task.capitalize()} Detection"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f"metrics_bar_{task}.png"))


# ─────────────────────────────────────────────
# FIGURE 3 — CONFUSION MATRICES GRID
# ─────────────────────────────────────────────

def plot_confusion_matrices(all_metrics: dict, output_dir: str,
                             models: list[str] | None = None, subtitle: str = ""):
    tasks = list(all_metrics.keys())

    strategy_f1: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    for task_data in all_metrics.values():
        for row in task_data:
            if row["strategy"] in strategy_f1 and row.get("f1"):
                strategy_f1[row["strategy"]].append(row["f1"])
    best_strat = max(
        strategy_f1,
        key=lambda s: (sum(strategy_f1[s]) / len(strategy_f1[s])) if strategy_f1[s] else 0
    )
    logger.info("Using strategy '%s' for confusion matrices", best_strat)

    first_task_data = list(all_metrics.values())[0]
    all_models = models or _models_in_data(first_task_data)
    n_rows, n_cols = len(tasks), len(all_models)

    # If there are no models to plot (n_cols == 0) or no tasks, skip gracefully
    if n_cols <= 0 or n_rows <= 0:
        logger.warning(
            "No models/tasks available for confusion matrices (n_rows=%d, n_cols=%d). Skipping.",
            n_rows,
            n_cols,
        )
        return

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 4 * n_rows),
                              squeeze=False)

    for ri, task in enumerate(tasks):
        task_data = all_metrics[task]
        for ci, model in enumerate(all_models):
            ax = axes[ri][ci]
            row = next((r for r in task_data
                        if r["model"] == model and r["strategy"] == best_strat), None)
            if row is None:
                ax.axis("off")
                continue
            cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: No", "Pred: Yes"], fontsize=9)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["True: No", "True: Yes"], fontsize=9)
            for ii in range(2):
                for jj in range(2):
                    ax.text(jj, ii, str(cm[ii, jj]), ha="center", va="center",
                            fontsize=13, fontweight="bold",
                            color="white" if cm[ii, jj] > cm.max() / 2 else "black")
            ax.set_title(f"{_model_label(model)} | {task[:5]}", fontsize=9, fontweight="bold")

    suptitle = f"Confusion Matrices  [{best_strat}]"
    if subtitle:
        suptitle += f"  —  {subtitle}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "confusion_matrices.png"))


# ─────────────────────────────────────────────
# FIGURE 4 — ENSEMBLE VS INDIVIDUAL
# ─────────────────────────────────────────────

def plot_ensemble_vs_individual(all_metrics: dict, output_dir: str,
                                 models: list[str] | None = None, subtitle: str = ""):
    tasks = list(all_metrics.keys())
    fig, axes = plt.subplots(1, len(tasks),
                              figsize=(7 * len(tasks), 5), sharey=True,
                              squeeze=False)

    for ax, task in zip(axes[0], tasks):
        task_data  = all_metrics[task]
        all_models = models or _models_in_data(task_data)
        colors     = _model_color(all_models)
        best       = _best_per_model(task_data)
        present    = [m for m in all_models if m in best]

        f1_vals    = [best[m]["f1"] for m in present]
        bar_colors = [colors.get(m, "#888") for m in present]
        bars = ax.bar(range(len(present)), f1_vals,
                      color=bar_colors, alpha=0.85, width=0.5)

        for bar, v in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([_model_label(m) for m in present], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"{task.capitalize()} Detection\n(Best F1 per model)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Model", fontsize=11)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)

    axes[0][0].set_ylabel("F1-score", fontsize=11)
    suptitle = "Ensemble vs Individual Models  (Best F1)"
    if subtitle:
        suptitle += f"  —  {subtitle}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "ensemble_vs_individual.png"))


# ─────────────────────────────────────────────
# FIGURE 5 — TITLE_ONLY vs TITLE_DESC COMPARISON
# ─────────────────────────────────────────────

def plot_input_mode_comparison(
    combined_metrics_by_mode: dict[str, dict],   # {input_mode: {task: [rows]}}
    output_dir: str,
    models: list[str] | None = None,
    subtitle: str = "",
):
    """
    Bar chart comparing F1 between title_only and title_desc for each model × task.
    combined_metrics_by_mode = {"title_only": {task: [rows]}, "title_desc": {task: [rows]}}
    """
    if not HAS_MPL:
        return
    if len(combined_metrics_by_mode) < 2:
        return

    tasks = list(next(iter(combined_metrics_by_mode.values())).keys())
    modes = list(combined_metrics_by_mode.keys())

    mode_colors = {"title_only": "#4C72B0", "title_desc": "#DD8452"}

    for task in tasks:
        # Gather best F1 per (model, mode)
        model_set: set[str] = set()
        for mode_data in combined_metrics_by_mode.values():
            for row in mode_data.get(task, []):
                model_set.add(row["model"])
        all_models_sorted = models or sorted(model_set)

        x      = np.arange(len(all_models_sorted))
        width  = 0.35
        fig, ax = plt.subplots(figsize=(max(10, len(all_models_sorted) * 3), 5))

        for i, mode in enumerate(modes):
            task_data = combined_metrics_by_mode.get(mode, {}).get(task, [])
            best = _best_per_model(task_data)
            vals = [best.get(m, {}).get("f1", 0) for m in all_models_sorted]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=mode.replace("_", " "),
                          color=mode_colors.get(mode, "#888"), alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([_model_label(m) for m in all_models_sorted], fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("F1-score (best strategy)", fontsize=11)
        title = f"Title-only vs Title+Description — {task.capitalize()} Detection"
        if subtitle:
            title += f"\n{subtitle}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=11)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout()
        _save(fig, os.path.join(output_dir, f"input_mode_comparison_{task}.png"))


# ─────────────────────────────────────────────
# GENERATE ALL FIGURES FOR ONE SLICE
# ─────────────────────────────────────────────

def generate_figures_for_slice(
    metrics_json: str,
    output_dir: str,
    models: list[str] | None = None,
    subtitle: str = "",
):
    if not os.path.exists(metrics_json):
        logger.warning("Metrics file not found: %s — skipping", metrics_json)
        return
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = load_metrics(metrics_json)

    if models is None:
        first_task = next(iter(all_metrics.values()))
        models = _models_in_data(first_task)

    for task, task_data in all_metrics.items():
        plot_f1_heatmap(task_data, task, output_dir, models, subtitle)
        plot_metrics_bar(task_data, task, output_dir, models, subtitle)

    plot_confusion_matrices(all_metrics, output_dir, models, subtitle)
    plot_ensemble_vs_individual(all_metrics, output_dir, models, subtitle)
    logger.info("Figures saved to '%s/'", output_dir)


# ─────────────────────────────────────────────
# GENERATE ALL (per-slice + combined + comparison)
# ─────────────────────────────────────────────

def generate_all_figures(
    eval_dir: str = "evaluation",
    output_dir: str = "figures",
    models: list[str] | None = None,
    projects: list[str] | None = None,
    input_modes: list[str] | None = None,
):
    if not HAS_MPL:
        logger.error("matplotlib / numpy not installed: pip install matplotlib numpy")
        sys.exit(1)

    if input_modes is None:
        input_modes = INPUT_MODES

    # Auto-discover projects from eval_dir if not specified
    if projects is None:
        projects = []
        if os.path.isdir(eval_dir):
            for name in sorted(os.listdir(eval_dir)):
                if name.startswith("project") and os.path.isdir(os.path.join(eval_dir, name)):
                    projects.append(name)

    # Per-slice figures
    for project in projects:
        for input_mode in input_modes:
            metrics_json = os.path.join(eval_dir, project, input_mode, "full_metrics.json")
            fig_dir      = os.path.join(output_dir, project, input_mode)
            subtitle     = f"{project} / {input_mode.replace('_', ' ')}"
            generate_figures_for_slice(metrics_json, fig_dir, models, subtitle)

    # Combined figures
    combined_json = os.path.join(eval_dir, "combined", "full_metrics.json")
    generate_figures_for_slice(
        combined_json,
        os.path.join(output_dir, "combined"),
        models,
        subtitle="All Projects Combined",
    )

    # Title-only vs title+desc comparison (using combined metrics)
    if os.path.exists(combined_json) and HAS_MPL:
        combined_data = load_metrics(combined_json)

        # Split rows by input_mode
        mode_split: dict[str, dict] = {m: {t: [] for t in TASKS} for m in input_modes}
        for task, rows in combined_data.items():
            for row in rows:
                mode = row.get("input_mode", "")
                if mode in mode_split:
                    mode_split[mode][task].append(row)

        comparison_dir = os.path.join(output_dir, "combined")
        plot_input_mode_comparison(mode_split, comparison_dir, models,
                                   subtitle="All Projects Combined")

    logger.info("\nAll figures written under '%s/'", output_dir)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate figures from evaluation metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_figures.py
  python scripts/generate_figures.py --project project_01 --input_mode title_only
  python scripts/generate_figures.py --models llama3.1 qwen2.5
  # Single slice from a specific metrics JSON:
  python scripts/generate_figures.py --metrics_json evaluation/project_01/title_only/full_metrics.json --output_dir figures/project_01/title_only
        """,
    )
    p.add_argument("--eval_dir",     default="evaluation",
                   help="Root evaluation directory (default: evaluation)")
    p.add_argument("--output_dir",   default="figures")
    p.add_argument("--models",       nargs="+", default=None)
    p.add_argument("--project",      default=None, help="Restrict to one project")
    p.add_argument("--input_mode",   choices=INPUT_MODES, default=None)
    # Legacy / direct path override
    p.add_argument("--metrics_json", default=None,
                   help="Point directly to a full_metrics.json for a single-slice run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.metrics_json:
        # Single-slice mode
        generate_figures_for_slice(
            metrics_json=args.metrics_json,
            output_dir=args.output_dir,
            models=args.models,
        )
    else:
        projects    = [args.project]    if args.project    else None
        input_modes = [args.input_mode] if args.input_mode else None
        generate_all_figures(
            eval_dir=args.eval_dir,
            output_dir=args.output_dir,
            models=args.models,
            projects=projects,
            input_modes=input_modes,
        )