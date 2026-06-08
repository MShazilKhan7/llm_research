"""
scripts/generate_figures.py
Generate publication-quality figures from evaluation/full_metrics.json.

Model-aware: reads whatever models are present in the metrics JSON —
no hardcoded model list. Works whether you have 1 model or all 3.

Figures generated
-----------------
  1. f1_heatmap_<task>.png          – F1 heatmap: models × strategies
  2. metrics_bar_<task>.png         – Grouped bar per model (best strategy)
  3. confusion_matrices.png         – Grid of confusion matrices
  4. ensemble_vs_individual.png     – Ensemble vs individual best F1

Usage
-----
  python scripts/generate_figures.py
  python scripts/generate_figures.py --models llama3.1 qwen2.5
  python scripts/generate_figures.py --metrics_json evaluation/full_metrics.json
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

STRATEGIES = ["zero_shot", "few_shot", "cot", "few_shot_cot"]

# Colour palette — enough for any number of models
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

def _model_color(models: list[str]) -> dict[str, str]:
    colors = {}
    for i, m in enumerate(models):
        colors[m] = _PALETTE[i % len(_PALETTE)]
    colors["ensemble"] = "#2d2d2d"   # always dark for ensemble
    return colors

def _model_label(alias: str) -> str:
    """Short display label: alias + description on second line."""
    if alias == "ensemble":
        return "Ensemble"
    desc = get_model_description(alias)
    return f"{alias}\n({desc})"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

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
    """Return {model: best_row} keyed on highest F1."""
    best: dict[str, dict] = {}
    for row in data:
        m = row["model"]
        if row.get("f1") is None:
            continue
        if m not in best or row["f1"] > best[m]["f1"]:
            best[m] = row
    return best


def _models_in_data(data: list[dict]) -> list[str]:
    """Ordered unique model names found in the metrics data."""
    seen, result = set(), []
    for row in data:
        m = row["model"]
        if m not in seen:
            seen.add(m)
            result.append(m)
    # Put ensemble last
    if "ensemble" in result:
        result.remove("ensemble")
        result.append("ensemble")
    return result


# ─────────────────────────────────────────────
# FIGURE 1 — F1 HEATMAP
# ─────────────────────────────────────────────

def plot_f1_heatmap(task_data: list[dict], task: str, output_dir: str,
                    models: list[str] | None = None):
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

    ax.set_title(f"F1-score Heatmap — {task.capitalize()} Detection",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Prompting Strategy", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f"f1_heatmap_{task}.png"))


# ─────────────────────────────────────────────
# FIGURE 2 — GROUPED METRICS BAR
# ─────────────────────────────────────────────

def plot_metrics_bar(task_data: list[dict], task: str, output_dir: str,
                     models: list[str] | None = None):
    all_models = models or _models_in_data(task_data)
    best = _best_per_model(task_data)
    present = [m for m in all_models if m in best]
    if not present:
        return

    metrics     = ["accuracy", "precision", "recall", "f1"]
    m_colors    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    x           = np.arange(len(present))
    width       = 0.18
    offsets     = [-1.5, -0.5, 0.5, 1.5]

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
    ax.set_title(f"Best Performance per Model — {task.capitalize()} Detection",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f"metrics_bar_{task}.png"))


# ─────────────────────────────────────────────
# FIGURE 3 — CONFUSION MATRICES GRID
# ─────────────────────────────────────────────

def plot_confusion_matrices(all_metrics: dict, output_dir: str,
                             models: list[str] | None = None):
    tasks = list(all_metrics.keys())

    # Find best overall strategy
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

    all_models = models or _models_in_data(list(all_metrics.values())[0])
    n_rows, n_cols = len(tasks), len(all_models)

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

    fig.suptitle(f"Confusion Matrices  [{best_strat}]", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "confusion_matrices.png"))


# ─────────────────────────────────────────────
# FIGURE 4 — ENSEMBLE VS INDIVIDUAL
# ─────────────────────────────────────────────

def plot_ensemble_vs_individual(all_metrics: dict, output_dir: str,
                                 models: list[str] | None = None):
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

        f1_vals = [best[m]["f1"] for m in present]
        bar_colors = [colors.get(m, "#888") for m in present]
        bars = ax.bar(range(len(present)), f1_vals,
                      color=bar_colors, alpha=0.85, width=0.5)

        for bar, v in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([_model_label(m) for m in present], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"{task.capitalize()} Detection\n(Best F1 per model)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Model", fontsize=11)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)

    axes[0][0].set_ylabel("F1-score", fontsize=11)
    fig.suptitle("Ensemble vs Individual Models  (Best F1)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "ensemble_vs_individual.png"))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def generate_all_figures(
    metrics_json: str = "evaluation/full_metrics.json",
    output_dir: str = "figures",
    models: list[str] | None = None,
):
    if not HAS_MPL:
        logger.error("matplotlib / numpy not installed: pip install matplotlib numpy")
        sys.exit(1)
    if not os.path.exists(metrics_json):
        logger.error("Metrics file not found: %s — run evaluate_all.py first", metrics_json)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    all_metrics = load_metrics(metrics_json)

    # If models not specified, derive from JSON
    if models is None:
        first_task = next(iter(all_metrics.values()))
        models = _models_in_data(first_task)
    logger.info("Generating figures for models: %s", models)

    for task, task_data in all_metrics.items():
        plot_f1_heatmap(task_data, task, output_dir, models)
        plot_metrics_bar(task_data, task, output_dir, models)

    plot_confusion_matrices(all_metrics, output_dir, models)
    plot_ensemble_vs_individual(all_metrics, output_dir, models)

    logger.info("All figures saved to '%s/'", output_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate figures from evaluation metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_figures.py
  python scripts/generate_figures.py --models llama3.1 qwen2.5
        """,
    )
    p.add_argument("--metrics_json", default="evaluation/full_metrics.json")
    p.add_argument("--output_dir",   default="figures")
    p.add_argument("--models", nargs="+", default=None,
                   help="Restrict figures to these model aliases")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all_figures(args.metrics_json, args.output_dir, args.models)
