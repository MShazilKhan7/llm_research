"""
scripts/reproducibility.py
Helpers for reproducible experiments.

Adds two new dimensions vs. the original:
  - project     : which project CSV the issues come from  (e.g. "project_01")
  - input_mode  : "title_only" or "title_desc"

Ground-truth column mapping (from the real project CSVs)
---------------------------------------------------------
  ambiguity / title_only   →  Final_amb_T
  ambiguity / title_desc   →  Final_amb_TD
  incompleteness / title_only  →  Final_inc_T
  incompleteness / title_desc  →  Final_inc_TD
"""

from __future__ import annotations
import hashlib
import json
import os
import subprocess
import datetime

# ─────────────────────────────────────────────
# GROUND-TRUTH COLUMN MAP
# ─────────────────────────────────────────────

GROUND_TRUTH_COL: dict[tuple[str, str], str] = {
    ("ambiguity",      "title_only"): "Final_amb_T",
    ("ambiguity",      "title_desc"): "Final_amb_TD",
    ("incompleteness", "title_only"): "Final_inc_T",
    ("incompleteness", "title_desc"): "Final_inc_TD",
}

# ─────────────────────────────────────────────
# RESULT CSV FIELDNAMES
# ─────────────────────────────────────────────

EXTENDED_FIELDNAMES = [
    "issue_id",
    "title",
    "ground_truth",
    "prediction",
    "raw_response",
    "prompt_hash",
    "model_version",
    "temperature",
    "timestamp_utc",
    # new
    "project",
    "input_mode",
    "task",
    "strategy",
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def dataset_checksum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_dataset_unchanged(path: str, expected: str):
    actual = dataset_checksum(path)
    if actual != expected:
        raise RuntimeError(
            f"Dataset checksum mismatch for '{path}'!\n"
            f"  Expected : {expected}\n"
            f"  Got      : {actual}\n"
            "The ground-truth file changed since inference started. "
            "Restore the original file or remove the model results folder and re-run."
        )


def make_result_row(
    issue: dict,
    label_col: str,
    prediction: int,
    raw_response: str,
    prompt_text: str,
    model_version: str,
    project: str,
    input_mode: str,
    task: str,
    strategy: str,
) -> dict:
    prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
    return {
        "issue_id":      issue["Issue_ID"],
        "title":         issue["Title"],
        "ground_truth":  issue[label_col],
        "prediction":    prediction,
        "raw_response":  raw_response,
        "prompt_hash":   prompt_hash,
        "model_version": model_version,
        "temperature":   0,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "project":       project,
        "input_mode":    input_mode,
        "task":          task,
        "strategy":      strategy,
    }


def save_run_config(
    output_dir: str,
    dataset_paths: list[str],
    models: list[str],
    tasks: list[str],
    strategies: list[str],
    input_modes: list[str],
    projects: list[str],
    model_versions: dict[str, str],
    extra: dict | None = None,
):
    config = {
        "timestamp_utc":  datetime.datetime.utcnow().isoformat(),
        "dataset_paths":  dataset_paths,
        "models":         models,
        "tasks":          tasks,
        "strategies":     strategies,
        "input_modes":    input_modes,
        "projects":       projects,
        "model_versions": model_versions,
        **(extra or {}),
    }
    path = os.path.join(output_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def save_env_snapshot(output_dir: str):
    try:
        out = subprocess.check_output(
            ["pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        out = "# could not capture pip freeze\n"
    path = os.path.join(output_dir, "requirements_snapshot.txt")
    with open(path, "w") as f:
        f.write(out)