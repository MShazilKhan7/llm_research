"""
scripts/reproducibility.py
Everything needed to make experiments fully reproducible and auditable.

Responsibilities
----------------
1. save_run_config()   – write a JSON record of every setting used in a run
2. dataset_checksum()  – MD5 of the ground truth file (detects silent changes)
3. prompt_hash()       – SHA256 of a prompt string (verify prompts didn't drift)
4. save_env_snapshot() – pip freeze → requirements_snapshot.txt
5. load_run_config()   – reload a past config for comparison or re-run
6. assert_dataset_unchanged() – hard-fail if the dataset was modified since last run

All artefacts are written to outputs/run_<timestamp>/ by default so every
experiment has its own folder and nothing overwrites a previous run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# CHECKSUMS
# ─────────────────────────────────────────────────────────────────

def dataset_checksum(path: str, algorithm: str = "md5") -> str:
    """
    Return hex digest of the dataset file.
    Use MD5 for speed (this is integrity checking, not security).
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_hash(prompt_text: str) -> str:
    """SHA-256 of a prompt string — store alongside predictions to verify prompts."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


def assert_dataset_unchanged(path: str, expected_checksum: str, algorithm: str = "md5"):
    """
    Raise RuntimeError if the dataset file differs from the expected checksum.
    Call this at the start of ensemble / evaluate steps to guard against
    accidentally swapping in a different ground truth file.
    """
    actual = dataset_checksum(path, algorithm)
    if actual != expected_checksum:
        raise RuntimeError(
            f"Dataset integrity check FAILED for '{path}'.\n"
            f"  Expected checksum : {expected_checksum}\n"
            f"  Actual checksum   : {actual}\n"
            "The ground truth file has changed since inference was run. "
            "Re-run inference or restore the original file."
        )
    logger.info("Dataset integrity OK  (%s: %s)", algorithm, actual)


# ─────────────────────────────────────────────────────────────────
# ENVIRONMENT SNAPSHOT
# ─────────────────────────────────────────────────────────────────

def get_env_info() -> dict:
    """Collect Python version, platform, and installed package versions."""
    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform":       platform.platform(),
        "executable":     sys.executable,
    }

    # Key library versions (best-effort)
    for lib in ["openai", "google.generativeai", "groq", "pandas", "numpy", "matplotlib"]:
        try:
            mod = __import__(lib.replace(".", "_") if "." in lib else lib)
            ver = getattr(mod, "__version__", "unknown")
        except ImportError:
            ver = "not installed"
        info[f"pkg_{lib.split('.')[0]}"] = ver

    return info


def save_env_snapshot(output_dir: str) -> str:
    """
    Run `pip freeze` and save to <output_dir>/requirements_snapshot.txt.
    Returns the path.
    """
    out_path = os.path.join(output_dir, "requirements_snapshot.txt")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30,
        )
        with open(out_path, "w") as f:
            f.write(result.stdout)
        logger.info("Environment snapshot → %s", out_path)
    except Exception as exc:
        logger.warning("Could not save pip freeze: %s", exc)
        out_path = None
    return out_path


# ─────────────────────────────────────────────────────────────────
# RUN CONFIG
# ─────────────────────────────────────────────────────────────────

def save_run_config(
    output_dir: str,
    dataset_path: str,
    models: list[str],
    tasks: list[str],
    strategies: list[str],
    model_versions: dict[str, str],
    extra: dict | None = None,
) -> str:
    """
    Write a JSON file capturing everything needed to reproduce this run.

    Parameters
    ----------
    output_dir      : where to write run_config.json
    dataset_path    : path to ground truth CSV
    models          : list of model names used
    tasks           : list of tasks run
    strategies      : list of strategies run
    model_versions  : {'gpt': 'gpt-4o-mini', 'gemini': 'gemini-1.5-flash', ...}
    extra           : any additional key-value pairs to store

    Returns path to saved config.
    """
    os.makedirs(output_dir, exist_ok=True)

    config: dict[str, Any] = {
        "run_timestamp":    datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path":     dataset_path,
            "checksum": dataset_checksum(dataset_path),
            "algorithm": "md5",
            "n_issues": _count_rows(dataset_path),
        },
        "models":          models,
        "tasks":           tasks,
        "strategies":      strategies,
        "model_versions":  model_versions,
        "llm_temperature": 0,        # all clients use temperature=0
        "environment":     get_env_info(),
        **(extra or {}),
    }

    out_path = os.path.join(output_dir, "run_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Run config saved → %s", out_path)
    return out_path


def load_run_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────
# PER-PREDICTION METADATA  (appended to result CSVs)
# ─────────────────────────────────────────────────────────────────

# Extended fieldnames for result CSVs — include these alongside the
# original fields to make each row fully self-documenting.
EXTENDED_FIELDNAMES = [
    "issue_id",
    "title",
    "ground_truth",
    "prediction",
    "raw_response",        # FULL response — never truncated
    "prompt_hash",         # first 16 chars of SHA-256 of the prompt
    "model_version",       # exact model string used (e.g. gpt-4o-mini)
    "temperature",         # always 0 — explicit in the file
    "timestamp_utc",       # when this prediction was made
]


def make_result_row(
    issue: dict,
    label_col: str,
    prediction: int,
    raw_response: str,
    prompt_text: str,
    model_version: str,
) -> dict:
    """Build one result row with full reproducibility metadata."""
    return {
        "issue_id":      issue["issue_id"],
        "title":         issue["title"],
        "ground_truth":  int(issue[label_col]),
        "prediction":    prediction,
        "raw_response":  raw_response,            # full, not truncated
        "prompt_hash":   prompt_hash(prompt_text),
        "model_version": model_version,
        "temperature":   0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _count_rows(csv_path: str) -> int:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def make_run_output_dir(base: str = "outputs") -> str:
    """
    Create a timestamped run directory, e.g. outputs/run_20250528_143012/
    so every experiment is isolated and nothing overwrites past results.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"run_{ts}")
    os.makedirs(path, exist_ok=True)
    return path
