"""
evaluation/metrics.py
Compute Accuracy, Precision, Recall, F1-score, and Confusion Matrix
for any (ground_truth, predictions) pair.

All functions work with plain Python lists — no sklearn dependency required
(though sklearn is used for the confusion-matrix helper when available).
"""

from __future__ import annotations
import math
from typing import Optional


# ─────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    positive_label: int = 1,
) -> dict:
    """
    Compute binary classification metrics.

    Parameters
    ----------
    y_true : list[int]  Ground truth labels (0 or 1)
    y_pred : list[int]  Predicted labels (0, 1, or -1 for parse failures)
    positive_label : int  Which label is treated as "positive" (default 1)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, tp, fp, tn, fn,
                    n_total, n_valid, n_invalid
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have equal length"

    # Separate valid and invalid predictions
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]
    n_invalid   = len(y_true) - len(valid_pairs)
    n_valid     = len(valid_pairs)

    if n_valid == 0:
        return {
            "accuracy": None, "precision": None, "recall": None, "f1": None,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "n_total": len(y_true), "n_valid": 0, "n_invalid": n_invalid,
        }

    tp = sum(1 for t, p in valid_pairs if t == positive_label and p == positive_label)
    fp = sum(1 for t, p in valid_pairs if t != positive_label and p == positive_label)
    tn = sum(1 for t, p in valid_pairs if t != positive_label and p != positive_label)
    fn = sum(1 for t, p in valid_pairs if t == positive_label and p != positive_label)

    accuracy  = (tp + tn) / n_valid if n_valid else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_total":   len(y_true),
        "n_valid":   n_valid,
        "n_invalid": n_invalid,
    }


def confusion_matrix_dict(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Return confusion matrix as a labelled dict."""
    return {
        "TP (True Positive)":  tp,
        "FP (False Positive)": fp,
        "TN (True Negative)":  tn,
        "FN (False Negative)": fn,
    }


# ─────────────────────────────────────────────
# PRETTY PRINT
# ─────────────────────────────────────────────

def format_metrics(metrics: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"{'─'*50}")
        lines.append(f"  {title}")
        lines.append(f"{'─'*50}")
    lines.append(f"  Accuracy  : {_fmt(metrics['accuracy'])}")
    lines.append(f"  Precision : {_fmt(metrics['precision'])}")
    lines.append(f"  Recall    : {_fmt(metrics['recall'])}")
    lines.append(f"  F1-score  : {_fmt(metrics['f1'])}")
    lines.append(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")
    lines.append(
        f"  Valid preds: {metrics['n_valid']}/{metrics['n_total']} "
        f"({metrics['n_invalid']} parse failures)"
    )
    return "\n".join(lines)


def _fmt(v) -> str:
    return f"{v:.4f}" if v is not None else "N/A"
