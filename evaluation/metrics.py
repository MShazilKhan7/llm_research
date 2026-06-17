"""
evaluation/metrics.py
Standard binary-classification metrics.
"""

from __future__ import annotations


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """
    Compute Accuracy, Precision, Recall, F1, and confusion-matrix counts
    for binary classification (positive class = 1).

    Parse failures (prediction == -1) should be filtered out before calling this.
    """
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp)    if (tp + fp) else 0.0
    recall    = tp / (tp + fn)    if (tp + fn) else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_evaluated": total,
    }


def format_metrics(m: dict, title: str = "") -> str:
    header = f"  {title}" if title else "  Metrics"
    return (
        f"\n{header}\n"
        f"  {'-' * 50}\n"
        f"  Accuracy  : {m['accuracy']:.4f}\n"
        f"  Precision : {m['precision']:.4f}\n"
        f"  Recall    : {m['recall']:.4f}\n"
        f"  F1-score  : {m['f1']:.4f}\n"
        f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}"
        f"  (n={m['n_evaluated']})\n"
    )