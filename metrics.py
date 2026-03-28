"""
metrics.py
==========
Evaluation metrics for binary classification under episodic few-shot settings.

All metrics are implemented in pure PyTorch/NumPy to avoid additional dependencies.
AUROC and AUPRC are the primary metrics throughout the paper.
"""

import math
from typing import Dict, List

import torch
import torch.nn.functional as F


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


@torch.no_grad()
def auroc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """Compute AUROC via the rank-sum formula.

    Args:
        y_true:  Binary ground-truth labels (0/1), shape (N,).
        y_score: Predicted scores (higher = more positive), shape (N,).

    Returns:
        AUROC value in [0, 1], or NaN if only one class is present.
    """
    y = y_true.view(-1).float()
    s = y_score.view(-1).float()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float, device=s.device)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def auprc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """Compute AUPRC (Average Precision) via precision-recall curve.

    Args:
        y_true:  Binary ground-truth labels (0/1), shape (N,).
        y_score: Predicted scores (higher = more positive), shape (N,).

    Returns:
        AUPRC value in [0, 1], or NaN if no positive samples.
    """
    y = y_true.view(-1).float()
    s = y_score.view(-1).float()
    n_pos = float((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(s, descending=True)
    y_sorted = y[order]
    tp = torch.cumsum(y_sorted, dim=0)
    fp = torch.cumsum(1 - y_sorted, dim=0)
    precision = tp / torch.clamp(tp + fp, min=1.0)
    return float((precision * y_sorted).sum() / n_pos)


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """Compute all evaluation metrics from raw logits.

    Args:
        logits: Raw model output (before sigmoid), shape (N,).
        y:      Binary ground-truth labels (0/1), shape (N,).

    Returns:
        Dictionary with keys: Accuracy, Loss, AUROC, AUPRC, F1, BALACC.
    """
    y_true = y.view(-1).float()
    prob   = torch.sigmoid(logits.view(-1).float())
    y_pred = (prob >= 0.5).float()

    acc  = float((y_pred == y_true).float().mean())
    loss = float(F.binary_cross_entropy_with_logits(logits.view(-1).float(), y_true))

    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())

    prec   = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1     = _safe_div(2 * prec * recall, prec + recall) if (prec + recall) > 0 else 0.0
    tpr    = _safe_div(tp, tp + fn)
    tnr    = _safe_div(tn, tn + fp)
    balacc = float("nan") if any(math.isnan(v) for v in [tpr, tnr]) else (tpr + tnr) / 2

    return {
        "Accuracy": acc,
        "Loss":     loss,
        "AUROC":    auroc(y_true, prob),
        "AUPRC":    auprc(y_true, prob),
        "F1":       float(f1),
        "BALACC":   balacc,
    }


def mean_ignore_nan(values: List[float]) -> float:
    """Compute mean while ignoring NaN values."""
    valid = [v for v in values if not math.isnan(float(v))]
    return sum(valid) / len(valid) if valid else float("nan")


def summarize_episode_results(records: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-episode metric dictionaries into mean values.

    Args:
        records: List of metric dicts, one per episode.

    Returns:
        Dict mapping metric name to mean value across episodes.
    """
    if not records:
        return {}
    keys = [k for k in records[0] if k not in ("Task", "Episode")]
    return {k: mean_ignore_nan([r[k] for r in records]) for k in keys}
