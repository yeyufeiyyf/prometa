#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
utils.py — ProMeta utility functions

Covers:
  - Random seed setting
  - Binary classification metrics (AUROC, AUPRC, Accuracy, F1, BALACC)
  - Result summarisation helpers
"""

import math
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def mean_ignore_nan(xs: List[float]) -> float:
    vals = [float(x) for x in xs if not math.isnan(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


# ── Core metrics ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _auroc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y, s    = y_true.view(-1).float(), y_score.view(-1).float()
    n_pos   = int((y == 1).sum())
    n_neg   = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order   = torch.argsort(s)
    ranks   = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float, device=s.device)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def _auprc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y, s  = y_true.view(-1).float(), y_score.view(-1).float()
    n_pos = float((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order    = torch.argsort(s, descending=True)
    y_sorted = y[order]
    tp       = torch.cumsum(y_sorted, 0)
    fp       = torch.cumsum(1 - y_sorted, 0)
    prec     = tp / torch.clamp(tp + fp, min=1.0)
    return float((prec * y_sorted).sum() / n_pos)


# ── Public metric functions ───────────────────────────────────────────────────

@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> dict:
    """
    Compute all evaluation metrics from raw logits and binary labels.

    Returns a dict with keys: Accuracy, Loss, AUROC, AUPRC, F1, BALACC.
    """
    y_true  = y.view(-1).float()
    prob    = torch.sigmoid(logits.view(-1).float())
    y_pred  = (prob >= 0.5).float()

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
        "AUROC":    _auroc(y_true, prob),
        "AUPRC":    _auprc(y_true, prob),
        "F1":       float(f1),
        "BALACC":   balacc,
    }


def summarize_records(df, variant: str):
    """Aggregate per-episode metrics into a single-row summary DataFrame."""
    import pandas as pd
    cols = ["Accuracy", "Loss", "AUROC", "AUPRC", "F1", "BALACC"]
    row  = {"Variant": variant}
    for c in cols:
        row[c] = mean_ignore_nan(df[c].tolist()) if c in df.columns else float("nan")
    return pd.DataFrame([row])
