"""Support-only RDKit 3D-descriptor augmentation used in response Table S8."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


DESCRIPTOR_NAMES = (
    "PMI1",
    "PMI2",
    "PMI3",
    "NPR1",
    "NPR2",
    "RadiusOfGyration",
    "InertialShapeFactor",
    "Eccentricity",
    "Asphericity",
    "SpherocityIndex",
    "PBF",
)


def descriptor_index(path: str | Path) -> dict[str, np.ndarray]:
    """Load, median-impute, and globally z-score the released descriptor cache."""
    frame = pd.read_csv(path, low_memory=False)
    raw = (
        frame[list(DESCRIPTOR_NAMES)]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
        .to_numpy()
    )
    median = np.nanmedian(raw, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(raw), raw, median[None, :])
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    normalized = np.nan_to_num(
        (filled - mean[None, :]) / std[None, :],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    return {
        str(compound_id): normalized[index]
        for index, compound_id in enumerate(frame["compound_id"].astype(str))
    }


def descriptor_matrix(index: dict[str, np.ndarray], compound_ids: list[str]) -> np.ndarray:
    missing = [compound_id for compound_id in compound_ids if compound_id not in index]
    if missing:
        raise KeyError(f"Missing 3D descriptors for {missing[:3]}")
    return np.stack([index[compound_id] for compound_id in compound_ids]).astype(np.float32)


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(float(value), 1e-6)))


def normalize(values: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    return values / values.norm(dim=1, keepdim=True).clamp_min(epsilon)


def prototype_logits(
    query: torch.Tensor,
    support: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    negative = support[labels.long() == 0].mean(dim=0)
    positive = support[labels.long() == 1].mean(dim=0)
    distance_negative = ((query - negative) ** 2).sum(dim=1)
    distance_positive = ((query - positive) ** 2).sum(dim=1)
    return torch.stack([-distance_negative, -distance_positive], dim=1)


def fused_features(
    embedding: torch.Tensor,
    fingerprint: torch.Tensor,
    descriptors: torch.Tensor,
    fingerprint_weight: torch.Tensor,
    descriptor_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            normalize(embedding),
            fingerprint_weight * normalize(fingerprint),
            descriptor_weight * normalize(descriptors),
        ],
        dim=1,
    )


def combined_logits(
    query_embedding: torch.Tensor,
    query_fingerprint: torch.Tensor,
    query_descriptors: torch.Tensor,
    support_embedding: torch.Tensor,
    support_fingerprint: torch.Tensor,
    support_descriptors: torch.Tensor,
    labels: torch.Tensor,
    fingerprint_weight: torch.Tensor,
    fingerprint_prior_weight: torch.Tensor,
    descriptor_weight: torch.Tensor,
    descriptor_prior_weight: torch.Tensor,
) -> torch.Tensor:
    support = fused_features(
        support_embedding,
        support_fingerprint,
        support_descriptors,
        fingerprint_weight,
        descriptor_weight,
    )
    query = fused_features(
        query_embedding,
        query_fingerprint,
        query_descriptors,
        fingerprint_weight,
        descriptor_weight,
    )
    logits = prototype_logits(query, support, labels)
    logits += fingerprint_prior_weight * prototype_logits(
        normalize(query_fingerprint), normalize(support_fingerprint), labels
    )
    logits += descriptor_prior_weight * prototype_logits(
        normalize(query_descriptors), normalize(support_descriptors), labels
    )
    return logits


def leave_one_out_loss(
    embedding: torch.Tensor,
    fingerprint: torch.Tensor,
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor | None:
    if int((labels == 0).sum()) < 2 or int((labels == 1).sum()) < 2:
        return None
    logits = []
    targets = []
    for index in range(len(labels)):
        keep = torch.ones(len(labels), dtype=torch.bool, device=labels.device)
        keep[index] = False
        logits.append(
            combined_logits(
                embedding[index : index + 1],
                fingerprint[index : index + 1],
                descriptors[index : index + 1],
                embedding[keep],
                fingerprint[keep],
                descriptors[keep],
                labels[keep],
                *weights,
            )
        )
        targets.append(labels[index : index + 1])
    return F.cross_entropy(torch.cat(logits), torch.cat(targets))


def learn_support_weights(
    support_embedding: np.ndarray,
    support_fingerprint: np.ndarray,
    support_descriptors: np.ndarray,
    support_labels: np.ndarray,
    device: torch.device,
    initial: tuple[float, float, float, float] = (0.5, 0.5, 0.05, 0.05),
    steps: int = 100,
    learning_rate: float = 0.05,
    regularization: float = 0.1,
) -> dict[str, Any]:
    embedding = torch.as_tensor(support_embedding, dtype=torch.float32, device=device)
    fingerprint = torch.as_tensor(support_fingerprint, dtype=torch.float32, device=device)
    descriptors = torch.as_tensor(support_descriptors, dtype=torch.float32, device=device)
    labels = torch.as_tensor(support_labels, dtype=torch.long, device=device)
    if int((labels == 0).sum()) < 2 or int((labels == 1).sum()) < 2:
        return {
            "weights": initial,
            "status": "fallback_insufficient_support_for_loo",
            "loo_loss_start": float("nan"),
            "loo_loss_end": float("nan"),
        }
    parameters = [
        torch.nn.Parameter(
            torch.tensor(inverse_softplus(value), dtype=torch.float32, device=device)
        )
        for value in initial
    ]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    initial_logs = [math.log(max(value, 1e-6)) for value in initial]
    first_loss = float("nan")
    last_loss = float("nan")
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        weights = tuple(F.softplus(parameter) + 1e-6 for parameter in parameters)
        loss = leave_one_out_loss(embedding, fingerprint, descriptors, labels, weights)
        if loss is None:
            break
        penalty = regularization * sum(
            (torch.log(weight) - initial_log).pow(2)
            for weight, initial_log in zip(weights, initial_logs)
        )
        if step == 0:
            first_loss = float(loss.detach().cpu())
        (loss + penalty).backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return {
        "weights": tuple(
            float((F.softplus(parameter) + 1e-6).detach().cpu())
            for parameter in parameters
        ),
        "status": "learned_loo_support",
        "loo_loss_start": first_loss,
        "loo_loss_end": last_loss,
    }


def predict(
    support_embedding: np.ndarray,
    support_fingerprint: np.ndarray,
    support_descriptors: np.ndarray,
    support_labels: np.ndarray,
    query_embedding: np.ndarray,
    query_fingerprint: np.ndarray,
    query_descriptors: np.ndarray,
    learned: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        logits = combined_logits(
            torch.as_tensor(query_embedding, dtype=torch.float32, device=device),
            torch.as_tensor(query_fingerprint, dtype=torch.float32, device=device),
            torch.as_tensor(query_descriptors, dtype=torch.float32, device=device),
            torch.as_tensor(support_embedding, dtype=torch.float32, device=device),
            torch.as_tensor(support_fingerprint, dtype=torch.float32, device=device),
            torch.as_tensor(support_descriptors, dtype=torch.float32, device=device),
            torch.as_tensor(support_labels, dtype=torch.long, device=device),
            *[
                torch.tensor(value, dtype=torch.float32, device=device)
                for value in learned["weights"]
            ],
        )
        return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(float)
