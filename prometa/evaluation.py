"""ProMeta support-conditioned inference and metric aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch_geometric.data import Batch, Data

from .data import graphs_for_examples, iter_episodes, read_split
from .model import GCNEncoder, ProteinEncoder


# RDKit emits one deprecation line per molecule for the historical ECFP API.
# Keep that API because it is the exact fingerprint route used for the paper,
# but silence only its warning channel during batch reproduction.
RDLogger.DisableLog("rdApp.warning")


def l2_normalize(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    denominator = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denominator, epsilon)


def smiles_to_ecfp4(smiles: str, bits: int = 512) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=bits)
    return np.asarray(fingerprint, dtype=np.float32)


def fingerprint_index(contexts: pd.DataFrame, bits: int = 512) -> dict[str, np.ndarray]:
    index: dict[str, np.ndarray] = {}
    for _, row in contexts.drop_duplicates("Compound ID").iterrows():
        fingerprint = smiles_to_ecfp4(row["Smiles"], bits=bits)
        if fingerprint is not None:
            index[str(row["Compound ID"])] = fingerprint
    return index


def load_encoder(checkpoint: str | Path, device: torch.device) -> GCNEncoder:
    encoder = GCNEncoder().to(device)
    try:
        state: Any = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        for key in ["encoder", "state_dict", "model", "backbone"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
        state = {
            key.removeprefix("module.").removeprefix("encoder."): value
            for key, value in state.items()
        }
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    return encoder


def load_protein_encoder(
    checkpoint: str | Path,
    device: torch.device,
    embed_dim: int = 64,
    out_dim: int = 128,
) -> ProteinEncoder:
    """Load a retained POI or E3 sequence checkpoint with strict key checks."""
    encoder = ProteinEncoder(embed_dim=embed_dim, out_dim=out_dim).to(device)
    try:
        state: Any = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        for key in ["encoder", "state_dict", "model", "module"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
        state = {key.removeprefix("module."): value for key, value in state.items()}
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    return encoder


@torch.no_grad()
def encode_graphs(
    encoder: GCNEncoder,
    graphs: list[Data],
    device: torch.device,
) -> np.ndarray:
    batch = Batch.from_data_list(graphs).to(device)
    embedding = encoder(batch.x, batch.edge_index, batch.batch)
    return embedding.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def encode_graphs_sequence_aware(
    encoder: GCNEncoder,
    poi_encoder: ProteinEncoder,
    e3_encoder: ProteinEncoder,
    graphs: list[Data],
    poi_index: dict[str, torch.LongTensor],
    e3_index: dict[str, torch.LongTensor],
    device: torch.device,
) -> np.ndarray:
    """Encode graphs with the retained additive POI/E3 sequence branches."""
    batch = Batch.from_data_list(graphs).to(device)
    molecular = encoder(batch.x, batch.edge_index, batch.batch)
    compound_ids = [str(getattr(graph, "cid", "")) for graph in graphs]
    poi = poi_encoder.encode_compound_ids(compound_ids, poi_index, device)
    e3 = e3_encoder.encode_compound_ids(compound_ids, e3_index, device)
    embedding = molecular + poi_encoder.alpha * poi + e3_encoder.alpha * e3
    return embedding.detach().cpu().numpy().astype(np.float32)


def fused_features(
    embeddings: np.ndarray,
    compound_ids: list[str],
    fingerprints: dict[str, np.ndarray],
    fingerprint_weight: float = 0.5,
) -> np.ndarray:
    fp = np.stack([fingerprints[compound_id] for compound_id in compound_ids])
    return np.concatenate(
        [l2_normalize(embeddings), fingerprint_weight * l2_normalize(fp)],
        axis=1,
    )


def prototype_probabilities(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    positive = support_features[support_labels == 1].mean(axis=0)
    negative = support_features[support_labels == 0].mean(axis=0)
    distance_positive = ((query_features - positive) ** 2).sum(axis=1)
    distance_negative = ((query_features - negative) ** 2).sum(axis=1)
    logits = np.stack([-distance_negative, -distance_positive], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def linear_head_probabilities(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    classifier = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
    )
    classifier.fit(support_features, support_labels)
    return classifier.predict_proba(query_features)[:, 1]


def evaluate_split(
    split_path: str | Path,
    checkpoint: str | Path,
    graph_index: dict[str, Data],
    contexts: pd.DataFrame,
    device: torch.device,
    head: str = "prototype",
    fingerprint_bits: int = 512,
    fingerprint_weight: float = 0.5,
    phase: str = "meta_test",
    metadata_overrides: dict[str, Any] | None = None,
    poi_checkpoint: str | Path | None = None,
    e3_checkpoint: str | Path | None = None,
    poi_index: dict[str, torch.LongTensor] | None = None,
    e3_index: dict[str, torch.LongTensor] | None = None,
) -> pd.DataFrame:
    split_path = Path(split_path)
    split = read_split(split_path)
    meta = {**split.get("meta", {}), **(metadata_overrides or {})}
    encoder = load_encoder(checkpoint, device)
    sequence_arguments = [poi_checkpoint, e3_checkpoint, poi_index, e3_index]
    if any(argument is not None for argument in sequence_arguments) and not all(
        argument is not None for argument in sequence_arguments
    ):
        raise ValueError(
            "Sequence-aware inference requires both checkpoints and both sequence indices"
        )
    sequence_aware = all(argument is not None for argument in sequence_arguments)
    poi_encoder = (
        load_protein_encoder(poi_checkpoint, device) if poi_checkpoint is not None else None
    )
    e3_encoder = (
        load_protein_encoder(e3_checkpoint, device) if e3_checkpoint is not None else None
    )
    fingerprints = fingerprint_index(contexts, bits=fingerprint_bits)
    fallback_labels = dict(zip(contexts["Compound ID"], contexts["Label_bin"]))
    rows: list[dict[str, Any]] = []

    for task, episode_index, support_examples, query_examples in iter_episodes(split, phase=phase):
        support, support_ids, support_labels = graphs_for_examples(
            support_examples, graph_index, fallback_labels
        )
        query, query_ids, query_labels = graphs_for_examples(
            query_examples, graph_index, fallback_labels
        )
        if not support or not query or set(support_labels.tolist()) != {0, 1}:
            continue
        if any(compound_id not in fingerprints for compound_id in support_ids + query_ids):
            continue
        support_embedding = (
            encode_graphs_sequence_aware(
                encoder, poi_encoder, e3_encoder, support, poi_index, e3_index, device
            )
            if sequence_aware
            else encode_graphs(encoder, support, device)
        )
        query_embedding = (
            encode_graphs_sequence_aware(
                encoder, poi_encoder, e3_encoder, query, poi_index, e3_index, device
            )
            if sequence_aware
            else encode_graphs(encoder, query, device)
        )
        support_features = fused_features(
            support_embedding,
            support_ids,
            fingerprints,
            fingerprint_weight,
        )
        query_features = fused_features(
            query_embedding,
            query_ids,
            fingerprints,
            fingerprint_weight,
        )
        if head == "prototype":
            probabilities = prototype_probabilities(support_features, support_labels, query_features)
        elif head == "linear":
            probabilities = linear_head_probabilities(support_features, support_labels, query_features)
        else:
            raise ValueError(f"Unsupported head: {head}")
        for compound_id, label, probability in zip(query_ids, query_labels, probabilities):
            rows.append(
                {
                    "split": split_path.stem,
                    "direction": meta.get("direction", ""),
                    "source": meta.get("source", ""),
                    "target": meta.get("target", ""),
                    "k_shot": meta.get("k_shot", ""),
                    "q_query": meta.get("q_query", ""),
                    "query_setting": meta.get("query_setting", ""),
                    "seed": meta.get("seed", ""),
                    "method": (
                        "ProMeta (sequence-aware)"
                        if sequence_aware and head == "prototype"
                        else "ProMeta + linear head (sequence-aware)"
                        if sequence_aware
                        else "ProMeta"
                        if head == "prototype"
                        else "ProMeta + linear head"
                    ),
                    "task": task,
                    "episode": episode_index,
                    "compound_id": compound_id,
                    "y_true": int(label),
                    "prob": float(probability),
                }
            )
    return pd.DataFrame(rows)


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["y_true"].astype(int).to_numpy()
    probabilities = frame["prob"].astype(float).to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    has_two_classes = len(set(labels.tolist())) == 2
    return {
        "n_query_eval": float(len(frame)),
        "AUROC": float(roc_auc_score(labels, probabilities)) if has_two_classes else np.nan,
        "AUPRC": float(average_precision_score(labels, probabilities)) if has_two_classes else np.nan,
        "BALACC": float(balanced_accuracy_score(labels, predictions)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "Accuracy": float(accuracy_score(labels, predictions)),
    }


def split_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "split", "direction", "k_shot", "q_query", "query_setting", "seed", "method"
    ]
    rows = []
    for key, subset in predictions.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, key))
        row.update(metric_values(subset))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["direction", "k_shot", "q_query", "query_setting", "method"]
    rows: list[dict[str, Any]] = []
    for key, subset in metrics.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, key))
        row["n_seeds"] = int(subset["seed"].nunique())
        row["mean_n_query_eval"] = float(subset["n_query_eval"].mean())
        for metric in ["AUROC", "AUPRC", "BALACC", "F1", "Accuracy"]:
            values = subset[metric].dropna().astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[metric] = f"{mean:.3f} +/- {std:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)
