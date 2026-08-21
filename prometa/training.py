"""Training routines for episodic and non-episodic ProMeta encoders."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from .data import graphs_for_examples
from .model import GCNEncoder, ProteinEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(encoder: GCNEncoder, graphs: list[Data], device: torch.device) -> torch.Tensor:
    batch = Batch.from_data_list(graphs).to(device)
    return encoder(batch.x, batch.edge_index, batch.batch)


def encode_sequence_aware(
    encoder: GCNEncoder,
    poi_encoder: ProteinEncoder,
    e3_encoder: ProteinEncoder,
    graphs: list[Data],
    poi_index: dict[str, torch.LongTensor],
    e3_index: dict[str, torch.LongTensor],
    device: torch.device,
) -> torch.Tensor:
    """Apply the reported additive molecule/POI/E3 fusion."""
    molecular = encode(encoder, graphs, device)
    compound_ids = [str(getattr(graph, "cid", "")) for graph in graphs]
    poi = poi_encoder.encode_compound_ids(compound_ids, poi_index, device)
    e3 = e3_encoder.encode_compound_ids(compound_ids, e3_index, device)
    return molecular + poi_encoder.alpha * poi + e3_encoder.alpha * e3


def source_tasks(
    contexts: pd.DataFrame,
    graph_index: dict[str, Data],
    source_e3: str,
    k_shot: int,
    q_query: int,
) -> tuple[dict[str, pd.DataFrame], str]:
    source = contexts[
        (contexts["E3 ligase"].astype(str).str.strip() == source_e3)
        & contexts["Compound ID"].astype(str).isin(graph_index)
    ].copy()
    required = int(k_shot) + int(q_query)
    tasks: dict[str, pd.DataFrame] = {}
    for target, subset in source.groupby("Target"):
        positive = int((subset["Label_bin"] == 1).sum())
        negative = int((subset["Label_bin"] == 0).sum())
        if positive >= required and negative >= required:
            tasks[str(target)] = subset.reset_index(drop=True)
    if tasks:
        return tasks, "target"
    positive = int((source["Label_bin"] == 1).sum())
    negative = int((source["Label_bin"] == 0).sum())
    if positive >= required and negative >= required:
        return {f"{source_e3}_aggregate": source.reset_index(drop=True)}, "aggregate"
    raise RuntimeError(
        f"No source task has K+Q={required} examples per class for {source_e3}"
    )


def sample_episode(
    task: pd.DataFrame,
    graph_index: dict[str, Data],
    k_shot: int,
    q_query: int,
    device: torch.device,
) -> tuple[list[Data], torch.Tensor, list[Data], torch.Tensor]:
    required = int(k_shot) + int(q_query)
    positive = task[task["Label_bin"] == 1].sample(required, replace=False)
    negative = task[task["Label_bin"] == 0].sample(required, replace=False)
    support_rows = pd.concat([positive.iloc[:k_shot], negative.iloc[:k_shot]])
    query_rows = pd.concat([positive.iloc[k_shot:], negative.iloc[k_shot:]])
    support_examples = support_rows[["Compound ID", "Label_bin"]].to_dict("records")
    query_examples = query_rows[["Compound ID", "Label_bin"]].to_dict("records")
    support, _, support_labels = graphs_for_examples(support_examples, graph_index)
    query, _, query_labels = graphs_for_examples(query_examples, graph_index)
    return (
        support,
        torch.as_tensor(support_labels, dtype=torch.float32, device=device),
        query,
        torch.as_tensor(query_labels, dtype=torch.float32, device=device),
    )


def train_episodic(
    contexts: pd.DataFrame,
    graph_index: dict[str, Data],
    source_e3: str,
    k_shot: int,
    q_query: int,
    seed: int,
    device: torch.device,
    epochs: int = 100,
    episodes_per_epoch: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
) -> tuple[GCNEncoder, dict[str, Any]]:
    set_seed(seed)
    tasks, task_mode = source_tasks(
        contexts, graph_index, source_e3, k_shot, q_query
    )
    encoder = GCNEncoder().to(device)
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss()
    task_names = list(tasks)
    for _ in range(int(epochs)):
        encoder.train()
        for _ in range(int(episodes_per_epoch)):
            support, support_labels, query, query_labels = sample_episode(
                tasks[random.choice(task_names)],
                graph_index,
                k_shot,
                q_query,
                device,
            )
            support_embedding = encode(encoder, support, device)
            query_embedding = encode(encoder, query, device)
            positive = support_embedding[support_labels == 1].mean(0, keepdim=True)
            negative = support_embedding[support_labels == 0].mean(0, keepdim=True)
            logits = -(
                torch.norm(query_embedding - positive, dim=1)
                - torch.norm(query_embedding - negative, dim=1)
            )
            loss = loss_function(logits, query_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    encoder.eval()
    return encoder, {
        "source_e3": source_e3,
        "seed": int(seed),
        "k_shot": int(k_shot),
        "train_q": int(q_query),
        "task_mode": task_mode,
        "n_source_tasks": len(tasks),
        "epochs": int(epochs),
        "episodes_per_epoch": int(episodes_per_epoch),
    }


def train_episodic_sequence_aware(
    contexts: pd.DataFrame,
    graph_index: dict[str, Data],
    poi_index: dict[str, torch.LongTensor],
    e3_index: dict[str, torch.LongTensor],
    source_e3: str,
    k_shot: int,
    q_query: int,
    seed: int,
    device: torch.device,
    epochs: int = 100,
    episodes_per_epoch: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    protein_embed_dim: int = 64,
) -> tuple[GCNEncoder, ProteinEncoder, ProteinEncoder, dict[str, Any]]:
    """Train the sequence-aware episodic model used in the original runs.

    All GCN, POI, E3, and scalar-alpha parameters are registered in the same
    Adam optimizer.  This explicit routine is retained separately from the
    GCN-only numerical-audit path to make the two release scopes unambiguous.
    """
    set_seed(seed)
    tasks, task_mode = source_tasks(
        contexts, graph_index, source_e3, k_shot, q_query
    )
    encoder = GCNEncoder().to(device)
    poi_encoder = ProteinEncoder(embed_dim=protein_embed_dim).to(device)
    e3_encoder = ProteinEncoder(embed_dim=protein_embed_dim).to(device)
    optimizer = torch.optim.Adam(
        [
            *encoder.parameters(),
            *poi_encoder.parameters(),
            *e3_encoder.parameters(),
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()
    task_names = list(tasks)
    for _ in range(int(epochs)):
        encoder.train()
        poi_encoder.train()
        e3_encoder.train()
        for _ in range(int(episodes_per_epoch)):
            support, support_labels, query, query_labels = sample_episode(
                tasks[random.choice(task_names)],
                graph_index,
                k_shot,
                q_query,
                device,
            )
            support_embedding = encode_sequence_aware(
                encoder, poi_encoder, e3_encoder, support, poi_index, e3_index, device
            )
            query_embedding = encode_sequence_aware(
                encoder, poi_encoder, e3_encoder, query, poi_index, e3_index, device
            )
            positive = support_embedding[support_labels == 1].mean(0, keepdim=True)
            negative = support_embedding[support_labels == 0].mean(0, keepdim=True)
            logits = -(
                torch.norm(query_embedding - positive, dim=1)
                - torch.norm(query_embedding - negative, dim=1)
            )
            loss = loss_function(logits, query_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    encoder.eval()
    poi_encoder.eval()
    e3_encoder.eval()
    return encoder, poi_encoder, e3_encoder, {
        "source_e3": source_e3,
        "seed": int(seed),
        "k_shot": int(k_shot),
        "train_q": int(q_query),
        "task_mode": task_mode,
        "n_source_tasks": len(tasks),
        "epochs": int(epochs),
        "episodes_per_epoch": int(episodes_per_epoch),
        "sequence_aware": True,
        "poi_alpha": float(poi_encoder.alpha.detach().cpu()),
        "e3_alpha": float(e3_encoder.alpha.detach().cpu()),
        "optimizer_parameter_groups": ["gcn", "poi_encoder", "e3_encoder"],
    }


def train_non_episodic(
    contexts: pd.DataFrame,
    graph_index: dict[str, Data],
    source_e3: str,
    seed: int,
    device: torch.device,
    epochs: int = 120,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[GCNEncoder, dict[str, Any]]:
    """Train the encoder used by the ``ProMeta w/o episodic training`` control."""
    set_seed(seed)
    source = contexts[
        (contexts["E3 ligase"].astype(str).str.strip() == source_e3)
        & contexts["Compound ID"].astype(str).isin(graph_index)
    ]
    examples = source[["Compound ID", "Label_bin"]].to_dict("records")
    graphs, _, labels = graphs_for_examples(examples, graph_index)
    for graph, label in zip(graphs, labels):
        graph.y = torch.tensor([int(label)], dtype=torch.long)
    encoder = GCNEncoder().to(device)
    head = nn.Linear(128, 1).to(device)
    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], device=device)
    )
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *head.parameters()],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    for _ in range(int(epochs)):
        encoder.train()
        head.train()
        for batch in loader:
            batch = batch.to(device)
            logits = head(encoder(batch.x, batch.edge_index, batch.batch)).view(-1)
            loss = loss_function(logits, batch.y.view(-1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    encoder.eval()
    return encoder, {
        "source_e3": source_e3,
        "seed": int(seed),
        "n_source": len(graphs),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
    }
