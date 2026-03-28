"""
models.py
=========
Model definitions for Pro-PROTAC.

Classes
-------
- GCNEncoder  : Three-layer graph convolutional encoder with global mean pooling.
- ProtoNet    : Episodic meta-learning wrapper implementing prototype-based
                classification and training.
"""

import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from data_utils import build_crbn_tasks, encode_batch, sample_episode
from metrics import compute_metrics, mean_ignore_nan


# ---------------------------------------------------------------------------
# GCN Encoder
# ---------------------------------------------------------------------------

class GCNEncoder(nn.Module):
    """Three-layer Graph Convolutional Network encoder.

    Produces a fixed-dimensional molecular embedding via global mean pooling
    over final node representations.

    Architecture
    ------------
    GCNConv(in_dim → hidden_dim)
    → ReLU → Dropout
    → GCNConv(hidden_dim → hidden_dim)  [repeated for num_layers - 2]
    → ReLU → Dropout
    → GCNConv(hidden_dim → out_dim)
    → GlobalMeanPool

    Args:
        in_dim:     Input node feature dimension (default: 9 for ECFP atom types).
        hidden_dim: Hidden layer dimension (default: 128).
        out_dim:    Output embedding dimension (default: 128).
        num_layers: Total number of GCN layers (default: 3).
        dropout:    Dropout probability applied between layers (default: 0.1).
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:          Node feature matrix, shape (N_nodes, in_dim).
            edge_index: Graph connectivity, shape (2, N_edges).
            batch:      Batch vector mapping each node to its graph index.

        Returns:
            Graph-level embeddings, shape (N_graphs, out_dim).
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


# ---------------------------------------------------------------------------
# ProtoNet
# ---------------------------------------------------------------------------

class ProtoNet(nn.Module):
    """Prototype-based meta-learning model for cross-ligase PROTAC activity
    prediction.

    At meta-training time, the encoder is optimized episodically on CRBN tasks.
    At inference time, class prototypes are computed from the support set and
    query molecules are classified by nearest-prototype assignment---requiring
    no parameter updates.

    Args:
        encoder: A ``GCNEncoder`` instance used to embed PROTAC molecules.
    """

    def __init__(self, encoder: GCNEncoder):
        super().__init__()
        self.encoder = encoder

    # ------------------------------------------------------------------
    # Prototype computation and classification
    # ------------------------------------------------------------------

    def compute_prototypes(
        self,
        support_data: List,
        support_labels: torch.Tensor,
        device: torch.device,
    ) -> Dict[int, torch.Tensor]:
        """Compute class prototypes as mean support embeddings.

        Args:
            support_data:   List of PyG Data objects (support set).
            support_labels: Float tensor of binary labels, shape (2K,).
            device:         Target device.

        Returns:
            Dict mapping class index (0/1) to prototype vector.
        """
        z = encode_batch(self.encoder, support_data, device)
        return {
            1: z[support_labels == 1].mean(dim=0, keepdim=True),
            0: z[support_labels == 0].mean(dim=0, keepdim=True),
        }

    def classify(
        self,
        query_data: List,
        prototypes: Dict[int, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Classify query molecules by nearest-prototype Euclidean distance.

        Classification logit = -(d_pos - d_neg), where d_c is the Euclidean
        distance from the query embedding to prototype c.  A positive logit
        indicates the query is closer to the active prototype.

        Args:
            query_data: List of PyG Data objects (query set).
            prototypes: Class prototype dict from ``compute_prototypes``.
            device:     Target device.

        Returns:
            Logit tensor, shape (N_query,).
        """
        z_q = encode_batch(self.encoder, query_data, device)
        d_pos = torch.norm(z_q - prototypes[1], dim=1)
        d_neg = torch.norm(z_q - prototypes[0], dim=1)
        return -(d_pos - d_neg)

    # ------------------------------------------------------------------
    # Meta-training
    # ------------------------------------------------------------------

    def meta_train(
        self,
        df: "pd.DataFrame",
        graph_index: Dict,
        device: torch.device,
        K: int = 2,
        Q: int = 3,
        epochs: int = 100,
        episodes_per_epoch: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        save_path: Optional[str] = None,
    ) -> None:
        """Run episodic meta-training on CRBN-associated compounds.

        Training follows the prototypical network episodic protocol
        (Snell et al., 2017).  At each episode, K support and Q query
        molecules are sampled per class; the encoder is updated to minimise
        cross-entropy loss on query predictions.

        Args:
            df:                  Full PROTAC dataset DataFrame.
            graph_index:         Compound-id → PyG graph mapping.
            device:              Training device.
            K:                   Support samples per class per episode.
            Q:                   Query samples per class per episode.
            epochs:              Number of meta-training epochs.
            episodes_per_epoch:  Episodes sampled per epoch.
            lr:                  Adam learning rate.
            weight_decay:        Adam weight decay.
            save_path:           If provided, save encoder state dict here
                                 after training (used for UMAP visualization).
        """
        crbn_tasks = build_crbn_tasks(df, graph_index, K=K, Q=Q)
        if not crbn_tasks:
            raise RuntimeError("No valid CRBN tasks found for meta-training.")

        optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=lr, weight_decay=weight_decay
        )
        bce = nn.BCEWithLogitsLoss()
        task_names = list(crbn_tasks.keys())

        print(f"Meta-training Pro-PROTAC | epochs={epochs} | "
              f"episodes/epoch={episodes_per_epoch} | K={K} Q={Q}")

        for epoch in range(1, epochs + 1):
            self.encoder.train()
            losses = []

            for _ in range(episodes_per_epoch):
                ep = sample_episode(
                    crbn_tasks[random.choice(task_names)],
                    graph_index, K=K, Q=Q, device=device,
                )
                if ep is None:
                    continue

                sup_data, sup_labels, qry_data, qry_labels = ep
                prototypes = self.compute_prototypes(sup_data, sup_labels, device)
                logits = self.classify(qry_data, prototypes, device)

                loss = bce(logits, qry_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            avg = mean_ignore_nan(losses)
            print(f"Epoch {epoch:03d}/{epochs} | loss = {avg:.4f}")

        if save_path is not None:
            torch.save(self.encoder.state_dict(), save_path)
            print(f"Encoder saved → {save_path}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        split_path: str,
        graph_index: Dict,
        label_map: Dict,
        device: torch.device,
        phase: str = "meta_test",
    ) -> "pd.DataFrame":
        """Evaluate Pro-PROTAC on a pre-generated episodic split file.

        At inference, class prototypes are computed from the support set
        and query molecules are classified without any parameter update.

        Args:
            split_path:  Path to a JSON split file (CRBN→VHL or rare E3 format).
            graph_index: Compound-id → PyG graph mapping.
            label_map:   Compound-id → binary label mapping.
            device:      Inference device.
            phase:       Split phase to evaluate (``"meta_test"`` by default).

        Returns:
            DataFrame with per-episode metric values.
        """
        import json
        import pandas as pd
        from data_utils import make_data_list

        with open(split_path) as f:
            split = json.load(f)
        phase_obj = split.get(phase, {})

        self.encoder.eval()
        records, skipped, total = [], 0, 0

        for task, episodes in phase_obj.items():
            for ep_idx, ep in enumerate(episodes):
                total += 1
                sup_data = make_data_list(
                    ep.get("support", []), graph_index, label_map
                )
                qry_data = make_data_list(
                    ep.get("query", []), graph_index, label_map
                )
                if not sup_data or not qry_data:
                    skipped += 1
                    continue

                sup_labels = torch.tensor(
                    [g.y.item() for g in sup_data],
                    dtype=torch.float32, device=device,
                )
                qry_labels = torch.tensor(
                    [g.y.item() for g in qry_data],
                    dtype=torch.float32, device=device,
                )

                if not ((sup_labels == 1).any() and (sup_labels == 0).any()):
                    skipped += 1
                    continue

                prototypes = self.compute_prototypes(sup_data, sup_labels, device)
                logits = self.classify(qry_data, prototypes, device)

                rec = {"Task": task, "Episode": ep_idx}
                rec.update(compute_metrics(logits, qry_labels))
                records.append(rec)

        print(f"Evaluation done | valid={len(records)} | "
              f"skipped={skipped} | total={total}")
        return pd.DataFrame(records)
