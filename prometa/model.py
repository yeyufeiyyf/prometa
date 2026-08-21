"""Molecular and protein encoders used by ProMeta."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# Padding and non-standard residues use index 0.  The ordering is retained
# from the sequence-aware experimental source so that its checkpoints load
# without any key or vocabulary conversion.
AA_VOCAB = {amino_acid: index + 1 for index, amino_acid in enumerate("ACDEFGHIKLMNPQRSTVWY")}
AA_VOCAB_SIZE = len(AA_VOCAB) + 1


def sequence_to_tensor(sequence: str, max_length: int = 2000) -> torch.LongTensor:
    """Convert an amino-acid sequence to the indices used by ProteinEncoder."""
    indices = [AA_VOCAB.get(amino_acid.upper(), 0) for amino_acid in str(sequence)[:max_length]]
    return torch.tensor(indices, dtype=torch.long)


class ProteinEncoder(nn.Module):
    """Lightweight POI/E3 sequence encoder from the reported experiments.

    Each protein branch embeds amino acids, mean-pools the sequence, projects
    it to the molecular embedding dimension, and exposes a trainable scalar
    ``alpha``.  Separate instances are used for POI and E3 sequences.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        out_dim: int = 128,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(AA_VOCAB_SIZE, embed_dim, padding_idx=0)
        self.proj = nn.Linear(embed_dim, out_dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, sequence: torch.LongTensor) -> torch.Tensor:
        embedded = self.embedding(sequence)
        pooled = embedded.mean(dim=0)
        return F.relu(self.proj(pooled))

    def encode_compound_ids(
        self,
        compound_ids: list[str],
        sequence_index: dict[str, torch.LongTensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode compound-linked sequences, using zero for a missing sequence."""
        vectors: list[torch.Tensor] = []
        for compound_id in compound_ids:
            sequence = sequence_index.get(str(compound_id))
            if sequence is None:
                vectors.append(torch.zeros(self.proj.out_features, device=device))
            else:
                vectors.append(self.forward(sequence.to(device)))
        return torch.stack(vectors, dim=0)


class GCNEncoder(nn.Module):
    """Three-layer molecular GCN used for episodic representation learning.

    The defaults reproduce the revision experiments: nine atom-type features,
    128 hidden/output dimensions, three graph-convolution layers, and dropout
    after every non-final layer.
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            GCNConv(dims[index], dims[index + 1])
            for index in range(num_layers)
        )
        self.dropout = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)
