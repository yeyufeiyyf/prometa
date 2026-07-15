"""Neural encoder used in the released ProMeta experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


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
