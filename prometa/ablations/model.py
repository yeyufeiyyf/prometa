"""Lightweight, context-aware models for the ProMeta E3-v2 experiments.

The module is intentionally self contained: it depends only on PyTorch and
PyTorch Geometric and does not require changes to the historical data or runner
code.  A single :class:`E3V2Model` exposes all E3 ablations through the same
``forward`` and ``score_episode`` APIs.

Two protein input routes are supported:

* integer residue tokens ``[batch, length]`` are encoded into contextual token
  representations and an attention-pooled protein vector; and
* frozen precomputed features (1280-dimensional ESM vectors by default) are
  detached from their source graph and passed through a small shared adapter.

``e3_atom_attention`` means E3-token-to-atom attention.  It is deliberately not
called E3-*ligand* attention unless an explicit ``e3_ligand_mask`` is supplied.
When that mask is available, only masked atoms contribute to the interaction
pool and per-graph mask coverage is returned in the diagnostics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch


E3Variant = Literal[
    "no_e3",
    "additive",
    "frozen_random_e3",
    "film",
    "conditional_metric",
    "e3_atom_attention",
    "interaction",
    "grounded_interaction",
    "grounded_additive",
    "grounded_no_global",
    "grounded_no_component",
    "grounded_no_poi_context",
    "grounded_no_e3_context",
    "grounded_no_ligand_localization",
    "grounded_source_linear",
    "grounded_constant_e3",
    "grounded_poi_control",
]
DistanceName = Literal["squared_euclidean", "euclidean", "cosine"]

VALID_VARIANTS = frozenset(
    {
        "no_e3",
        "additive",
        "frozen_random_e3",
        "film",
        "conditional_metric",
        "e3_atom_attention",
        "interaction",
        "grounded_interaction",
        "grounded_additive",
        "grounded_no_global",
        "grounded_no_component",
        "grounded_no_poi_context",
        "grounded_no_e3_context",
        "grounded_no_ligand_localization",
        "grounded_source_linear",
        "grounded_constant_e3",
        "grounded_poi_control",
    }
)
VALID_DISTANCES = frozenset({"squared_euclidean", "euclidean", "cosine"})


@dataclass(frozen=True)
class E3V2Config:
    """Configuration shared by every E3-v2 variant."""

    variant: E3Variant = "interaction"
    node_dim: int = 9
    edge_dim: int = 0
    embedding_dim: int = 128
    gnn_layers: int = 3
    protein_vocab_size: int = 22
    protein_padding_idx: int = 0
    protein_max_length: int = 2048
    protein_layers: int = 2
    pretrained_protein_dim: int = 1280
    protein_adapter_dim: int = 128
    attention_heads: int = 4
    interaction_rank: int = 32
    dropout: float = 0.1
    distance: DistanceName = "squared_euclidean"
    temperature: float = 1.0
    learnable_temperature: bool = False
    num_auxiliary_outputs: int = 0

    def validate(self) -> None:
        if self.variant not in VALID_VARIANTS:
            raise ValueError(
                f"Unknown E3-v2 variant {self.variant!r}; expected one of "
                f"{sorted(VALID_VARIANTS)}"
            )
        if self.distance not in VALID_DISTANCES:
            raise ValueError(
                f"Unknown distance {self.distance!r}; expected one of "
                f"{sorted(VALID_DISTANCES)}"
            )
        if self.node_dim < 1 or self.embedding_dim < 1:
            raise ValueError("node_dim and embedding_dim must be positive")
        if self.edge_dim < 0:
            raise ValueError("edge_dim cannot be negative")
        if self.gnn_layers < 1 or self.protein_layers < 1:
            raise ValueError("gnn_layers and protein_layers must be positive")
        if self.protein_max_length < 1:
            raise ValueError("protein_max_length must be positive")
        if self.pretrained_protein_dim < 1 or self.protein_adapter_dim < 1:
            raise ValueError("protein feature dimensions must be positive")
        if self.attention_heads < 1:
            raise ValueError("attention_heads must be positive")
        if self.embedding_dim % self.attention_heads != 0:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        if self.interaction_rank < 1:
            raise ValueError("interaction_rank must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.num_auxiliary_outputs < 0:
            raise ValueError("num_auxiliary_outputs cannot be negative")


@dataclass
class MoleculeEncoding:
    atom_embeddings: Tensor
    graph_embedding: Tensor


@dataclass
class ProteinEncoding:
    token_embeddings: Tensor
    token_mask: Tensor
    pooled: Tensor
    pool_weights: Tensor | None
    input_mode: str


class MoleculeEncoder(nn.Module):
    """A small bond-aware GINE encoder returning atom and graph features.

    ``edge_dim=0`` keeps the module compatible with historical graph caches
    that have no bond attributes: every edge receives an all-zero feature with
    the dimensionality expected by the current layer.  For new graphs, set
    ``edge_dim`` to the explicit bond-feature width (14 in ``e3_v2.data``).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        embedding_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.edge_dim = int(edge_dim)
        dimensions = [node_dim] + [embedding_dim] * num_layers
        self.convolutions = nn.ModuleList(
            GINEConv(
                nn=nn.Sequential(
                    nn.Linear(dimensions[index], dimensions[index + 1]),
                    nn.GELU(),
                    nn.Linear(dimensions[index + 1], dimensions[index + 1]),
                ),
                edge_dim=self.edge_dim if self.edge_dim > 0 else None,
            )
            for index in range(num_layers)
        )
        self.normalizations = nn.ModuleList(
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        )
        self.dropout = float(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor | None = None,
    ) -> MoleculeEncoding:
        if edge_attr is not None:
            if edge_attr.ndim != 2 or edge_attr.size(0) != edge_index.size(1):
                raise ValueError("edge_attr must have shape [n_edges, edge_dim]")
            if self.edge_dim == 0:
                raise ValueError(
                    "edge_attr was provided but E3V2Config.edge_dim is zero"
                )
            if edge_attr.size(1) != self.edge_dim:
                raise ValueError(
                    f"expected {self.edge_dim} bond features, got {edge_attr.size(1)}"
                )
        hidden = x
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            if edge_attr is None:
                zero_width = self.edge_dim if self.edge_dim > 0 else hidden.size(1)
                layer_edge_attr = hidden.new_zeros(edge_index.size(1), zero_width)
            else:
                layer_edge_attr = edge_attr.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            updated = convolution(hidden, edge_index, layer_edge_attr)
            updated = normalization(updated)
            updated = F.gelu(updated)
            hidden = F.dropout(updated, p=self.dropout, training=self.training)
        batch_size = int(batch.max().item()) + 1
        pooled = global_mean_pool(hidden, batch, size=batch_size)
        return MoleculeEncoding(atom_embeddings=hidden, graph_embedding=pooled)


class TokenConvBlock(nn.Module):
    """Linear-cost contextualization that preserves one vector per residue."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=3,
            padding=1,
            groups=dimension,
        )
        self.pointwise = nn.Conv1d(dimension, 2 * dimension, kernel_size=1)
        self.normalization = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        update = tokens.transpose(1, 2)
        update = self.depthwise(update)
        update = F.glu(self.pointwise(update), dim=1).transpose(1, 2)
        tokens = self.normalization(tokens + self.dropout(update))
        return tokens * mask.unsqueeze(-1).to(tokens.dtype)


class ProteinSequenceEncoder(nn.Module):
    """Encode proteins without discarding residue-level representations.

    Unlike the historical mean-pooling encoder, this encoder adds positional
    information, applies lightweight residual token convolutions, retains all
    contextual residue vectors, and uses learned attention pooling only for the
    optional sequence-level summary.
    """

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        max_length: int,
        embedding_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.padding_idx = int(padding_idx)
        self.max_length = int(max_length)
        self.residue_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.input_normalization = nn.LayerNorm(embedding_dim)
        self.blocks = nn.ModuleList(
            TokenConvBlock(embedding_dim, dropout) for _ in range(num_layers)
        )
        self.pool_score = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> ProteinEncoding:
        if tokens.ndim != 2:
            raise ValueError(
                f"protein tokens must have shape [batch, length], got {tuple(tokens.shape)}"
            )
        if tokens.size(1) < 1:
            raise ValueError("protein token length must be at least one")
        if tokens.size(1) > self.max_length:
            raise ValueError(
                f"protein token length {tokens.size(1)} exceeds configured "
                f"maximum {self.max_length}"
            )
        if mask is None:
            mask = tokens.ne(self.padding_idx)
        else:
            if mask.shape != tokens.shape:
                raise ValueError("protein mask must have the same shape as tokens")
            mask = mask.to(dtype=torch.bool, device=tokens.device)

        positions = torch.arange(tokens.size(1), device=tokens.device)
        hidden = self.residue_embedding(tokens.long())
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = self.input_normalization(hidden)
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, mask)

        scores = self.pool_score(hidden).squeeze(-1)
        minimum = torch.finfo(scores.dtype).min
        weights = torch.softmax(scores.masked_fill(~mask, minimum), dim=1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        return ProteinEncoding(
            token_embeddings=hidden,
            token_mask=mask,
            pooled=pooled,
            pool_weights=weights,
            input_mode="tokens",
        )


class PrecomputedProteinAdapter(nn.Module):
    """Small shared adapter for frozen ESM-style protein features."""

    def __init__(self, input_dim: int, adapter_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: Tensor) -> ProteinEncoding:
        if features.ndim != 2 or features.size(1) != self.input_dim:
            raise ValueError(
                "precomputed protein features must have shape "
                f"[batch, {self.input_dim}], got {tuple(features.shape)}"
            )
        # The feature bank is a frozen input.  Gradients still update this small
        # adapter but never propagate into an upstream ESM graph or feature tensor.
        pooled = self.network(features.detach())
        token_embeddings = pooled.unsqueeze(1)
        token_mask = torch.ones(
            (pooled.size(0), 1),
            dtype=torch.bool,
            device=pooled.device,
        )
        return ProteinEncoding(
            token_embeddings=token_embeddings,
            token_mask=token_mask,
            pooled=pooled,
            pool_weights=None,
            input_mode="precomputed_features",
        )


class EpisodicDistance(nn.Module):
    """One distance implementation shared by training and evaluation."""

    def __init__(
        self,
        name: DistanceName,
        temperature: float,
        learnable_temperature: bool,
    ) -> None:
        super().__init__()
        if name not in VALID_DISTANCES:
            raise ValueError(f"unsupported distance: {name}")
        self.name = name
        log_temperature = torch.tensor(float(temperature)).log()
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_temperature)
        else:
            self.register_buffer("log_temperature", log_temperature)

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(min=1e-4, max=1e4)

    @staticmethod
    def _validate_scale(scale: Tensor | None, values: Tensor, name: str) -> None:
        if scale is not None and scale.shape != values.shape:
            raise ValueError(
                f"{name} must have shape {tuple(values.shape)}, got {tuple(scale.shape)}"
            )

    def forward(
        self,
        query: Tensor,
        reference: Tensor,
        query_scale: Tensor | None = None,
        reference_scale: Tensor | None = None,
    ) -> Tensor:
        if query.ndim != 2 or reference.ndim != 2:
            raise ValueError("query and reference embeddings must be rank-2")
        if query.size(1) != reference.size(1):
            raise ValueError("query and reference embedding dimensions differ")
        self._validate_scale(query_scale, query, "query_scale")
        self._validate_scale(reference_scale, reference, "reference_scale")

        difference = query[:, None, :] - reference[None, :, :]
        if query_scale is not None or reference_scale is not None:
            q_scale = torch.ones_like(query) if query_scale is None else query_scale
            r_scale = (
                torch.ones_like(reference)
                if reference_scale is None
                else reference_scale
            )
            effective_scale = (
                q_scale[:, None, :].clamp_min(1e-8)
                * r_scale[None, :, :].clamp_min(1e-8)
            ).sqrt()
            difference = difference * effective_scale

        if self.name == "squared_euclidean":
            distance = difference.square().sum(dim=-1)
        elif self.name == "euclidean":
            distance = difference.square().sum(dim=-1).clamp_min(1e-12).sqrt()
        else:
            if query_scale is not None or reference_scale is not None:
                q_scale = torch.ones_like(query) if query_scale is None else query_scale
                r_scale = (
                    torch.ones_like(reference)
                    if reference_scale is None
                    else reference_scale
                )
                query = query * q_scale
                reference = reference * r_scale
            query_normalized = F.normalize(query, dim=-1)
            reference_normalized = F.normalize(reference, dim=-1)
            distance = 1.0 - query_normalized @ reference_normalized.transpose(0, 1)
        return distance


class PrototypeHead(nn.Module):
    """Class prototypes and logits using a single configured distance metric."""

    def __init__(self, distance: EpisodicDistance) -> None:
        super().__init__()
        self.distance = distance

    @staticmethod
    def _class_means(values: Tensor, labels: Tensor, classes: Tensor) -> Tensor:
        return torch.stack([values[labels.eq(label)].mean(dim=0) for label in classes])

    def forward(
        self,
        support_embedding: Tensor,
        support_labels: Tensor,
        query_embedding: Tensor,
        support_metric_scale: Tensor | None = None,
        query_metric_scale: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if support_embedding.ndim != 2 or query_embedding.ndim != 2:
            raise ValueError("support and query embeddings must be rank-2")
        if support_labels.ndim != 1 or support_labels.numel() != support_embedding.size(0):
            raise ValueError("support_labels must contain one label per support row")
        classes = torch.unique(support_labels, sorted=True)
        if classes.numel() < 2:
            raise ValueError("prototype scoring requires at least two support classes")
        prototypes = self._class_means(support_embedding, support_labels, classes)
        prototype_scale = None
        if support_metric_scale is not None:
            if support_metric_scale.shape != support_embedding.shape:
                raise ValueError("support_metric_scale shape differs from support embedding")
            prototype_scale = self._class_means(
                support_metric_scale,
                support_labels,
                classes,
            )
        distances = self.distance(
            query_embedding,
            prototypes,
            query_scale=query_metric_scale,
            reference_scale=prototype_scale,
        )
        logits = -distances / self.distance.temperature
        return {
            "logits": logits,
            "distances": distances,
            "prototypes": prototypes,
            "classes": classes,
            "temperature": self.distance.temperature,
        }


class E3V2Model(nn.Module):
    """Unified molecule/POI/E3 encoder with explicit E3 fusion variants.

    ``forward`` accepts either a PyG ``Data``/``Batch`` object or the standard
    ``x, edge_index, batch`` tensor triplet.  For each protein, callers must pass
    at most one of token IDs or frozen precomputed features.  E3-aware variants
    require one E3 input; POI inputs are optional and default to a zero context.

    The returned mapping always contains ``embedding``, ``metric_scale``,
    ``auxiliary_logits``, and ``diagnostics``.  Call :meth:`score_episode` with
    support/query mappings from this same forward path so training and evaluation
    cannot silently use different distance definitions.
    """

    def __init__(self, config: E3V2Config | None = None, **overrides: Any) -> None:
        super().__init__()
        if config is not None and overrides:
            raise ValueError("pass either E3V2Config or keyword overrides, not both")
        self.config = config if config is not None else E3V2Config(**overrides)
        self.config.validate()
        dimension = self.config.embedding_dim

        self.molecule_encoder = MoleculeEncoder(
            node_dim=self.config.node_dim,
            edge_dim=self.config.edge_dim,
            embedding_dim=dimension,
            num_layers=self.config.gnn_layers,
            dropout=self.config.dropout,
        )
        self.protein_encoder = ProteinSequenceEncoder(
            vocab_size=self.config.protein_vocab_size,
            padding_idx=self.config.protein_padding_idx,
            max_length=self.config.protein_max_length,
            embedding_dim=dimension,
            num_layers=self.config.protein_layers,
            dropout=self.config.dropout,
        )
        self.precomputed_protein_adapter = PrecomputedProteinAdapter(
            input_dim=self.config.pretrained_protein_dim,
            adapter_dim=self.config.protein_adapter_dim,
            output_dim=dimension,
        )
        # The repaired interaction receives an E3-specific adapter.  The legacy
        # variants deliberately retain the shared adapter so their historical
        # checkpoints and ablation semantics remain reproducible.
        if self.config.variant in {
            "grounded_interaction",
            "grounded_additive",
            "grounded_no_global",
            "grounded_no_component",
            "grounded_no_poi_context",
            "grounded_no_e3_context",
            "grounded_no_ligand_localization",
            "grounded_source_linear",
            "grounded_constant_e3",
            "grounded_poi_control",
        }:
            self.e3_protein_encoder = copy.deepcopy(self.protein_encoder)
            self.e3_precomputed_protein_adapter = copy.deepcopy(
                self.precomputed_protein_adapter
            )
        self.poi_projection = nn.Linear(dimension, dimension, bias=False)
        self.e3_projection = nn.Linear(dimension, dimension, bias=False)
        self.poi_scale = nn.Parameter(torch.tensor(1.0))
        self.e3_scale = nn.Parameter(torch.tensor(1.0))

        variant = self.config.variant
        if variant == "frozen_random_e3":
            self.random_e3_encoder = ProteinSequenceEncoder(
                vocab_size=self.config.protein_vocab_size,
                padding_idx=self.config.protein_padding_idx,
                max_length=self.config.protein_max_length,
                embedding_dim=dimension,
                num_layers=self.config.protein_layers,
                dropout=0.0,
            )
            self.random_e3_feature_adapter = PrecomputedProteinAdapter(
                input_dim=self.config.pretrained_protein_dim,
                adapter_dim=self.config.protein_adapter_dim,
                output_dim=dimension,
            )
            self.random_e3_projection = nn.Linear(dimension, dimension, bias=False)
            for module in (
                self.random_e3_encoder,
                self.random_e3_feature_adapter,
                self.random_e3_projection,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

        if variant == "film":
            self.film = nn.Sequential(
                nn.LayerNorm(2 * dimension),
                nn.Linear(2 * dimension, dimension),
                nn.GELU(),
                nn.Linear(dimension, 2 * dimension),
            )
        elif variant == "conditional_metric":
            self.metric_conditioner = nn.Sequential(
                nn.LayerNorm(2 * dimension),
                nn.Linear(2 * dimension, dimension),
                nn.GELU(),
                nn.Linear(dimension, dimension),
            )
        elif variant == "e3_atom_attention":
            self.atom_attention = nn.MultiheadAttention(
                embed_dim=dimension,
                num_heads=self.config.attention_heads,
                dropout=self.config.dropout,
                batch_first=True,
            )
            self.atom_attention_projection = nn.Sequential(
                nn.LayerNorm(dimension),
                nn.Linear(dimension, dimension, bias=False),
            )
        elif variant == "interaction":
            rank = self.config.interaction_rank
            self.molecule_low_rank = nn.Linear(dimension, rank, bias=False)
            self.poi_low_rank = nn.Linear(dimension, rank, bias=False)
            self.e3_low_rank = nn.Linear(dimension, rank, bias=False)
            self.interaction_projection = nn.Sequential(
                nn.LayerNorm(4 * rank),
                nn.Linear(4 * rank, dimension),
                nn.GELU(),
                nn.Linear(dimension, dimension),
            )
        elif variant in {
            "grounded_interaction",
            "grounded_additive",
            "grounded_no_global",
            "grounded_no_component",
            "grounded_no_poi_context",
            "grounded_no_e3_context",
            "grounded_no_ligand_localization",
            "grounded_source_linear",
            "grounded_constant_e3",
            "grounded_poi_control",
        }:
            rank = self.config.interaction_rank
            self.grounded_molecule_input_norm = nn.LayerNorm(dimension)
            self.grounded_poi_input_norm = nn.LayerNorm(dimension)
            self.grounded_e3_input_norm = nn.LayerNorm(dimension)
            self.grounded_recruiter_input_norm = nn.LayerNorm(dimension)
            self.grounded_molecule_low_rank = nn.Linear(
                dimension, rank, bias=False
            )
            self.grounded_poi_low_rank = nn.Linear(dimension, rank, bias=False)
            self.grounded_e3_low_rank = nn.Linear(dimension, rank, bias=False)
            self.grounded_recruiter_low_rank = nn.Linear(
                dimension, rank, bias=False
            )
            self.grounded_molecule_code_norm = nn.LayerNorm(rank)
            self.grounded_poi_code_norm = nn.LayerNorm(rank)
            self.grounded_e3_code_norm = nn.LayerNorm(rank)
            self.grounded_recruiter_code_norm = nn.LayerNorm(rank)
            self.grounded_interaction_projection = nn.Sequential(
                nn.LayerNorm(4 * rank),
                nn.Linear(4 * rank, dimension),
                nn.GELU(),
                nn.Linear(dimension, dimension),
            )
            self.grounded_recruiter_e3_projection = nn.Sequential(
                nn.LayerNorm(rank),
                nn.Linear(rank, dimension),
                nn.GELU(),
                nn.Linear(dimension, dimension),
            )
            self.grounded_compatibility_head = nn.Sequential(
                nn.LayerNorm(rank),
                nn.Linear(rank, 1),
            )
            if variant == "grounded_additive":
                hidden = 3 * rank
                self.grounded_additive_projection = nn.Sequential(
                    nn.LayerNorm(4 * dimension),
                    nn.Linear(4 * dimension, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, dimension),
                )
            if variant == "grounded_source_linear":
                self.source_classifier = nn.Linear(dimension, 2)

        self.prototype_head = PrototypeHead(
            EpisodicDistance(
                name=self.config.distance,
                temperature=self.config.temperature,
                learnable_temperature=self.config.learnable_temperature,
            )
        )
        self.auxiliary_head = (
            nn.Linear(dimension, self.config.num_auxiliary_outputs)
            if self.config.num_auxiliary_outputs > 0
            else None
        )

    @staticmethod
    def _graph_tensors(
        x: Tensor | Data,
        edge_index: Tensor | None,
        batch: Tensor | None,
        edge_attr: Tensor | None,
        e3_ligand_mask: Tensor | None,
        e3_ligand_match_ok: Tensor | None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        if isinstance(x, Data):
            graph = x
            if edge_index is not None or batch is not None:
                raise ValueError(
                    "when the first argument is a PyG Data/Batch, do not also pass "
                    "edge_index or batch"
                )
            edge_index = graph.edge_index
            batch = getattr(graph, "batch", None)
            if edge_attr is None:
                edge_attr = getattr(graph, "edge_attr", None)
            if e3_ligand_mask is None:
                e3_ligand_mask = getattr(graph, "e3_ligand_mask", None)
            if e3_ligand_match_ok is None:
                e3_ligand_match_ok = getattr(graph, "e3_ligand_match_ok", None)
            x = graph.x
        if edge_index is None:
            raise ValueError("edge_index is required")
        if x.ndim != 2:
            raise ValueError("node features must have shape [n_atoms, node_dim]")
        if x.size(0) < 1:
            raise ValueError("at least one atom is required")
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if batch.ndim != 1 or batch.numel() != x.size(0):
            raise ValueError("batch must contain one graph index per atom")
        return (
            x,
            edge_index,
            batch.long(),
            edge_attr,
            e3_ligand_mask,
            e3_ligand_match_ok,
        )

    @staticmethod
    def _validate_protein_sources(
        role: str,
        tokens: Tensor | None,
        features: Tensor | None,
    ) -> None:
        if tokens is not None and features is not None:
            raise ValueError(
                f"{role} received both tokens and precomputed features; pass only one"
            )

    def _protein_encoding(
        self,
        *,
        role: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        tokens: Tensor | None,
        token_mask: Tensor | None,
        features: Tensor | None,
        frozen_random: bool = False,
    ) -> ProteinEncoding:
        self._validate_protein_sources(role, tokens, features)
        if features is not None:
            if features.size(0) != batch_size:
                raise ValueError(f"{role} feature batch size differs from graph batch")
            if frozen_random:
                adapter = self.random_e3_feature_adapter
            elif role == "E3" and hasattr(self, "e3_precomputed_protein_adapter"):
                adapter = self.e3_precomputed_protein_adapter
            else:
                adapter = self.precomputed_protein_adapter
            return adapter(features)
        if tokens is not None:
            if tokens.size(0) != batch_size:
                raise ValueError(f"{role} token batch size differs from graph batch")
            if frozen_random:
                encoder = self.random_e3_encoder
            elif role == "E3" and hasattr(self, "e3_protein_encoder"):
                encoder = self.e3_protein_encoder
            else:
                encoder = self.protein_encoder
            return encoder(tokens, token_mask)

        zeros = torch.zeros(batch_size, self.config.embedding_dim, device=device, dtype=dtype)
        return ProteinEncoding(
            token_embeddings=zeros.unsqueeze(1),
            token_mask=torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            pooled=zeros,
            pool_weights=None,
            input_mode="none",
        )

    @staticmethod
    def _safe_attention_tokens(encoding: ProteinEncoding) -> tuple[Tensor, Tensor]:
        tokens = encoding.token_embeddings
        mask = encoding.token_mask
        empty = ~mask.any(dim=1)
        if empty.any():
            tokens = tokens.clone()
            mask = mask.clone()
            tokens[empty, 0] = 0.0
            mask[empty, 0] = True
        return tokens, mask

    @staticmethod
    def _dense_atom_selection(
        e3_ligand_mask: Tensor | None,
        atom_mask: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        provided = e3_ligand_mask is not None
        if e3_ligand_mask is None:
            selected = atom_mask
        elif e3_ligand_mask.ndim == 1:
            if e3_ligand_mask.numel() != batch.numel():
                raise ValueError(
                    "flat e3_ligand_mask must contain one value per graph atom"
                )
            dense_mask, _ = to_dense_batch(
                e3_ligand_mask.to(
                    device=batch.device,
                    dtype=torch.bool,
                ).unsqueeze(-1),
                batch,
                max_num_nodes=atom_mask.size(1),
            )
            selected = dense_mask.squeeze(-1) & atom_mask
        elif e3_ligand_mask.shape == atom_mask.shape:
            selected = e3_ligand_mask.to(dtype=torch.bool, device=atom_mask.device) & atom_mask
        else:
            raise ValueError(
                "e3_ligand_mask must be flat [n_atoms] or dense [batch, max_atoms]"
            )
        totals = atom_mask.sum(dim=1)
        selected_counts = selected.sum(dim=1)
        coverage = selected_counts.to(torch.float32) / totals.clamp_min(1).to(torch.float32)
        provided_flag = torch.tensor(
            provided,
            dtype=torch.bool,
            device=atom_mask.device,
        )
        return selected, selected_counts, totals, coverage, provided_flag

    @staticmethod
    def _component_validity(
        match_ok: Tensor | None,
        selected_counts: Tensor,
        batch_size: int,
        device: torch.device,
        mask_provided: bool,
    ) -> Tensor:
        """Return one audited component-valid flag per molecular graph.

        Direct model callers that provide a non-empty mask but no audit flag
        are treated as explicitly validated.  The experiment runner always
        supplies the audit flag so its all-atom fallback can never be mistaken
        for a verified E3-ligand annotation.
        """

        if match_ok is None:
            if not mask_provided:
                return torch.zeros(batch_size, dtype=torch.bool, device=device)
            return selected_counts.gt(0)
        valid = torch.as_tensor(match_ok, dtype=torch.bool, device=device).reshape(-1)
        if valid.numel() == 1 and batch_size != 1:
            valid = valid.expand(batch_size)
        if valid.numel() != batch_size:
            raise ValueError(
                "e3_ligand_match_ok must contain one boolean per graph; "
                f"expected {batch_size}, got {valid.numel()}"
            )
        return valid

    @staticmethod
    def _grounded_code(raw: Tensor, normalization: nn.Module) -> Tensor:
        # Unlike tanh, this path cannot collapse large positive/negative inputs
        # to the same +/-1 corner.  Unit normalization also prevents a protein
        # adapter from winning purely through scale.
        return F.normalize(F.silu(normalization(raw)), dim=-1, eps=1e-6)

    def forward(
        self,
        x: Tensor | Data,
        edge_index: Tensor | None = None,
        batch: Tensor | None = None,
        edge_attr: Tensor | None = None,
        *,
        poi_tokens: Tensor | None = None,
        e3_tokens: Tensor | None = None,
        poi_mask: Tensor | None = None,
        e3_mask: Tensor | None = None,
        poi_features: Tensor | None = None,
        e3_features: Tensor | None = None,
        e3_ligand_mask: Tensor | None = None,
        e3_ligand_match_ok: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> dict[str, Any]:
        (
            x,
            edge_index,
            batch,
            edge_attr,
            e3_ligand_mask,
            e3_ligand_match_ok,
        ) = self._graph_tensors(
            x,
            edge_index,
            batch,
            edge_attr,
            e3_ligand_mask,
            e3_ligand_match_ok,
        )
        if x.size(1) != self.config.node_dim:
            raise ValueError(
                f"expected {self.config.node_dim} node features, got {x.size(1)}"
            )
        molecule = self.molecule_encoder(x, edge_index, batch, edge_attr=edge_attr)
        batch_size = molecule.graph_embedding.size(0)
        poi = self._protein_encoding(
            role="POI",
            batch_size=batch_size,
            device=molecule.graph_embedding.device,
            dtype=molecule.graph_embedding.dtype,
            tokens=poi_tokens,
            token_mask=poi_mask,
            features=poi_features,
        )
        variant = self.config.variant
        poi_context = self.poi_scale * self.poi_projection(poi.pooled)
        if variant == "grounded_no_poi_context":
            poi_context = torch.zeros_like(poi_context)
        base_embedding = molecule.graph_embedding + poi_context

        e3: ProteinEncoding | None = None
        e3_context: Tensor | None = None
        metric_scale: Tensor | None = None
        attention_weights: Tensor | None = None
        mask_diagnostics: dict[str, Any] = {}
        compatibility_logit: Tensor | None = None
        component_valid = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=molecule.graph_embedding.device,
        )
        grounded_diagnostics: dict[str, Any] = {}

        if variant in {"grounded_constant_e3", "grounded_poi_control"}:
            # Capacity-matched negative control: keep the complete grounded E3
            # pathway active, but replace E3 identity/sequence with one shared
            # constant feature.  The experiment runner supplies the source-E3
            # feature (which is training-side and in-distribution); direct model
            # callers fall back to an all-zero constant.  Every interaction
            # branch remains trainable while biological E3 identity is absent.
            if e3_features is None:
                constant_e3_features = torch.zeros(
                    batch_size,
                    self.config.pretrained_protein_dim,
                    device=molecule.graph_embedding.device,
                    dtype=molecule.graph_embedding.dtype,
                )
            else:
                if e3_features.size(0) != batch_size:
                    raise ValueError("E3 control feature batch size differs from graph batch")
                reference = e3_features[:1].expand_as(e3_features)
                if not torch.equal(e3_features, reference):
                    raise ValueError(
                        f"{variant} requires one constant E3 feature "
                        "repeated across the batch"
                    )
                constant_e3_features = e3_features
            e3 = self._protein_encoding(
                role="E3",
                batch_size=batch_size,
                device=molecule.graph_embedding.device,
                dtype=molecule.graph_embedding.dtype,
                tokens=None,
                token_mask=None,
                features=constant_e3_features,
            )
            e3_context = self.e3_scale * self.e3_projection(e3.pooled)
        elif variant != "no_e3":
            if e3_tokens is None and e3_features is None:
                raise ValueError(f"variant {variant!r} requires E3 tokens or features")
            e3 = self._protein_encoding(
                role="E3",
                batch_size=batch_size,
                device=molecule.graph_embedding.device,
                dtype=molecule.graph_embedding.dtype,
                tokens=e3_tokens,
                token_mask=e3_mask,
                features=e3_features,
                frozen_random=(variant == "frozen_random_e3"),
            )
            if variant == "frozen_random_e3":
                e3_context = self.random_e3_projection(e3.pooled)
            else:
                e3_context = self.e3_scale * self.e3_projection(e3.pooled)
            if variant == "grounded_no_e3_context":
                e3_context = torch.zeros_like(e3_context)

        if variant == "no_e3":
            embedding = base_embedding
        elif variant in {"additive", "frozen_random_e3"}:
            embedding = base_embedding + e3_context
        elif variant == "film":
            condition = torch.cat([poi_context, e3_context], dim=-1)
            gamma, beta = self.film(condition).chunk(2, dim=-1)
            embedding = (
                molecule.graph_embedding * (1.0 + 0.5 * torch.tanh(gamma))
                + beta
                + poi_context
            )
        elif variant == "conditional_metric":
            embedding = base_embedding
            raw_scale = self.metric_conditioner(
                torch.cat([poi_context, e3_context], dim=-1)
            )
            metric_scale = F.softplus(raw_scale) + 1e-4
            metric_scale = metric_scale / metric_scale.mean(dim=-1, keepdim=True).clamp_min(
                1e-6
            )
        elif variant == "e3_atom_attention":
            dense_atoms, atom_mask = to_dense_batch(
                molecule.atom_embeddings,
                batch,
                batch_size=batch_size,
            )
            e3_attention_tokens, e3_attention_mask = self._safe_attention_tokens(e3)
            (
                selected_atoms,
                selected_counts,
                total_counts,
                coverage,
                mask_provided,
            ) = self._dense_atom_selection(e3_ligand_mask, atom_mask, batch)
            component_valid = self._component_validity(
                e3_ligand_match_ok,
                selected_counts,
                batch_size,
                atom_mask.device,
                bool(mask_provided.item()),
            )
            # MultiheadAttention cannot accept a row whose every key is masked.
            # Give such rows one zero-valued sentinel atom, then explicitly zero
            # their interaction below so there is no silent all-atom fallback.
            safe_atoms = dense_atoms
            safe_selection = selected_atoms
            empty_selection = ~selected_atoms.any(dim=1)
            if empty_selection.any():
                safe_atoms = dense_atoms.clone()
                safe_selection = selected_atoms.clone()
                safe_atoms[empty_selection, 0] = 0.0
                safe_selection[empty_selection, 0] = True
            attended_atoms, attention_weights = self.atom_attention(
                query=e3_attention_tokens,
                key=safe_atoms,
                value=safe_atoms,
                key_padding_mask=~safe_selection,
                need_weights=return_diagnostics,
                average_attn_weights=True,
            )
            # Retain an explicit E3-by-molecule interaction.  This remains
            # non-additive when the ESM route provides one summary token.
            token_interaction = e3_attention_tokens * attended_atoms
            token_weights = e3_attention_mask.unsqueeze(-1).to(
                token_interaction.dtype
            )
            pooled_interaction = (token_interaction * token_weights).sum(dim=1)
            pooled_interaction = pooled_interaction / token_weights.sum(
                dim=1
            ).clamp_min(1.0)
            # A graph with an explicitly empty ligand mask contributes no
            # interaction rather than silently falling back to all atoms.
            pooled_interaction = pooled_interaction * selected_counts.gt(0).unsqueeze(-1)
            embedding = base_embedding + self.atom_attention_projection(pooled_interaction)
            mask_diagnostics = {
                "atom_attention_scope": (
                    "validated_ligand_else_explicit_fallback"
                    if e3_ligand_match_ok is not None
                    else (
                        "provided_mask" if bool(mask_provided.item()) else "all_atoms"
                    )
                ),
                "e3_ligand_mask_provided": mask_provided,
                "e3_ligand_match_ok": component_valid,
                "e3_ligand_fallback_used": ~component_valid,
                "e3_ligand_mask_selected_atoms": selected_counts,
                "e3_ligand_mask_total_atoms": total_counts,
                "e3_ligand_mask_coverage": coverage,
                "atom_attention_selection_mask": selected_atoms,
            }
        elif variant == "interaction":
            molecule_low = torch.tanh(
                self.molecule_low_rank(molecule.graph_embedding)
            )
            poi_low = torch.tanh(self.poi_low_rank(poi_context))
            e3_low = torch.tanh(self.e3_low_rank(e3_context))
            interactions = torch.cat(
                [
                    molecule_low * poi_low,
                    molecule_low * e3_low,
                    poi_low * e3_low,
                    molecule_low * poi_low * e3_low,
                ],
                dim=-1,
            )
            embedding = base_embedding + self.interaction_projection(interactions)
        elif variant in {
            "grounded_interaction",
            "grounded_additive",
            "grounded_no_global",
            "grounded_no_component",
            "grounded_no_poi_context",
            "grounded_no_e3_context",
            "grounded_no_ligand_localization",
            "grounded_source_linear",
            "grounded_constant_e3",
            "grounded_poi_control",
        }:
            molecule_raw = self.grounded_molecule_low_rank(
                self.grounded_molecule_input_norm(molecule.graph_embedding)
            )
            if variant == "grounded_no_poi_context":
                poi_raw = torch.zeros_like(molecule_raw)
            else:
                poi_raw = self.grounded_poi_low_rank(
                    self.grounded_poi_input_norm(poi_context)
                )
            molecule_low = self._grounded_code(
                molecule_raw, self.grounded_molecule_code_norm
            )
            poi_low = (
                torch.zeros_like(molecule_low)
                if variant == "grounded_no_poi_context"
                else self._grounded_code(poi_raw, self.grounded_poi_code_norm)
            )
            if variant == "grounded_no_e3_context":
                e3_raw = torch.zeros_like(molecule_raw)
                e3_low = torch.zeros_like(molecule_low)
            else:
                e3_raw = self.grounded_e3_low_rank(
                    self.grounded_e3_input_norm(e3_context)
                )
                e3_low = self._grounded_code(e3_raw, self.grounded_e3_code_norm)
            global_interactions = torch.cat(
                [
                    molecule_low * poi_low,
                    molecule_low * e3_low,
                    poi_low * e3_low,
                    molecule_low * poi_low * e3_low,
                ],
                dim=-1,
            )

            global_residual = self.grounded_interaction_projection(
                global_interactions
            )
            dense_atoms, atom_mask = to_dense_batch(
                molecule.atom_embeddings,
                batch,
                batch_size=batch_size,
            )
            (
                selected_atoms,
                selected_counts,
                total_counts,
                coverage,
                mask_provided,
            ) = self._dense_atom_selection(e3_ligand_mask, atom_mask, batch)
            component_valid = self._component_validity(
                e3_ligand_match_ok,
                selected_counts,
                batch_size,
                atom_mask.device,
                bool(mask_provided.item()),
            )
            recruiter_selection = (
                atom_mask
                if variant == "grounded_no_ligand_localization"
                else selected_atoms
            )
            selection_weights = recruiter_selection.unsqueeze(-1).to(dense_atoms.dtype)
            recruiter = (dense_atoms * selection_weights).sum(dim=1)
            recruiter = recruiter / selection_weights.sum(dim=1).clamp_min(1.0)
            recruiter_raw = self.grounded_recruiter_low_rank(
                self.grounded_recruiter_input_norm(recruiter)
            )
            recruiter_low = self._grounded_code(
                recruiter_raw, self.grounded_recruiter_code_norm
            )
            recruiter_e3 = recruiter_low * e3_low
            validity = component_valid.unsqueeze(-1).to(recruiter_e3.dtype)
            if variant == "grounded_no_e3_context":
                component_residual = torch.zeros_like(base_embedding)
                compatibility_logit = base_embedding.new_zeros(batch_size)
            else:
                component_residual = self.grounded_recruiter_e3_projection(
                    recruiter_e3
                ) * validity
                compatibility_logit = self.grounded_compatibility_head(
                    recruiter_e3
                ).squeeze(-1)
                compatibility_logit = compatibility_logit * component_valid.to(
                    compatibility_logit.dtype
                )
            if variant == "grounded_no_global":
                embedding = base_embedding + component_residual
            elif variant == "grounded_no_component":
                embedding = base_embedding + global_residual
            elif variant == "grounded_additive":
                recruiter_context = recruiter * validity
                additive_inputs = torch.cat(
                    [molecule.graph_embedding, poi_context, e3_context, recruiter_context],
                    dim=-1,
                )
                embedding = base_embedding + self.grounded_additive_projection(
                    additive_inputs
                )
            else:
                embedding = base_embedding + global_residual + component_residual
            mask_diagnostics = {
                "component_scope": (
                    "whole_molecule_on_verified_component_rows"
                    if variant == "grounded_no_ligand_localization"
                    else "verified_e3_ligand_only"
                ),
                "e3_ligand_mask_provided": mask_provided,
                "e3_ligand_match_ok": component_valid,
                "e3_ligand_fallback_used": ~component_valid,
                "e3_ligand_mask_selected_atoms": selected_counts,
                "e3_ligand_mask_total_atoms": total_counts,
                "e3_ligand_mask_coverage": coverage,
            }
            grounded_diagnostics = {
                "grounded_molecule_raw": molecule_raw,
                "grounded_poi_raw": poi_raw,
                "grounded_e3_raw": e3_raw,
                "grounded_recruiter_raw": recruiter_raw,
                "grounded_molecule_code": molecule_low,
                "grounded_poi_code": poi_low,
                "grounded_e3_code": e3_low,
                "grounded_recruiter_code": recruiter_low,
                "grounded_component_residual": component_residual,
            }
        else:  # pragma: no cover - guarded by E3V2Config.validate
            raise RuntimeError(f"unhandled E3-v2 variant: {variant}")

        auxiliary_logits = (
            self.auxiliary_head(embedding) if self.auxiliary_head is not None else None
        )
        diagnostics: dict[str, Any] = {}
        if return_diagnostics:
            diagnostics = {
                "variant": variant,
                "molecule_embedding": molecule.graph_embedding,
                "bond_features_used": edge_attr is not None,
                "configured_edge_dim": self.config.edge_dim,
                "poi_embedding": poi.pooled,
                "poi_input_mode": poi.input_mode,
                "poi_pool_weights": poi.pool_weights,
                "e3_embedding": None if e3 is None else e3.pooled,
                "e3_input_mode": "ignored" if e3 is None else e3.input_mode,
                "e3_pool_weights": None if e3 is None else e3.pool_weights,
                "atom_attention_weights": attention_weights,
                **mask_diagnostics,
                **grounded_diagnostics,
            }
        return {
            "embedding": embedding,
            "metric_scale": metric_scale,
            "auxiliary_logits": auxiliary_logits,
            "compatibility_logit": compatibility_logit,
            "component_valid": component_valid,
            "diagnostics": diagnostics,
        }

    def score_episode(
        self,
        support_output: Mapping[str, Any],
        support_labels: Tensor,
        query_output: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        """Score an episode with the exact metric used by this model.

        Both training and evaluation should call this method rather than
        reimplementing distance/prototype calculations in separate runners.
        """

        if self.config.variant == "grounded_source_linear":
            query_embedding = query_output["embedding"]
            logits = self.source_classifier(query_embedding)
            classes = torch.tensor([0, 1], device=logits.device, dtype=torch.long)
            prototypes = query_embedding.new_zeros((2, query_embedding.size(1)))
            return {
                "logits": logits,
                "distances": -logits,
                "prototypes": prototypes,
                "classes": classes,
                "temperature": query_embedding.new_tensor(1.0),
            }
        return self.prototype_head(
            support_embedding=support_output["embedding"],
            support_labels=support_labels.long(),
            query_embedding=query_output["embedding"],
            support_metric_scale=support_output.get("metric_scale"),
            query_metric_scale=query_output.get("metric_scale"),
        )


__all__ = [
    "E3V2Config",
    "E3V2Model",
    "EpisodicDistance",
    "MoleculeEncoder",
    "PrecomputedProteinAdapter",
    "ProteinSequenceEncoder",
    "PrototypeHead",
    "VALID_VARIANTS",
]
