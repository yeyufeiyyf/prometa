"""Data loading and episodic split helpers for ProMeta."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data

from .model import sequence_to_tensor


ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]


def atom_features(atom: rdchem.Atom) -> list[float]:
    atomic_number = atom.GetAtomicNum()
    return [1.0 if atomic_number == value else 0.0 for value in ATOM_LIST]


def mol_to_graph(mol: rdchem.Mol | None) -> Data | None:
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edges: list[list[int]] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[begin, end], [end, begin]])
    if not edges:
        edges = [[0, 0]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def graph_index_from_sdf(path: str | Path) -> dict[str, Data]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    index: dict[str, Data] = {}
    for mol in supplier:
        if mol is None or not mol.HasProp("_Name"):
            continue
        graph = mol_to_graph(mol)
        if graph is not None:
            index[str(mol.GetProp("_Name")).strip()] = graph
    return index


def save_graph_cache(index: dict[str, Data], path: str | Path) -> None:
    """Save exact SDF-derived graphs as a compact, non-pickled NPZ cache."""
    graph_ids = sorted(index)
    node_offsets = [0]
    edge_offsets = [0]
    node_blocks: list[np.ndarray] = []
    edge_blocks: list[np.ndarray] = []
    for compound_id in graph_ids:
        graph = index[compound_id]
        nodes = graph.x.detach().cpu().numpy().astype(np.float32, copy=False)
        edges = graph.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
        node_blocks.append(nodes)
        edge_blocks.append(edges)
        node_offsets.append(node_offsets[-1] + nodes.shape[0])
        edge_offsets.append(edge_offsets[-1] + edges.shape[1])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        graph_ids=np.asarray(graph_ids, dtype=str),
        node_offsets=np.asarray(node_offsets, dtype=np.int64),
        edge_offsets=np.asarray(edge_offsets, dtype=np.int64),
        node_features=np.concatenate(node_blocks, axis=0),
        edge_indices=np.concatenate(edge_blocks, axis=1),
    )


def graph_index_from_cache(path: str | Path) -> dict[str, Data]:
    """Load the compact graph cache created by :func:`save_graph_cache`."""
    with np.load(path, allow_pickle=False) as cache:
        graph_ids = cache["graph_ids"].astype(str)
        node_offsets = cache["node_offsets"]
        edge_offsets = cache["edge_offsets"]
        node_features = cache["node_features"]
        edge_indices = cache["edge_indices"]
        index: dict[str, Data] = {}
        for position, compound_id in enumerate(graph_ids):
            node_start, node_end = node_offsets[position : position + 2]
            edge_start, edge_end = edge_offsets[position : position + 2]
            index[str(compound_id)] = Data(
                x=torch.from_numpy(node_features[node_start:node_end].copy()),
                edge_index=torch.from_numpy(edge_indices[:, edge_start:edge_end].copy()),
            )
    return index


def read_contexts(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["Compound ID"] = frame["Compound ID"].astype(str)
    frame["Label_bin"] = frame["Label_bin"].astype(int)
    return frame


def protein_index_from_csv(
    path: str | Path,
    sequence_column: str,
    max_length: int = 2000,
) -> dict[str, torch.LongTensor]:
    """Build a compound-ID keyed protein-sequence index.

    The sequence table is kept separate from the evaluation context tables so
    that the latter remain compact and their historical split hashes remain
    unchanged.
    """
    frame = pd.read_csv(path, usecols=["Compound ID", sequence_column], low_memory=False)
    frame["Compound ID"] = frame["Compound ID"].astype(str)
    index: dict[str, torch.LongTensor] = {}
    for _, row in frame.iterrows():
        value = row[sequence_column]
        if pd.isna(value):
            continue
        sequence = str(value).strip()
        if not sequence or sequence.lower() in {"nan", "none"}:
            continue
        index[str(row["Compound ID"])] = sequence_to_tensor(sequence, max_length=max_length)
    return index


def example_id(example: Any) -> str:
    if isinstance(example, dict):
        for key in ["Compound ID", "compound_id", "cid", "id"]:
            if key in example:
                return str(example[key])
    return str(example)


def example_label(example: Any) -> int | None:
    if not isinstance(example, dict):
        return None
    for key in ["Label_bin", "label", "Label", "y_true", "y"]:
        if key in example and example[key] is not None:
            return int(float(example[key]))
    return None


def iter_episodes(
    split: dict[str, Any],
    phase: str = "meta_test",
) -> Iterator[tuple[str, int, list[Any], list[Any]]]:
    if "support_ids" in split and "query_ids" in split:
        yield "task", 0, split["support_ids"], split["query_ids"]
        return
    for task, episodes in split.get(phase, {}).items():
        for episode_index, episode in enumerate(episodes):
            yield str(task), episode_index, episode.get("support", []), episode.get("query", [])


def read_split(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def graphs_for_examples(
    examples: list[Any],
    graph_index: dict[str, Data],
    fallback_labels: dict[str, int] | None = None,
) -> tuple[list[Data], list[str], np.ndarray]:
    graphs: list[Data] = []
    compound_ids: list[str] = []
    labels: list[int] = []
    fallback_labels = fallback_labels or {}
    for example in examples:
        compound_id = example_id(example)
        label = example_label(example)
        if label is None:
            label = fallback_labels.get(compound_id)
        if compound_id not in graph_index or label is None:
            continue
        graph = graph_index[compound_id].clone()
        graph.cid = compound_id
        graph.y = torch.tensor([int(label)], dtype=torch.long)
        graphs.append(graph)
        compound_ids.append(compound_id)
        labels.append(int(label))
    return graphs, compound_ids, np.asarray(labels, dtype=np.int64)
