"""
data_utils.py
=============
Utilities for molecular graph construction, dataset loading,
and episodic few-shot data sampling.

Key functions
-------------
- build_graph_index   : Load an SDF file into a compound-id → PyG graph dict.
- build_label_map     : Build a compound-id → binary label dict from a CSV.
- build_crbn_tasks    : Group CRBN compounds by POI target for meta-training.
- sample_episode      : Sample a K-shot N-way episode from a task DataFrame.
- make_data_list      : Retrieve PyG graphs for a list of compound IDs.
"""

import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Batch, Data


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C N O F P S Cl Br I


# ---------------------------------------------------------------------------
# Molecular graph construction
# ---------------------------------------------------------------------------

def _atom_features(atom: rdchem.Atom) -> List[float]:
    """9-dimensional one-hot atom-type encoding."""
    z = atom.GetAtomicNum()
    return [1.0 if z == x else 0.0 for x in ATOM_LIST]


def mol_to_graph(mol: rdchem.Mol) -> Optional[Data]:
    """Convert an RDKit molecule to a PyG Data object.

    Nodes correspond to heavy atoms; edges correspond to covalent bonds
    (undirected, stored as two directed edges).

    Args:
        mol: RDKit molecule.

    Returns:
        PyG ``Data`` with attributes ``x`` (node features) and
        ``edge_index`` (connectivity), or ``None`` for invalid molecules.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor(
        [_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float
    )

    edge_list = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_list += [[i, j], [j, i]]
    if not edge_list:
        edge_list = [[0, 0]]  # self-loop for isolated atoms

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def build_graph_index(sdf_path: str) -> Dict[str, Data]:
    """Load an SDF file and build a compound-id → PyG graph mapping.

    The compound ID is read from the ``_Name`` property of each molecule
    (first line of the MOL block), which corresponds to the ``Compound ID``
    column in PROTAC-DB exports.

    Args:
        sdf_path: Path to the SDF file.

    Returns:
        Dict mapping compound ID strings to PyG Data objects.
    """
    print(f"Loading SDF: {sdf_path}")
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    graph_index: Dict[str, Data] = {}
    ok = fail = 0

    for mol in suppl:
        if mol is None or not mol.HasProp("_Name"):
            fail += 1
            continue
        cid = str(mol.GetProp("_Name")).strip()
        g = mol_to_graph(mol)
        if g is None:
            fail += 1
            continue
        graph_index[cid] = g
        ok += 1

    print(f"Graph index built: {ok} molecules | {fail} failed")
    return graph_index


# ---------------------------------------------------------------------------
# Label and dataset utilities
# ---------------------------------------------------------------------------

def build_label_map(df: pd.DataFrame) -> Dict[str, int]:
    """Build a compound-id → binary activity label mapping.

    Expects the DataFrame to contain a ``Compound ID`` column and either a
    ``Label_bin`` or ``Label`` column (values: 0 = inactive, 1 = active).

    Args:
        df: PROTAC dataset DataFrame.

    Returns:
        Dict mapping compound ID strings to integer labels {0, 1}.
    """
    if "Label_bin" not in df.columns:
        if "Label" in df.columns:
            df = df.copy()
            df["Label_bin"] = df["Label"].astype(int)
        else:
            raise ValueError("DataFrame must contain 'Label_bin' or 'Label' column.")

    return {
        str(row["Compound ID"]): int(row["Label_bin"])
        for _, row in df.iterrows()
        if not pd.isna(row.get("Label_bin"))
    }


def make_data_list(
    cids: List[str],
    graph_index: Dict[str, Data],
    label_map: Dict[str, int],
) -> List[Data]:
    """Retrieve PyG graphs with labels for a list of compound IDs.

    Compounds missing from either ``graph_index`` or ``label_map`` are silently
    skipped.

    Args:
        cids:        List of compound ID strings.
        graph_index: Compound-id → PyG graph mapping.
        label_map:   Compound-id → binary label mapping.

    Returns:
        List of PyG Data objects with ``y`` attribute set.
    """
    out = []
    for cid in [str(c) for c in cids]:
        if cid in graph_index and cid in label_map:
            g = graph_index[cid].clone()
            g.y = torch.tensor([label_map[cid]], dtype=torch.long)
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# CRBN task construction for meta-training
# ---------------------------------------------------------------------------

def build_crbn_tasks(
    df: pd.DataFrame,
    graph_index: Dict[str, Data],
    K: int,
    Q: int,
) -> Dict[str, pd.DataFrame]:
    """Group CRBN compounds by POI target for episodic meta-training.

    Only targets with at least K+Q active and K+Q inactive compounds
    (both present in graph_index) are retained.

    Args:
        df:          Full PROTAC dataset DataFrame.
        graph_index: Compound-id → PyG graph mapping.
        K:           Number of support samples per class.
        Q:           Number of query samples per class.

    Returns:
        Dict mapping POI target name to its subset DataFrame.
    """
    if "E3 ligase" not in df.columns or "Target" not in df.columns:
        raise ValueError("DataFrame must contain 'E3 ligase' and 'Target' columns.")

    df_crbn = df[
        (df["E3 ligase"] == "CRBN") &
        (df["Compound ID"].astype(str).isin(graph_index))
    ].copy()

    tasks: Dict[str, pd.DataFrame] = {}
    for tgt, sub in df_crbn.groupby("Target"):
        sub = sub.reset_index(drop=True)
        n_pos = int((sub["Label_bin"] == 1).sum())
        n_neg = int((sub["Label_bin"] == 0).sum())
        if n_pos >= K + Q and n_neg >= K + Q:
            tasks[str(tgt)] = sub

    print(f"CRBN meta-training tasks: {len(tasks)}")
    return tasks


# ---------------------------------------------------------------------------
# Episodic sampling
# ---------------------------------------------------------------------------

def sample_episode(
    df_target: pd.DataFrame,
    graph_index: Dict[str, Data],
    K: int,
    Q: int,
    device: torch.device,
    allow_replacement: bool = False,
) -> Optional[Tuple[List[Data], torch.Tensor, List[Data], torch.Tensor]]:
    """Sample a 2-way K-shot episode from a single POI-target subset.

    Args:
        df_target:         DataFrame for one POI target (contains Label_bin).
        graph_index:       Compound-id → PyG graph mapping.
        K:                 Support samples per class.
        Q:                 Query samples per class.
        device:            Torch device for label tensors.
        allow_replacement: If True, sample with replacement when samples are
                           scarce (used for bootstrap query sets).

    Returns:
        Tuple ``(support_data, support_labels, query_data, query_labels)``
        where labels are float32 tensors on ``device``, or ``None`` if the
        episode cannot be constructed.
    """
    df_pos = df_target[df_target["Label_bin"] == 1]
    df_neg = df_target[df_target["Label_bin"] == 0]
    need = K + Q

    if len(df_pos) == 0 or len(df_neg) == 0:
        return None

    rep_pos = allow_replacement and len(df_pos) < need
    rep_neg = allow_replacement and len(df_neg) < need
    if not allow_replacement and (len(df_pos) < need or len(df_neg) < need):
        return None

    pos = df_pos.sample(n=need, replace=rep_pos)
    neg = df_neg.sample(n=need, replace=rep_neg)

    sup_rows = pd.concat([pos.iloc[:K], neg.iloc[:K]])
    qry_rows = pd.concat([pos.iloc[K:], neg.iloc[K:]])

    def _to_tensors(rows):
        graphs, labels = [], []
        for _, r in rows.iterrows():
            cid = str(r["Compound ID"])
            if cid not in graph_index:
                return None, None
            graphs.append(graph_index[cid].clone())
            labels.append(int(r["Label_bin"]))
        return graphs, torch.tensor(labels, dtype=torch.float32, device=device)

    sup_data, sup_labels = _to_tensors(sup_rows)
    qry_data, qry_labels = _to_tensors(qry_rows)

    if sup_data is None or qry_data is None:
        return None

    return sup_data, sup_labels, qry_data, qry_labels


def encode_batch(
    encoder: torch.nn.Module,
    data_list: List[Data],
    device: torch.device,
) -> torch.Tensor:
    """Encode a list of PyG graphs into a batch embedding matrix.

    Args:
        encoder:   GNN encoder module.
        data_list: List of PyG Data objects.
        device:    Target device.

    Returns:
        Embedding matrix of shape ``(len(data_list), embedding_dim)``.
    """
    batch = Batch.from_data_list(data_list).to(device)
    return encoder(batch.x, batch.edge_index, batch.batch)
