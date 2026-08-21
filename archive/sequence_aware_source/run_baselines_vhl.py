#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_baselines_vhl.py  (+ protein sequence fusion)

在原有分子图编码基础上融合 POI 蛋白质序列信息：
  z_fused = z_mol + z_prot   （element-wise addition，维度均为 out_dim）

蛋白质编码器 (ProteinEncoder)：
  - 对氨基酸字符做可学习 embedding（20 种标准 AA + padding）
  - 全序列 mean-pool → Linear → out_dim

蛋白质索引通过 --seq-csv 指定的 CSV 文件构建，
该文件需含 "Compound ID" 列和蛋白质序列列（由 --seq-col 指定，默认 "POI_seq"）。
"""

import os
import copy
import json
import math
import argparse
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import rdchem, AllChem

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


# =========================================================
# Seed
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# RDKit -> PyG
# =========================================================
ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C N O F P S Cl Br I


def atom_to_feature_vector_protonet(atom: rdchem.Atom):
    z = atom.GetAtomicNum()
    return [1.0 if z == x else 0.0 for x in ATOM_LIST]


def mol_to_graph(mol: rdchem.Mol) -> Optional[Data]:
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor(
        [atom_to_feature_vector_protonet(a) for a in mol.GetAtoms()],
        dtype=torch.float,
    )
    edge_list = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_list += [[i, j], [j, i]]
    if not edge_list:
        edge_list = [[0, 0]]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def build_graph_index_from_sdf(sdf_path: str) -> Dict[str, Data]:
    print(f"📦 Loading SDF: {sdf_path}")
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    graph_index, ok, fail = {}, 0, 0
    for mol in suppl:
        if mol is None or not mol.HasProp("_Name"):
            fail += 1; continue
        cid = str(mol.GetProp("_Name")).strip()
        g = mol_to_graph(mol)
        if g is None:
            fail += 1; continue
        graph_index[cid] = g
        ok += 1
    print(f"✅ Graph index: {ok} ok | {fail} failed")
    return graph_index


# =========================================================
# ECFP4 fingerprint helpers
# =========================================================
def smiles_to_ecfp4(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(list(fp), dtype=np.float32)


def build_fp_index(df: pd.DataFrame, n_bits: int = 2048) -> Dict[str, np.ndarray]:
    smiles_col = None
    for c in ["SMILES", "smiles", "Smiles", "canonical_smiles"]:
        if c in df.columns:
            smiles_col = c; break
    if smiles_col is None:
        raise ValueError("CSV 中未找到 SMILES 列（SMILES/smiles/Smiles）")

    fp_index, ok, fail = {}, 0, 0
    for _, row in df.iterrows():
        cid = str(row["Compound ID"])
        fp  = smiles_to_ecfp4(str(row[smiles_col]), n_bits=n_bits)
        if fp is None:
            fail += 1; continue
        fp_index[cid] = fp
        ok += 1
    print(f"✅ ECFP4 index: {ok} ok | {fail} failed")
    return fp_index


def gather_fp(cids, fp_index, label_map):
    X, y = [], []
    for cid in [str(c) for c in cids]:
        if cid in fp_index and cid in label_map:
            X.append(fp_index[cid])
            y.append(label_map[cid])
    if not X:
        return None, None
    return np.stack(X), np.array(y, dtype=np.float32)


# =========================================================
# ★ Protein sequence helpers
# =========================================================

# 20 standard amino acids + unknown token (index 0 = padding/unknown)
AA_VOCAB = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
AA_VOCAB_SIZE = len(AA_VOCAB) + 1  # 21


def seq_to_tensor(seq: str, max_len: int = 2000) -> torch.LongTensor:
    """
    Amino-acid string → LongTensor of indices (truncated to max_len).
    Unknown characters map to index 0.
    """
    indices = [AA_VOCAB.get(aa.upper(), 0) for aa in seq[:max_len]]
    return torch.tensor(indices, dtype=torch.long)


def build_protein_index(
    seq_csv: str,
    seq_col: str = "POI_seq",
    max_len: int = 2000,
) -> Dict[str, torch.LongTensor]:
    """
    Read CSV with columns ['Compound ID', seq_col] and build a dict:
        { compound_id_str : LongTensor(seq_indices) }

    Rows with missing / empty sequences are silently skipped.
    """
    print(f"📦 Building protein index from: {seq_csv}  (col='{seq_col}')")
    df = pd.read_csv(seq_csv, low_memory=False)
    df["Compound ID"] = df["Compound ID"].astype(str)

    if seq_col not in df.columns:
        raise ValueError(
            f"Column '{seq_col}' not found in {seq_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    prot_index, ok, fail = {}, 0, 0
    for _, row in df.iterrows():
        cid = str(row["Compound ID"])
        seq = str(row[seq_col]) if not pd.isna(row[seq_col]) else ""
        seq = seq.strip()
        if not seq or seq.lower() in ("nan", "none", ""):
            fail += 1
            continue
        prot_index[cid] = seq_to_tensor(seq, max_len=max_len)
        ok += 1

    print(f"✅ Protein index: {ok} ok | {fail} skipped (missing seq)")
    return prot_index


class ProteinEncoder(nn.Module):
    """
    Light-weight amino-acid sequence encoder.

    Architecture:
        AA indices → Embedding(AA_VOCAB_SIZE, embed_dim)
        → mean-pool over sequence length
        → Linear(embed_dim, out_dim)
        → ReLU

    out_dim should match GCNEncoder.out_dim so that element-wise
    addition (z_mol + alpha * z_prot) is dimensionally consistent.

    alpha is a learnable scalar initialised to 0.1, so the model starts
    by trusting the molecule embedding and gradually learns how much
    protein context to incorporate.
    """

    def __init__(self, embed_dim: int = 64, out_dim: int = 128, alpha_init: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(AA_VOCAB_SIZE, embed_dim, padding_idx=0)
        self.proj      = nn.Linear(embed_dim, out_dim)
        # learnable fusion weight: z = z_mol + alpha * z_prot
        self.alpha     = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, seq_tensor: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            seq_tensor: LongTensor of shape (L,) — one sequence.
        Returns:
            Tensor of shape (out_dim,)
        """
        emb = self.embedding(seq_tensor)          # (L, embed_dim)
        pooled = emb.mean(dim=0)                  # (embed_dim,)
        return F.relu(self.proj(pooled))           # (out_dim,)

    def encode_batch_ids(
        self,
        cids: List[str],
        prot_index: Dict[str, torch.LongTensor],
        device: torch.device,
        out_dim: int,
    ) -> torch.Tensor:
        """
        Encode a list of compound IDs → stacked protein embeddings.

        Compounds whose sequence is missing get a zero vector so that
        the fusion falls back to the molecule embedding alone.
        """
        vecs = []
        for cid in cids:
            if cid in prot_index:
                t = prot_index[cid].to(device)
                vecs.append(self.forward(t))       # (out_dim,)
            else:
                vecs.append(torch.zeros(out_dim, device=device))
        return torch.stack(vecs, dim=0)            # (N, out_dim)


# =========================================================
# Metrics
# =========================================================
def _safe_div(a, b):
    return a / b if b != 0 else float("nan")


@torch.no_grad()
def binary_auc_roc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y, s = y_true.view(-1).float(), y_score.view(-1).float()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float, device=s.device)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def average_precision(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y, s = y_true.view(-1).float(), y_score.view(-1).float()
    n_pos = float((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(s, descending=True)
    y_sorted = y[order]
    tp = torch.cumsum(y_sorted, 0)
    fp = torch.cumsum(1 - y_sorted, 0)
    prec = tp / torch.clamp(tp + fp, min=1.0)
    return float((prec * y_sorted).sum() / n_pos)


@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> dict:
    y_true = y.view(-1).float()
    prob   = torch.sigmoid(logits.view(-1).float())
    y_pred = (prob >= 0.5).float()

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
        "Accuracy": acc, "Loss": loss,
        "AUROC": binary_auc_roc(y_true, prob),
        "AUPRC": average_precision(y_true, prob),
        "F1": float(f1), "BALACC": balacc,
    }


def metrics_from_proba(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(float)
    acc = float((y_pred == y_true).mean())

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

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        auroc = float("nan")
        auprc = float("nan")
    else:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
            auprc = float(average_precision_score(y_true, y_prob))
        except Exception:
            auroc = float("nan")
            auprc = float("nan")

    return {
        "Accuracy": acc, "Loss": float("nan"),
        "AUROC": auroc, "AUPRC": auprc,
        "F1": float(f1), "BALACC": balacc,
    }


def mean_ignore_nan(xs: List[float]) -> float:
    vals = [float(x) for x in xs if not math.isnan(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


# =========================================================
# Data helpers
# =========================================================
def build_label_map(df: pd.DataFrame) -> Dict[str, int]:
    if "Label_bin" not in df.columns and "Label" in df.columns:
        df["Label_bin"] = df["Label"].astype(int)
    return {
        str(row["Compound ID"]): int(row["Label_bin"])
        for _, row in df.iterrows()
        if not pd.isna(row.get("Label_bin"))
    }


def make_data_list(cids, graph_index, label_map) -> List[Data]:
    out = []
    for cid in [str(c) for c in cids]:
        if cid in graph_index and cid in label_map:
            g = graph_index[cid].clone()
            g.y = torch.tensor([label_map[cid]], dtype=torch.long)
            # stash the compound id on the graph so fusion can look it up
            g.cid = cid
            out.append(g)
    return out


# rare E3s to exclude when testing on rare E3 targets
RARE_E3_LIST = ["FEM1B", "IAP", "MDM2", "XIAP", "cIAP1",
                "DCAF1", "Keap1", "KLHL20", "DCAF16",
                "AhR", "BRD4", "RNF114", "KLHDC2",
                "DCAF11", "FBXO22", "UBR box", "RNF4"]


def build_meta_train_tasks(df, graph_index, K, Q, exclude_e3=None):
    """
    Build meta-train tasks, excluding specified E3 ligase(s) from training.

    exclude_e3: str or list of str.
      - "VHL"              → CRBN->VHL experiment (exclude VHL only)
      - RARE_E3_LIST       → CRBN->rareE3 experiment (only CRBN used for train)
      - None               → use all E3s
    """
    df = df[df["Compound ID"].astype(str).isin(graph_index)].copy()

    if exclude_e3 is None:
        exclude_set = set()
    elif isinstance(exclude_e3, str):
        exclude_set = {exclude_e3}
    else:
        exclude_set = set(exclude_e3)

    tasks = {}
    subset = df[~df["E3 ligase"].isin(exclude_set)]
    for (e3, tgt), sub in subset.groupby(["E3 ligase", "Target"]):
        sub = sub.reset_index(drop=True)
        n_pos = int((sub["Label_bin"] == 1).sum())
        n_neg = int((sub["Label_bin"] == 0).sum())
        if n_pos >= K + Q and n_neg >= K + Q:
            tasks[f"{e3}__{tgt}"] = sub
    print(f"  E3s in meta-train: {sorted(subset['E3 ligase'].unique())}")
    return tasks


def sample_episode(df_target, graph_index, K, Q, device, allow_replacement=False):
    df_pos = df_target[df_target["Label_bin"] == 1]
    df_neg = df_target[df_target["Label_bin"] == 0]
    need   = K + Q
    if len(df_pos) == 0 or len(df_neg) == 0:
        return None
    rep_pos = allow_replacement and len(df_pos) < need
    rep_neg = allow_replacement and len(df_neg) < need
    if not allow_replacement and (len(df_pos) < need or len(df_neg) < need):
        return None

    pos = df_pos.sample(n=need, replace=rep_pos)
    neg = df_neg.sample(n=need, replace=rep_neg)
    rows_s = pd.concat([pos.iloc[:K], neg.iloc[:K]])
    rows_q = pd.concat([pos.iloc[K:], neg.iloc[K:]])

    def to_tensors(rows):
        data_list, labels = [], []
        for _, r in rows.iterrows():
            cid = str(r["Compound ID"])
            if cid not in graph_index:
                return None, None
            g = graph_index[cid].clone()
            g.cid = cid
            data_list.append(g)
            labels.append(int(r["Label_bin"]))
        return data_list, torch.tensor(labels, dtype=torch.float32, device=device)

    sup_data, sup_labels = to_tensors(rows_s)
    qry_data, qry_labels = to_tensors(rows_q)
    if sup_data is None or qry_data is None:
        return None
    return sup_data, sup_labels, qry_data, qry_labels


# =========================================================
# Models
# =========================================================
class GCNEncoder(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=128, out_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs   = nn.ModuleList([GCNConv(dims[i], dims[i+1]) for i in range(num_layers)])
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


class SimpleGCNClassifier(nn.Module):
    def __init__(self, in_channels=9, hidden=128, dropout=0.1):
        super().__init__()
        self.conv1, self.conv2 = GCNConv(in_channels, hidden), GCNConv(hidden, hidden)
        self.lin, self.dropout = nn.Linear(hidden, 1), dropout

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = F.dropout(F.relu(self.conv1(x, ei)), p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, ei))
        return self.lin(global_mean_pool(x, b)).view(-1)


class MAMLModel(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=128, out_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout)
        self.head    = nn.Linear(out_dim, 1)

    def forward(self, data):
        return self.head(self.encoder(data.x, data.edge_index, data.batch)).view(-1)


class ECFPMLPClassifier(nn.Module):
    def __init__(self, in_dim=2048, hidden=512, hidden2=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


# =========================================================
# ★ Fused encoding helpers
# =========================================================

def _get_cids_from_data_list(data_list: List[Data]) -> List[str]:
    """Extract stored compound IDs from a list of PyG Data objects."""
    return [getattr(g, "cid", "") for g in data_list]


def encode_batch(encoder: GCNEncoder, data_list: List[Data], device) -> torch.Tensor:
    """Molecule-only encoding (used when protein_index is None)."""
    b = Batch.from_data_list(data_list).to(device)
    return encoder(b.x, b.edge_index, b.batch)


def encode_batch_fused(
    encoder: GCNEncoder,
    poi_encoder: ProteinEncoder,
    e3_encoder: ProteinEncoder,
    data_list: List[Data],
    poi_index: Dict[str, torch.LongTensor],
    e3_index:  Dict[str, torch.LongTensor],
    device,
    out_dim: int,
) -> torch.Tensor:
    """
    Fused encoding:
        z = z_mol + alpha_poi * z_poi + alpha_e3 * z_e3

    Falls back gracefully for compounds missing POI or E3 sequence.
    """
    b = Batch.from_data_list(data_list).to(device)
    z_mol = encoder(b.x, b.edge_index, b.batch)            # (N, out_dim)

    cids  = _get_cids_from_data_list(data_list)
    z_poi = poi_encoder.encode_batch_ids(cids, poi_index, device, out_dim)  # (N, out_dim)
    z_e3  = e3_encoder.encode_batch_ids( cids, e3_index,  device, out_dim)  # (N, out_dim)

    return z_mol + poi_encoder.alpha * z_poi + e3_encoder.alpha * z_e3


def _make_encode_fn(encoder, poi_encoder, e3_encoder, poi_index, e3_index, device, out_dim):
    """
    Return a unified encode function that accepts data_list and returns embeddings.
    Chooses fused or molecule-only depending on whether poi_index is provided.
    """
    if poi_index is not None:
        def fn(data_list):
            return encode_batch_fused(
                encoder, poi_encoder, e3_encoder,
                data_list, poi_index, e3_index, device, out_dim,
            )
    else:
        def fn(data_list):
            return encode_batch(encoder, data_list, device)
    return fn


# =========================================================
# Checkpoint loading
# =========================================================
def extract_encoder_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["encoder", "state_dict", "model", "backbone"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                ckpt_obj = ckpt_obj[key]; break
    cleaned = {}
    for k, v in ckpt_obj.items():
        nk = k
        for prefix in ["encoder.", "backbone.", "gnn.", "module.encoder.",
                        "module.backbone.", "module.gnn.", "module."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        if any(nk.startswith(p) for p in
               ["projection_head", "proj", "predictor", "classifier", "lin."]):
            continue
        cleaned[nk] = v
    return cleaned


def load_pretrained_encoder(encoder, ckpt_path, device):
    print(f"📥 Loading pretrained: {ckpt_path}")
    state = extract_encoder_state_dict(torch.load(ckpt_path, map_location=device))
    msg = encoder.load_state_dict(state, strict=False)
    print(f"✅ missing={msg.missing_keys} | unexpected={msg.unexpected_keys}")


# =========================================================
# ProtoNet
# =========================================================
def prototypical_logits(encode_fn, sup_data, sup_labels, qry_data):
    """Compute prototypical logits using provided encode_fn."""
    with torch.no_grad():
        z_s = encode_fn(sup_data)
        z_q = encode_fn(qry_data)
        pp  = z_s[sup_labels == 1].mean(0, keepdim=True)
        pn  = z_s[sup_labels == 0].mean(0, keepdim=True)
        return -(torch.norm(z_q - pp, dim=1) - torch.norm(z_q - pn, dim=1))


def meta_train_protonet(
    encoder, poi_encoder, e3_encoder, meta_tasks, graph_index,
    poi_index, e3_index, device,
    epochs, episodes_per_epoch, K, Q, lr, weight_decay, out_dim,
):
    print("🚀 Meta-training ProtoNet on CRBN ...")
    params = list(encoder.parameters())
    if poi_encoder is not None and poi_index is not None:
        params += list(poi_encoder.parameters())
    if e3_encoder is not None and e3_index is not None:
        params += list(e3_encoder.parameters())
    opt        = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    bce        = nn.BCEWithLogitsLoss()
    task_names = list(meta_tasks.keys())

    encode_fn_train = _make_encode_fn(encoder, poi_encoder, e3_encoder, poi_index, e3_index, device, out_dim)

    for epoch in range(1, epochs + 1):
        encoder.train()
        if poi_encoder is not None:
            poi_encoder.train()
        if e3_encoder is not None:
            e3_encoder.train()
        losses = []
        for _ in range(episodes_per_epoch):
            ep = sample_episode(meta_tasks[random.choice(task_names)],
                                graph_index, K, Q, device)
            if ep is None: continue
            sd, sl, qd, ql = ep
            z_s = encode_fn_train(sd)
            z_q = encode_fn_train(qd)
            pp  = z_s[sl == 1].mean(0, keepdim=True)
            pn  = z_s[sl == 0].mean(0, keepdim=True)
            logits = -(torch.norm(z_q - pp, dim=1) - torch.norm(z_q - pn, dim=1))
            loss = bce(logits, ql)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch:03d} | loss={mean_ignore_nan(losses):.4f}")


def eval_protonet_on_split(
    encoder, poi_encoder, e3_encoder, split_path, graph_index, label_map,
    poi_index, e3_index, device, out_dim, phase="meta_test",
) -> pd.DataFrame:
    print(f"🧪 ProtoNet eval | phase={phase}")
    encoder.eval()
    if poi_encoder is not None:
        poi_encoder.eval()
    if e3_encoder is not None:
        e3_encoder.eval()

    encode_fn = _make_encode_fn(encoder, poi_encoder, e3_encoder, poi_index, e3_index, device, out_dim)

    phase_obj = json.load(open(split_path)).get(phase, {})
    records, skipped, total = [], 0, 0

    for task, episodes in phase_obj.items():
        for ep_idx, ep in enumerate(episodes):
            total += 1
            sd = make_data_list(ep.get("support", []), graph_index, label_map)
            qd = make_data_list(ep.get("query",   []), graph_index, label_map)
            if not sd or not qd: skipped += 1; continue
            ys = torch.tensor([g.y.item() for g in sd], dtype=torch.float32, device=device)
            yq = torch.tensor([g.y.item() for g in qd], dtype=torch.float32, device=device)
            if not ((ys == 1).any() and (ys == 0).any()): skipped += 1; continue
            logits = prototypical_logits(encode_fn, sd, ys, qd)
            rec = {"Task": task, "Episode": ep_idx}
            rec.update(metrics_from_logits(logits, yq))
            records.append(rec)

    print(f"✅ valid={len(records)} | skipped={skipped} | total={total}")
    return pd.DataFrame(records)


# =========================================================
# GNN baseline
# =========================================================
def eval_gnn_on_split(
    split_path, graph_index, label_map,
    poi_index, e3_index, poi_encoder, e3_encoder, device,
    hidden, dropout, lr, weight_decay, steps,
    batch_size, out_dim, phase="meta_test",
) -> pd.DataFrame:
    """
    GNN baseline with optional protein fusion (POI + E3).
    """
    print(f"🧪 GNN baseline eval | phase={phase} | protein_fusion={poi_index is not None}")
    phase_obj = json.load(open(split_path)).get(phase, {})
    bce = nn.BCEWithLogitsLoss()
    records, skip_e, skip_c, total = [], 0, 0, 0

    if poi_encoder is not None:
        poi_encoder.eval()
    if e3_encoder is not None:
        e3_encoder.eval()

    for task, episodes in phase_obj.items():
        for ep_idx, ep in enumerate(episodes):
            total += 1
            sd = make_data_list(ep.get("support", []), graph_index, label_map)
            qd = make_data_list(ep.get("query",   []), graph_index, label_map)
            if not sd or not qd: skip_e += 1; continue
            ys = torch.tensor([g.y.item() for g in sd], dtype=torch.long)
            if not ((ys == 1).any() and (ys == 0).any()): skip_c += 1; continue

            model = SimpleGCNClassifier(9, hidden, dropout).to(device)
            opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            sb    = Batch.from_data_list(sd).to(device)
            yb    = sb.y.view(-1).float()

            # --- fine-tune on support ---
            model.train()
            for _ in range(steps):
                # get molecule pooled rep before the linear head
                x, ei, b = sb.x, sb.edge_index, sb.batch
                x_h = F.dropout(F.relu(model.conv1(x, ei)), p=model.dropout, training=True)
                z   = F.relu(model.conv2(x_h, ei))
                z_pool = global_mean_pool(z, b)       # (N, hidden)

                if poi_index is not None and poi_encoder is not None:
                    cids_s = _get_cids_from_data_list(sd)
                    z_poi = poi_encoder.encode_batch_ids(cids_s, poi_index, device, out_dim)
                    z_e3  = e3_encoder.encode_batch_ids( cids_s, e3_index,  device, out_dim)
                    z_pool = z_pool + poi_encoder.alpha * z_poi + e3_encoder.alpha * z_e3

                logits_sup = model.lin(z_pool).view(-1)
                loss = bce(logits_sup, yb)
                opt.zero_grad(); loss.backward(); opt.step()

            # --- eval on query ---
            model.eval()
            with torch.no_grad():
                qb = Batch.from_data_list(qd).to(device)
                x, ei, b = qb.x, qb.edge_index, qb.batch
                x_h = F.relu(model.conv1(x, ei))
                z   = F.relu(model.conv2(x_h, ei))
                z_pool = global_mean_pool(z, b)

                if poi_index is not None and poi_encoder is not None:
                    cids_q = _get_cids_from_data_list(qd)
                    z_poi = poi_encoder.encode_batch_ids(cids_q, poi_index, device, out_dim)
                    z_e3  = e3_encoder.encode_batch_ids( cids_q, e3_index,  device, out_dim)
                    z_pool = z_pool + poi_encoder.alpha * z_poi + e3_encoder.alpha * z_e3

                logits = model.lin(z_pool).view(-1)
                yq     = qb.y.view(-1).float().cpu()

            rec = {"Task": task, "Episode": ep_idx}
            rec.update(metrics_from_logits(logits.cpu(), yq))
            records.append(rec)

    print(f"✅ valid={len(records)} | skip_empty={skip_e} | skip_cls={skip_c} | total={total}")
    return pd.DataFrame(records)


# =========================================================
# FOMAML
# =========================================================
def _inner_loop(model_clone, sup_batch, y_sup, inner_lr, inner_steps):
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(model_clone.parameters(), lr=inner_lr)
    model_clone.train()
    for _ in range(inner_steps):
        loss = bce(model_clone(sup_batch), y_sup)
        opt.zero_grad(); loss.backward(); opt.step()
    return model_clone


def meta_train_maml(
    model, poi_encoder, e3_encoder, meta_tasks, graph_index,
    poi_index, e3_index, device,
    epochs, episodes_per_epoch, K, Q,
    inner_lr, inner_steps, outer_lr, weight_decay, out_dim,
):
    print("🚀 Meta-training FOMAML on CRBN ...")
    bce        = nn.BCEWithLogitsLoss()
    params     = list(model.parameters())
    if poi_encoder is not None and poi_index is not None:
        params += list(poi_encoder.parameters())
    if e3_encoder is not None and e3_index is not None:
        params += list(e3_encoder.parameters())
    outer_opt  = torch.optim.Adam(params, lr=outer_lr, weight_decay=weight_decay)
    task_names = list(meta_tasks.keys())

    for epoch in range(1, epochs + 1):
        losses = []
        for _ in range(episodes_per_epoch):
            ep = sample_episode(meta_tasks[random.choice(task_names)],
                                graph_index, K, Q, device)
            if ep is None: continue
            sd, sl, qd, ql = ep
            sb = Batch.from_data_list(sd).to(device)
            qb = Batch.from_data_list(qd).to(device)

            # For MAML we adapt the molecule encoder; protein encoder is shared
            model_clone = copy.deepcopy(model)
            model_clone = _inner_loop(model_clone, sb, sl, inner_lr, inner_steps)

            model.train()
            outer_opt.zero_grad()

            # Query logits with fused embeddings
            z_q = model_clone.encoder(qb.x, qb.edge_index, qb.batch)
            if poi_encoder is not None and poi_index is not None:
                poi_encoder.train()
                e3_encoder.train()
                cids_q = _get_cids_from_data_list(qd)
                z_poi = poi_encoder.encode_batch_ids(cids_q, poi_index, device, out_dim)
                z_e3  = e3_encoder.encode_batch_ids( cids_q, e3_index,  device, out_dim)
                z_q = z_q + poi_encoder.alpha * z_poi + e3_encoder.alpha * z_e3
            logits_q = model_clone.head(z_q).view(-1)

            loss = bce(logits_q, ql)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            outer_opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch:03d} | MAML loss={mean_ignore_nan(losses):.4f}")


def eval_maml_on_split(
    model, poi_encoder, e3_encoder, split_path, graph_index, label_map,
    poi_index, e3_index, device, inner_lr, inner_steps, out_dim,
    phase="meta_test",
) -> pd.DataFrame:
    print(f"🧪 MAML eval | phase={phase}")
    phase_obj = json.load(open(split_path)).get(phase, {})
    records, skipped, total = [], 0, 0

    if poi_encoder is not None:
        poi_encoder.eval()
    if e3_encoder is not None:
        e3_encoder.eval()

    for task, episodes in phase_obj.items():
        for ep_idx, ep in enumerate(episodes):
            total += 1
            sd = make_data_list(ep.get("support", []), graph_index, label_map)
            qd = make_data_list(ep.get("query",   []), graph_index, label_map)
            if not sd or not qd: skipped += 1; continue
            ys = torch.tensor([g.y.item() for g in sd], dtype=torch.float32, device=device)
            yq = torch.tensor([g.y.item() for g in qd], dtype=torch.float32, device=device)
            if not ((ys == 1).any() and (ys == 0).any()): skipped += 1; continue

            model_clone = copy.deepcopy(model)
            sb = Batch.from_data_list(sd).to(device)
            with torch.enable_grad():
                model_clone = _inner_loop(model_clone, sb, ys, inner_lr, inner_steps)

            model_clone.eval()
            with torch.no_grad():
                qb     = Batch.from_data_list(qd).to(device)
                z_q    = model_clone.encoder(qb.x, qb.edge_index, qb.batch)
                if poi_encoder is not None and poi_index is not None:
                    cids_q = _get_cids_from_data_list(qd)
                    z_poi = poi_encoder.encode_batch_ids(cids_q, poi_index, device, out_dim)
                    z_e3  = e3_encoder.encode_batch_ids( cids_q, e3_index,  device, out_dim)
                    z_q = z_q + poi_encoder.alpha * z_poi + e3_encoder.alpha * z_e3
                logits = model_clone.head(z_q).view(-1)

            rec = {"Task": task, "Episode": ep_idx}
            rec.update(metrics_from_logits(logits.cpu(), yq.cpu()))
            records.append(rec)

    print(f"✅ valid={len(records)} | skipped={skipped} | total={total}")
    return pd.DataFrame(records)


# =========================================================
# ECFP4 + MLP  (protein not fused — fingerprint methods stay independent)
# =========================================================
def eval_ecfp_mlp_on_split(split_path, fp_index, label_map, device,
                            n_bits=2048, hidden=512, hidden2=128, dropout=0.1,
                            lr=1e-3, weight_decay=1e-4, steps=50,
                            phase="meta_test") -> pd.DataFrame:
    print(f"🧪 ECFP4+MLP eval | phase={phase}")
    phase_obj = json.load(open(split_path)).get(phase, {})
    bce = nn.BCEWithLogitsLoss()
    records, skip_e, skip_c, total = [], 0, 0, 0

    for task, episodes in phase_obj.items():
        for ep_idx, ep in enumerate(episodes):
            total += 1
            sup_X, sup_y = gather_fp(ep.get("support", []), fp_index, label_map)
            qry_X, qry_y = gather_fp(ep.get("query",   []), fp_index, label_map)
            if sup_X is None or qry_X is None: skip_e += 1; continue
            if not (1 in sup_y and 0 in sup_y): skip_c += 1; continue

            sup_xt = torch.tensor(sup_X, dtype=torch.float32, device=device)
            sup_yt = torch.tensor(sup_y, dtype=torch.float32, device=device)
            qry_xt = torch.tensor(qry_X, dtype=torch.float32, device=device)
            qry_yt = torch.tensor(qry_y, dtype=torch.float32)

            model = ECFPMLPClassifier(n_bits, hidden, hidden2, dropout).to(device)
            opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            model.train()
            for _ in range(steps):
                loss = bce(model(sup_xt), sup_yt)
                opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                logits = model(qry_xt)

            rec = {"Task": task, "Episode": ep_idx}
            rec.update(metrics_from_logits(logits.cpu(), qry_yt))
            records.append(rec)

    print(f"✅ ECFP4+MLP done | valid={len(records)} | skip_empty={skip_e} | skip_cls={skip_c} | total={total}")
    return pd.DataFrame(records)


# =========================================================
# Random Forest + ECFP4
# =========================================================
def eval_rf_on_split(
    split_path: str,
    fp_index: Dict[str, np.ndarray],
    label_map: Dict[str, int],
    n_estimators: int = 500,
    max_features: str = "sqrt",
    phase: str = "meta_test",
) -> pd.DataFrame:
    print(f"🧪 RF+ECFP4 eval | phase={phase} | n_estimators={n_estimators}")
    phase_obj = json.load(open(split_path)).get(phase, {})
    records, skip_e, skip_c, total = [], 0, 0, 0

    for task, episodes in phase_obj.items():
        for ep_idx, ep in enumerate(episodes):
            total += 1
            sup_X, sup_y = gather_fp(ep.get("support", []), fp_index, label_map)
            qry_X, qry_y = gather_fp(ep.get("query",   []), fp_index, label_map)
            if sup_X is None or qry_X is None: skip_e += 1; continue
            if not (1 in sup_y and 0 in sup_y): skip_c += 1; continue

            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                random_state=0,
                n_jobs=-1,
            )
            clf.fit(sup_X, sup_y.astype(int))
            qry_prob = clf.predict_proba(qry_X)[:, 1]

            rec = {"Task": task, "Episode": ep_idx}
            rec.update(metrics_from_proba(qry_y, qry_prob))
            records.append(rec)

    print(f"✅ RF done | valid={len(records)} | skip_empty={skip_e} | skip_cls={skip_c} | total={total}")
    return pd.DataFrame(records)


# =========================================================
# Save helpers
# =========================================================
def summarize_records(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    cols = ["Accuracy", "Loss", "AUROC", "AUPRC", "F1", "BALACC"]
    row  = {"Variant": variant}
    for c in cols:
        row[c] = mean_ignore_nan(df[c].tolist()) if c in df.columns else float("nan")
    return pd.DataFrame([row])


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["gnn", "protonet", "protonet_supcon_all",
                             "protonet_supcon_crbn", "maml", "ecfp_mlp", "rf"])
    ap.add_argument("--csv",    required=True)
    ap.add_argument("--sdf",    required=True)
    ap.add_argument("--split",  required=True)
    ap.add_argument("--phase",  default="meta_test",
                    choices=["meta_test", "meta_valid", "meta_train"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--outdir", required=True)

    # ★ Protein sequence args
    ap.add_argument("--seq-csv",  default="",
                    help="Path to CSV with protein sequences "
                         "(e.g. autodl-tmp/research/protac_with_seq.csv). "
                         "If empty, protein fusion is disabled.")
    ap.add_argument("--seq-col",    default="POI_seq",
                    help="Column name for POI sequence in seq-csv")
    ap.add_argument("--e3-seq-col",  default="E3_seq",
                    help="Column name for E3 sequence in seq-csv")
    ap.add_argument("--exclude-e3", nargs="+", default=["VHL"],
                    help="E3 ligase(s) to hold out as test set. "
                         "Use 'rare' as shorthand for all rare E3s. "
                         "Examples: --exclude-e3 VHL  or  --exclude-e3 rare")
    ap.add_argument("--prot-embed-dim", type=int, default=64,
                    help="AA embedding dimension inside ProteinEncoder")
    ap.add_argument("--prot-max-len",   type=int, default=2000,
                    help="Max sequence length to encode (truncated beyond this)")

    # Shared ProtoNet / MAML args
    ap.add_argument("--meta-epochs",        type=int,   default=100)
    ap.add_argument("--episodes-per-epoch", type=int,   default=100)
    ap.add_argument("--meta-k",             type=int,   default=2)
    ap.add_argument("--meta-q",             type=int,   default=3)
    ap.add_argument("--encoder-hidden",     type=int,   default=128)
    ap.add_argument("--encoder-out",        type=int,   default=128)
    ap.add_argument("--encoder-layers",     type=int,   default=3)
    ap.add_argument("--encoder-dropout",    type=float, default=0.1)
    ap.add_argument("--meta-lr",            type=float, default=1e-3)
    ap.add_argument("--meta-weight-decay",  type=float, default=1e-5)
    ap.add_argument("--pretrained-ckpt",    default="")

    # MAML args
    ap.add_argument("--maml-inner-lr",    type=float, default=0.01)
    ap.add_argument("--maml-inner-steps", type=int,   default=5)

    # GNN args
    ap.add_argument("--gnn-hidden",       type=int,   default=128)
    ap.add_argument("--gnn-dropout",      type=float, default=0.1)
    ap.add_argument("--gnn-lr",           type=float, default=1e-3)
    ap.add_argument("--gnn-weight-decay", type=float, default=1e-4)
    ap.add_argument("--gnn-steps",        type=int,   default=50)
    ap.add_argument("--gnn-batch-size",   type=int,   default=32)

    # ECFP4+MLP args
    ap.add_argument("--fp-bits",          type=int,   default=2048)
    ap.add_argument("--fp-hidden",        type=int,   default=512)
    ap.add_argument("--fp-hidden2",       type=int,   default=128)
    ap.add_argument("--fp-dropout",       type=float, default=0.1)
    ap.add_argument("--fp-lr",            type=float, default=1e-3)
    ap.add_argument("--fp-weight-decay",  type=float, default=1e-4)
    ap.add_argument("--fp-steps",         type=int,   default=50)

    # RF args
    ap.add_argument("--rf-n-estimators",  type=int,   default=500)
    ap.add_argument("--rf-max-features",  type=str,   default="sqrt")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    out_dim = args.encoder_out
    print(f"🖥  device={device} | variant={args.variant} | K={args.meta_k} Q={args.meta_q}")

    df = pd.read_csv(args.csv, low_memory=False)
    df["Compound ID"] = df["Compound ID"].astype(str)
    if "Label_bin" not in df.columns and "Label" in df.columns:
        df["Label_bin"] = df["Label"].astype(int)
    label_map   = build_label_map(df)
    graph_index = build_graph_index_from_sdf(args.sdf)
    df = df[df["Compound ID"].isin(graph_index)].copy()
    print(f"✅ {len(df)} samples after SDF alignment")

    # ── Protein index (optional) ──────────────────────────
    poi_index   = None
    e3_index    = None
    poi_encoder = None
    e3_encoder  = None
    if args.seq_csv and args.variant not in ("rf", "ecfp_mlp"):
        poi_index = build_protein_index(
            args.seq_csv, seq_col=args.seq_col, max_len=args.prot_max_len
        )
        e3_index = build_protein_index(
            args.seq_csv, seq_col=args.e3_seq_col, max_len=args.prot_max_len
        )
        poi_encoder = ProteinEncoder(embed_dim=args.prot_embed_dim, out_dim=out_dim).to(device)
        e3_encoder  = ProteinEncoder(embed_dim=args.prot_embed_dim, out_dim=out_dim).to(device)
        print(f"✅ POI + E3 ProteinEncoder ready | embed_dim={args.prot_embed_dim} out_dim={out_dim}")
    else:
        if args.variant not in ("rf", "ecfp_mlp"):
            print("ℹ️  --seq-csv not provided → protein fusion disabled")

    # ── Fingerprint index (rf / ecfp_mlp) ────────────────
    fp_index = None
    if args.variant in ("rf", "ecfp_mlp"):
        fp_index = build_fp_index(df, n_bits=args.fp_bits)

    # ── dispatch ──────────────────────────────────────────
    if args.variant == "rf":
        records = eval_rf_on_split(
            split_path=args.split,
            fp_index=fp_index,
            label_map=label_map,
            n_estimators=args.rf_n_estimators,
            max_features=args.rf_max_features,
            phase=args.phase,
        )

    elif args.variant == "ecfp_mlp":
        records = eval_ecfp_mlp_on_split(
            split_path=args.split,
            fp_index=fp_index,
            label_map=label_map,
            device=device,
            n_bits=args.fp_bits,
            hidden=args.fp_hidden,
            hidden2=args.fp_hidden2,
            dropout=args.fp_dropout,
            lr=args.fp_lr,
            weight_decay=args.fp_weight_decay,
            steps=args.fp_steps,
            phase=args.phase,
        )

    elif args.variant == "gnn":
        records = eval_gnn_on_split(
            args.split, graph_index, label_map,
            poi_index, e3_index, poi_encoder, e3_encoder, device,
            args.gnn_hidden, args.gnn_dropout, args.gnn_lr,
            args.gnn_weight_decay, args.gnn_steps, args.gnn_batch_size,
            out_dim, args.phase,
        )

    elif args.variant == "maml":
        model = MAMLModel(
            in_dim=9, hidden_dim=args.encoder_hidden, out_dim=out_dim,
            num_layers=args.encoder_layers, dropout=args.encoder_dropout,
        ).to(device)
        _exclude = RARE_E3_LIST if args.exclude_e3 == ["rare"] else args.exclude_e3
        meta_tasks = build_meta_train_tasks(df, graph_index, args.meta_k, args.meta_q, exclude_e3=_exclude)
        print(f"🧩 Meta-train tasks: {len(meta_tasks)}")
        meta_train_maml(
            model, poi_encoder, e3_encoder, meta_tasks, graph_index,
            poi_index, e3_index, device,
            epochs=args.meta_epochs, episodes_per_epoch=args.episodes_per_epoch,
            K=args.meta_k, Q=args.meta_q,
            inner_lr=args.maml_inner_lr, inner_steps=args.maml_inner_steps,
            outer_lr=args.meta_lr, weight_decay=args.meta_weight_decay,
            out_dim=out_dim,
        )
        records = eval_maml_on_split(
            model, poi_encoder, e3_encoder, args.split, graph_index, label_map,
            poi_index, e3_index, device,
            inner_lr=args.maml_inner_lr, inner_steps=args.maml_inner_steps,
            out_dim=out_dim, phase=args.phase,
        )

    else:  # protonet / protonet_supcon_*
        encoder = GCNEncoder(
            in_dim=9, hidden_dim=args.encoder_hidden, out_dim=out_dim,
            num_layers=args.encoder_layers, dropout=args.encoder_dropout,
        ).to(device)
        if args.variant in ["protonet_supcon_all", "protonet_supcon_crbn"]:
            if not args.pretrained_ckpt:
                raise ValueError(f"{args.variant} requires --pretrained-ckpt")
            load_pretrained_encoder(encoder, args.pretrained_ckpt, device)
        _exclude = RARE_E3_LIST if args.exclude_e3 == ["rare"] else args.exclude_e3
        meta_tasks = build_meta_train_tasks(df, graph_index, args.meta_k, args.meta_q, exclude_e3=_exclude)
        print(f"🧩 Meta-train tasks: {len(meta_tasks)}")
        meta_train_protonet(
            encoder, poi_encoder, e3_encoder, meta_tasks, graph_index,
            poi_index, e3_index, device,
            epochs=args.meta_epochs, episodes_per_epoch=args.episodes_per_epoch,
            K=args.meta_k, Q=args.meta_q,
            lr=args.meta_lr, weight_decay=args.meta_weight_decay,
            out_dim=out_dim,
        )
        # Save encoder weights (for UMAP visualisation etc.)
        ckpt_path = os.path.join(args.outdir,
                                 f"{args.variant}_encoder_seed{args.seed}.pt")
        torch.save(encoder.state_dict(), ckpt_path)
        print(f"💾 Encoder saved → {ckpt_path}")
        if poi_encoder is not None:
            poi_ckpt = os.path.join(args.outdir,
                                    f"{args.variant}_poi_encoder_seed{args.seed}.pt")
            e3_ckpt  = os.path.join(args.outdir,
                                    f"{args.variant}_e3_encoder_seed{args.seed}.pt")
            torch.save(poi_encoder.state_dict(), poi_ckpt)
            torch.save(e3_encoder.state_dict(),  e3_ckpt)
            print(f"💾 POI encoder saved → {poi_ckpt}")
            print(f"💾 E3  encoder saved → {e3_ckpt}")
            print(f"🔬 alpha_poi = {poi_encoder.alpha.item():.4f} | alpha_e3 = {e3_encoder.alpha.item():.4f}")

        records = eval_protonet_on_split(
            encoder, poi_encoder, e3_encoder, args.split, graph_index, label_map,
            poi_index, e3_index, device, out_dim, args.phase,
        )

    if records.empty:
        raise RuntimeError("No valid evaluation records produced.")

    ep_csv  = os.path.join(args.outdir, f"{args.variant}_episodes_seed{args.seed}.csv")
    sum_csv = os.path.join(args.outdir, f"{args.variant}_summary_seed{args.seed}.csv")
    records.to_csv(ep_csv, index=False)
    summary = summarize_records(records, args.variant)
    summary.to_csv(sum_csv, index=False)

    print(f"\n📝 Episodes → {ep_csv}")
    print(f"📝 Summary  → {sum_csv}")
    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()