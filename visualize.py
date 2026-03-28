"""
visualize.py
============
UMAP visualization of Pro-PROTAC molecular embeddings.

Generates a two-panel figure showing VHL compound embeddings colored by:
  (A) POI target (Top 8 targets + Others)
  (B) Degradation activity (Active vs Inactive)

Prerequisites
-------------
- A trained encoder checkpoint produced by train_eval.py (--save-encoder flag).
- umap-learn package: pip install umap-learn

Usage
-----
    python visualize.py \\
        --ckpt  results/seed42/proprotac_encoder_seed42.pt \\
        --csv   protac_filtered_balanced.csv \\
        --sdf   protac.sdf \\
        --seed  42 \\
        --out   figures/umap_embedding.png
"""

import argparse
import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

from data_utils import build_graph_index, mol_to_graph

try:
    import umap
except ImportError:
    raise ImportError(
        "umap-learn is required for visualization. "
        "Install it with: pip install umap-learn"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Top 8 POI targets by compound count in VHL subset of PROTAC-DB
TOP_TARGETS = [
    "SMARCA2", "SMARCA4", "BRD4", "ER",
    "NAMPT", "EGFR e19d", "BCR-ABL", "BCL-xL",
]

# Color palette: 8 named targets + Others (gray)
PALETTE = [
    "#E07B39",  # SMARCA2   orange
    "#2D1B6B",  # SMARCA4   deep purple
    "#5B9BD5",  # BRD4      blue
    "#D94F3D",  # ER        red
    "#4CAF50",  # NAMPT     green
    "#9B7FCC",  # EGFR e19d light purple
    "#F4C842",  # BCR-ABL   yellow
    "#00BCD4",  # BCL-xL    cyan
    "#BBBBBB",  # Others    gray
]

ACT_COLORS = {
    "Active":   "#2D1B6B",
    "Inactive": "#CCCCCC",
}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Encoder (must match architecture used during training)
# ---------------------------------------------------------------------------

class GCNEncoder(nn.Module):
    """GCN encoder — must match the architecture in models.py exactly."""

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
        self.convs   = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


def load_encoder(ckpt_path: str, device: torch.device) -> GCNEncoder:
    """Load a trained GCNEncoder from a checkpoint file.

    Compatible with checkpoints saved via ``torch.save(encoder.state_dict(), path)``.

    Args:
        ckpt_path: Path to the ``.pt`` checkpoint file.
        device:    Target device.

    Returns:
        GCNEncoder with loaded weights, set to eval mode.
    """
    print(f"Loading encoder checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Handle common checkpoint wrapper formats
    if isinstance(ckpt, dict):
        for key in ["encoder", "state_dict", "model", "backbone"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    # Strip common prefixes and skip non-encoder keys
    cleaned = {}
    for k, v in ckpt.items():
        nk = k
        for prefix in ["encoder.", "backbone.", "gnn.", "module."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        if any(nk.startswith(p) for p in
               ["projection_head", "proj", "predictor", "classifier",
                "lin.", "head."]):
            continue
        cleaned[nk] = v

    encoder = GCNEncoder()
    msg = encoder.load_state_dict(cleaned, strict=False)
    if msg.missing_keys:
        print(f"Warning — missing keys: {msg.missing_keys}")

    encoder.to(device).eval()
    print("Encoder loaded successfully.")
    return encoder


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    encoder: GCNEncoder,
    graph_index: Dict[str, Data],
    cids: List[str],
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Extract molecular embeddings for a list of compound IDs.

    Args:
        encoder:    Trained GCNEncoder in eval mode.
        graph_index: Compound-id → PyG graph mapping.
        cids:       Ordered list of compound IDs to embed.
        device:     Target device.
        batch_size: Number of molecules processed per forward pass.

    Returns:
        Embedding matrix of shape ``(len(cids), embedding_dim)``.
    """
    valid_cids = [c for c in cids if c in graph_index]
    all_embs   = []

    for i in range(0, len(valid_cids), batch_size):
        chunk = valid_cids[i: i + batch_size]
        batch = Batch.from_data_list(
            [graph_index[c] for c in chunk]
        ).to(device)
        emb = encoder(batch.x, batch.edge_index, batch.batch)
        all_embs.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embs)
    print(f"Embeddings extracted: {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# POI label assignment
# ---------------------------------------------------------------------------

def get_poi_label(target: str) -> int:
    """Map a POI target name to a palette index.

    Args:
        target: POI target string from the dataset.

    Returns:
        Index into TOP_TARGETS (0–7) or ``len(TOP_TARGETS)`` for Others.
    """
    return TOP_TARGETS.index(target) if target in TOP_TARGETS else len(TOP_TARGETS)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_umap(
    coords: np.ndarray,
    poi_labels: np.ndarray,
    activity_labels: np.ndarray,
    out_path: str,
) -> None:
    """Render and save the two-panel UMAP figure.

    Panel (A): VHL compounds colored by POI target (Top 8 + Others).
    Panel (B): VHL compounds colored by degradation activity.

    Args:
        coords:           2D UMAP coordinates, shape (N, 2).
        poi_labels:       Integer POI label per compound (0–8).
        activity_labels:  Binary activity label per compound (0/1).
        out_path:         Output file path (PNG recommended, 300 dpi).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(wspace=0.35)

    # ── (A) POI target ────────────────────────────────────────────────────
    ax = axes[0]
    label_names = TOP_TARGETS + ["Others"]
    for idx, (name, color) in enumerate(zip(label_names, PALETTE)):
        mask = poi_labels == idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, s=16, alpha=0.75,
            label=name, rasterized=True, zorder=2,
        )

    ax.set_title(
        "(A) Colored by POI Target (VHL compounds)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.legend(
        fontsize=8, markerscale=2, frameon=False,
        loc="upper left", ncol=1, bbox_to_anchor=(0.0, 1.0),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)

    # ── (B) Degradation activity ──────────────────────────────────────────
    ax = axes[1]
    for label, name, color in [
        (1, "Active",   ACT_COLORS["Active"]),
        (0, "Inactive", ACT_COLORS["Inactive"]),
    ]:
        mask = activity_labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, s=16, alpha=0.75,
            label=name, rasterized=True, zorder=2,
        )

    ax.set_title(
        "(B) Colored by Degradation Activity (VHL compounds)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.legend(fontsize=10, markerscale=2, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="UMAP visualization of Pro-PROTAC molecular embeddings."
    )
    ap.add_argument("--ckpt",   required=True,
                    help="Path to trained encoder checkpoint (.pt).")
    ap.add_argument("--csv",    required=True,
                    help="Path to protac_filtered_balanced.csv.")
    ap.add_argument("--sdf",    required=True,
                    help="Path to protac.sdf.")
    ap.add_argument("--seed",   type=int, default=42,
                    help="Random seed for UMAP (default: 42).")
    ap.add_argument("--device", default="cuda",
                    help="Compute device (default: cuda).")
    ap.add_argument("--n-neighbors", type=int,   default=15,
                    help="UMAP n_neighbors parameter (default: 15).")
    ap.add_argument("--min-dist",    type=float, default=0.1,
                    help="UMAP min_dist parameter (default: 0.1).")
    ap.add_argument("--out",    default="figures/umap_embedding.png",
                    help="Output figure path (default: figures/umap_embedding.png).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device
        if (args.device != "cuda" or torch.cuda.is_available())
        else "cpu"
    )
    print(f"Device: {device}")

    # ── Load data (VHL compounds only) ───────────────────────────────────
    df = pd.read_csv(args.csv, low_memory=False)
    df["Compound ID"] = df["Compound ID"].astype(str)
    if "Label_bin" not in df.columns and "Label" in df.columns:
        df["Label_bin"] = df["Label"].astype(int)

    df_vhl = (
        df[df["E3 ligase"] == "VHL"]
        .dropna(subset=["Label_bin", "Target"])
        .copy()
    )
    print(f"VHL compounds: {len(df_vhl)}")

    graph_index = build_graph_index(args.sdf)
    df_vhl = df_vhl[df_vhl["Compound ID"].isin(graph_index)].reset_index(drop=True)
    print(f"After SDF filter: {len(df_vhl)}")

    # ── Extract embeddings ───────────────────────────────────────────────
    encoder    = load_encoder(args.ckpt, device)
    cids       = df_vhl["Compound ID"].tolist()
    embeddings = extract_embeddings(encoder, graph_index, cids, device)

    # ── Build label arrays ───────────────────────────────────────────────
    poi_labels      = np.array([get_poi_label(t) for t in df_vhl["Target"]])
    activity_labels = df_vhl["Label_bin"].values.astype(int)

    # ── UMAP dimensionality reduction ────────────────────────────────────
    print("Running UMAP ...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        metric="euclidean",
    )
    coords = reducer.fit_transform(embeddings)
    print("UMAP complete.")

    # ── Plot ─────────────────────────────────────────────────────────────
    plot_umap(coords, poi_labels, activity_labels, args.out)


if __name__ == "__main__":
    main()
