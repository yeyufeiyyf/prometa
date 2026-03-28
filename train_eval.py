"""
train_eval.py
=============
Unified entry point for training and evaluating Pro-PROTAC.

This script handles the full pipeline:
  1. Data loading and preprocessing
  2. Meta-training the GCN encoder on CRBN episodes
  3. Evaluating on held-out E3 ligase splits (VHL or rare E3 ligases)
  4. Saving per-episode and summary CSV results

Usage
-----
CRBN → VHL (K=2, Q=3):
    python train_eval.py \\
        --csv  protac_filtered_balanced.csv \\
        --sdf  protac.sdf \\
        --split splits/crbn_vhl_K2Q3_seed42.json \\
        --seed 42 --device cuda --outdir results/seed42

CRBN → VHL bootstrap (K=2, Q=5†):
    python train_eval.py \\
        --csv  protac_filtered_balanced.csv \\
        --sdf  protac.sdf \\
        --split splits/crbn_vhl_K2Q5_bootstrap_seed42.json \\
        --meta-q 5 \\
        --seed 42 --device cuda --outdir results/seed42

Rare E3 ligases (one-shot):
    python train_eval.py \\
        --csv  protac_filtered_balanced.csv \\
        --sdf  protac.sdf \\
        --split splits/crbn_rareE3_K1Q1_seed42.json \\
        --meta-k 1 --meta-q 1 \\
        --seed 42 --device cuda --outdir results/seed42

Run all three seeds:
    for seed in 42 2025 3407; do
        python train_eval.py \\
            --csv  protac_filtered_balanced.csv \\
            --sdf  protac.sdf \\
            --split splits/crbn_vhl_K2Q3_seed${seed}.json \\
            --seed ${seed} --device cuda --outdir results/seed${seed}
    done
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from data_utils import build_graph_index, build_label_map
from metrics import mean_ignore_nan
from models import GCNEncoder, ProtoNet


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(
    records: pd.DataFrame,
    seed: int,
    outdir: str,
) -> None:
    """Save per-episode CSV and aggregated summary CSV.

    Args:
        records: DataFrame with one row per episode.
        seed:    Random seed used in this run (included in filenames).
        outdir:  Output directory.
    """
    os.makedirs(outdir, exist_ok=True)

    ep_path  = os.path.join(outdir, f"proprotac_episodes_seed{seed}.csv")
    sum_path = os.path.join(outdir, f"proprotac_summary_seed{seed}.csv")

    records.to_csv(ep_path, index=False)

    metric_cols = ["Accuracy", "Loss", "AUROC", "AUPRC", "F1", "BALACC"]
    summary_row = {"Model": "Pro-PROTAC"}
    for col in metric_cols:
        summary_row[col] = (
            mean_ignore_nan(records[col].tolist())
            if col in records.columns
            else float("nan")
        )

    summary = pd.DataFrame([summary_row])
    summary.to_csv(sum_path, index=False)

    print(f"\nEpisode results  → {ep_path}")
    print(f"Summary results  → {sum_path}")
    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Train and evaluate Pro-PROTAC for cross-ligase PROTAC "
            "degradation activity prediction."
        )
    )

    # ── Data ──────────────────────────────────────────────────────────────
    ap.add_argument(
        "--csv", required=True,
        help="Path to the balanced PROTAC dataset CSV "
             "(e.g. protac_filtered_balanced.csv).",
    )
    ap.add_argument(
        "--sdf", required=True,
        help="Path to the PROTAC SDF file (e.g. protac.sdf).",
    )
    ap.add_argument(
        "--split", required=True,
        help="Path to the episodic split JSON file.",
    )
    ap.add_argument(
        "--phase", default="meta_test",
        choices=["meta_test", "meta_valid", "meta_train"],
        help="Split phase to evaluate (default: meta_test).",
    )

    # ── Output ────────────────────────────────────────────────────────────
    ap.add_argument(
        "--outdir", required=True,
        help="Directory for output CSV files.",
    )
    ap.add_argument(
        "--save-encoder", default="",
        help=(
            "If set, save the trained encoder weights to this path after "
            "meta-training. Useful for downstream UMAP visualization."
        ),
    )

    # ── Reproducibility ───────────────────────────────────────────────────
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    ap.add_argument(
        "--device", default="cuda",
        help="Compute device: 'cuda' or 'cpu' (default: cuda).",
    )

    # ── Encoder architecture ──────────────────────────────────────────────
    ap.add_argument(
        "--encoder-hidden", type=int, default=128,
        help="GCN hidden layer dimension (default: 128).",
    )
    ap.add_argument(
        "--encoder-out", type=int, default=128,
        help="GCN output embedding dimension (default: 128).",
    )
    ap.add_argument(
        "--encoder-layers", type=int, default=3,
        help="Number of GCN message-passing layers (default: 3).",
    )
    ap.add_argument(
        "--encoder-dropout", type=float, default=0.1,
        help="Dropout probability applied between GCN layers (default: 0.1).",
    )

    # ── Meta-training ─────────────────────────────────────────────────────
    ap.add_argument(
        "--meta-epochs", type=int, default=100,
        help="Number of meta-training epochs (default: 100).",
    )
    ap.add_argument(
        "--episodes-per-epoch", type=int, default=100,
        help="Episodes sampled per meta-training epoch (default: 100).",
    )
    ap.add_argument(
        "--meta-k", type=int, default=2,
        help="Support samples per class per episode (default: 2).",
    )
    ap.add_argument(
        "--meta-q", type=int, default=3,
        help="Query samples per class per episode (default: 3).",
    )
    ap.add_argument(
        "--meta-lr", type=float, default=1e-3,
        help="Adam learning rate for meta-training (default: 1e-3).",
    )
    ap.add_argument(
        "--meta-weight-decay", type=float, default=1e-5,
        help="Adam weight decay for meta-training (default: 1e-5).",
    )

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Setup ────────────────────────────────────────────────────────────
    set_seed(args.seed)
    device = torch.device(
        args.device
        if (args.device != "cuda" or torch.cuda.is_available())
        else "cpu"
    )
    print(f"Device : {device}")
    print(f"Seed   : {args.seed}")
    print(f"Split  : {args.split}")
    print(f"Phase  : {args.phase}")

    # ── Load data ────────────────────────────────────────────────────────
    print("\nLoading data ...")
    df = pd.read_csv(args.csv, low_memory=False)
    df["Compound ID"] = df["Compound ID"].astype(str)
    if "Label_bin" not in df.columns and "Label" in df.columns:
        df["Label_bin"] = df["Label"].astype(int)

    label_map   = build_label_map(df)
    graph_index = build_graph_index(args.sdf)
    df = df[df["Compound ID"].isin(graph_index)].copy()
    print(f"{len(df)} samples after SDF alignment")

    # ── Build model ──────────────────────────────────────────────────────
    encoder = GCNEncoder(
        in_dim=9,
        hidden_dim=args.encoder_hidden,
        out_dim=args.encoder_out,
        num_layers=args.encoder_layers,
        dropout=args.encoder_dropout,
    ).to(device)

    model = ProtoNet(encoder)

    # ── Meta-training ────────────────────────────────────────────────────
    print("\nStarting meta-training ...")
    model.meta_train(
        df=df,
        graph_index=graph_index,
        device=device,
        K=args.meta_k,
        Q=args.meta_q,
        epochs=args.meta_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        lr=args.meta_lr,
        weight_decay=args.meta_weight_decay,
        save_path=args.save_encoder if args.save_encoder else None,
    )

    # ── Evaluation ───────────────────────────────────────────────────────
    print("\nEvaluating ...")
    records = model.evaluate(
        split_path=args.split,
        graph_index=graph_index,
        label_map=label_map,
        device=device,
        phase=args.phase,
    )

    if records.empty:
        raise RuntimeError("No valid evaluation records produced.")

    save_results(records, seed=args.seed, outdir=args.outdir)


if __name__ == "__main__":
    main()
