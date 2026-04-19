#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py — ProMeta meta-training entry point

Meta-trains a GCNEncoder + ProteinEncoder on CRBN episodes using
prototypical networks, then saves the encoder checkpoints for evaluation.

Example:
    python train.py \\
        --csv     protac_filtered_balanced.csv \\
        --sdf     protac.sdf \\
        --seq-csv protac_with_seq.csv \\
        --split   splits/crbn_vhl_K2Q3_seed42.json \\
        --seed    42 \\
        --outdir  results/vhl_K2Q3_seed42
"""

import json
import os
import random
import argparse

import torch
import torch.nn as nn
import pandas as pd

from torch_geometric.data import Batch

from models import GCNEncoder, ProteinEncoder
from data import (
    build_graph_index,
    build_protein_index,
    build_label_map,
    build_meta_tasks,
    sample_episode,
    RARE_E3_LIST,
)
from utils import set_seed, mean_ignore_nan


# ── Encoding helpers ──────────────────────────────────────────────────────────

def _get_cids(data_list):
    return [getattr(g, "cid", "") for g in data_list]


def encode_batch(encoder, data_list, device):
    b = Batch.from_data_list(data_list).to(device)
    return encoder(b.x, b.edge_index, b.batch)


def encode_batch_fused(encoder, poi_enc, e3_enc, data_list,
                       poi_index, e3_index, device, out_dim):
    """z = z_mol + alpha_poi * z_poi + alpha_e3 * z_e3"""
    b     = Batch.from_data_list(data_list).to(device)
    z_mol = encoder(b.x, b.edge_index, b.batch)
    cids  = _get_cids(data_list)
    z_poi = poi_enc.encode_batch_ids(cids, poi_index, device, out_dim)
    z_e3  = e3_enc.encode_batch_ids(cids,  e3_index,  device, out_dim)
    return z_mol + poi_enc.alpha * z_poi + e3_enc.alpha * z_e3


def make_encode_fn(encoder, poi_enc, e3_enc, poi_index, e3_index, device, out_dim):
    if poi_index is not None:
        return lambda dl: encode_batch_fused(
            encoder, poi_enc, e3_enc, dl, poi_index, e3_index, device, out_dim
        )
    return lambda dl: encode_batch(encoder, dl, device)


# ── Meta-training ─────────────────────────────────────────────────────────────

def meta_train(
    encoder, poi_enc, e3_enc,
    meta_tasks, graph_index, poi_index, e3_index,
    device, epochs, episodes_per_epoch, K, Q,
    lr, weight_decay, out_dim,
):
    print("🚀 Meta-training ProMeta ...")
    params = list(encoder.parameters())
    if poi_enc is not None and poi_index is not None:
        params += list(poi_enc.parameters())
    if e3_enc is not None and e3_index is not None:
        params += list(e3_enc.parameters())

    opt        = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    bce        = nn.BCEWithLogitsLoss()
    task_names = list(meta_tasks.keys())
    encode_fn  = make_encode_fn(encoder, poi_enc, e3_enc,
                                poi_index, e3_index, device, out_dim)

    for epoch in range(1, epochs + 1):
        encoder.train()
        if poi_enc  is not None: poi_enc.train()
        if e3_enc   is not None: e3_enc.train()

        losses = []
        for _ in range(episodes_per_epoch):
            ep = sample_episode(
                meta_tasks[random.choice(task_names)],
                graph_index, K, Q, device,
            )
            if ep is None:
                continue
            sup_data, sup_labels, qry_data, qry_labels = ep

            z_s     = encode_fn(sup_data)
            z_q     = encode_fn(qry_data)
            pp      = z_s[sup_labels == 1].mean(0, keepdim=True)
            pn      = z_s[sup_labels == 0].mean(0, keepdim=True)
            logits  = -(torch.norm(z_q - pp, dim=1) - torch.norm(z_q - pn, dim=1))
            loss    = bce(logits, qry_labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"Epoch {epoch:03d} | loss = {mean_ignore_nan(losses):.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="ProMeta — meta-training")

    # Data
    ap.add_argument("--csv",     required=True, help="Preprocessed PROTAC CSV")
    ap.add_argument("--sdf",     required=True, help="PROTAC SDF file")
    ap.add_argument("--split",   required=True, help="Episodic split JSON")
    ap.add_argument("--seq-csv", default="",
                    help="CSV with POI_seq and E3_seq columns "
                         "(leave empty to disable protein fusion)")
    ap.add_argument("--seq-col",    default="POI_seq")
    ap.add_argument("--e3-seq-col", default="E3_seq")
    ap.add_argument("--exclude-e3", nargs="+", default=["VHL"],
                    help="E3 ligase(s) to hold out. Use 'rare' for all rare E3s.")

    # Output
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--device", default="cuda")

    # Encoder architecture
    ap.add_argument("--encoder-hidden",  type=int,   default=128)
    ap.add_argument("--encoder-out",     type=int,   default=128)
    ap.add_argument("--encoder-layers",  type=int,   default=3)
    ap.add_argument("--encoder-dropout", type=float, default=0.1)

    # Protein encoder
    ap.add_argument("--prot-embed-dim", type=int, default=64)
    ap.add_argument("--prot-max-len",   type=int, default=2000)

    # Meta-training
    ap.add_argument("--meta-epochs",        type=int,   default=100)
    ap.add_argument("--episodes-per-epoch", type=int,   default=100)
    ap.add_argument("--meta-k",             type=int,   default=2)
    ap.add_argument("--meta-q",             type=int,   default=3)
    ap.add_argument("--meta-lr",            type=float, default=1e-3)
    ap.add_argument("--meta-weight-decay",  type=float, default=1e-5)

    return ap.parse_args()


def main():
    args   = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device  = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    out_dim = args.encoder_out
    print(f"🖥  device={device} | K={args.meta_k} Q={args.meta_q} | seed={args.seed}")

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv, low_memory=False)
    df["Compound ID"] = df["Compound ID"].astype(str)
    if "Label_bin" not in df.columns and "Label" in df.columns:
        df["Label_bin"] = df["Label"].astype(int)

    graph_index = build_graph_index(args.sdf)
    df          = df[df["Compound ID"].isin(graph_index)].copy()
    print(f"✅ {len(df)} samples after SDF alignment")

    # ── Protein index (optional) ───────────────────────────────────────────────
    poi_index, e3_index, poi_enc, e3_enc = None, None, None, None
    if args.seq_csv:
        poi_index = build_protein_index(args.seq_csv, args.seq_col,  args.prot_max_len)
        e3_index  = build_protein_index(args.seq_csv, args.e3_seq_col, args.prot_max_len)
        poi_enc   = ProteinEncoder(args.prot_embed_dim, out_dim).to(device)
        e3_enc    = ProteinEncoder(args.prot_embed_dim, out_dim).to(device)
        print(f"✅ ProteinEncoder ready | embed_dim={args.prot_embed_dim} out_dim={out_dim}")
    else:
        print("ℹ️  --seq-csv not provided → protein fusion disabled")

    # ── Build meta-train tasks ─────────────────────────────────────────────────
    exclude = RARE_E3_LIST if args.exclude_e3 == ["rare"] else args.exclude_e3
    meta_tasks = build_meta_tasks(df, graph_index, args.meta_k, args.meta_q,
                                  exclude_e3=exclude)
    print(f"🧩 Meta-train tasks: {len(meta_tasks)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    encoder = GCNEncoder(
        in_dim=9,
        hidden_dim=args.encoder_hidden,
        out_dim=out_dim,
        num_layers=args.encoder_layers,
        dropout=args.encoder_dropout,
    ).to(device)

    # ── Meta-train ─────────────────────────────────────────────────────────────
    meta_train(
        encoder, poi_enc, e3_enc,
        meta_tasks, graph_index, poi_index, e3_index,
        device=device,
        epochs=args.meta_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        K=args.meta_k,
        Q=args.meta_q,
        lr=args.meta_lr,
        weight_decay=args.meta_weight_decay,
        out_dim=out_dim,
    )

    # ── Save checkpoints ───────────────────────────────────────────────────────
    enc_path = os.path.join(args.outdir, f"encoder_seed{args.seed}.pt")
    torch.save(encoder.state_dict(), enc_path)
    print(f"💾 Encoder saved → {enc_path}")

    if poi_enc is not None:
        poi_path = os.path.join(args.outdir, f"poi_encoder_seed{args.seed}.pt")
        e3_path  = os.path.join(args.outdir, f"e3_encoder_seed{args.seed}.pt")
        torch.save(poi_enc.state_dict(), poi_path)
        torch.save(e3_enc.state_dict(),  e3_path)
        print(f"💾 POI encoder saved → {poi_path}")
        print(f"💾 E3  encoder saved → {e3_path}")
        print(f"🔬 alpha_poi = {poi_enc.alpha.item():.4f} | "
              f"alpha_e3 = {e3_enc.alpha.item():.4f}")


if __name__ == "__main__":
    main()
