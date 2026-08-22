#!/usr/bin/env python3
"""Train episodic ProMeta or its non-episodic control on released data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prometa.data import graph_index_from_cache, protein_index_from_csv, read_contexts
from prometa.training import (
    train_episodic,
    train_episodic_sequence_aware,
    train_non_episodic,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts",
        type=Path,
        default=ROOT / "data" / "strict_crbn_vhl_contexts.csv",
    )
    parser.add_argument("--graph-cache", type=Path, default=ROOT / "data" / "prometa_graphs.npz")
    parser.add_argument("--source-e3", choices=["CRBN", "VHL"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--train-q", type=int, default=3)
    parser.add_argument("--mode", choices=["episodic", "non_episodic"], default="episodic")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Limit PyTorch intra-op CPU threads; useful for parallel reproducibility runs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sequence-csv",
        type=Path,
        default=ROOT / "data" / "protac_with_seq.csv",
        help="Compound-linked POI/E3 sequences used by the reported episodic model.",
    )
    parser.add_argument("--poi-sequence-column", default="POI_seq")
    parser.add_argument("--e3-sequence-column", default="E3_seq")
    parser.add_argument("--protein-max-length", type=int, default=2000)
    parser.add_argument("--protein-embed-dim", type=int, default=64)
    parser.add_argument(
        "--without-protein-sequences",
        action="store_true",
        help="Run the explicitly labeled GCN-only ablation/audit path.",
    )
    args = parser.parse_args()

    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be at least 1")
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)

    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    contexts = read_contexts(args.contexts)
    graph_index = graph_index_from_cache(args.graph_cache)
    poi_encoder = None
    e3_encoder = None
    if args.mode == "episodic" and not args.without_protein_sequences:
        poi_index = protein_index_from_csv(
            args.sequence_csv,
            args.poi_sequence_column,
            max_length=args.protein_max_length,
        )
        e3_index = protein_index_from_csv(
            args.sequence_csv,
            args.e3_sequence_column,
            max_length=args.protein_max_length,
        )
        encoder, poi_encoder, e3_encoder, audit = train_episodic_sequence_aware(
            contexts,
            graph_index,
            poi_index,
            e3_index,
            source_e3=args.source_e3,
            k_shot=args.k,
            q_query=args.train_q,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            episodes_per_epoch=args.episodes_per_epoch,
            protein_embed_dim=args.protein_embed_dim,
        )
    elif args.mode == "episodic":
        encoder, audit = train_episodic(
            contexts,
            graph_index,
            source_e3=args.source_e3,
            k_shot=args.k,
            q_query=args.train_q,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            episodes_per_epoch=args.episodes_per_epoch,
        )
    else:
        encoder, audit = train_non_episodic(
            contexts,
            graph_index,
            source_e3=args.source_e3,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit["cpu_threads"] = args.cpu_threads
    torch.save(encoder.state_dict(), args.output)
    if poi_encoder is not None and e3_encoder is not None:
        poi_output = args.output.with_name(f"{args.output.stem}_poi{args.output.suffix}")
        e3_output = args.output.with_name(f"{args.output.stem}_e3{args.output.suffix}")
        torch.save(poi_encoder.state_dict(), poi_output)
        torch.save(e3_encoder.state_dict(), e3_output)
        audit["poi_checkpoint"] = str(poi_output)
        audit["e3_checkpoint"] = str(e3_output)
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
