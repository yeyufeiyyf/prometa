#!/usr/bin/env python3
"""Train episodic ProMeta or its non-episodic control on released data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prometa.data import graph_index_from_cache, read_contexts
from prometa.training import train_episodic, train_non_episodic


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    contexts = read_contexts(args.contexts)
    graph_index = graph_index_from_cache(args.graph_cache)
    if args.mode == "episodic":
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
    torch.save(encoder.state_dict(), args.output)
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
