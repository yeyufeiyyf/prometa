#!/usr/bin/env python3
"""Run retained or reproducibly retrained POI/E3 sequence-aware inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from prometa.data import graph_index_from_cache, protein_index_from_csv, read_contexts
from prometa.evaluation import evaluate_split, split_metrics, summarize_metrics


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    "s3_crbn_vhl_k2q3",
    "s3_crbn_vhl_k2q5",
    "s3_vhl_crbn_k2q3",
    "s3_vhl_crbn_k2q5",
)


def checkpoint_triplet(
    family: str,
    source_e3: str,
    seed: int,
) -> tuple[Path, Path, Path]:
    """Resolve a matched GCN/POI/E3 triplet from one training run."""
    if family == "retained_seed42":
        if seed != 42:
            raise ValueError("The retained main sequence-aware family contains only seed 42")
        root = ROOT / "checkpoints" / "sequence_aware" / "main_seed42"
        return (
            root / "protonet_encoder_seed42.pt",
            root / "protonet_poi_encoder_seed42.pt",
            root / "protonet_e3_encoder_seed42.pt",
        )
    root = (
        ROOT
        / "checkpoints"
        / "sequence_aware"
        / "retrained_main_2026"
        / f"{source_e3.lower()}_source"
    )
    stem = f"prometa_{source_e3.lower()}_seed{seed}"
    return (
        root / f"{stem}.pt",
        root / f"{stem}_poi.pt",
        root / f"{stem}_e3.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--family",
        choices=["retained_seed42", "retrained_main_2026"],
        default="retained_seed42",
        help=(
            "Use the retained original seed-42 triplet or the direction-specific "
            "three-seed reproducibly retrained family."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.output_root is None:
        output_name = (
            "sequence_aware_seed42"
            if args.family == "retained_seed42"
            else "sequence_aware_retrained_main_2026"
        )
        args.output_root = ROOT / "results" / "recomputed" / output_name

    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    contexts = read_contexts(ROOT / "data" / "strict_crbn_vhl_contexts.csv")
    graph_index = graph_index_from_cache(ROOT / "data" / "prometa_graphs.npz")
    sequence_csv = ROOT / "data" / "protac_with_seq.csv"
    poi_index = protein_index_from_csv(sequence_csv, "POI_seq")
    e3_index = protein_index_from_csv(sequence_csv, "E3_seq")
    registry = pd.read_csv(ROOT / "experiments" / "response_letter_experiments.csv")
    registry = registry[registry["experiment_id"].isin(EXPERIMENTS)]
    seeds = (42,) if args.family == "retained_seed42" else (42, 2025, 3407)

    prediction_frames: list[pd.DataFrame] = []
    for _, row in registry.iterrows():
        experiment_id = str(row["experiment_id"])
        for seed in seeds:
            split_path = (
                ROOT
                / "splits"
                / "by_experiment"
                / experiment_id
                / f"{experiment_id}_seed{seed}.json"
            )
            gcn_checkpoint, poi_checkpoint, e3_checkpoint = checkpoint_triplet(
                args.family, str(row["source"]), seed
            )
            for checkpoint in (gcn_checkpoint, poi_checkpoint, e3_checkpoint):
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
            predictions = evaluate_split(
                split_path=split_path,
                checkpoint=gcn_checkpoint,
                graph_index=graph_index,
                contexts=contexts,
                device=device,
                head="prototype",
                fingerprint_bits=int(row["fingerprint_bits"]),
                fingerprint_weight=float(row["fingerprint_weight"]),
                metadata_overrides={
                    "direction": row["direction"],
                    "source": row["source"],
                    "target": row["target"],
                    "k_shot": int(row["k_shot"]),
                    "q_query": int(row["q_query"]),
                    "query_setting": row["query_setting"],
                    "seed": seed,
                },
                poi_checkpoint=poi_checkpoint,
                e3_checkpoint=e3_checkpoint,
                poi_index=poi_index,
                e3_index=e3_index,
            )
            predictions["experiment_id"] = experiment_id
            predictions["checkpoint_family"] = args.family
            predictions["gcn_checkpoint"] = str(gcn_checkpoint.relative_to(ROOT))
            predictions["poi_checkpoint"] = str(poi_checkpoint.relative_to(ROOT))
            predictions["e3_checkpoint"] = str(e3_checkpoint.relative_to(ROOT))
            prediction_frames.append(predictions)

    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = split_metrics(predictions)
    summary = summarize_metrics(metrics)
    predictions.to_csv(args.output_root / "query_predictions.csv", index=False)
    metrics.to_csv(args.output_root / "split_metrics.csv", index=False)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    print(metrics[["direction", "k_shot", "q_query", "seed", "AUROC", "AUPRC"]].to_string(index=False))


if __name__ == "__main__":
    main()
