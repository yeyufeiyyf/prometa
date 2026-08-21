#!/usr/bin/env python3
"""Run the retained main seed-42 POI/E3 sequence-aware inference path."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "recomputed" / "sequence_aware_seed42",
    )
    args = parser.parse_args()

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
    checkpoint_root = ROOT / "checkpoints" / "sequence_aware" / "main_seed42"
    gcn_checkpoint = checkpoint_root / "protonet_encoder_seed42.pt"
    poi_checkpoint = checkpoint_root / "protonet_poi_encoder_seed42.pt"
    e3_checkpoint = checkpoint_root / "protonet_e3_encoder_seed42.pt"
    registry = pd.read_csv(ROOT / "experiments" / "response_letter_experiments.csv")
    registry = registry[registry["experiment_id"].isin(EXPERIMENTS)]

    prediction_frames: list[pd.DataFrame] = []
    for _, row in registry.iterrows():
        experiment_id = str(row["experiment_id"])
        split_path = (
            ROOT
            / "splits"
            / "by_experiment"
            / experiment_id
            / f"{experiment_id}_seed42.json"
        )
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
                "seed": 42,
            },
            poi_checkpoint=poi_checkpoint,
            e3_checkpoint=e3_checkpoint,
            poi_index=poi_index,
            e3_index=e3_index,
        )
        predictions["experiment_id"] = experiment_id
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
